"""Read side of the metrics system: turns `metrics_logs` rows (written by
app.pipeline.metrics) into the shapes the API and dashboard return.

Everything here is plain SQL that runs on Postgres and SQLite (the tests
use SQLite) plus Python aggregation — no Postgres-only functions.

Vocabulary:
  tokens_in / tokens_out  Ollama's prompt_eval_count / eval_count, summed over LLM calls
  est_cost                tokens * per-1K rate (see settings.token_rate_*; a reference price for local models)
  total_ms                wall time of the whole run (`pipeline` row)
  llm_ms / tool_ms / db_ms / embed_ms   time inside those calls; other_ms = node time none of them explain
  network_ms              wall time of LLM calls minus Ollama's own server-side time
                          (HTTP transport + JSON + queueing overhead)
"""
import csv
import io
import json
from collections import defaultdict
from statistics import mean
from typing import Any, Optional

from sqlalchemy import bindparam, text

from app import db
from app.config import settings
from app.pipeline.metrics import estimate_cost

CSV_COLUMNS = [
    "project_id", "name", "status", "communication_mode", "model", "framework", "task_types", "tasks", "tasks_passed",
    "pass_rate", "attempts", "avg_attempts", "escalations", "repeated_errors", "run_ok", "ifs_avg", "ifs_strict_avg", "ifs_planner", "memory_hits",
    "tokens_in", "tokens_out",
    "tokens_total", "est_cost_usd", "llm_calls", "total_ms", "llm_ms", "created_at",
]


def _meta(v: Any) -> dict:
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v:
        try:
            m = json.loads(v)
            return m if isinstance(m, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _r(x: Optional[float], n: int = 2) -> Optional[float]:
    return None if x is None else round(x, n)


def _avg(xs: list[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


# ------------------------------------------------------- one project --------

def _ifs_row(stage: str, task_order: Optional[int], attempt: Optional[int], m: dict) -> dict:
    """One recorded IFS measurement (see app.ifs.record_ifs), trimmed for the API."""
    return {"stage": stage, "task_order": task_order, "attempt": attempt, "scorable": bool(m.get("scorable")),
            **{k: m.get(k) for k in ("entities", "preserved", "score", "strict_preserved", "strict_score",
                                     "missing", "missing_strict", "communication_mode")}}


def _ifs_summary(rows: list[dict]) -> dict:
    """Per-stage view of a run's IFS records: every record, plus the Planner score and, per
    task, the Developer's LAST attempt (the version QA finally judged)."""
    scorable = [r for r in rows if r["scorable"]]
    planner = [r for r in scorable if r["stage"] == "planner"]
    last_dev: dict[Any, dict] = {}
    for r in scorable:
        if r["stage"] == "developer":
            if r["task_order"] not in last_dev or (r["attempt"] or 0) >= (last_dev[r["task_order"]]["attempt"] or 0):
                last_dev[r["task_order"]] = r
    dev = [last_dev[k] for k in sorted(last_dev, key=lambda x: (x is None, x))]
    return {
        "records": rows,
        "planner": {"score": _r(_avg([r["score"] for r in planner]), 3),
                    "strict_score": _r(_avg([r["strict_score"] for r in planner]), 3),
                    "missing": sorted({e for r in planner for e in (r["missing"] or [])})} if planner else None,
        "developer": {"score": _r(_avg([r["score"] for r in dev]), 3),
                      "strict_score": _r(_avg([r["strict_score"] for r in dev]), 3),
                      "by_task": dev} if dev else None,
        "note": "score = substring matcher (as in SE-Blackboard); strict_score = whole-token / difflib matcher. "
                "None = the request has no extractable technical entities, so IFS is undefined for it.",
    }


def ifs_records(project_id: str) -> Optional[dict]:
    """Just the IFS measurements of one project (GET /projects/{id}/ifs)."""
    with db.engine.connect() as conn:
        rows = conn.execute(
            text("SELECT name, task_order, attempt, meta FROM metrics_logs WHERE project_id = :p AND kind = 'ifs' "
                 "ORDER BY t_start_ms, attempt"), {"p": project_id}).fetchall()
    if not rows:
        return None
    return {"project_id": project_id, **_ifs_summary([_ifs_row(r.name, r.task_order, r.attempt, _meta(r.meta)) for r in rows])}


def project_metrics(project_id: str) -> Optional[dict]:
    """Full metrics for one project: tokens & cost per agent, latency
    breakdown, per-node timeline (flamegraph data), tool usage, tasks."""
    with db.engine.connect() as conn:
        rows = conn.execute(
            text("SELECT kind, agent, task_order, attempt, name, tokens_in, tokens_out, duration_ms, "
                 "t_start_ms, ok, meta FROM metrics_logs WHERE project_id = :p ORDER BY t_start_ms, kind"),
            {"p": project_id},
        ).fetchall()
        proj = conn.execute(
            text("SELECT name, request, status, created_at FROM projects WHERE id = :p"), {"p": project_id}
        ).fetchone()
    if not rows:
        return None

    t_zero = min((r.t_start_ms for r in rows if r.t_start_ms is not None), default=0)
    agents: dict[str, dict] = defaultdict(lambda: {"calls": 0, "tokens_in": 0, "tokens_out": 0, "llm_ms": 0, "est_cost": 0.0})
    tools: dict[str, dict] = defaultdict(lambda: {"calls": 0, "total_ms": 0, "errors": 0})
    nodes, tasks, llm_calls = [], [], []
    lat = {"llm_ms": 0, "tool_ms": 0, "db_ms": 0, "embed_ms": 0, "other_ms": 0, "network_ms": 0,
           "ollama_compute_ms": 0, "model_load_ms": 0}
    embed_tokens = 0
    paused_ms = 0
    pipeline: dict = {}
    model = None
    ifs_rows: list[dict] = []

    for r in rows:
        m = _meta(r.meta)
        if r.kind == "llm":
            a = agents[r.agent or "?"]
            a["calls"] += 1
            a["tokens_in"] += r.tokens_in
            a["tokens_out"] += r.tokens_out
            a["llm_ms"] += r.duration_ms
            a["est_cost"] += estimate_cost(r.tokens_in, r.tokens_out)
            lat["network_ms"] += m.get("network_ms") or 0
            lat["ollama_compute_ms"] += (m.get("ollama_ms") or 0) - (m.get("load_ms") or 0)
            lat["model_load_ms"] += m.get("load_ms") or 0
            model = model or r.name
            llm_calls.append({
                "agent": r.agent, "task_order": r.task_order, "attempt": r.attempt, "model": r.name,
                "tokens_in": r.tokens_in, "tokens_out": r.tokens_out, "duration_ms": r.duration_ms,
                "tok_per_s": m.get("tok_per_s"), "estimated": bool(m.get("estimated")), "ok": bool(r.ok),
                "start_ms": (r.t_start_ms or t_zero) - t_zero,
            })
        elif r.kind == "embed":
            embed_tokens += r.tokens_in
        elif r.kind == "pause":
            paused_ms += r.duration_ms   # time the run waited for a human (excluded from total_ms)
        elif r.kind == "tool":
            t = tools[r.name or "?"]
            t["calls"] += 1
            t["total_ms"] += r.duration_ms
            t["errors"] += 0 if r.ok else 1
        elif r.kind == "node":
            for k in ("llm_ms", "tool_ms", "db_ms", "embed_ms", "other_ms"):
                lat[k] += m.get(k) or 0
            nodes.append({
                "node": r.name, "task_order": r.task_order, "attempt": r.attempt,
                "start_ms": (r.t_start_ms or t_zero) - t_zero, "duration_ms": r.duration_ms,
                "llm_ms": m.get("llm_ms", 0), "tool_ms": m.get("tool_ms", 0), "db_ms": m.get("db_ms", 0),
                "embed_ms": m.get("embed_ms", 0), "other_ms": m.get("other_ms", 0),
                "llm_calls": m.get("llm_calls", 0), "tokens_in": r.tokens_in, "tokens_out": r.tokens_out,
                "ok": bool(r.ok),
            })
        elif r.kind == "task":
            tasks.append({
                "order": r.task_order, "type": r.name, "passed": bool(r.ok), "attempts": r.attempt,
                "escalated": bool(m.get("escalated")), "failures": m.get("failures", 0),
                "repeated_errors": m.get("repeated_errors", 0), "timeouts": m.get("timeouts", 0),
                "categories": m.get("categories", []), "used_similar_fix": bool(m.get("used_similar_fix")),
            })
        elif r.kind == "pipeline":
            pipeline = {"mode": r.name, "total_ms": r.duration_ms, "ok": bool(r.ok), "model": m.get("model"),
                        "framework": m.get("framework")}
        elif r.kind == "ifs":
            ifs_rows.append(_ifs_row(r.name, r.task_order, r.attempt, m))

    model = pipeline.get("model") or model
    tokens_in = sum(a["tokens_in"] for a in agents.values())
    tokens_out = sum(a["tokens_out"] for a in agents.values())
    node_total = sum(n["duration_ms"] for n in nodes)
    if pipeline.get("total_ms"):
        total_ms = pipeline["total_ms"]
    else:
        ends = [(r.t_start_ms or 0) + r.duration_ms for r in rows if r.t_start_ms]
        total_ms = (max(ends) - t_zero) if ends else node_total

    for a in agents.values():
        a["est_cost"] = _r(a["est_cost"], 6)
    by_node: dict[str, dict] = defaultdict(lambda: {"executions": 0, "total_ms": 0})
    for n in nodes:
        b = by_node[n["node"]]
        b["executions"] += 1
        b["total_ms"] += n["duration_ms"]
    for b in by_node.values():
        b["avg_ms"] = round(b["total_ms"] / b["executions"])
    for t in tools.values():
        t["avg_ms"] = round(t["total_ms"] / t["calls"])

    done = len(tasks)
    passed = sum(1 for t in tasks if t["passed"])
    return {
        "project_id": project_id,
        "name": proj.name if proj else None,
        "request": proj.request if proj else None,
        "status": proj.status if proj else None,
        "created_at": str(proj.created_at) if proj else None,
        "communication_mode": pipeline.get("mode"),
        "model": model,
        "framework": pipeline.get("framework"),
        "tokens": {
            "tokens_in": tokens_in, "tokens_out": tokens_out, "total": tokens_in + tokens_out,
            "embed_tokens_est": embed_tokens, "llm_calls": len(llm_calls),
            "by_agent": {k: v for k, v in sorted(agents.items())},
        },
        "cost": {
            "est_cost_usd": _r(estimate_cost(tokens_in, tokens_out), 6),
            "rate_in_per_1k": settings.token_rate_in_per_1k, "rate_out_per_1k": settings.token_rate_out_per_1k,
            "note": "tokens x per-1K rate; local Ollama has no API bill, so this is a cloud-equivalent reference price",
        },
        "latency": {"total_ms": total_ms, "node_total_ms": node_total, "paused_ms": paused_ms,
                    **{k: int(v) for k, v in lat.items()},
                    "by_node": dict(by_node)},
        "tools": dict(tools),
        "nodes": nodes,
        "llm_calls": llm_calls[:300],
        "tasks": tasks,
        "ifs": _ifs_summary(ifs_rows),
        "success": {
            "tasks": done, "tasks_passed": passed, "pass_rate": _r(passed / done, 3) if done else None,
            "avg_attempts": _r(_avg([t["attempts"] for t in tasks]), 2),
            "escalations": sum(1 for t in tasks if t["escalated"]),
            "repeated_errors": sum(t["repeated_errors"] for t in tasks),
            "run_ok": pipeline.get("ok"),
        },
    }


# ------------------------------------------------------------ many projects -

_LIST_SQL = """
SELECT p.id AS id, p.name AS name, p.request AS request, p.status AS status, p.created_at AS created_at,
  SUM(CASE WHEN m.kind = 'llm' THEN m.tokens_in ELSE 0 END)  AS tokens_in,
  SUM(CASE WHEN m.kind = 'llm' THEN m.tokens_out ELSE 0 END) AS tokens_out,
  SUM(CASE WHEN m.kind = 'llm' THEN m.duration_ms ELSE 0 END) AS llm_ms,
  SUM(CASE WHEN m.kind = 'llm' THEN 1 ELSE 0 END)            AS llm_calls,
  MAX(CASE WHEN m.kind = 'pipeline' THEN m.duration_ms END)  AS total_ms,
  SUM(CASE WHEN m.kind = 'task' THEN 1 ELSE 0 END)           AS tasks,
  SUM(CASE WHEN m.kind = 'task' AND m.ok THEN 1 ELSE 0 END)  AS tasks_passed,
  SUM(CASE WHEN m.kind = 'task' THEN m.attempt ELSE 0 END)   AS attempts
FROM projects p JOIN metrics_logs m ON m.project_id = p.id
{where}GROUP BY p.id, p.name, p.request, p.status, p.created_at
ORDER BY p.created_at DESC
LIMIT :limit OFFSET :offset
"""

_DETAIL_SQL = text(
    "SELECT project_id, kind, name, ok, meta FROM metrics_logs "
    "WHERE kind IN ('pipeline', 'task', 'ifs') AND project_id IN :ids"
).bindparams(bindparam("ids", expanding=True))


def list_projects(limit: int = 100, offset: int = 0, status: Optional[str] = None,
                  mode: Optional[str] = None, ids: Optional[list[str]] = None) -> list[dict]:
    """One summary row per project that has metrics, newest first.
    `ids` restricts it to those projects (used for a batch's results)."""
    if ids is not None and not ids:
        return []
    with db.engine.connect() as conn:
        if ids is not None:
            stmt = text(_LIST_SQL.format(where="WHERE p.id IN :ids\n")).bindparams(bindparam("ids", expanding=True))
            base = conn.execute(stmt, {"limit": max(limit, len(ids)), "offset": offset, "ids": list(ids)}).fetchall()
        else:
            base = conn.execute(text(_LIST_SQL.format(where="")), {"limit": limit, "offset": offset}).fetchall()
        ids = [str(r.id) for r in base]
        detail = conn.execute(_DETAIL_SQL, {"ids": ids}).fetchall() if ids else []

    per: dict[str, dict] = defaultdict(lambda: {"mode": None, "model": None, "framework": None, "types": set(),
                                                "esc": 0, "rep": 0, "run_ok": None, "ifs": [], "ifs_strict": [],
                                                "ifs_planner": [], "mem": 0})
    for d in detail:
        m, p = _meta(d.meta), per[str(d.project_id)]
        if d.kind == "pipeline":
            p["mode"], p["model"], p["framework"] = d.name, m.get("model"), m.get("framework")
            p["run_ok"] = bool(d.ok)
        elif d.kind == "ifs":
            if d.name == "planner" and m.get("scorable"):
                p["ifs_planner"].append(float(m["score"]))
        else:
            p["types"].add(d.name)
            p["esc"] += 1 if m.get("escalated") else 0
            p["rep"] += m.get("repeated_errors", 0)
            if m.get("ifs") is not None:
                p["ifs"].append(float(m["ifs"]))
            if m.get("ifs_strict") is not None:
                p["ifs_strict"].append(float(m["ifs_strict"]))
            p["mem"] += 1 if m.get("memory_used") else 0

    out = []
    for r in base:
        p = per[str(r.id)]
        tin, tout = int(r.tokens_in or 0), int(r.tokens_out or 0)
        tasks = int(r.tasks or 0)
        row = {
            "project_id": str(r.id), "name": r.name, "request": r.request, "status": r.status,
            "created_at": str(r.created_at), "communication_mode": p["mode"], "model": p["model"],
            "framework": p["framework"], "task_types": ",".join(sorted(p["types"])), "tasks": tasks, "tasks_passed": int(r.tasks_passed or 0),
            "pass_rate": _r(int(r.tasks_passed or 0) / tasks, 3) if tasks else None,
            "attempts": int(r.attempts or 0), "avg_attempts": _r(int(r.attempts or 0) / tasks, 2) if tasks else None,
            "escalations": p["esc"], "repeated_errors": p["rep"], "run_ok": p["run_ok"],
            "ifs_avg": _r(_avg(p["ifs"]), 3), "ifs_strict_avg": _r(_avg(p["ifs_strict"]), 3),
            "ifs_planner": _r(_avg(p["ifs_planner"]), 3), "memory_hits": p["mem"],
            "tokens_in": tin, "tokens_out": tout, "tokens_total": tin + tout,
            "est_cost_usd": _r(estimate_cost(tin, tout), 6), "llm_calls": int(r.llm_calls or 0),
            "total_ms": int(r.total_ms) if r.total_ms is not None else None, "llm_ms": int(r.llm_ms or 0),
        }
        if status and row["status"] != status:
            continue
        if mode and row["communication_mode"] != mode:
            continue
        out.append(row)
    return out


def compare(project_ids: list[str]) -> dict:
    """Side-by-side metrics for 2+ projects, with the change from the first
    (baseline) to each other project."""
    projects = [p for p in (project_metrics(i) for i in project_ids) if p]
    if len(projects) < 2:
        return {"projects": projects, "deltas": []}

    def pick(p: dict) -> dict:
        return {
            "tokens_total": p["tokens"]["total"], "est_cost_usd": p["cost"]["est_cost_usd"],
            "total_ms": p["latency"]["total_ms"], "llm_ms": p["latency"]["llm_ms"],
            "tool_ms": p["latency"]["tool_ms"], "attempts": p["success"]["avg_attempts"],
            "pass_rate": p["success"]["pass_rate"],
        }

    base = pick(projects[0])
    deltas = []
    for p in projects[1:]:
        cur = pick(p)
        deltas.append({
            "project_id": p["project_id"],
            **{k: {"baseline": base[k], "value": cur[k],
                   "pct_change": _r((cur[k] - base[k]) / base[k] * 100, 1) if base[k] not in (None, 0) and cur[k] is not None else None}
               for k in base},
        })
    return {"projects": projects, "deltas": deltas}


# ---------------------------------------------------------- success metrics -

def _task_rows(limit: int) -> list:
    with db.engine.connect() as conn:
        return conn.execute(
            text("SELECT project_id, name, ok, attempt, meta, t_start_ms FROM metrics_logs "
                 "WHERE kind = 'task' ORDER BY t_start_ms DESC LIMIT :n"), {"n": limit}
        ).fetchall()


def _group_stats(rows: list) -> dict:
    n = len(rows)
    if not n:
        return {"tasks": 0}
    metas = [_meta(r.meta) for r in rows]
    passed = [r for r in rows if r.ok]
    failures = sum(m.get("failures", 0) for m in metas)
    return {
        "tasks": n, "pass_rate": _r(len(passed) / n, 3),
        "avg_attempts": _r(_avg([r.attempt for r in rows]), 2),
        "avg_attempts_to_pass": _r(_avg([r.attempt for r in passed]), 2),
        "escalation_rate": _r(sum(1 for m in metas if m.get("escalated")) / n, 3),
        "timeout_rate": _r(sum(1 for m in metas if m.get("timeouts")) / n, 3),
        "repeat_error_rate": _r(sum(m.get("repeated_errors", 0) for m in metas) / failures, 3) if failures else None,
    }


def success_summary(limit: int = 500) -> dict:
    """Pass rate / attempts / timeout & escalation rate by task type and by
    model, plus the Failure Recovery effect (same-error repeats; attempts
    with vs without a similar past fix from memory)."""
    rows = _task_rows(limit)
    by_type, by_model, by_framework = defaultdict(list), defaultdict(list), defaultdict(list)
    with_fix, without_fix = [], []
    for r in rows:
        m = _meta(r.meta)
        by_type[r.name or "?"].append(r)
        by_model[m.get("model") or "?"].append(r)
        by_framework[m.get("framework") or "none"].append(r)
        if m.get("failures"):
            (with_fix if m.get("used_similar_fix") else without_fix).append(r)
    return {
        "overall": _group_stats(rows),
        "by_task_type": {k: _group_stats(v) for k, v in sorted(by_type.items())},
        "by_model": {k: _group_stats(v) for k, v in sorted(by_model.items())},
        # pass rate / attempts / timeouts per web framework ("none" = plain code tasks)
        "by_framework": {k: _group_stats(v) for k, v in sorted(by_framework.items())},
        "failure_recovery": {
            "tasks_with_failures": len(with_fix) + len(without_fix),
            "avg_attempts_with_similar_fix": _r(_avg([r.attempt for r in with_fix]), 2),
            "avg_attempts_without_similar_fix": _r(_avg([r.attempt for r in without_fix]), 2),
        },
        # tasks by number of attempts used (1 = passed first time)
        "attempt_histogram": {str(a): sum(1 for r in rows if r.attempt == a)
                              for a in sorted({r.attempt for r in rows})},
    }


# ------------------------------------------------------------ model stats ---

def _llm_rows(model: Optional[str] = None, limit: int = 5000) -> list:
    sql = ("SELECT project_id, name, tokens_out, duration_ms, ok, meta, t_start_ms FROM metrics_logs "
           "WHERE kind = 'llm'")
    params: dict[str, Any] = {"n": limit}
    if model:
        sql += " AND name = :model"
        params["model"] = model
    with db.engine.connect() as conn:
        return conn.execute(text(sql + " ORDER BY t_start_ms DESC LIMIT :n"), params).fetchall()


def model_stats(slow_factor: float = 1.25) -> dict:
    """Per-model speed and quality, to answer 'which model performs best?'
    and to spot a model that is slower than another."""
    by_model = defaultdict(list)
    for r in _llm_rows():
        by_model[r.name].append(r)
    success = success_summary()["by_model"]
    models = {}
    for name, rs in by_model.items():
        real = [r for r in rs if not _meta(r.meta).get("estimated") and r.ok]
        tout = sum(r.tokens_out for r in real)
        dur = sum(r.duration_ms for r in real)
        eval_ms = sum(_meta(r.meta).get("eval_ms") or 0 for r in real)
        models[name] = {
            "llm_calls": len(rs), "errors": sum(1 for r in rs if not r.ok),
            "avg_call_ms": _r(_avg([r.duration_ms for r in rs]), 0),
            "ms_per_output_token": _r(dur / tout, 2) if tout else None,
            "tokens_per_s_generation": _r(tout / (eval_ms / 1000), 2) if eval_ms else None,
            "cold_loads": sum(1 for r in rs if (_meta(r.meta).get("load_ms") or 0) > 1000),
            **{f"task_{k}": v for k, v in success.get(name, {}).items()},
        }
    speeds = [m["ms_per_output_token"] for m in models.values() if m["ms_per_output_token"]]
    fastest = min(speeds) if speeds else None
    for m in models.values():
        v = m["ms_per_output_token"]
        m["slowdown_vs_fastest"] = _r(v / fastest, 2) if v and fastest else None
        m["slower_than_fastest"] = bool(v and fastest and v / fastest > slow_factor)
    return {"models": models, "slow_factor": slow_factor}


def regression(model: Optional[str] = None, window: int = 5, threshold: float = 0.2) -> dict:
    """Has generation gotten slower? Compares the average ms-per-output-token
    of the last `window` projects that used `model` with the `window`
    before them; flags a regression if it is more than `threshold` slower."""
    rows = _llm_rows(model)
    if model is None and rows:
        model = rows[0].name  # most recently used model
        rows = [r for r in rows if r.name == model]
    per: dict[str, dict] = {}
    for r in rows:
        if r.ok and not _meta(r.meta).get("estimated"):
            p = per.setdefault(str(r.project_id), {"tok": 0, "ms": 0, "t": r.t_start_ms or 0})
            p["tok"] += r.tokens_out
            p["ms"] += r.duration_ms
            p["t"] = min(p["t"], r.t_start_ms or p["t"])
    series = sorted(
        ({"project_id": k, "ms_per_output_token": _r(v["ms"] / v["tok"], 2), "t": v["t"]}
         for k, v in per.items() if v["tok"]), key=lambda x: x["t"])
    out = {"model": model, "window": window, "threshold": threshold, "series": series,
           "regression": False, "recent_ms_per_token": None, "baseline_ms_per_token": None, "pct_change": None}
    if len(series) < 2:
        out["note"] = "need at least 2 projects with token data to compare"
        return out
    recent = series[-window:]
    baseline = series[max(0, len(series) - 2 * window): len(series) - len(recent)] or series[:1]
    rv, bv = _avg([s["ms_per_output_token"] for s in recent]), _avg([s["ms_per_output_token"] for s in baseline])
    out.update(recent_ms_per_token=_r(rv), baseline_ms_per_token=_r(bv),
               pct_change=_r((rv - bv) / bv * 100, 1) if bv else None,
               regression=bool(bv and (rv - bv) / bv > threshold))
    return out


# ------------------------------------------------------------------ export --

def to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in CSV_COLUMNS})
    return buf.getvalue()
