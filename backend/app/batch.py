"""Batch processing: run many requests at once and compare their results.

    POST /batch                    start N runs concurrently        -> batch_id
    GET  /batch/{id}               live status + per-project state
    GET  /batch/{id}/results       comparison table + aggregates (json | csv)
    POST /batch/{id}/cancel        stop the runs that haven't finished

A batch is a set of projects, each an ordinary run_pipeline() call with its own
project id, workspace and event stream, so everything built for single runs
(live WebSocket, metrics, intervention, downloads) works per project. What the
batch adds is concurrency control, isolation (one project crashing doesn't stop
the others), persistence (`batches` / `batch_projects`) and the comparison view
on top of the metrics each run already records.

Concurrency: up to `max_concurrency` pipelines are in flight (default
BATCH_MAX_CONCURRENCY=5). The pipelines overlap in this process; whether the
LLM calls inside them overlap depends on your Ollama (OLLAMA_NUM_PARALLEL) — a
default single-slot Ollama serialises generation, so concurrency then mostly
hides the non-LLM time (tool calls, QA, installs). Latencies measured inside a
concurrent batch include queueing for the model; compare them only against runs
made under the same concurrency (experiments do this for you).

Single process, like the event bus: a backend restart while a batch is running
marks it (and its unfinished projects) `interrupted` on the next start.
"""
import asyncio
import csv
import io
import json
import logging
import time
import uuid
from typing import Any, Awaitable, Callable, Optional

from sqlalchemy import text

from app import db, metrics_report, run_config, stats
from app.config import settings
from app.schemas import BatchItem, RunSettings

logger = logging.getLogger("aiopsforge.batch")

FINAL = ("done", "failed", "cancelled", "interrupted")


def _j(value: Any) -> Any:
    """JSONB comes back decoded from Postgres but as text from other drivers."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def resolve_items(requests: list, defaults: Optional[RunSettings] = None) -> list[dict]:
    """Merges each item with the batch defaults and the system defaults.
    Returns plain dicts: request, project_name, communication_mode, framework, overrides."""
    defaults = defaults or RunSettings()
    out = []
    for r in requests:
        item = BatchItem(request=r) if isinstance(r, str) else r

        def pick(field: str):
            v = getattr(item, field)
            return v if v is not None else getattr(defaults, field)
        overrides = {k: pick(k) for k in ("model", "max_qa_attempts", "use_memory", "failure_recovery", "temperature", "seed", "num_ctx") if pick(k) is not None}
        out.append({
            "request": item.request, "project_name": item.project_name,
            "communication_mode": pick("communication_mode") or "blackboard",
            "framework": pick("framework"), "overrides": run_config.validate(overrides),
        })
    return out


RunFn = Callable[..., Awaitable[Any]]


class BatchManager:
    def __init__(self, run_fn: Optional[RunFn] = None):
        self._run_fn = run_fn
        self._tasks: dict[str, asyncio.Task] = {}

    # ---------------------------------------------------------------- start --
    def _runner(self) -> RunFn:
        if self._run_fn is None:
            from app.pipeline.graph import run_pipeline  # local: graph imports a lot
            return run_pipeline
        return self._run_fn

    async def start(self, items: list[dict], *, name: Optional[str] = None, max_concurrency: Optional[int] = None,
                    experiment_id: Optional[str] = None, level: Optional[str] = None,
                    config: Optional[dict] = None) -> str:
        """Creates the batch (rows first, so it is visible immediately) and
        starts running it in the background. Returns the batch id."""
        if not items:
            raise ValueError("a batch needs at least one request")
        if len(items) > settings.batch_max_runs:
            raise ValueError(f"{len(items)} runs exceeds the limit of {settings.batch_max_runs} (BATCH_MAX_RUNS)")
        conc = max(1, min(max_concurrency or settings.batch_max_concurrency, len(items)))
        batch_id = str(uuid.uuid4())
        snapshot = {**(config or {}), "max_concurrency": conc, "total": len(items)}
        await asyncio.to_thread(self._insert, batch_id, name, experiment_id, level, snapshot, items)
        task = asyncio.create_task(self._run_batch(batch_id, name or batch_id[:8], items, conc))
        self._tasks[batch_id] = task
        task.add_done_callback(lambda t: self._tasks.pop(batch_id, None))
        return batch_id

    @staticmethod
    def _insert(batch_id, name, experiment_id, level, config, items) -> None:
        with db.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO batches (id, name, status, experiment_id, level, config, total) "
                     "VALUES (:id, :name, 'running', :exp, :level, :config, :total)"),
                {"id": batch_id, "name": name, "exp": experiment_id, "level": level,
                 "config": json.dumps(config), "total": len(items)})
            for i, it in enumerate(items):
                conn.execute(
                    text("INSERT INTO batch_projects (batch_id, idx, request, settings, status) "
                         "VALUES (:b, :i, :req, :settings, 'pending')"),
                    {"b": batch_id, "i": i, "req": it["request"], "settings": json.dumps(
                        {k: it.get(k) for k in ("communication_mode", "framework", "overrides", "rep", "level", "position")
                         if it.get(k) is not None})})

    async def wait(self, batch_id: str) -> None:
        task = self._tasks.get(batch_id)
        if task:
            await asyncio.wait([task])

    def is_running(self, batch_id: str) -> bool:
        return batch_id in self._tasks

    # ------------------------------------------------------------------ run --
    async def _run_batch(self, batch_id: str, name: str, items: list[dict], conc: int) -> None:
        sem = asyncio.Semaphore(conc)
        cancelled = False
        try:
            results = await asyncio.gather(
                *(self._run_item(batch_id, name, i, it, sem) for i, it in enumerate(items)), return_exceptions=True)
            cancelled = any(isinstance(r, asyncio.CancelledError) for r in results)
        except asyncio.CancelledError:
            cancelled = True
        finally:
            await asyncio.shield(asyncio.to_thread(self._finalize, batch_id, cancelled))

    async def _run_item(self, batch_id: str, name: str, idx: int, item: dict, sem: asyncio.Semaphore) -> None:
        try:
            async with sem:
                project_id = str(uuid.uuid4())
                await asyncio.to_thread(self._start_item, batch_id, idx, project_id, f"{name} #{idx + 1}", item)
                try:
                    result = await self._runner()(
                        project_id, item["request"], item["communication_mode"],
                        framework=item.get("framework"), overrides=item.get("overrides") or None)
                    status = "passed" if (result.done and not result.needs_human_review) else "needs_review"
                    await asyncio.to_thread(self._end_item, batch_id, idx, status, None, project_id)
                except asyncio.CancelledError:
                    await asyncio.shield(asyncio.to_thread(self._end_item, batch_id, idx, "cancelled", None, project_id))
                    raise
                except Exception as exc:  # noqa: BLE001 — one project failing must not stop the others
                    logger.exception("batch %s item %d failed", batch_id, idx)
                    await asyncio.to_thread(self._end_item, batch_id, idx, "error", f"{type(exc).__name__}: {exc}"[:500], project_id)
        except asyncio.CancelledError:
            # cancelled while still waiting for a concurrency slot: it never started
            await asyncio.shield(asyncio.to_thread(self._end_item, batch_id, idx, "cancelled", None, None, only_pending=True))
            raise

    @staticmethod
    def _start_item(batch_id, idx, project_id, project_name, item) -> None:
        try:
            db.create_project(project_id, item.get("project_name") or project_name, item["request"])
        except Exception:  # noqa: BLE001 — the run itself will surface a real DB problem
            logger.warning("could not create project row for %s", project_id)
        with db.engine.begin() as conn:
            conn.execute(text("UPDATE batch_projects SET status = 'running', project_id = :p, started_at = CURRENT_TIMESTAMP "
                              "WHERE batch_id = :b AND idx = :i"), {"p": project_id, "b": batch_id, "i": idx})

    @staticmethod
    def _end_item(batch_id, idx, status, error, project_id, only_pending: bool = False) -> None:
        with db.engine.begin() as conn:
            sql = ("UPDATE batch_projects SET status = :s, error = :e, finished_at = CURRENT_TIMESTAMP "
                   "WHERE batch_id = :b AND idx = :i")
            if only_pending:
                sql += " AND status = 'pending'"
            conn.execute(text(sql), {"s": status, "e": error, "b": batch_id, "i": idx})
        if project_id and status in ("passed", "needs_review"):
            try:
                db.update_project_status(project_id, "done" if status == "passed" else "needs_human_review")
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _finalize(batch_id: str, cancelled: bool) -> None:
        with db.engine.begin() as conn:
            rows = conn.execute(text("SELECT status FROM batch_projects WHERE batch_id = :b"), {"b": batch_id}).fetchall()
            statuses = [r[0] for r in rows]
            if cancelled:
                status = "cancelled"
                conn.execute(text("UPDATE batch_projects SET status = 'cancelled' WHERE batch_id = :b "
                                  "AND status IN ('pending', 'running')"), {"b": batch_id})
            elif statuses and all(s == "error" for s in statuses):
                status = "failed"
            else:
                status = "done"
            conn.execute(text("UPDATE batches SET status = :s, finished_at = CURRENT_TIMESTAMP WHERE id = :b"),
                         {"s": status, "b": batch_id})

    async def cancel(self, batch_id: str) -> bool:
        """Cancels a running batch. Runs already in flight are stopped; ones that
        haven't started never will. Returns False if it isn't running."""
        task = self._tasks.get(batch_id)
        if not task:
            return False
        task.cancel()
        await asyncio.wait([task])
        return True


# --------------------------------------------------------------- queries ----

def status(batch_id: str) -> Optional[dict]:
    """Live status of a batch: counts, progress, and every project's state."""
    with db.engine.connect() as conn:
        b = conn.execute(text("SELECT id, name, status, experiment_id, level, config, total, created_at, finished_at "
                              "FROM batches WHERE id = :b"), {"b": batch_id}).fetchone()
        if b is None:
            return None
        rows = conn.execute(text("SELECT idx, project_id, request, settings, status, error FROM batch_projects "
                                 "WHERE batch_id = :b ORDER BY idx"), {"b": batch_id}).fetchall()
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    finished = sum(counts.get(s, 0) for s in ("passed", "needs_review", "error", "cancelled"))
    return {
        "batch_id": str(b.id), "name": b.name, "status": b.status, "experiment_id": str(b.experiment_id) if b.experiment_id else None,
        "level": b.level, "total": b.total, "finished": finished, "running": counts.get("running", 0),
        "pending": counts.get("pending", 0), "passed": counts.get("passed", 0),
        "needs_review": counts.get("needs_review", 0), "errors": counts.get("error", 0),
        "progress": round(finished / b.total, 3) if b.total else 1.0,
        "created_at": str(b.created_at), "finished_at": str(b.finished_at) if b.finished_at else None,
        "config": _j(b.config),
        "projects": [{"idx": r.idx, "project_id": str(r.project_id) if r.project_id else None,
                      "request": r.request[:200], "status": r.status, "error": r.error, "settings": _j(r.settings) or {}}
                     for r in rows],
    }


def list_batches(limit: int = 50, experiment_id: Optional[str] = None) -> list[dict]:
    sql = "SELECT id FROM batches "
    params: dict[str, Any] = {"n": limit}
    if experiment_id:
        sql += "WHERE experiment_id = :e "
        params["e"] = experiment_id
    with db.engine.connect() as conn:
        ids = [str(r[0]) for r in conn.execute(text(sql + "ORDER BY created_at DESC LIMIT :n"), params).fetchall()]
    return [s for s in (status(i) for i in ids) if s]


def results(batch_id: str) -> Optional[dict]:
    """The comparison table for a batch: one row per request (status, framework,
    tokens, latency, pass rate, cost, attempts, ...) plus aggregates."""
    st = status(batch_id)
    if st is None:
        return None
    ids = [p["project_id"] for p in st["projects"] if p["project_id"]]
    by_id = {r["project_id"]: r for r in metrics_report.list_projects(ids=ids)} if ids else {}
    rows = []
    for p in st["projects"]:
        m = by_id.get(p["project_id"], {})
        s = p["settings"]
        rows.append({
            "idx": p["idx"], "request": p["request"], "status": p["status"], "error": p["error"],
            "project_id": p["project_id"],
            "communication_mode": m.get("communication_mode") or s.get("communication_mode"),
            "framework": m.get("framework") or s.get("framework"),
            "model": m.get("model") or (s.get("overrides") or {}).get("model"),
            "overrides": s.get("overrides") or {},
            **{k: m.get(k) for k in (
                "tasks", "tasks_passed", "pass_rate", "avg_attempts", "attempts", "escalations", "repeated_errors",
                "run_ok", "ifs_avg", "ifs_strict_avg", "ifs_planner", "memory_hits", "tokens_in", "tokens_out",
                "tokens_total", "est_cost_usd", "llm_calls", "total_ms", "llm_ms")},
        })
    return {"batch": {k: v for k, v in st.items() if k != "projects"}, "rows": rows, "summary": summarize(rows)}


def _col(rows: list[dict], key: str) -> list[float]:
    return [r[key] for r in rows if r.get(key) is not None]


def summarize(rows: list[dict]) -> dict:
    """Aggregates over the runs that produced metrics."""
    measured = [r for r in rows if r.get("tokens_total") is not None]
    n = len(rows)
    outcome = {s: sum(1 for r in rows if r["status"] == s) for s in ("passed", "needs_review", "error", "cancelled")}
    return {
        "runs": n, "measured": len(measured), **{f"{k}": v for k, v in outcome.items()},
        "run_pass_rate": round(outcome["passed"] / n, 3) if n else None,
        "tokens_total": sum(_col(measured, "tokens_total")),
        "est_cost_usd": round(sum(_col(measured, "est_cost_usd")), 6),
        "total_ms": stats.describe(_col(measured, "total_ms")),
        "tokens": stats.describe(_col(measured, "tokens_total")),
        "avg_attempts": stats.describe(_col(measured, "avg_attempts")),
        "ifs_avg": stats.describe(_col(measured, "ifs_avg")),
        "ifs_strict_avg": stats.describe(_col(measured, "ifs_strict_avg")),
        "by_framework": {fw: sum(1 for r in rows if (r.get("framework") or "none") == fw)
                         for fw in sorted({(r.get("framework") or "none") for r in rows})},
    }


OVERRIDE_COLUMNS = ["use_memory", "max_qa_attempts", "failure_recovery", "temperature", "seed", "num_ctx"]

CSV_COLUMNS = ["batch_id", "idx", "status", "error", "request", "framework", "communication_mode", "model",
               *OVERRIDE_COLUMNS, "tasks", "tasks_passed", "pass_rate", "run_ok", "attempts",
               "avg_attempts", "escalations", "repeated_errors", "ifs_avg", "ifs_strict_avg", "ifs_planner",
               "memory_hits", "tokens_in", "tokens_out",
               "tokens_total", "est_cost_usd", "llm_calls", "total_ms", "llm_ms", "project_id"]


def override_values(overrides: Optional[dict]) -> dict:
    """The per-run settings columns of the CSV exports (blank = not overridden)."""
    return {k: (overrides or {}).get(k) for k in OVERRIDE_COLUMNS}


def results_csv(res: dict) -> str:
    """Raw per-run data, one row per project — suitable for re-analysis in R / pandas."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for r in res["rows"]:
        row = {**r, "batch_id": res["batch"]["batch_id"], **override_values(r["overrides"])}
        w.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in CSV_COLUMNS})
    return buf.getvalue()


def recover_stale() -> dict:
    """Run at startup: anything still 'running' belonged to a process that no
    longer exists. Marks it `interrupted` so it doesn't look alive forever."""
    out = {}
    with db.engine.begin() as conn:
        for table in ("batches", "experiments"):
            r = conn.execute(text(f"UPDATE {table} SET status = 'interrupted', finished_at = CURRENT_TIMESTAMP "
                                  f"WHERE status = 'running'"))
            out[table] = r.rowcount
        conn.execute(text("UPDATE batch_projects SET status = 'cancelled', error = 'interrupted by a backend restart' "
                          "WHERE status IN ('pending', 'running')"))
    return out


manager = BatchManager()
