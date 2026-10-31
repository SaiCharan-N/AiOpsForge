"""Experiments: controlled comparisons with statistics, built on batches.

Designs (POST /experiments, or a `preset`):

  ab            The SAME requests are run under each level of ONE factor —
                communication_mode (blackboard vs message_passing), model,
                max_qa_attempts (retry policy), use_memory, failure_recovery,
                temperature, num_ctx, or framework.
                One batch per level, run one after another; the first level is
                the baseline. Each metric is compared baseline-vs-level with
                Welch's t-test (+ Mann-Whitney), Holm-corrected across levels;
                pass rate with Fisher's exact test.
  repeat        (was experiments/exp1_memory_speedup.py) every request is run
                twice, cold then warm, memory on. Paired t-test on attempts,
                latency and tokens. Pairs whose first run already had a memory
                hit are excluded (the "cold" run wasn't cold).
  sequence      (was exp3_repeat_bug_trend.py) related requests run one at a
                time; does effort fall as memory accumulates? Linear trend of
                attempts / latency / tokens against position.
  mcp_overhead  (was exp2_mcp_overhead.py) latency of a write_file through the
                MCP server vs a direct file write.

Fairness rules the runner enforces (and records in the result's `design`
block, because a reviewer will ask):
  * Memory isolation. Long-term memory is shared across projects, so a run in
    one arm could otherwise learn from an earlier arm. For every `ab`
    experiment whose factor is not use_memory, memory is switched OFF in all
    arms (unless you set base.use_memory yourself). repeat/sequence need memory
    and force it ON.
  * Everything not being varied is identical between arms (same requests, same
    base settings, same concurrency).
  * Arms run sequentially; latencies include queueing for the model at the
    stated concurrency. Time drift between arms (model warm-up, thermals) is
    NOT randomised away — repeat the experiment with the levels in reverse
    order to check for an order effect.
  * Errored runs (exceptions) count as failures for pass rate and are excluded
    from the continuous metrics (they have none); the count is reported.

The whole configuration + environment (default model, embedding model,
thresholds, Python/platform, start time) is stored with the experiment so it
can be reproduced and cited; results stay retrievable after a restart.
"""
import asyncio
import csv
import io
import json
import logging
import math
import platform
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import text

from app import batch, db, run_config, stats
from app.batch import BatchManager, _j
from app.config import settings
from app.schemas import COMMUNICATION_MODES, ExperimentRequest

logger = logging.getLogger("aiopsforge.experiments")

DEFAULT_AB_REQUESTS = [
    "a function that checks if a number is prime, with a test",
    "a function that flattens a nested list, with a test",
    "a function that removes duplicate items from a list while preserving order, with a test",
    "a function that reverses a string, with a test",
    "a function that returns the n-th Fibonacci number, with a test",
]
# The request sets of the original scripts, verbatim.
EXP1_REQUESTS = DEFAULT_AB_REQUESTS[:3]
EXP3_REQUESTS = [
    "a function that sums all elements in a list using a manually indexed loop, with a test",
    "a function that finds the last element of a list using index arithmetic, with a test",
    "a function that reverses a list in place using a manually indexed loop, with a test",
    "a function that copies every element of an array into a new array using an indexed loop, with a test",
    "a function that computes a running total over a list using index-based iteration, with a test",
]

PRESETS: dict[str, dict] = {
    "communication_mode": {
        "description": "Blackboard (shared context) vs message passing (isolated agents), memory off, 3 repetitions.",
        "design": "ab", "factor": "communication_mode", "levels": ["blackboard", "message_passing"],
        "requests": DEFAULT_AB_REQUESTS, "repetitions": 3},
    "memory": {
        "description": "Long-term memory on vs off. Runs one at a time so earlier runs can help later ones.",
        "design": "ab", "factor": "use_memory", "levels": [False, True],
        "requests": EXP3_REQUESTS, "repetitions": 1, "max_concurrency": 1},
    "retry_policy": {
        "description": "How many QA attempts a task gets before escalating: 1 vs 3 vs 5, memory off.",
        "design": "ab", "factor": "max_qa_attempts", "levels": [1, 3, 5],
        "requests": DEFAULT_AB_REQUESTS, "repetitions": 3},
    "memory_speedup": {
        "description": "Experiment 1: each request run twice, cold then warm (paired comparison).",
        "design": "repeat", "requests": EXP1_REQUESTS},
    "repeat_bug_trend": {
        "description": "Experiment 3: related requests in sequence; does effort fall as memory accumulates?",
        "design": "sequence", "requests": EXP3_REQUESTS},
    "mcp_overhead": {
        "description": "Experiment 2: latency of an MCP write_file call vs a direct file write.",
        "design": "mcp_overhead", "calls": 20},
}

# key, label, kind, lower_is_better, scale-for-display (value -> shown)
METRICS = [
    ("tokens_total", "Tokens", "continuous", True),
    ("total_ms", "Latency (ms)", "continuous", True),
    ("est_cost_usd", "Est. cost (USD)", "continuous", True),
    ("avg_attempts", "Attempts per task", "continuous", True),
    ("ifs_avg", "Instruction fidelity (IFS)", "continuous", False),
    ("ifs_strict_avg", "Instruction fidelity, strict matcher", "continuous", False),
    ("ifs_planner", "Planner-stage IFS", "continuous", False),
    ("pass", "Run passed", "binary", False),
]
LABELS = {k: lbl for k, lbl, _, _ in METRICS}


# ------------------------------------------------------------------ plan ----

def _label(factor: str, value: Any) -> str:
    if factor in ("use_memory", "failure_recovery"):
        return "on" if value else "off"
    return str(value)


def _validate_level(factor: str, value: Any) -> Any:
    if factor == "communication_mode":
        if value not in COMMUNICATION_MODES:
            raise ValueError(f"communication_mode levels must be from {list(COMMUNICATION_MODES)}")
        return value
    if factor == "framework":
        from app.webapp_detect import validate_framework
        fw = validate_framework(value)
        if fw is None:
            raise ValueError("framework levels must name a framework (fastapi, flask, django, express, next)")
        return fw
    return run_config.validate({factor: value})[factor]


def build_plan(req: ExperimentRequest) -> dict:
    """Resolves preset + explicit fields into a validated, fully explicit plan.
    Raises ValueError (-> HTTP 422) for anything unusable."""
    preset: dict = {}
    if req.preset:
        if req.preset not in PRESETS:
            raise ValueError(f"unknown preset {req.preset!r}; available: {', '.join(PRESETS)}")
        preset = PRESETS[req.preset]
    given = req.model_fields_set

    def pick(field: str, default=None):
        if field in given and getattr(req, field) is not None:
            return getattr(req, field)
        return preset.get(field, default)

    design = pick("design")
    if design is None:
        raise ValueError("choose a `design` (ab, repeat, sequence, mcp_overhead) or a `preset`")
    plan: dict[str, Any] = {
        "design": design, "preset": req.preset, "name": req.name or (req.preset or design),
        "alpha": req.alpha, "repetitions": req.repetitions if "repetitions" in given else preset.get("repetitions", 1),
        "base": {k: v for k, v in req.base.model_dump().items() if v is not None},
    }
    if design == "mcp_overhead":
        plan["calls"] = pick("calls", 20)
        plan["total_runs"] = 0
        return plan

    requests = [r.strip() for r in (pick("requests") or []) if r and r.strip()]
    if not requests:
        raise ValueError("`requests` must list at least one request (or use a preset that provides them)")
    plan["requests"] = requests
    conc_default = pick("max_concurrency", settings.batch_max_concurrency)

    if design == "ab":
        factor = pick("factor")
        if not factor:
            raise ValueError("an `ab` experiment needs a `factor` (communication_mode, framework, model, max_qa_attempts, use_memory, failure_recovery, temperature, num_ctx)")
        levels = pick("levels")
        if not levels or len(levels) < 2:
            raise ValueError("`levels` must list at least two values of the factor (the first is the baseline)")
        if len(levels) > 6:
            raise ValueError("at most 6 levels")
        levels = [_validate_level(factor, v) for v in levels]
        labels = [_label(factor, v) for v in levels]
        if len(set(labels)) != len(labels):
            raise ValueError("levels must be distinct")
        if factor in plan["base"]:
            raise ValueError(f"base.{factor} conflicts with the factor being varied")
        isolation = []
        if factor != "use_memory" and "use_memory" not in plan["base"]:
            plan["base"]["use_memory"] = False
            isolation.append("long-term memory switched OFF in every arm so one arm cannot learn from another")
        plan.update(factor=factor, levels=levels, labels=labels, baseline=labels[0])
        plan["isolation"] = isolation
        plan["max_concurrency"] = 1 if factor == "use_memory" else conc_default
        plan["total_runs"] = len(requests) * len(levels) * plan["repetitions"]
    elif design == "repeat":
        plan["base"]["use_memory"] = True
        plan["max_concurrency"] = conc_default
        plan["repetitions"] = 1
        plan["isolation"] = ["memory ON; pairs whose first run already hit memory are excluded from the paired test"]
        plan["total_runs"] = 2 * len(requests)
    else:  # sequence
        plan["base"]["use_memory"] = True
        plan["max_concurrency"] = 1
        plan["repetitions"] = 1
        plan["isolation"] = ["memory ON; requests run strictly one after another"]
        plan["total_runs"] = len(requests)
    if plan["total_runs"] > settings.batch_max_runs:
        raise ValueError(f"this experiment would create {plan['total_runs']} runs; the limit is {settings.batch_max_runs} (BATCH_MAX_RUNS)")
    return plan


def environment() -> dict:
    """Everything a reader needs to reproduce or fairly cite the run (no secrets)."""
    return {
        "started_at": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
        "platform": platform.platform(), "default_model": settings.ollama_model,
        "embedding_model": settings.embedding_model, "ollama_base_url": settings.ollama_base_url,
        "default_max_qa_attempts": settings.max_qa_attempts,
        "memory_similarity_threshold": settings.memory_similarity_threshold, "memory_top_k": settings.memory_top_k,
        "failure_similarity_threshold": settings.failure_similarity_threshold,
        "batch_max_concurrency": settings.batch_max_concurrency,
        "failure_analyzer_llm_mode": settings.failure_analyzer_llm_mode,
        # None = Ollama's own default (random sampling, small context) — say so in the paper.
        "ollama_temperature": settings.ollama_temperature, "ollama_seed": settings.ollama_seed,
        "ollama_num_ctx": settings.ollama_num_ctx,
    }


def _clean(obj: Any) -> Any:
    """JSON-safe: inf/nan -> None, tuples -> lists."""
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


# ---------------------------------------------------------------- results ---

def _values(rows: list[dict], key: str) -> list[float]:
    if key == "pass":
        return [1.0 if r["status"] == "passed" else 0.0 for r in rows]
    return [float(r[key]) for r in rows if r.get(key) is not None]


def _arm_summary(label: str, batch_id: str, rows: list[dict]) -> dict:
    return {
        "label": label, "batch_id": batch_id, "n": len(rows),
        "errors": sum(1 for r in rows if r["status"] == "error"),
        "passed": sum(1 for r in rows if r["status"] == "passed"),
        "cancelled": sum(1 for r in rows if r["status"] == "cancelled"),
        "metrics": {key: stats.describe(_values(rows, key)) for key, *_ in METRICS},
        "est_cost_usd_total": round(sum(_values(rows, "est_cost_usd")), 6),
    }


def compute_ab(plan: dict, arms: list[dict]) -> dict:
    """arms: [{label, batch_id, rows}] in level order (first = baseline)."""
    baseline = plan["baseline"]
    comparisons, verdict, warnings = {}, [], []
    for key, label, kind, lower_better in METRICS:
        groups = {a["label"]: _values(a["rows"], key) for a in arms}
        if any(len(v) == 0 for v in groups.values()):
            comparisons[key] = {"skipped": "at least one condition has no measurements for this metric"}
            continue
        comp = stats.compare_groups(groups, baseline, kind=kind, alpha=plan["alpha"])
        comparisons[key] = comp
        base_mean = comp["groups"][baseline].get("mean")
        for c in comp["comparisons"]:
            diff = c["effect"]["diff"]
            pct = (diff / base_mean * 100.0) if (base_mean and diff is not None and kind == "continuous") else None
            c["pct_change"] = pct
            p_adj = c["p_adjusted"]
            if p_adj is None:
                continue
            sig = "significant" if c["significant"] else "not significant"
            if kind == "binary":
                verdict.append(f"{c['level']} vs {baseline}: {label.lower()} {comp['groups'][baseline]['mean']:.0%} → "
                               f"{comp['groups'][c['level']]['mean']:.0%} ({sig}, Fisher p={c['p']:.3g}, Holm p={p_adj:.3g})")
            else:
                d = c["effect"].get("cohens_d")
                verdict.append(f"{c['level']} vs {baseline}: {label} {pct:+.1f}% ({sig}, Holm p={p_adj:.3g}"
                               + (f", d={d:.2f} {c['effect']['label']}" if d is not None else "") + ")"
                               if pct is not None else f"{c['level']} vs {baseline}: {label} ({sig}, Holm p={p_adj:.3g})")
        if any(c["low_power"] for c in comp["comparisons"]):
            warnings.append(f"{label}: fewer than {stats.MIN_N_WARNING} observations in a condition — inconclusive whatever p says")
    n_err = sum(a["errors"] for a in [_arm_summary(a["label"], a["batch_id"], a["rows"]) for a in arms])
    if n_err:
        warnings.append(f"{n_err} run(s) errored: counted as failures for pass rate, excluded from the other metrics")
    conc = plan["max_concurrency"]
    if conc > 1:
        warnings.append(f"runs executed {conc} at a time: latency includes queueing for the model at that concurrency")
    warnings.append("arms ran one after another in the given order; repeat with the levels reversed to check for an order effect")
    return {"design": "ab", "factor": plan["factor"], "baseline": baseline,
            "arms": [_arm_summary(a["label"], a["batch_id"], a["rows"]) for a in arms],
            "comparisons": comparisons, "metric_labels": LABELS, "verdict": verdict, "warnings": warnings}


def compute_repeat(plan: dict, first: list[dict], repeat: list[dict]) -> dict:
    by_idx = {r["idx"]: r for r in repeat}
    pairs, excluded = [], []
    for f in first:
        r = by_idx.get(f["idx"])
        if r is None or f.get("tokens_total") is None or r.get("tokens_total") is None:
            excluded.append({"request": f["request"], "reason": "a run produced no measurements"})
        elif (f.get("memory_hits") or 0) > 0:
            excluded.append({"request": f["request"], "reason": "the first run already had a memory hit (not cold)"})
        else:
            pairs.append((f, r))
    results, verdict = {}, []
    for key, label, _, _ in METRICS[:5]:
        a = [p[0].get(key) for p in pairs if p[0].get(key) is not None and p[1].get(key) is not None]
        b = [p[1].get(key) for p in pairs if p[0].get(key) is not None and p[1].get(key) is not None]
        t = stats.paired_t_test(a, b)
        t["mean_first"], t["mean_repeat"] = (stats.mean(a) if a else None), (stats.mean(b) if b else None)
        t["pct_change"] = ((t["mean_repeat"] - t["mean_first"]) / t["mean_first"] * 100) if t["mean_first"] else None
        results[key] = t
        if t["p"] is not None and t["pct_change"] is not None:
            verdict.append(f"{label}: first {t['mean_first']:.4g} → repeat {t['mean_repeat']:.4g} ({t['pct_change']:+.1f}%, paired p={t['p']:.3g})")
    warnings = []
    if len(pairs) < stats.MIN_N_WARNING:
        warnings.append(f"only {len(pairs)} usable pair(s): inconclusive whatever p says")
    if excluded:
        warnings.append(f"{len(excluded)} pair(s) excluded (see `excluded`)")
    return {"design": "repeat", "pairs": [{"request": f["request"], "first": {k: f.get(k) for k in ("total_ms", "avg_attempts", "tokens_total", "ifs_avg", "memory_hits", "status")},
                                           "repeat": {k: r.get(k) for k in ("total_ms", "avg_attempts", "tokens_total", "ifs_avg", "memory_hits", "status")}} for f, r in pairs],
            "excluded": excluded, "memory_hit_in_repeat": sum(1 for _, r in pairs if (r.get("memory_hits") or 0) > 0),
            "comparisons": results, "metric_labels": LABELS, "verdict": verdict, "warnings": warnings,
            "arms": [_arm_summary("first", "", first), _arm_summary("repeat", "", repeat)]}


def compute_sequence(plan: dict, rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda r: r["idx"])
    series = [{"position": r["idx"] + 1, "request": r["request"], "status": r["status"],
               **{k: r.get(k) for k in ("avg_attempts", "total_ms", "tokens_total", "memory_hits", "ifs_avg")}} for r in rows]
    trends, verdict = {}, []
    for key, label, _, _ in METRICS[:4]:
        pts = [(s["position"], s[key]) for s in series if s.get(key) is not None]
        t = stats.linear_trend([p[0] for p in pts], [p[1] for p in pts])
        trends[key] = t
        if t["p"] is not None:
            verdict.append(f"{label}: slope {t['slope']:+.4g} per position (r={t['r']:.2f}, p={t['p']:.3g})")
    half = len(series) // 2
    halves = {}
    if half >= 1:
        for key in ("avg_attempts", "total_ms", "tokens_total"):
            a = [s[key] for s in series[:half] if s.get(key) is not None]
            b = [s[key] for s in series[half:] if s.get(key) is not None]
            halves[key] = stats.welch_t_test(a, b)
    warnings = ["a trend over so few positions is only suggestive; run the sequence several times"] if len(series) < 8 else []
    return {"design": "sequence", "series": series, "trends": trends, "first_vs_second_half": halves,
            "memory_hit_rate": (sum(1 for s in series if (s.get("memory_hits") or 0) > 0) / len(series)) if series else None,
            "metric_labels": LABELS, "verdict": verdict, "warnings": warnings}


async def run_mcp_overhead(calls: int, alpha: float = 0.05) -> dict:
    """Latency of write_file via the MCP server vs a direct write to a temp dir."""
    from app.mcp_client import call_tool, discover_tools
    pid = f"exp-mcp-overhead-{uuid.uuid4().hex[:8]}"
    payload = "x" * 200
    mcp_ms, direct_ms = [], []
    # One-time cost of discovering the server's tools (what a session pays once, before any call).
    discovery_ms, tools_found = None, None
    try:
        t0 = time.perf_counter()
        tools_found = len(await discover_tools())
        discovery_ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:  # noqa: BLE001 — still measure the calls
        logger.warning("tool discovery failed during the MCP overhead experiment: %s", exc)
    for i in range(2):   # warm-up (first MCP call opens the session, imports...)
        await call_tool("write_file", {"project_id": pid, "path": f"warmup_{i}.txt", "content": payload})
    try:
        for i in range(calls):
            t0 = time.perf_counter()
            await call_tool("write_file", {"project_id": pid, "path": f"bench_{i}.txt", "content": payload})
            mcp_ms.append((time.perf_counter() - t0) * 1000)
    finally:
        try:
            await call_tool("run_shell_command", {"project_id": pid, "command": "rm -f bench_*.txt warmup_*.txt", "timeout": 30})
        except Exception:  # noqa: BLE001 — cleanup is best effort
            pass
    with tempfile.TemporaryDirectory() as d:
        for i in range(calls):
            t0 = time.perf_counter()
            (Path(d) / f"bench_{i}.txt").write_text(payload)
            direct_ms.append((time.perf_counter() - t0) * 1000)
    mean_mcp, mean_direct = stats.mean(mcp_ms), stats.mean(direct_ms)
    welch, mwu = stats.welch_t_test(direct_ms, mcp_ms), stats.mann_whitney_u(direct_ms, mcp_ms)
    return {
        "design": "mcp_overhead", "calls": calls, "discovery_ms": discovery_ms, "tools_found": tools_found,
        "mcp_ms": stats.describe(mcp_ms), "direct_ms": stats.describe(direct_ms),
        "p95_mcp_ms": sorted(mcp_ms)[min(len(mcp_ms) - 1, int(0.95 * len(mcp_ms)))],
        "overhead_ms": mean_mcp - mean_direct, "ratio": (mean_mcp / mean_direct) if mean_direct else None,
        "tests": {"welch": welch, "mann_whitney": mwu},
        "samples": {"mcp_ms": mcp_ms, "direct_ms": direct_ms},
        "verdict": [f"MCP write_file: {mean_mcp:.2f} ms vs direct {mean_direct:.3f} ms per call "
                    f"(+{mean_mcp - mean_direct:.2f} ms, {mean_mcp / mean_direct:.0f}x), Welch p={welch['p']:.3g}"
                    if welch["p"] is not None else f"MCP write_file: {mean_mcp:.2f} ms vs direct {mean_direct:.3f} ms per call"],
        "warnings": ["file writes are tiny; this measures the transport + session cost per tool call, which is what the pipeline pays"],
    }


# ---------------------------------------------------------------- manager ---

class ExperimentManager:
    def __init__(self, batches: Optional[BatchManager] = None):
        self._batches = batches
        self._tasks: dict[str, asyncio.Task] = {}

    @property
    def batches(self) -> BatchManager:
        return self._batches or batch.manager

    async def start(self, req: ExperimentRequest) -> str:
        plan = build_plan(req)
        exp_id = str(uuid.uuid4())
        config = {**plan, "environment": environment()}
        await asyncio.to_thread(self._insert, exp_id, plan["name"], plan["design"], config)
        task = asyncio.create_task(self._run(exp_id, plan))
        self._tasks[exp_id] = task
        task.add_done_callback(lambda t: self._tasks.pop(exp_id, None))
        return exp_id

    @staticmethod
    def _insert(exp_id, name, design, config) -> None:
        with db.engine.begin() as conn:
            conn.execute(text("INSERT INTO experiments (id, name, design, status, config) VALUES (:id, :n, :d, 'running', :c)"),
                         {"id": exp_id, "n": name, "d": design, "c": json.dumps(_clean(config))})

    @staticmethod
    def _finish(exp_id, status, result=None, error=None) -> None:
        with db.engine.begin() as conn:
            conn.execute(text("UPDATE experiments SET status = :s, result = :r, error = :e, finished_at = CURRENT_TIMESTAMP WHERE id = :id"),
                         {"s": status, "r": json.dumps(_clean(result)) if result is not None else None, "e": error, "id": exp_id})

    async def wait(self, exp_id: str) -> None:
        t = self._tasks.get(exp_id)
        if t:
            await asyncio.wait([t])

    async def cancel(self, exp_id: str) -> bool:
        t = self._tasks.get(exp_id)
        if not t:
            return False
        t.cancel()
        await asyncio.wait([t])
        return True

    # -------------------------------------------------------------- running --
    async def _run(self, exp_id: str, plan: dict) -> None:
        try:
            design = plan["design"]
            if design == "mcp_overhead":
                result = await run_mcp_overhead(plan["calls"], plan["alpha"])
            elif design == "ab":
                result = await self._run_ab(exp_id, plan)
            elif design == "repeat":
                result = await self._run_repeat(exp_id, plan)
            else:
                result = await self._run_sequence(exp_id, plan)
            result["design_notes"] = plan.get("isolation", [])
            await asyncio.to_thread(self._finish, exp_id, "done", result)
        except asyncio.CancelledError:
            await asyncio.shield(asyncio.to_thread(self._finish, exp_id, "cancelled"))
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("experiment %s failed", exp_id)
            await asyncio.to_thread(self._finish, exp_id, "failed", None, f"{type(exc).__name__}: {exc}"[:500])

    async def _arm(self, exp_id: str, plan: dict, label: str, items: list[dict]) -> dict:
        bid = await self.batches.start(
            items, name=f"{plan['name']}:{label}", max_concurrency=plan["max_concurrency"],
            experiment_id=exp_id, level=label, config={"experiment": plan["name"], "level": label})
        try:
            await self.batches.wait(bid)
        except asyncio.CancelledError:
            await asyncio.shield(self.batches.cancel(bid))
            raise
        res = await asyncio.to_thread(batch.results, bid)
        return {"label": label, "batch_id": bid, "rows": res["rows"]}

    def _items(self, plan: dict, requests: list[str], **extra) -> list[dict]:
        base = plan["base"]
        items = batch.resolve_items(requests, _settings(base))
        for it, ex in zip(items, extra.get("per_item") or [{}] * len(items)):
            it.update(ex)
        return items

    async def _run_ab(self, exp_id: str, plan: dict) -> dict:
        arms = []
        for value, label in zip(plan["levels"], plan["labels"]):
            items = []
            for rep in range(plan["repetitions"]):
                for it in batch.resolve_items(plan["requests"], _settings({**plan["base"]})):
                    _apply_factor(it, plan["factor"], value)
                    it["rep"], it["level"] = rep + 1, label
                    items.append(it)
            arms.append(await self._arm(exp_id, plan, label, items))
        return compute_ab(plan, arms)

    async def _run_repeat(self, exp_id: str, plan: dict) -> dict:
        reqs = plan["requests"]
        first = await self._arm(exp_id, plan, "first", self._items(plan, reqs))
        repeat = await self._arm(exp_id, plan, "repeat", self._items(plan, reqs))
        res = compute_repeat(plan, first["rows"], repeat["rows"])
        res["arms"][0]["batch_id"], res["arms"][1]["batch_id"] = first["batch_id"], repeat["batch_id"]
        return res

    async def _run_sequence(self, exp_id: str, plan: dict) -> dict:
        items = self._items(plan, plan["requests"])
        for i, it in enumerate(items):
            it["position"] = i + 1
        arm = await self._arm(exp_id, plan, "sequence", items)
        res = compute_sequence(plan, arm["rows"])
        res["batch_id"] = arm["batch_id"]
        return res


def _settings(base: dict):
    from app.schemas import RunSettings
    return RunSettings(**base)


def _apply_factor(item: dict, factor: str, value: Any) -> None:
    if factor in ("communication_mode", "framework"):
        item[factor] = value
    else:
        item["overrides"] = {**item.get("overrides", {}), factor: value}


# ---------------------------------------------------------------- queries ---

def status(exp_id: str) -> Optional[dict]:
    with db.engine.connect() as conn:
        e = conn.execute(text("SELECT id, name, design, status, config, error, created_at, finished_at FROM experiments WHERE id = :i"),
                         {"i": exp_id}).fetchone()
    if e is None:
        return None
    cfg = _j(e.config) or {}
    arms = batch.list_batches(100, experiment_id=exp_id)
    arms.sort(key=lambda a: a["created_at"])
    done_runs = sum(a["finished"] for a in arms)
    total = cfg.get("total_runs") or 0
    return {"experiment_id": str(e.id), "name": e.name, "design": e.design, "status": e.status, "error": e.error,
            "created_at": str(e.created_at), "finished_at": str(e.finished_at) if e.finished_at else None,
            "total_runs": total, "finished_runs": done_runs,
            "progress": round(done_runs / total, 3) if total else (1.0 if e.status == "done" else 0.0),
            "config": cfg,
            "arms": [{k: a[k] for k in ("batch_id", "level", "status", "total", "finished", "passed", "needs_review", "errors")} for a in arms]}


def results(exp_id: str) -> Optional[dict]:
    """The stored result (computed when the experiment finished) — available
    after a restart. `result` is None while it is still running."""
    st = status(exp_id)
    if st is None:
        return None
    with db.engine.connect() as conn:
        r = conn.execute(text("SELECT result FROM experiments WHERE id = :i"), {"i": exp_id}).fetchone()
    return {"experiment": {k: v for k, v in st.items()}, "result": _j(r[0]) if r and r[0] else None}


def list_experiments(limit: int = 50) -> list[dict]:
    with db.engine.connect() as conn:
        rows = conn.execute(text("SELECT id, name, design, status, created_at, finished_at FROM experiments "
                                 "ORDER BY created_at DESC LIMIT :n"), {"n": limit}).fetchall()
    return [{"experiment_id": str(r.id), "name": r.name, "design": r.design, "status": r.status,
             "created_at": str(r.created_at), "finished_at": str(r.finished_at) if r.finished_at else None} for r in rows]


# ---------------------------------------------------------------- exports ---

RUN_CSV_COLUMNS = ["experiment_id", "design", "level", "rep"] + batch.CSV_COLUMNS


def runs_csv(exp_id: str) -> Optional[str]:
    """Raw per-run data of every arm, one row per project — for re-analysis in R / pandas."""
    st = status(exp_id)
    if st is None:
        return None
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=RUN_CSV_COLUMNS, extrasaction="ignore")
    w.writeheader()
    if st["design"] == "mcp_overhead":
        res = results(exp_id)["result"] or {}
        buf = io.StringIO()
        w2 = csv.writer(buf)
        w2.writerow(["experiment_id", "path", "call", "ms"])
        for path in ("mcp_ms", "direct_ms"):
            for i, v in enumerate((res.get("samples") or {}).get(path, [])):
                w2.writerow([exp_id, path.replace("_ms", ""), i + 1, v])
        return buf.getvalue()
    for arm in st["arms"]:
        res = batch.results(arm["batch_id"])
        status_rows = {p["idx"]: p for p in batch.status(arm["batch_id"])["projects"]}
        for r in res["rows"]:
            ov = r["overrides"]
            row = {**r, "experiment_id": exp_id, "design": st["design"], "level": arm["level"],
                   "rep": (status_rows[r["idx"]]["settings"] or {}).get("rep"),
                   "batch_id": arm["batch_id"], **batch.override_values(ov)}
            w.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in RUN_CSV_COLUMNS})
    return buf.getvalue()


_TEX_ESC = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


def _tex(s: Any) -> str:
    return "".join(_TEX_ESC.get(c, c) for c in str(s))


def _fmt(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "--"
    if x != 0 and (abs(x) >= 1e5 or abs(x) < 1e-3):
        return f"{x:.2e}"
    return f"{x:.{digits}f}"


def latex(exp_id: str) -> Optional[str]:
    """LaTeX (booktabs) tables for the experiment: a per-condition summary and the
    statistical comparisons. Paste into a paper; check the captions and column
    choice, they are a starting point."""
    res = results(exp_id)
    if res is None:
        return None
    exp, r = res["experiment"], res["result"]
    if not r:
        return "% experiment has no result yet (status: %s)\n" % exp["status"]
    name, tag = _tex(exp["name"]), exp_id[:8]
    out = [f"% Generated by AIOpsForge for experiment {exp_id} ({exp['design']})"]
    if r["design"] == "ab":
        cols = [("tokens_total", "Tokens", 1.0, 0), ("total_ms", "Latency (s)", 1e-3, 1), ("avg_attempts", "Attempts/task", 1.0, 2), ("est_cost_usd", "Cost (USD)", 1.0, 5)]
        out += [r"\begin{table}[t]", r"\centering", fr"\caption{{{name}: results per {_tex(r['factor'])} level (mean $\pm$ SD).}}",
                fr"\label{{tab:{tag}-summary}}", r"\begin{tabular}{l" + "r" * (2 + len(cols)) + "}", r"\toprule",
                "Condition & $n$ & Pass rate & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
        for a in r["arms"]:
            cells = []
            for key, _, scale, dig in cols:
                m = a["metrics"][key]
                cells.append("--" if not m.get("n") else f"{_fmt(m['mean'] * scale, dig)} $\\pm$ {_fmt((m['sd'] or 0) * scale, dig)}")
            out.append(f"{_tex(a['label'])} & {a['n']} & {_fmt(a['passed'] / a['n'] * 100 if a['n'] else None, 0)}\\% & " + " & ".join(cells) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        out += [r"\begin{table}[t]", r"\centering",
                fr"\caption{{{name}: comparison against the baseline (\texttt{{{_tex(r['baseline'])}}}). $p$ is Holm-adjusted; $d$ is Cohen's $d$.}}",
                fr"\label{{tab:{tag}-tests}}", r"\begin{tabular}{llrrrr}", r"\toprule",
                r"Metric & Condition & $\Delta$ (\%) & 95\% CI of $\Delta$ & $p$ & $d$ \\", r"\midrule"]
        for key, label, kind, _ in METRICS:
            comp = r["comparisons"].get(key)
            if not comp or "comparisons" not in comp:
                continue
            for c in comp["comparisons"]:
                ci = c["effect"].get("ci95")
                d = c["effect"].get("cohens_d")
                delta = _fmt(c.get("pct_change"), 1) if kind == "continuous" else f"{_fmt((c['effect']['diff'] or 0) * 100, 1)} pp"
                out.append(f"{_tex(label)} & {_tex(c['level'])} & {delta} & "
                           f"{('[' + _fmt(ci[0], 3) + ', ' + _fmt(ci[1], 3) + ']') if ci else '--'} & {_fmt(c['p_adjusted'], 3)} & {_fmt(d, 2)} \\\\")
        out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    elif r["design"] == "repeat":
        out += [r"\begin{table}[t]", r"\centering", fr"\caption{{{name}: first (cold) vs repeat (warm) run, paired $t$-test.}}",
                fr"\label{{tab:{tag}}}", r"\begin{tabular}{lrrrr}", r"\toprule", r"Metric & First & Repeat & $\Delta$ (\%) & $p$ \\", r"\midrule"]
        for key, label, *_ in METRICS[:4]:
            t = r["comparisons"].get(key, {})
            out.append(f"{_tex(label)} & {_fmt(t.get('mean_first'), 3)} & {_fmt(t.get('mean_repeat'), 3)} & {_fmt(t.get('pct_change'), 1)} & {_fmt(t.get('p'), 3)} \\\\")
        out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    elif r["design"] == "sequence":
        out += [r"\begin{table}[t]", r"\centering", fr"\caption{{{name}: linear trend of each metric over the position in the sequence.}}",
                fr"\label{{tab:{tag}}}", r"\begin{tabular}{lrrr}", r"\toprule", r"Metric & Slope & $r$ & $p$ \\", r"\midrule"]
        for key, label, *_ in METRICS[:4]:
            t = r["trends"].get(key, {})
            out.append(f"{_tex(label)} & {_fmt(t.get('slope'), 4)} & {_fmt(t.get('r'), 2)} & {_fmt(t.get('p'), 3)} \\\\")
        out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    else:
        out += [r"\begin{table}[t]", r"\centering", fr"\caption{{{name}: tool-call latency (ms), {r['calls']} calls each.}}",
                fr"\label{{tab:{tag}}}", r"\begin{tabular}{lrrr}", r"\toprule", r"Path & Mean & SD & Median \\", r"\midrule"]
        for lab, k in (("MCP write\\_file", "mcp_ms"), ("Direct write", "direct_ms")):
            m = r[k]
            out.append(f"{lab} & {_fmt(m['mean'], 3)} & {_fmt(m['sd'], 3)} & {_fmt(m['median'], 3)} \\\\")
        out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def recover_stale() -> dict:
    return batch.recover_stale()


manager = ExperimentManager()
