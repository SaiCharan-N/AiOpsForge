"""Cost & performance metrics collection (Observable AI system).

What is measured, per project:

  llm    every Ollama generate call: tokens in/out (Ollama's own
         prompt_eval_count / eval_count), wall time, and Ollama's server-side
         breakdown (total / model-load / prompt-eval / generation), from which
         the transport+queueing overhead (wall - server) is derived
  embed  every embedding call (token count is estimated: the legacy
         /api/embeddings endpoint doesn't report one)
  tool   every MCP tool call (write_file, run_tests, ...), incl. session setup
  db     application-DB / memory queries (net of any embedding time inside)
  node   one row per LangGraph node execution, with its LLM / tool / DB /
         embed subtotals and the unaccounted remainder ("other")
  task   one row per finished task: passed?, attempts, failures, timeouts...
  pipeline  one row per run: total wall time, outcome, model, mode

How it stays out of the way:

  * Nothing changes any function's return type. Call sites don't pass
    metrics around; a per-node *scope* (a contextvar, which asyncio.to_thread
    copies into worker threads) tells `record()` which project / agent /
    task / attempt a call belongs to.
  * Events are appended to an in-memory buffer (a few microseconds) and
    written to Postgres in ONE batched insert when a node finishes.
  * Every public function swallows its own errors. Metrics collection can
    never fail the pipeline; `settings.metrics_enabled=False` turns it into
    a no-op.

Reporting lives in app.metrics_report; the tables are declared in
db/init.sql and db.run_schema_migrations.
"""
import functools
import json
import logging
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Optional

from sqlalchemy import text

from app import run_config
from app.config import settings

logger = logging.getLogger("aiopsforge.metrics")

MAX_BUFFERED_PER_PROJECT = 5000


class Scope:
    """Identifies who is running (project / agent / task / attempt) and
    accumulates that node's subtotals as calls are recorded inside it."""

    __slots__ = ("project_id", "agent", "task_order", "attempt", "acc")

    def __init__(self, project_id: Optional[str], agent: str,
                 task_order: Optional[int] = None, attempt: Optional[int] = None):
        self.project_id = project_id
        self.agent = agent
        self.task_order = task_order
        self.attempt = attempt
        self.acc = {"llm": 0.0, "tool": 0.0, "db": 0.0, "embed": 0.0,
                    "tokens_in": 0, "tokens_out": 0, "llm_calls": 0}


_scope: ContextVar[Optional[Scope]] = ContextVar("aiopsforge_metrics_scope", default=None)
_buffers: dict[str, list[dict]] = {}
_lock = threading.Lock()


def current_scope() -> Optional[Scope]:
    return _scope.get()


@contextmanager
def scope(project_id: Optional[str], agent: str, task_order: Optional[int] = None,
          attempt: Optional[int] = None):
    """Sets the metrics scope for the enclosed code (and any thread it
    hands work to via asyncio.to_thread)."""
    s = Scope(project_id, agent, task_order, attempt)
    token = _scope.set(s)
    try:
        yield s
    finally:
        _scope.reset(token)


def estimate_tokens(text_: str) -> int:
    """~4 characters per token — only used when Ollama gave no real count."""
    return max(1, len(text_ or "") // 4)


def estimate_cost(tokens_in: float, tokens_out: float) -> float:
    """tokens * rate_per_1k. Local Ollama inference has no API bill, so the
    default rates are a cloud-equivalent reference price, useful for
    comparing runs/models against each other; set both to 0 to disable."""
    return (tokens_in / 1000.0) * settings.token_rate_in_per_1k + \
           (tokens_out / 1000.0) * settings.token_rate_out_per_1k


# ------------------------------------------------------------- recording ----

def record(kind: str, name: str, duration_ms: float, *, tokens_in: int = 0, tokens_out: int = 0,
           ok: bool = True, meta: Optional[dict] = None, scope_: Optional[Scope] = None,
           task_order: Optional[int] = None, attempt: Optional[int] = None,
           t_start_ms: Optional[int] = None) -> None:
    """Buffers one metrics event. Never raises."""
    try:
        if not settings.metrics_enabled:
            return
        s = scope_ or _scope.get()
        if s is None or not s.project_id:
            return
        now_ms = int(time.time() * 1000)
        ev = {
            "project_id": s.project_id, "kind": kind, "agent": s.agent, "node": s.agent,
            "task_order": task_order if task_order is not None else s.task_order,
            "attempt": attempt if attempt is not None else s.attempt,
            "name": name, "tokens_in": int(tokens_in), "tokens_out": int(tokens_out),
            "duration_ms": int(round(duration_ms)),
            "t_start_ms": t_start_ms if t_start_ms is not None else int(now_ms - duration_ms),
            "ok": bool(ok), "meta": meta,
        }
        if kind in s.acc:
            s.acc[kind] += duration_ms
        if kind == "llm":
            s.acc["tokens_in"] += int(tokens_in)
            s.acc["tokens_out"] += int(tokens_out)
            s.acc["llm_calls"] += 1
        with _lock:
            buf = _buffers.setdefault(s.project_id, [])
            buf.append(ev)
            if len(buf) > MAX_BUFFERED_PER_PROJECT:
                del buf[: len(buf) - MAX_BUFFERED_PER_PROJECT]
    except Exception as exc:  # noqa: BLE001 — metrics must never break the pipeline
        logger.debug("metrics.record failed: %s", exc)


def record_llm(model: str, wall_ms: float, body: Optional[dict], prompt: str = "",
               output: str = "", system: Optional[str] = None, ok: bool = True) -> None:
    """Records one Ollama generate call from its response body.

    Token counts are Ollama's own (`prompt_eval_count`, `eval_count`); if a
    response lacks them they are estimated and flagged `estimated`.
    `network_ms` = wall time - Ollama's server-side total_duration, i.e. the
    HTTP transport + JSON + queueing overhead around the model itself.
    """
    try:
        body = body or {}
        tin, tout = body.get("prompt_eval_count"), body.get("eval_count")
        estimated = tin is None or tout is None
        if tin is None:
            tin = estimate_tokens((system or "") + prompt)
        if tout is None:
            tout = estimate_tokens(output) if output else 0

        def ms(key: str) -> float:
            return float(body.get(key) or 0) / 1e6

        server_ms, eval_ms = ms("total_duration"), ms("eval_duration")
        meta = {
            "model": model, "estimated": estimated,
            "ollama_ms": round(server_ms, 1), "load_ms": round(ms("load_duration"), 1),
            "prompt_eval_ms": round(ms("prompt_eval_duration"), 1), "eval_ms": round(eval_ms, 1),
            "network_ms": round(max(wall_ms - server_ms, 0.0), 1) if server_ms else None,
            "tok_per_s": round(tout / (eval_ms / 1000.0), 2) if eval_ms > 0 and not estimated else None,
        }
        record("llm", model, wall_ms, tokens_in=tin, tokens_out=tout, ok=ok, meta=meta)
    except Exception as exc:  # noqa: BLE001
        logger.debug("metrics.record_llm failed: %s", exc)


def record_embed(model: str, wall_ms: float, text_: str, ok: bool = True) -> None:
    record("embed", model, wall_ms, tokens_in=estimate_tokens(text_), ok=ok, meta={"estimated": True})


@contextmanager
def timed(kind: str, name: str, meta: Optional[dict] = None):
    """Times the enclosed block and records it as `kind` (tool | db | ...).
    For kind='db' the time spent in nested embedding calls is subtracted,
    so embedding latency isn't counted twice."""
    s = _scope.get()
    t_wall = time.time()
    t0 = time.perf_counter()
    embed0 = s.acc["embed"] if s is not None else 0.0
    ok = True
    try:
        yield
    except BaseException:
        ok = False
        raise
    finally:
        dur = (time.perf_counter() - t0) * 1000.0
        if kind == "db" and s is not None:
            dur = max(dur - (s.acc["embed"] - embed0), 0.0)
        record(kind, name, dur, ok=ok, meta=meta, t_start_ms=int(t_wall * 1000))


def timed_db(name: Optional[str] = None) -> Callable:
    """Decorator: times a synchronous DB/memory function as kind='db'."""
    def deco(fn: Callable) -> Callable:
        label = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with timed("db", label):
                return fn(*args, **kwargs)
        return wrapper
    return deco


def record_task(state: dict, task: dict, status: str, attempts: int) -> None:
    """One row per finished task — the source of pass rate, attempts,
    escalation / timeout rates and the Failure Recovery repeat-error rate."""
    try:
        hist = state.get("_failure_history") or []
        record(
            "task", task.get("type", "code"), 0, ok=(status == "passed"),
            task_order=task["order"], attempt=attempts,
            meta={
                "status": status, "model": run_config.model(),
                "escalated": bool(state.get("_last_escalated")),
                "failures": len(hist),
                "repeated_errors": sum(1 for h in hist if h.get("repeat")),
                "timeouts": sum(1 for h in hist if h.get("category") == "timeout"),
                "categories": [h.get("category") for h in hist],
                "used_similar_fix": any(h.get("used_similar_fix") for h in hist),
                "memory_used": bool(state.get("memory_used")),
                "ifs": state.get("ifs_score"),
                "ifs_strict": state.get("ifs_strict"),
                "framework": task.get("framework") or state.get("framework"),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("metrics.record_task failed: %s", exc)


# --------------------------------------------------------- node wrapper -----

def instrument_node(name: str, fn: Callable) -> Callable:
    """Wraps a LangGraph node: opens a scope for it, times it, and on exit
    writes one `node` row (with LLM / tool / DB / embed subtotals) and
    flushes the project's buffered events in a single insert."""
    import asyncio

    @functools.wraps(fn)
    async def wrapper(state: dict):
        pid = state.get("project_id") if isinstance(state, dict) else None
        task = (state.get("current_task") or {}) if isinstance(state, dict) else {}
        s = Scope(pid, name, task.get("order"), (state.get("attempt_count", 0) + 1) if isinstance(state, dict) else None)
        token = _scope.set(s)
        t_wall = time.time()
        t0 = time.perf_counter()
        ok = True
        try:
            return await fn(state)
        except BaseException:
            ok = False
            raise
        finally:
            dur = (time.perf_counter() - t0) * 1000.0
            _scope.reset(token)
            try:
                a = s.acc
                accounted = a["llm"] + a["tool"] + a["db"] + a["embed"]
                record(
                    "node", name, dur, tokens_in=a["tokens_in"], tokens_out=a["tokens_out"], ok=ok,
                    scope_=s, t_start_ms=int(t_wall * 1000),
                    meta={"llm_ms": round(a["llm"]), "tool_ms": round(a["tool"]), "db_ms": round(a["db"]),
                          "embed_ms": round(a["embed"]), "llm_calls": a["llm_calls"],
                          "other_ms": round(max(dur - accounted, 0.0))},
                )
                if pid:
                    await asyncio.to_thread(flush, pid)
            except Exception as exc:  # noqa: BLE001
                logger.debug("node metrics finalisation failed: %s", exc)
    return wrapper


# ------------------------------------------------------------------ flush ---

_INSERT = text(
    "INSERT INTO metrics_logs (project_id, kind, agent, node, task_order, attempt, name, tokens_in, "
    "tokens_out, duration_ms, t_start_ms, ok, meta) VALUES (:project_id, :kind, :agent, :node, "
    ":task_order, :attempt, :name, :tokens_in, :tokens_out, :duration_ms, :t_start_ms, :ok, :meta)"
)


def flush(project_id: str) -> int:
    """Writes all buffered events for a project in one batched INSERT.
    On a DB failure the events are put back (bounded) and retried at the
    next flush. Returns the number written."""
    with _lock:
        events = _buffers.pop(project_id, [])
    if not events:
        return 0
    try:
        from app import db  # lazy: db.py imports this module for @timed_db

        rows = [{**e, "meta": json.dumps(e["meta"]) if e.get("meta") is not None else None} for e in events]
        with db.engine.begin() as conn:
            conn.execute(_INSERT, rows)
        return len(events)
    except Exception as exc:  # noqa: BLE001
        logger.warning("metrics flush failed for %s (%d event(s) kept for retry): %s", project_id, len(events), exc)
        with _lock:
            buf = _buffers.setdefault(project_id, [])
            buf[:0] = events
            if len(buf) > MAX_BUFFERED_PER_PROJECT:
                del buf[: len(buf) - MAX_BUFFERED_PER_PROJECT]
        return 0


def discard(project_id: str) -> None:
    with _lock:
        _buffers.pop(project_id, None)
