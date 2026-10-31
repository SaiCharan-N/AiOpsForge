"""User intervention: pause points, feedback, resume.

The pipeline used to run end to end with no way to say "wait, I meant X".
This module lets a user steer a run while it is in flight:

  Automatic pause points (opt-in per request: `interactive=true`)
    planner     after the Planner's task list   "Does this plan make sense?"
    developer   after the FIRST generated file  "Is this the right approach?"
    qa_failure  after a QA failure has been analysed, before the retry:
                "Retry with the analyzer's fix, add guidance, or skip?"
  On-demand pause (any run, interactive or not)
    POST /projects/{id}/intervention {"pause_after": "next" | <node>}
    pauses the run the next time that node finishes.

At a pause the user answers with POST /projects/{id}/feedback:
    approve     continue
    revise      guidance / clarification (or an edited task list at the plan):
                the Planner re-plans / the Developer regenerates the first
                file; elsewhere the text is added to the Developer's context
    skip_task   abandon the current task (flagged for human review)
    abort       stop the run

How it works
  * Each graph node is wrapped by `interactive_node`. After the node
    returns, the gate decides whether to pause. To pause it parks on an
    asyncio Future; the endpoint resolves that Future. The graph state is
    an ordinary in-memory dict that simply isn't touched while waiting, so
    partial results cannot be lost on pause/resume, and the SAME node's
    result is what continues (or is re-run) afterwards.
  * A best-effort snapshot of the state is written to Redis at each pause
    (visible to debugging / inspection) and every pause + answer is logged
    to the `interventions` table, so the project keeps a history of what
    the user was asked and what they said.
  * The gate wraps OUTSIDE the metrics wrapper, so time spent waiting for a
    human is not counted as node latency; it is recorded separately
    (`pause` rows / `paused_ms`) and excluded from the run's total_ms.
  * Everything is in one process (like app.events). A backend restart
    while a run is paused ends that run — see the README limitation note.
  * A pause nobody answers auto-approves after
    settings.intervention_timeout_seconds.
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app import db, events, short_term_memory as stm
from app import run_config
from app.config import settings
from app.pipeline import event_types as ev, metrics
from app.schemas import Task

logger = logging.getLogger("aiopsforge.intervention")

AUTO_POINTS = ("planner", "developer", "qa_failure")
MANUAL_TARGETS = ("next", "planner", "developer", "qa", "failure_analyzer", "next_task")

_OPTIONS = {
    "planner": ["approve", "revise", "abort"],
    "developer": ["approve", "revise", "abort"],
    "qa": ["approve", "revise", "skip_task", "abort"],
    "failure_analyzer": ["approve", "revise", "skip_task", "abort"],
    "next_task": ["approve", "revise", "abort"],
}


class NotPaused(Exception):
    """Feedback arrived but the run isn't waiting for any."""


class PipelineAborted(Exception):
    """The user chose 'abort' at a pause point. Carries the state so the
    partial results (plan, task history, files already written) survive."""

    def __init__(self, message: str = "aborted by user", state: Optional[dict] = None):
        super().__init__(message)
        self.state = state or {}


@dataclass
class Feedback:
    action: str = "approve"
    guidance: Optional[str] = None
    clarification: Optional[str] = None
    tasks: Optional[list[str]] = None
    source: str = "user"  # "user" | "timeout"


@dataclass
class PauseInfo:
    point: str            # planner | developer | qa_failure | manual
    node: str             # the node that just finished
    question: str
    options: list[str]
    payload: dict
    timeout_s: int
    since: float = field(default_factory=time.time)


class Control:
    """Per-run intervention state."""

    def __init__(self, project_id: str, auto_points: set[str]):
        self.project_id = project_id
        self.auto = auto_points
        self.pending: set[str] = set()       # manual pause requests: "next" or a node name
        self.waiting: Optional[PauseInfo] = None
        self._future: Optional[asyncio.Future] = None
        self._answered = False
        self.dev_first_done = False
        self.paused_ms = 0.0
        self.pauses = 0


_controls: dict[str, Control] = {}


# ----------------------------------------------------------- registry -------

def register(project_id: str, interactive: bool = False, pause_points: Optional[list[str]] = None) -> Control:
    points = set(pause_points if pause_points is not None else AUTO_POINTS) & set(AUTO_POINTS) if interactive else set()
    ctl = Control(project_id, points)
    _controls[project_id] = ctl
    return ctl


def get(project_id: str) -> Optional[Control]:
    return _controls.get(project_id)


def release(project_id: str) -> None:
    _controls.pop(project_id, None)


def request_pause(project_id: str, pause_after: str = "next") -> Control:
    """Asks the run to pause when `pause_after` next finishes. KeyError if
    the project has no active run; ValueError for an unknown target."""
    ctl = _controls.get(project_id)
    if ctl is None:
        raise KeyError(project_id)
    if pause_after not in MANUAL_TARGETS:
        raise ValueError(f"pause_after must be one of {list(MANUAL_TARGETS)}")
    ctl.pending.add(pause_after)
    return ctl


def cancel_pause_requests(project_id: str) -> Control:
    ctl = _controls.get(project_id)
    if ctl is None:
        raise KeyError(project_id)
    ctl.pending.clear()
    return ctl


def status(project_id: str) -> dict:
    """What the status endpoint / a reconnecting client needs to know."""
    ctl = _controls.get(project_id)
    if ctl is None:
        return {"active": False, "paused": False, "interactive": False, "pending_pause": [], "pause": None}
    w = ctl.waiting
    paused = w is not None and not ctl._answered
    return {
        "active": True, "paused": paused, "interactive": bool(ctl.auto), "auto_points": sorted(ctl.auto),
        "pending_pause": sorted(ctl.pending), "pauses": ctl.pauses,
        "pause": {"point": w.point, "node": w.node, "question": w.question, "options": w.options,
                  "payload": w.payload, "since": w.since, "timeout_s": w.timeout_s} if paused else None,
    }


def submit_feedback(project_id: str, fb: Feedback) -> PauseInfo:
    """Answers the current pause. KeyError: no active run. NotPaused: run
    isn't waiting (or was already answered). ValueError: this answer isn't
    valid at this pause point."""
    ctl = _controls.get(project_id)
    if ctl is None:
        raise KeyError(project_id)
    w, fut = ctl.waiting, ctl._future
    if w is None or fut is None or ctl._answered or fut.done():
        raise NotPaused(project_id)
    if fb.action not in w.options:
        raise ValueError(f"'{fb.action}' is not available at this pause point; choose one of {w.options}")
    if fb.action == "revise" and not (fb.guidance or fb.clarification or fb.tasks):
        raise ValueError("'revise' needs guidance, a clarification, or an edited task list")
    if fb.tasks and w.node != "planner":
        raise ValueError("an edited task list can only be submitted at the plan review")
    ctl._answered = True
    fut.get_loop().call_soon_threadsafe(lambda: fut.done() or fut.set_result(fb))
    return w


# --------------------------------------------------------- guidance ---------

def guidance_for(state: dict, task_order: Optional[int] = None) -> list[str]:
    """The user's guidance/clarifications that apply now: everything given
    at project level plus anything given for THIS task."""
    out: list[str] = []
    for e in state.get("user_feedback") or []:
        applies = e.get("scope") == "project" or (e.get("scope") == "task" and e.get("task_order") == task_order)
        if not applies:
            continue
        if e.get("guidance"):
            out.append(e["guidance"])
        if e.get("clarification"):
            out.append(f"Clarification: {e['clarification']}")
    return out


# ------------------------------------------------------------- the gate -----

def _decide(ctl: Control, name: str, state: dict) -> Optional[str]:
    """Which pause point (if any) applies now that node `name` finished."""
    if name == "next_task" and state.get("done"):
        ctl.pending.clear()
        return None  # nothing left to steer
    if name in ctl.pending or "next" in ctl.pending:
        ctl.pending.discard(name)
        ctl.pending.discard("next")
        return "manual"
    if name == "planner" and "planner" in ctl.auto:
        return "planner"
    if name == "developer" and "developer" in ctl.auto and not ctl.dev_first_done \
            and state.get("attempt_count", 0) == 0:
        return "developer"
    if name == "failure_analyzer" and "qa_failure" in ctl.auto:
        return "qa_failure"
    return None


def _payload(name: str, state: dict) -> dict:
    task = state.get("current_task") or {}
    base: dict[str, Any] = {
        "task_order": task.get("order"), "task": task.get("description"),
        "attempts_failed": state.get("attempt_count", 0), "max_attempts": run_config.max_qa_attempts(),
    }
    if name == "planner":
        base["tasks"] = [{"order": t["order"], "description": t["description"], "type": t.get("type", "code")}
                         for t in state.get("task_list") or []]
    elif name == "developer":
        code = state.get("code") or ""
        base.update(filename=state.get("code_filename"), code=code[:6000], code_length=len(code))
    elif name in ("qa", "failure_analyzer"):
        tr = state.get("test_result") or {}
        base["passed"] = bool(tr.get("passed"))
        base["error"] = (tr.get("error") or tr.get("output") or "")[:800]
        a = state.get("_failure_analysis") or {}
        if a:
            base["analysis"] = {k: a.get(k) for k in (
                "error_type", "error_category", "code_location", "hint", "root_cause", "repeat_error", "snippet")}
    elif name == "next_task":
        hist = state.get("task_history") or []
        base["last_task"] = hist[-1] if hist else None
    return base


def _question(point: str, name: str, state: dict) -> str:
    task = state.get("current_task") or {}
    if point == "planner":
        return "Here is the plan. Does it make sense? Approve it, edit the tasks, or tell me what to change."
    if point == "developer":
        return "Here is the first file. Is this the right approach?"
    if point == "qa_failure":
        return (f"Task {task.get('order')} failed QA (attempt {state.get('attempt_count')} of "
                f"{run_config.max_qa_attempts()}). Retry with the analyzer's fix, add your own guidance, or skip this task?")
    return f"Paused after {name.replace('_', ' ')} as you requested. Continue, or give guidance?"


def _build_info(point: str, name: str, state: dict) -> PauseInfo:
    return PauseInfo(
        point=point, node=name, question=_question(point, name, state), options=list(_OPTIONS[name]),
        payload=_payload(name, state), timeout_s=int(settings.intervention_timeout_seconds or 0),
    )


def _snapshot(state: dict, info: PauseInfo) -> str:
    """JSON view of the state at the pause — enough to inspect or recover
    the partial results (plan, code so far, task history, user feedback)."""
    snap = {k: state.get(k) for k in (
        "project_id", "request", "communication_mode", "task_list", "task_index", "current_task", "code_filename",
        "attempt_count", "task_history", "user_feedback", "test_result", "needs_human_review")}
    snap["code"] = (state.get("code") or "")[:20000]
    snap["paused"] = {"point": info.point, "node": info.node, "since": info.since}
    return json.dumps(snap, default=str)


async def _pause(ctl: Control, state: dict, info: PauseInfo) -> Feedback:
    pid = state["project_id"]
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    ctl.waiting, ctl._future, ctl._answered = info, fut, False
    ctl.pauses += 1
    state["paused_at"] = info.point

    try:  # best-effort snapshot for inspection / partial-result recovery
        await asyncio.to_thread(
            stm.set_working, pid, "paused_snapshot", _snapshot(state, info),
            (info.timeout_s or 3600) + 3600,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not snapshot state at pause for %s: %s", pid, exc)

    await events.emit(
        pid, ev.PIPELINE_PAUSED, "system", info.question,
        {"point": info.point, "node": info.node, "question": info.question, "options": info.options,
         "payload": info.payload, "timeout_s": info.timeout_s, "since": info.since},
    )
    logger.info("project %s paused at %s (after %s)", pid, info.point, info.node)

    t0 = time.perf_counter()
    try:
        fb = await asyncio.wait_for(fut, info.timeout_s or None)
    except asyncio.TimeoutError:
        fb = Feedback(action="approve", source="timeout")
        logger.info("pause at %s for %s timed out; auto-approving", info.point, pid)
    finally:
        waited_ms = (time.perf_counter() - t0) * 1000.0
        ctl.paused_ms += waited_ms
        ctl.waiting, ctl._future = None, None
        state["paused_at"] = None

    metrics.record("pause", info.point, waited_ms, scope_=metrics.Scope(pid, "intervention"))
    try:
        await asyncio.to_thread(
            db.log_intervention, pid, "pause", info.point, info.node, fb.action, fb.guidance, fb.clarification,
            fb.tasks, fb.source, info.question, int(waited_ms),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not log intervention for %s: %s", pid, exc)

    await events.emit(
        pid, ev.PIPELINE_RESUMED, "system",
        _resumed_message(fb), {"point": info.point, "action": fb.action, "guidance": fb.guidance,
                                "clarification": fb.clarification, "edited_plan": bool(fb.tasks),
                                "source": fb.source, "waited_ms": int(waited_ms)},
    )
    return fb


def _resumed_message(fb: Feedback) -> str:
    if fb.source == "timeout":
        return "No answer received — continuing automatically."
    return {
        "approve": "Approved — continuing.",
        "revise": "Feedback received — applying it.",
        "skip_task": "Skipping this task at your request.",
        "abort": "Run cancelled at your request.",
    }.get(fb.action, "Continuing.")


async def _apply_task_edit(state: dict, descriptions: list[str]) -> None:
    """Replaces the plan with the user's edited task list. Types are kept
    by position (so an edited webapp task stays a webapp task); new tasks
    default to plain code tasks."""
    pid = state["project_id"]
    old = state.get("task_list") or []
    new = []
    for i, desc in enumerate(descriptions, start=1):
        ttype = old[i - 1].get("type", "code") if i - 1 < len(old) else "code"
        fw = (old[i - 1].get("framework") if i - 1 < len(old) else None) or state.get("framework")
        new.append(Task(order=i, description=desc.strip(), type=ttype, framework=fw).model_dump())
    state["task_list"] = new
    state["task_index"] = 0
    state["current_task"] = new[0]
    state["attempt_count"] = 0

    def persist() -> None:
        db.delete_tasks_after(pid, len(new))
        for t in new:
            db.upsert_task(pid, t["order"], t["description"], "pending", task_type=t["type"])
    try:
        await asyncio.to_thread(persist)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not persist edited plan for %s: %s", pid, exc)

    lines = "\n".join(f"{t['order']}. {t['description']}" for t in new)
    await events.emit(pid, ev.PLAN_REVISED, "planner", f"You revised the plan — {len(new)} task(s):\n{lines}",
                      {"tasks": new, "by": "user"})


async def _apply(name: str, point: str, state: dict, fb: Feedback) -> bool:
    """Applies the user's answer to the state. Returns True if the node
    that just finished should be RE-RUN (re-plan / regenerate)."""
    if fb.action == "abort":
        raise PipelineAborted("aborted by user", state)

    task = state.get("current_task") or {}
    if fb.guidance or fb.clarification:
        state.setdefault("user_feedback", []).append({
            "point": point, "node": name, "action": fb.action, "guidance": fb.guidance,
            "clarification": fb.clarification,
            "scope": "project" if name in ("planner", "next_task") or not task else "task",
            "task_order": task.get("order"), "at": time.time(),
        })

    if fb.action == "skip_task":
        if name == "qa":
            if state.get("_qa_decision") in ("analyze_failure", "retry"):
                state["_qa_decision"] = "escalate"
                state["_skipped_by_user"] = True
        elif name == "failure_analyzer":
            state["_user_skip"] = True
            state["_skipped_by_user"] = True
        return False

    if fb.action == "revise":
        if name == "planner":
            if fb.tasks:
                await _apply_task_edit(state, fb.tasks)
                return False
            return bool(fb.guidance or fb.clarification)  # re-plan with the guidance
        if name == "developer":
            if fb.guidance or fb.clarification:
                state["_guided_regen"] = True
                return True
    return False


def interactive_node(name: str, fn: Callable) -> Callable:
    """Wraps a graph node with the pause gate. Zero cost when nothing is
    pending; see the module docstring for the semantics."""
    async def wrapper(state: dict) -> dict:
        state = await fn(state)
        ctl = _controls.get(state.get("project_id"))
        if ctl is None:
            return state
        forced: Optional[str] = None
        revisions = 0
        while True:
            point = forced or _decide(ctl, name, state)
            if point is None:
                return state
            info = _build_info(point, name, state)
            fb = await _pause(ctl, state, info)
            rerun = await _apply(name, point, state, fb)
            if rerun and revisions < settings.intervention_max_revisions:
                revisions += 1
                forced = point
                state = await fn(state)
                continue
            if point == "developer":
                ctl.dev_first_done = True
            return state
    wrapper.__name__ = f"interactive_{name}"
    return wrapper
