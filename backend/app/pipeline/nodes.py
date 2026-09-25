"""The actual LangGraph node functions. Each does ONE thing, mirroring the
Planner/Developer/QA agent boundary.

Phase 11 changes:
  - Failure Analyzer: QA failures go through targeted analysis before retry
  - Deterministic QA: syntactic checks run first, before LLM-based QA
  - Memory enhancements: execution context captured and reused

developer_node and qa_node are deliberately thin: they hold the logic that
is the SAME regardless of task type (loading state, short-term-memory
bookkeeping, event narration, long-term memory retrieval, deciding
retry/escalate/next) and defer everything task-type-specific to
app.pipeline.task_handlers — see that package for how to add a new task
type without touching this file.
"""
import asyncio
import json
import logging
import time
from pathlib import Path

from app.config import settings
from app.mcp_client import call_tool
from app.pipeline import event_types as ev
from app.pipeline.task_handlers import get_task_handler
from app.schemas import Task, TestResult
from app.agents import planner, qa, failure_analyzer
from app.agents.qa_deterministic import run_deterministic_qa
from app import db, events, memory, short_term_memory as stm

logger = logging.getLogger("aiopsforge.pipeline.nodes")


# ------------------------------------------------------- shared helpers ----

def _log_run_safe(project_id: str, agent: str, input_data: dict, output_data: dict,
                   status: str = "ok", duration_ms: int | None = None) -> None:
    """Best-effort run logging — a DB hiccup should never take down the
    pipeline itself, so failures here are logged and swallowed.
    """
    try:
        db.log_run(project_id, agent, input_data, output_data, status, duration_ms)
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to log run for agent=%s: %s", agent, exc)


def _retrieve_memory_safe(problem_signature: str) -> list[dict]:
    """Best-effort long-term memory lookup — an embedding-service hiccup
    should degrade to 'no memory context' rather than break the pipeline.
    """
    try:
        return memory.retrieve_similar(problem_signature)
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory retrieval failed, continuing without it: %s", exc)
        return []


def _save_memory_safe(project_id: str, problem_signature: str, solution_summary: str) -> None:
    try:
        memory.save_memory(project_id, problem_signature, solution_summary)
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to save memory entry, continuing: %s", exc)


# ---------------------------------------------------------------- nodes ----

async def planner_node(state: dict) -> dict:
    t0 = time.time()
    await events.emit(
        state["project_id"], ev.PLANNER_START, "planner",
        "Reading your request and breaking it into tasks…",
    )
    tasks = await asyncio.to_thread(planner.plan, state["request"])
    state["task_list"] = [t.model_dump() for t in tasks]
    state["task_index"] = 0
    state["current_task"] = state["task_list"][0] if state["task_list"] else None
    state["attempt_count"] = 0
    duration_ms = int((time.time() - t0) * 1000)
    logger.info("planner: %d task(s) in %.1fs", len(tasks), duration_ms / 1000)
    _log_run_safe(
        state["project_id"], "planner",
        {"request": state["request"]},
        {"task_count": len(tasks), "tasks": [t.model_dump() for t in tasks]},
        duration_ms=duration_ms,
    )
    # Insert one 'pending' row per task right away, so a polling dashboard
    # sees the full plan immediately rather than tasks appearing one at a time.
    for t in tasks:
        try:
            db.upsert_task(state["project_id"], t.order, t.description, "pending")
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to upsert pending task %d: %s", t.order, exc)

    plan_lines = "\n".join(f"{t.order}. {t.description}" for t in tasks)
    await events.emit(
        state["project_id"], ev.PLANNER_DONE, "planner",
        f"Planned {len(tasks)} task(s):\n{plan_lines}",
        {"tasks": [t.model_dump() for t in tasks], "duration_ms": duration_ms},
    )
    return state


async def developer_node(state: dict) -> dict:
    """Shared prefix (task loading, short-term-memory bookkeeping, event
    narration, long-term memory retrieval) that applies regardless of task
    type, then defers to that task type's handler for the actual work.
    """
    t0 = time.time()
    task = Task(**state["current_task"])
    previous_failure = (
        TestResult(**state["test_result"]) if state.get("test_result") else None
    )
    is_retry = state["attempt_count"] > 0

    # Week 8: record working state in short-term memory as we go — this is
    # what a live dashboard or a debugging session would read mid-run.
    # Namespaced per project_id, so it can never collide with another
    # project's in-flight state.
    stm.set_working(state["project_id"], "current_task", task.description)
    stm.set_working(state["project_id"], "attempt_count", str(state["attempt_count"] + 1))
    try:
        db.upsert_task(state["project_id"], task.order, task.description, "in_progress", state["attempt_count"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to upsert in_progress task %d: %s", task.order, exc)

    retry_reason = ""
    if is_retry and previous_failure is not None:
        detail = (previous_failure.error or previous_failure.output or "").strip()
        if detail:
            retry_reason = f" Fixing: {detail[:200]}"

    await events.emit(
        state["project_id"],
        ev.DEVELOPER_RETRY if is_retry else ev.DEVELOPER_START,
        "developer",
        (
            f"Retrying task {task.order} (attempt {state['attempt_count'] + 1}): "
            f"{task.description}{retry_reason}"
            if is_retry else
            f"Writing code for task {task.order}: {task.description}"
        ),
        {"task_order": task.order, "attempt": state["attempt_count"] + 1, "is_retry": is_retry},
    )

    # Week 7: on a FRESH attempt (not a retry), check long-term memory for a
    # similar problem solved on any past project. On a retry we already have
    # a specific failure to fix, so memory context would just be noise.
    # Computed generically here (not inside a handler) since it's the same
    # regardless of task type — a handler is free to ignore it, as
    # WebappTaskHandler currently does.
    memory_context = None
    used_memory = False
    if not is_retry:
        memory_context = await asyncio.to_thread(_retrieve_memory_safe, task.description)
        used_memory = bool(memory_context)
        if used_memory:
            await events.emit(
                state["project_id"], ev.MEMORY_HIT, "developer",
                f"Found {len(memory_context)} similar fix(es) in long-term memory — reusing as a hint.",
                {"matches": memory_context},
            )

    comm_mode = state.get("communication_mode", "blackboard")
    handler = get_task_handler(task.type)
    return await handler.generate(
        state, task, t0, is_retry, previous_failure, comm_mode, memory_context, used_memory,
    )


async def qa_node(state: dict) -> dict:
    """Runs the check appropriate to this task's type and ALSO decides the
    retry/escalate/next outcome here, storing it in state['_qa_decision'].

    Phase 11 fixes (two bugs found together while wiring this up):

    1. `code_filepath` was passed as the bare filename (e.g. "foo.py"),
       but the Developer writes files via MCP into
       {workspace_root}/{project_id}/{filename} (see mcp_server/workspace.py),
       not the backend container's CWD. The deterministic check could
       therefore never actually find the file and always failed with a
       spurious "File not found", silently defeating its entire purpose
       (never catching a real syntax error early, never skipping the
       expensive real test run for one). Fixed by resolving the same
       workspace path the MCP tools use.

    2. Once (1) is fixed and the file IS found, the deterministic check
       passing was being treated as the FINAL result — the actual test
       suite (handler.qa_tool_name via MCP run_tests/check_webapp) was
       skipped entirely on a deterministic pass. That's backwards: "the
       file compiles and its imports parse" says nothing about whether the
       code is CORRECT, only that it's not obviously broken. Deterministic
       checks now only fast-fail (skip the real test run when we already
       know the file won't even import — no point spending an MCP round
       trip on that) and otherwise fall through to the real, authoritative
       test execution — matching the target flow's "Developer -> QA" step
       being one deterministic-first check, not two independent QA passes.
       Note: what this module's comments call "LLM-based QA" below is a
       legacy label from before this rework — handler.qa_tool_name (pytest
       via MCP run_tests, or check_webapp's boot-and-probe) involves no LLM
       reasoning at all; it's the real, deterministic test execution. The
       only LLM in the failure path is failure_analyzer_node, and it only
       runs after a genuine failure, per "LLM-based QA only where reasoning
       is actually required."
    """
    t0 = time.time()
    task_type = state["current_task"].get("type", "code")
    handler = get_task_handler(task_type)
    # Captured before it's overwritten below with THIS attempt's result —
    # needed if this attempt passes, to record what failure it fixed.
    prior_test_result = state.get("test_result")

    await events.emit(
        state["project_id"], ev.QA_START, "qa",
        handler.describe_qa_start(state, state["current_task"]),
        {"filename": state.get("code_filename"), "tool": handler.qa_tool_name},
    )

    # Fast pre-flight check (syntax, imports, file existence) — catches
    # obviously-broken output without the cost of an MCP round trip and a
    # subprocess pytest run. Resolve the SAME path the MCP write_file tool
    # actually wrote to (workspace_root/project_id/filename), not a bare
    # filename relative to this container's CWD.
    code_filepath = None
    if state.get("code_filename"):
        code_filepath = str(Path(settings.workspace_root) / state["project_id"] / state["code_filename"])

    det_passed, det_msg = await asyncio.to_thread(
        run_deterministic_qa,
        state["project_id"],
        task_type,
        code_filepath=code_filepath,
    )

    if not det_passed:
        # Fail fast: don't spend an MCP call + subprocess pytest run on code
        # that doesn't even compile/import. This IS the QA result — it goes
        # through the normal retry/escalate accounting below, same as a
        # real test failure would.
        await events.emit(
            state["project_id"], ev.QA_RESULT, "qa",
            f"Deterministic pre-flight check failed: {det_msg}",
            {"passed": False, "deterministic": True},
        )
        result = TestResult(passed=False, output=det_msg, error=det_msg)
        state["test_result"] = result.model_dump()
        duration_ms = int((time.time() - t0) * 1000)
        _log_run_safe(
            state["project_id"], "qa",
            {"task": state["current_task"]["description"], "attempt": state["attempt_count"] + 1},
            {"passed": False, "error": det_msg, "deterministic": True},
            status="error",
            duration_ms=duration_ms,
        )
        if state["attempt_count"] + 1 >= settings.max_qa_attempts:
            state["_qa_decision"] = "escalate"
        else:
            state["attempt_count"] += 1
            state["_qa_decision"] = "analyze_failure"
            await events.emit(
                state["project_id"], ev.RETRY_SCHEDULED, "system",
                f"Sending failure to Failure Analyzer before retry (attempt {state['attempt_count'] + 1}/{settings.max_qa_attempts}).",
                {"attempt": state["attempt_count"] + 1, "max_attempts": settings.max_qa_attempts},
            )
        return state

    logger.info("deterministic pre-flight passed: %s — running real test suite", det_msg)

    raw = await call_tool(handler.qa_tool_name, {"project_id": state["project_id"]})
    result = qa.parse_run_tests_output(raw)
    state["test_result"] = result.model_dump()
    duration_ms = int((time.time() - t0) * 1000)
    logger.info(
        "qa: task %d attempt %d -> passed=%s (%.1fs)",
        state["current_task"]["order"], state["attempt_count"] + 1, result.passed, duration_ms / 1000,
    )
    _log_run_safe(
        state["project_id"], "qa",
        {"task": state["current_task"]["description"], "attempt": state["attempt_count"] + 1},
        {"passed": result.passed, "error": result.error},
        status="ok" if result.passed else "error",
        duration_ms=duration_ms,
    )
    await events.emit(
        state["project_id"], ev.QA_RESULT, "qa",
        "Tests passed." if result.passed else f"Tests failed: {result.error}",
        {"passed": result.passed, "error": result.error, "duration_ms": duration_ms},
    )

    # Values here are the exact edge keys registered in add_conditional_edges
    # for the "qa" node in app.pipeline.graph — route_after_qa there just
    # echoes this back.
    if result.passed:
        state["_qa_decision"] = "next_task"
        # Week 7 / Phase 11: a task that just passed is worth remembering.
        # Capture execution context for future repairs — including, when
        # this was a retry, WHAT failure was fixed and the diagnosis that
        # fixed it, not just the final working code. This is what makes a
        # future failure_analyzer_node lookup ("similar past fixes")
        # actually useful for a repair strategy, not just a code snippet.
        analysis = state.get("_failure_analysis") or {}
        exec_context = {
            "files_changed": [state.get("code_filename")],
            "model": settings.ollama_model,
            "task_type": task_type,
            "attempt_count": state["attempt_count"] + 1,
            "outcome": "passed",
            "failure_fixed": (
                (prior_test_result.get("error") or prior_test_result.get("output") or "")[:300]
                if state["attempt_count"] > 0 and prior_test_result else None
            ),
            "root_cause": analysis.get("root_cause"),
            "repair_strategy": analysis.get("repair_strategy"),
        }
        await asyncio.to_thread(
            memory.save_memory,
            state["project_id"],
            state["current_task"]["description"],
            f"Working solution ({state.get('code_filename', 'file')}):\n{state.get('code', '')[:1000]}",
            execution_context=exec_context,
        )
    elif state["attempt_count"] + 1 >= settings.max_qa_attempts:
        state["_qa_decision"] = "escalate"
    else:
        state["attempt_count"] += 1
        state["_qa_decision"] = "analyze_failure"  # Phase 11: go to analyzer
        await events.emit(
            state["project_id"], ev.RETRY_SCHEDULED, "system",
            f"Sending failure to Failure Analyzer before retry (attempt {state['attempt_count'] + 1}/{settings.max_qa_attempts}).",
            {"attempt": state["attempt_count"] + 1, "max_attempts": settings.max_qa_attempts},
        )

    return state


async def next_task_node(state: dict) -> dict:
    """Records the outcome of the task just finished, then advances to the
    next task (or marks the whole run done).
    """
    task = state["current_task"]
    result = state.get("test_result")
    final_status = "passed" if (result and result["passed"]) else (
        "needs_human_review" if state.get("_last_escalated") else "failed"
    )
    await events.emit(
        state["project_id"], ev.TASK_COMPLETE, "system",
        f"Task {task['order']} finished: {final_status}.",
        {"task_order": task["order"], "status": final_status},
    )
    state["task_history"].append(
        {
            "task": task["description"],
            "order": task["order"],
            "passed": result["passed"] if result else None,
            "attempts": state["attempt_count"] + 1,
            "filename": state.get("code_filename"),
            "escalated": state.get("_last_escalated", False),
            "memory_used": state.get("memory_used", False),
            "communication_mode": state.get("communication_mode", "blackboard"),
            "ifs_score": state.get("ifs_score"),
        }
    )
    try:
        db.upsert_task(
            state["project_id"], task["order"], task["description"], final_status,
            state["attempt_count"] + 1, state.get("memory_used", False),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to upsert final task status for task %d: %s", task["order"], exc)
    state["_last_escalated"] = False
    state["memory_used"] = False
    state["ifs_score"] = None
    state["_failure_analysis"] = None

    state["task_index"] += 1
    state["attempt_count"] = 0
    state["test_result"] = None
    state["code"] = None
    state["code_filename"] = None

    if state["task_index"] < len(state["task_list"]):
        state["current_task"] = state["task_list"][state["task_index"]]
        state["done"] = False
    else:
        state["current_task"] = None
        state["done"] = True

    return state


async def failure_analyzer_node(state: dict) -> dict:
    """Phase 11: Analyzes a QA failure to determine a targeted repair strategy.
    Runs after QA fails but before retry, so Developer can do focused fixes.
    """
    t0 = time.time()
    result = TestResult(**state["test_result"])
    
    await events.emit(
        state["project_id"], ev.DEVELOPER_RETRY, "failure_analyzer",
        f"Analyzing failure for task {state['current_task']['order']} (attempt {state['attempt_count']})...",
        {"error": result.error},
    )
    
    analysis = await asyncio.to_thread(
        failure_analyzer.analyze_failure,
        result,
        code_filename=state.get("code_filename"),
    )
    
    state["_failure_analysis"] = analysis
    duration_ms = int((time.time() - t0) * 1000)
    
    logger.info(
        "failure analysis: file=%s strategy=%s similar=%d (%.1fs)",
        analysis.get("failed_file"), analysis.get("repair_strategy"),
        len(analysis.get("similar_fixes", [])), duration_ms / 1000,
    )
    
    hint = analysis.get("hint", "")
    similar_count = len(analysis.get("similar_fixes", []))
    msg = f"Failed file: {analysis.get('failed_file')}\n{hint}"
    if similar_count > 0:
        msg += f"\nFound {similar_count} similar fix(es) in memory."
    
    await events.emit(
        state["project_id"], ev.RETRY_SCHEDULED, "failure_analyzer",
        msg,
        {
            "analysis": analysis,
            "similar_fixes": similar_count,
            "duration_ms": duration_ms,
        },
    )
    
    return state


async def escalate_node(state: dict) -> dict:
    """Reached when a task exhausts its retry budget. Flags the whole run
    for human review but still lets the pipeline continue to remaining
    tasks rather than stopping dead.
    """
    logger.warning(
        "escalate: task %d needs human review after %d attempts",
        state["current_task"]["order"], state["attempt_count"] + 1,
    )
    await events.emit(
        state["project_id"], ev.ESCALATE, "system",
        f"Task {state['current_task']['order']} needs human review after "
        f"{state['attempt_count'] + 1} failed attempts.",
        {"task_order": state["current_task"]["order"], "attempts": state["attempt_count"] + 1},
    )
    state["needs_human_review"] = True
    state["_last_escalated"] = True
    return state


# ------------------------------------------------------------- routing ----

def route_after_qa(state: dict) -> str:
    """Pure lookup — all the actual decision logic (and the attempt_count
    bookkeeping) already happened in qa_node, for the reason explained there.
    
    Phase 11: Routes to 'analyze_failure' for targeted repair before retry.
    """
    return state["_qa_decision"]


def route_after_next_task(state: dict) -> str:
    return "developer" if not state["done"] else "end"
