"""The real AIOpsForge graph: Planner -> Developer -> QA, with targeted
failure analysis and smart retry (Phase 11).

Graph shape (Phase 11 with Failure Analyzer):

    planner --> developer --> qa --+-- (passed) ----------------------+
                   ^               |                                 v
                   |               +-- (failed, attempts<max) --> analyze_failure --> developer
                   |               |                                  |
                   |               +-- (failed, attempts>=max) --------+--> escalate --> next_task
                   |                                                  |
                   +----------------------------- next_task <---------+
                                          |
                                (more tasks) --> developer
                                          |
                                (no more tasks) --> END

Phase 11 improvements:
  - Deterministic QA runs first (fast, no LLM)
  - Failure Analyzer produces targeted repair strategies
  - Developer can do focused fixes via analysis hints
  - Memory captures execution context for future repairs

This module is deliberately just wiring: node behavior lives in
app.pipeline.nodes, task-type-specific generation logic lives in
app.pipeline.task_handlers. Nothing about a new task type, a new retry
policy detail, or a new event type should ever require editing this file.
"""
import asyncio
import logging
import time

from langgraph.graph import StateGraph, END

from app import run_config
from app.config import settings
from app.pipeline import event_types as ev
from app.pipeline.nodes import (
    planner_node, developer_node, qa_node, failure_analyzer_node,
    next_task_node, escalate_node,
    route_after_qa, route_after_analyzer, route_after_next_task,
)
from app.pipeline import intervention, metrics
from app.schemas import AgentState
from app import events, short_term_memory as stm

logger = logging.getLogger("aiopsforge.pipeline.graph")


def build_graph():
    graph = StateGraph(dict)

    def wire(name, fn, interactive=True):
        # Order matters: the intervention gate wraps OUTSIDE the metrics
        # wrapper, so time spent waiting for a human is never counted as
        # node latency. `escalate` has no pause point of its own.
        node = metrics.instrument_node(name, fn)
        graph.add_node(name, intervention.interactive_node(name, node) if interactive else node)

    wire("planner", planner_node)
    wire("developer", developer_node)
    wire("qa", qa_node)
    wire("failure_analyzer", failure_analyzer_node)
    wire("next_task", next_task_node)
    wire("escalate", escalate_node, interactive=False)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "developer")
    graph.add_edge("developer", "qa")

    graph.add_conditional_edges(
        "qa", route_after_qa,
        {
            "retry": "developer",
            "next_task": "next_task",
            "escalate": "escalate",
            "analyze_failure": "failure_analyzer",  # Phase 11
        },
    )
    
    # Phase 11: After failure analysis, retry to developer — or, if the
    # user chose "skip this task" at the qa_failure pause, escalate it.
    graph.add_conditional_edges(
        "failure_analyzer", route_after_analyzer,
        {"developer": "developer", "escalate": "escalate"},
    )
    graph.add_edge("escalate", "next_task")

    graph.add_conditional_edges(
        "next_task", route_after_next_task,
        {"developer": "developer", "end": END},
    )

    return graph.compile()


async def run_pipeline(project_id: str, request: str, communication_mode: str = "blackboard",
                       interactive: bool = False, pause_points: list[str] | None = None,
                       framework: str | None = None, overrides: dict | None = None) -> AgentState:
    """Runs one project to completion.

    `interactive=True` parks the run for the user's review at the chosen
    `pause_points` (see app.pipeline.intervention); with it off the run
    never pauses on its own, but can still be paused on demand via
    POST /projects/{id}/intervention. If the user aborts, the partial
    results are returned with `aborted=True` rather than raised.

    `framework` is the user's explicit framework choice (validated by the
    caller); None auto-detects it from the request text.

    `overrides` (see app.run_config): per-run model / max_qa_attempts /
    use_memory — how A/B experiments vary a setting between concurrent runs.
    """
    overrides_token = run_config.push(overrides)
    # Captured now: record_run() runs after the finally block below has restored the overrides.
    run_model, run_overrides = run_config.model(), run_config.current()
    ctl = intervention.register(project_id, interactive, pause_points)
    app = build_graph()
    initial: dict = AgentState(
        project_id=project_id, request=request, communication_mode=communication_mode,
        framework_choice=framework,
    ).model_dump()
    await events.emit(
        project_id, ev.PIPELINE_START, "system",
        f'New request ({communication_mode} mode): "{request}"',
        {"request": request, "communication_mode": communication_mode},
    )
    # Metrics: one `pipeline` row per run (total wall time / outcome / model
    # / mode). Node-level rows are written by metrics.instrument_node.
    run_scope = metrics.Scope(project_id, "pipeline")
    run_t0, run_wall = time.perf_counter(), time.time()

    async def record_run(ok: bool, fw: str | None = None) -> None:
        wall_ms = (time.perf_counter() - run_t0) * 1000
        # Time the run spent parked waiting for a human is not pipeline
        # latency: total_ms excludes it (and it is kept separately).
        metrics.record(
            "pipeline", communication_mode, max(wall_ms - ctl.paused_ms, 0.0), ok=ok, scope_=run_scope,
            t_start_ms=int(run_wall * 1000),
            meta={"model": run_model, "request": request[:200],
                  "paused_ms": int(ctl.paused_ms), "wall_ms": int(wall_ms), "pauses": ctl.pauses,
                  "interactive": interactive, "framework": fw, "overrides": run_overrides},
        )
        try:
            await asyncio.to_thread(metrics.flush, project_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("final metrics flush failed for %s: %s", project_id, exc)

    aborted = False
    try:
        final = await app.ainvoke(initial, config={"recursion_limit": 100})
    except intervention.PipelineAborted as exc:
        # The user pressed abort: keep everything produced so far.
        aborted = True
        final = {**initial, **exc.state, "aborted": True, "paused_at": None}
    except BaseException:
        await record_run(False)   # a crashed run still gets its row (ok=False)
        raise
    finally:
        intervention.release(project_id)
        run_config.pop(overrides_token)
        # Explicit cleanup the moment a run ends (success OR escalation),
        # rather than only relying on the TTL to eventually expire the
        # keys. Best-effort — a Redis hiccup here shouldn't mask the
        # pipeline's actual result.
        try:
            deleted = stm.clear_project(project_id)
            logger.info("cleared %d short-term memory key(s) for project %s", deleted, project_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to clear short-term memory for %s: %s", project_id, exc)

    result = AgentState(**final)
    await record_run(bool(result.done and not result.needs_human_review and not aborted), result.framework)
    if aborted:
        message = "Run cancelled. Files written so far are still available to download."
    elif result.done and not result.needs_human_review:
        message = "All tasks passed."
    else:
        message = "Finished — one or more tasks need human review."
    await events.emit(
        project_id, ev.PIPELINE_DONE, "system", message,
        {"done": result.done, "needs_human_review": result.needs_human_review, "aborted": aborted},
    )
    return result
