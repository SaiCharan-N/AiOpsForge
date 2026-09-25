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
import logging

from langgraph.graph import StateGraph, END

from app.pipeline import event_types as ev
from app.pipeline.nodes import (
    planner_node, developer_node, qa_node, failure_analyzer_node,
    next_task_node, escalate_node,
    route_after_qa, route_after_next_task,
)
from app.schemas import AgentState
from app import events, short_term_memory as stm

logger = logging.getLogger("aiopsforge.pipeline.graph")


def build_graph():
    graph = StateGraph(dict)

    graph.add_node("planner", planner_node)
    graph.add_node("developer", developer_node)
    graph.add_node("qa", qa_node)
    graph.add_node("failure_analyzer", failure_analyzer_node)
    graph.add_node("next_task", next_task_node)
    graph.add_node("escalate", escalate_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "developer")
    graph.add_edge("developer", "qa")

    graph.add_conditional_edges(
        "qa", route_after_qa,
        {
            "next_task": "next_task",
            "escalate": "escalate",
            "analyze_failure": "failure_analyzer",  # Phase 11
        },
    )
    
    # Phase 11: After failure analysis, retry to developer
    graph.add_edge("failure_analyzer", "developer")
    graph.add_edge("escalate", "next_task")

    graph.add_conditional_edges(
        "next_task", route_after_next_task,
        {"developer": "developer", "end": END},
    )

    return graph.compile()


async def run_pipeline(project_id: str, request: str, communication_mode: str = "blackboard") -> AgentState:
    app = build_graph()
    initial: dict = AgentState(
        project_id=project_id, request=request, communication_mode=communication_mode,
    ).model_dump()
    await events.emit(
        project_id, ev.PIPELINE_START, "system",
        f'New request ({communication_mode} mode): "{request}"',
        {"request": request, "communication_mode": communication_mode},
    )
    try:
        final = await app.ainvoke(initial, config={"recursion_limit": 100})
    finally:
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
    await events.emit(
        project_id, ev.PIPELINE_DONE, "system",
        (
            "All tasks passed."
            if result.done and not result.needs_human_review
            else "Finished — one or more tasks need human review."
        ),
        {"done": result.done, "needs_human_review": result.needs_human_review},
    )
    return result
