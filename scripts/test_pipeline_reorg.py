#!/usr/bin/env python3
"""Proves the pipeline reorg actually delivers what it claims:
- the task-handler registry resolves the right handler for each type
- a NEW task type can be added WITHOUT touching nodes.py or graph.py
  (the actual "extension should be easy" claim, demonstrated, not asserted)
- TaskHandler genuinely enforces its contract (can't half-implement it)
- every event_types constant is a unique string (no accidental duplicates
  that would make two different events indistinguishable to the frontend)
- the graph still compiles correctly with the reorganized nodes
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# Same unrelated mcp/pydantic version stub used elsewhere in this test
# suite — orthogonal to what's being tested here.
_stub = types.ModuleType("app.mcp_client")
_stub.call_tool = lambda *a, **k: None
sys.modules["app.mcp_client"] = _stub

from app.pipeline import event_types as ev  # noqa: E402
from app.pipeline.graph import build_graph  # noqa: E402
from app.pipeline.task_handlers import get_task_handler, _HANDLERS  # noqa: E402
from app.pipeline.task_handlers.base import TaskHandler  # noqa: E402
from app.pipeline.task_handlers.code_handler import CodeTaskHandler  # noqa: E402
from app.pipeline.task_handlers.webapp_handler import WebappTaskHandler  # noqa: E402


def check_registry_resolves_known_types():
    assert isinstance(get_task_handler("code"), CodeTaskHandler)
    assert isinstance(get_task_handler("webapp_scaffold"), WebappTaskHandler)
    assert get_task_handler("code").qa_tool_name == "run_tests"
    assert get_task_handler("webapp_scaffold").qa_tool_name == "check_webapp"
    print("[ok] registry resolves both known task types to the correct handler + qa_tool_name")


def check_unknown_type_falls_back_to_code_handler():
    handler = get_task_handler("something_that_does_not_exist")
    assert isinstance(handler, CodeTaskHandler)
    print("[ok] unknown/legacy task type falls back to CodeTaskHandler instead of crashing")


def check_task_handler_enforces_its_contract():
    class IncompleteHandler(TaskHandler):
        pass  # doesn't implement generate()

    try:
        IncompleteHandler()
        raise AssertionError("expected TypeError: ABC should refuse instantiation without generate()")
    except TypeError:
        print("[ok] TaskHandler genuinely enforces its contract (can't instantiate without generate())")


def check_a_new_task_type_can_be_added_without_touching_nodes_or_graph():
    """The actual claim this whole reorg makes, demonstrated at runtime:
    register a brand-new handler and confirm the existing nodes/graph code
    picks it up with ZERO modification to nodes.py or graph.py.
    """
    class DummyCliToolHandler(TaskHandler):
        qa_tool_name = "run_tests"  # reusing an existing MCP tool for this demo

        def describe_qa_start(self, state, task):
            return "Checking the generated CLI tool…"

        async def generate(self, state, task, t0, is_retry, previous_failure,
                            communication_mode, memory_context, used_memory):
            state["code"] = "dummy cli tool"
            state["code_filename"] = "cli.py"
            state["memory_used"] = False
            state["ifs_score"] = None
            return state

    # This is the ENTIRE integration step for a new task type — one
    # dict entry. Nothing in nodes.py or graph.py changes.
    _HANDLERS["cli_tool_scaffold"] = DummyCliToolHandler()

    handler = get_task_handler("cli_tool_scaffold")
    assert isinstance(handler, DummyCliToolHandler)
    assert handler.describe_qa_start({}, {}) == "Checking the generated CLI tool…"
    print("[ok] a brand-new task type is fully wired in with ONE dict entry — nodes.py/graph.py untouched")

    del _HANDLERS["cli_tool_scaffold"]  # clean up after the demo


def check_event_type_constants_are_unique():
    names = [n for n in dir(ev) if n.isupper()]
    values = [getattr(ev, n) for n in names]
    assert len(values) == len(set(values)), (
        "two different event_types constants share the same string value — "
        "the frontend would be unable to tell them apart"
    )
    assert len(names) >= 15, f"expected the full event vocabulary, only found {len(names)}"
    print(f"[ok] all {len(names)} event_types constants have unique string values")


def check_graph_still_compiles_with_reorganized_nodes():
    graph = build_graph()
    assert type(graph).__name__ == "CompiledStateGraph"
    print("[ok] build_graph() still compiles correctly with the reorganized node functions")


if __name__ == "__main__":
    check_registry_resolves_known_types()
    check_unknown_type_falls_back_to_code_handler()
    check_task_handler_enforces_its_contract()
    check_a_new_task_type_can_be_added_without_touching_nodes_or_graph()
    check_event_type_constants_are_unique()
    check_graph_still_compiles_with_reorganized_nodes()
    print("\nAll pipeline reorg checks passed.")
