"""The extension point for adding a new task type to the pipeline.

Before this existed, "which task types exist" was answered by grep-ing
graph.py for `task.type ==` and finding the if/else branches scattered
across developer_node AND qa_node separately (easy to update one and
forget the other). Now it's answered by reading this directory's __init__.

To add a new task type (say, "cli_tool_scaffold"):
  1. Write a new module here implementing TaskHandler (generate() +
     qa_tool_name at minimum).
  2. Register it in task_handlers/__init__.py's _HANDLERS dict.
  3. If it needs a new MCP tool for QA to call, add that tool to
     mcp_server/tools/ and register it in mcp_server/server.py.
That's it — app.pipeline.nodes.developer_node and qa_node need ZERO
changes; they look the handler up by task.type and defer to it.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional


class TaskHandler(ABC):
    """One implementation per Task.type value (see app.schemas.Task)."""

    #: MCP tool name qa_node should call to check this task type's output.
    #: Must return text in the same PASSED/EXIT_CODE/OUTPUT format as
    #: run_tests (see mcp_server/tools/run_tests.py) so app.agents.qa's
    #: existing parser needs no changes for a new task type.
    qa_tool_name: str = "run_tests"

    def describe_qa_start(self, state: dict, task: Any) -> str:
        """Human-readable message shown in the live event feed right
        before QA runs this task type's check. Override for task types
        where "running pytest" isn't the accurate description.
        """
        return f"Running pytest on {state.get('code_filename', 'the generated file')} via MCP…"

    @abstractmethod
    async def generate(
        self,
        state: dict,
        task: Any,
        t0: float,
        is_retry: bool,
        previous_failure: Optional[Any],
        communication_mode: str,
        memory_context: Optional[list[dict]],
        used_memory: bool,
    ) -> dict:
        """Does the actual generation work for this task type: calls the
        relevant app.agents.* function, writes output via
        app.mcp_client.call_tool, emits live events (app.events.emit) for
        anything the demo UI should show, and returns the updated state
        dict.

        Must set, at minimum, before returning:
          state["code"]           — human-readable summary of what was produced
          state["code_filename"]  — primary filename (used by QA's start message)
          state["memory_used"]    — bool
          state["ifs_score"]      — float | None (see app.ifs.compute_ifs)

        `memory_context`/`used_memory` are computed once, generically, by
        app.pipeline.nodes.developer_node before dispatch — a handler is
        free to ignore them (WebappTaskHandler currently does; scaffolding
        doesn't yet draw on long-term memory) or use them (CodeTaskHandler
        does).
        """
        raise NotImplementedError
