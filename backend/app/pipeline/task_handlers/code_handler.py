"""Handles Task(type="code") — the original, default task type: one file
generated for one independently-testable task. Logic is unchanged from the
pre-reorg _run_code_task in graph.py; this is a pure move, not a rewrite.
"""
import asyncio
import logging
import time

from app.agents import developer
from app.mcp_client import call_tool
from app import events, ifs
from app.pipeline import event_types as ev
from app.pipeline.task_handlers.base import TaskHandler

logger = logging.getLogger("aiopsforge.pipeline.code_handler")


class CodeTaskHandler(TaskHandler):
    qa_tool_name = "run_tests"

    async def generate(
        self, state, task, t0, is_retry, previous_failure, communication_mode,
        memory_context, used_memory,
    ) -> dict:
        filename_hint = state.get("code_filename") if is_retry else None
        # Phase 11 fix: the Failure Analyzer node (app.pipeline.nodes.
        # failure_analyzer_node) already runs before this handler on every
        # retry and computes a root cause / affected function / repair hint
        # — it was being stored in state["_failure_analysis"] and shown in
        # the live event feed, but never actually reaching the Developer's
        # prompt. Passing it through here is the fix for "developer isn't
        # getting enough context from the tester."
        failure_analysis = state.get("_failure_analysis") if is_retry else None
        code, filename = await asyncio.to_thread(
            developer.generate_code,
            task,
            previous_failure=previous_failure if is_retry else None,
            filename_hint=filename_hint,
            memory_context=memory_context,
            original_request=state["request"] if communication_mode == "blackboard" else None,
            failure_analysis=failure_analysis,
        )

        await call_tool(
            "write_file",
            {"project_id": state["project_id"], "path": filename, "content": code},
        )

        state["code"] = code
        state["code_filename"] = filename
        state["memory_used"] = used_memory
        # Phase 6: computed regardless of mode (not just in blackboard mode) so
        # both arms are directly comparable — the score measures how much of
        # the ORIGINAL request's technical detail survives into this output,
        # whether or not the Developer was even given that original text.
        state["ifs_score"] = ifs.compute_ifs(state["request"], f"{filename}\n{code}")
        duration_ms = int((time.time() - t0) * 1000)
        logger.info(
            "developer: task %d attempt %d -> %s (%.1fs)%s [ifs=%s]",
            task.order, state["attempt_count"] + 1, filename, duration_ms / 1000,
            " [memory used]" if used_memory else "", state["ifs_score"],
        )
        from app.pipeline.nodes import _log_run_safe  # local import avoids a circular import at module load
        _log_run_safe(
            state["project_id"], "developer",
            {"task": task.description, "attempt": state["attempt_count"] + 1, "is_retry": is_retry},
            {
                "filename": filename, "code_length": len(code), "memory_used": used_memory,
                "communication_mode": communication_mode, "ifs_score": state["ifs_score"],
            },
            duration_ms=duration_ms,
        )
        # Phase 7: full code included (not just filename/length) so the live
        # UI can actually show what got written, not just narrate that
        # something did. Capped so one giant file can't blow up an event
        # payload — the truncation only affects what streams to the demo UI,
        # never what's written to disk.
        await events.emit(
            state["project_id"], ev.DEVELOPER_DONE, "developer",
            f"Wrote {filename} ({len(code)} chars) via MCP write_file — sending to QA.",
            {
                "filename": filename, "code_length": len(code), "duration_ms": duration_ms,
                "ifs_score": state["ifs_score"], "code": code[:6000],
                "truncated": len(code) > 6000,
            },
        )
        return state
