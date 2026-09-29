"""Handles Task(type="webapp_scaffold") — a multi-file FastAPI+Jinja2 app,
written file-by-file with a live event per file. Logic is unchanged from
the pre-reorg _run_webapp_scaffold in graph.py; this is a pure move.
"""
import asyncio
import logging
import time

from app.agents import scaffolder
from app.mcp_client import call_tool
from app import events, ifs
from app.pipeline import event_types as ev
from app.pipeline.task_handlers.base import TaskHandler

logger = logging.getLogger("aiopsforge.pipeline.webapp_handler")


class WebappTaskHandler(TaskHandler):
    qa_tool_name = "check_webapp"

    def describe_qa_start(self, state, task) -> str:
        return "Checking the scaffolded web app compiles cleanly via MCP…"

    async def generate(
        self, state, task, t0, is_retry, previous_failure, communication_mode,
        memory_context, used_memory,
    ) -> dict:
        """`memory_context`/`used_memory` are accepted (per the TaskHandler
        contract) but deliberately unused — webapp scaffolding doesn't yet
        draw on long-term memory (see README "Implementation nice-to-haves").
        """
        files = await asyncio.to_thread(
            scaffolder.scaffold_webapp,
            task.description,
            previous_failure=previous_failure if is_retry else None,
            original_request=state["request"] if communication_mode == "blackboard" else None,
            failure_analysis=state.get("_failure_analysis") if is_retry else None,
        )

        for path, content in files.items():
            await call_tool(
                "write_file",
                {"project_id": state["project_id"], "path": path, "content": content},
            )
            await events.emit(
                state["project_id"], ev.DEVELOPER_FILE_WRITTEN, "developer",
                f"Wrote {path} ({len(content)} chars)",
                {"filename": path, "code": content[:6000], "truncated": len(content) > 6000},
            )

        # Best-effort per-project virtualenv + dependency install, matching
        # the project's stated per-project isolation goal. Failure here does
        # NOT fail the task — QA's check_webapp only verifies the code
        # itself, so a slow/offline pip install doesn't block the pipeline;
        # it just means the .venv won't be ready for check_webapp's runtime
        # boot-and-probe stage, which itself skips gracefully when that
        # happens (see mcp_server/tools/check_webapp.py:_probe_runtime).
        if "requirements.txt" in files:
            await events.emit(
                state["project_id"], ev.VENV_SETUP, "system",
                "Setting up an isolated virtualenv and installing dependencies…",
                {},
            )
            venv_result = await call_tool(
                "run_shell_command",
                {
                    "project_id": state["project_id"],
                    "command": "python3 -m venv .venv && .venv/bin/pip install --quiet -r requirements.txt",
                    "timeout": 120,
                },
            )
            venv_ok = "exit_code: 0" in str(venv_result)
            await events.emit(
                state["project_id"], ev.VENV_READY if venv_ok else ev.VENV_FAILED, "system",
                "Virtualenv ready." if venv_ok else "Virtualenv setup had issues (code was still written and checked).",
                {"ok": venv_ok},
            )

        all_content = "\n".join(files.values())
        combined_size = sum(len(c) for c in files.values())
        state["code"] = f"Scaffolded {len(files)} file(s): {', '.join(sorted(files))}"
        state["code_filename"] = "main.py" if "main.py" in files else next(iter(files), None)
        state["memory_used"] = False
        state["ifs_score"] = ifs.compute_ifs(state["request"], all_content)
        duration_ms = int((time.time() - t0) * 1000)
        logger.info(
            "scaffolder: task %d -> %d file(s), %d chars total (%.1fs) [ifs=%s]",
            task.order, len(files), combined_size, duration_ms / 1000, state["ifs_score"],
        )
        from app.pipeline.nodes import _log_run_safe  # local import avoids a circular import at module load
        _log_run_safe(
            state["project_id"], "scaffolder",
            {"task": task.description},
            {
                "files": sorted(files), "total_chars": combined_size,
                "communication_mode": communication_mode, "ifs_score": state["ifs_score"],
            },
            duration_ms=duration_ms,
        )
        await events.emit(
            state["project_id"], ev.DEVELOPER_DONE, "developer",
            f"Scaffolded {len(files)} file(s) for the web app — sending to QA.",
            {"files": sorted(files), "duration_ms": duration_ms, "ifs_score": state["ifs_score"]},
        )
        return state
