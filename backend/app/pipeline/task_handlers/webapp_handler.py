"""Handles Task(type="webapp_scaffold") — a multi-file web app, written
file-by-file with a live event per file. The framework (FastAPI by default;
Flask, Django, Express, Next.js) comes from the task (see app.scaffolds):
it decides which scaffold prompt/starter project is used, how dependencies
are installed (pip in a venv vs npm), and which test/boot runner QA uses.
"""
import asyncio
import logging
import time

from app.agents import scaffolder
from app.scaffolds import get_framework
from app.mcp_client import call_tool
from app import events, ifs
from app.pipeline import event_types as ev, intervention
from app.pipeline.task_handlers.base import TaskHandler

logger = logging.getLogger("aiopsforge.pipeline.webapp_handler")


class WebappTaskHandler(TaskHandler):
    qa_tool_name = "check_webapp"

    @staticmethod
    def _framework_id(state, task) -> str:
        fw = task.get("framework") if isinstance(task, dict) else getattr(task, "framework", None)
        return get_framework(fw or state.get("framework")).id

    def qa_tool_args(self, state, task) -> dict:
        # check_webapp picks the framework's test runner + boot command from this.
        return {"framework": self._framework_id(state, task)}

    def describe_qa_start(self, state, task) -> str:
        fw = get_framework(self._framework_id(state, task))
        steps = {"python": "compile, run its tests, boot it", "node": "check syntax, run its tests, boot it"}[fw.language]
        if fw.id == "next":
            steps = "build it and boot it"
        return f"Checking the scaffolded {fw.label} app via MCP ({steps})…"

    async def generate(
        self, state, task, t0, is_retry, previous_failure, communication_mode,
        memory_context, used_memory,
    ) -> dict:
        """`memory_context`/`used_memory` are accepted (per the TaskHandler
        contract) but deliberately unused — webapp scaffolding doesn't yet
        draw on long-term memory (see README "Implementation nice-to-haves").
        """
        fw = get_framework(self._framework_id(state, task))
        state["framework"] = fw.id
        files = await asyncio.to_thread(
            scaffolder.scaffold_webapp,
            task.description,
            previous_failure=previous_failure if is_retry else None,
            original_request=state["request"] if communication_mode == "blackboard" else None,
            failure_analysis=state.get("_failure_analysis") if is_retry else None,
            user_guidance=intervention.guidance_for(state, task.order),
            framework=fw.id,
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
        if fw.dep_file in files:
            node = fw.language == "node"
            await events.emit(
                state["project_id"], ev.VENV_SETUP, "system",
                "Installing npm dependencies…" if node
                else "Setting up an isolated virtualenv and installing dependencies…",
                {"framework": fw.id},
            )
            venv_result = await call_tool(
                "run_shell_command",
                {"project_id": state["project_id"], "command": fw.install_cmd, "timeout": fw.install_timeout},
            )
            venv_ok = "exit_code: 0" in str(venv_result)
            await events.emit(
                state["project_id"], ev.VENV_READY if venv_ok else ev.VENV_FAILED, "system",
                ("Dependencies installed." if node else "Virtualenv ready.") if venv_ok
                else ("Dependency install had issues" if node else "Virtualenv setup had issues")
                     + " (code was still written and checked).",
                {"ok": venv_ok, "framework": fw.id},
            )

        all_content = "\n".join(files.values())
        combined_size = sum(len(c) for c in files.values())
        state["code"] = f"Scaffolded {len(files)} file(s): {', '.join(sorted(files))}"
        state["code_filename"] = fw.entry_file if fw.entry_file in files else next(iter(files), None)
        state["memory_used"] = False
        ifs.record_ifs("developer", state["request"], all_content, state=state,
                       communication_mode=communication_mode)
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
                "ifs_strict": state.get("ifs_strict"), "framework": fw.id,
            },
            duration_ms=duration_ms,
        )
        await events.emit(
            state["project_id"], ev.DEVELOPER_DONE, "developer",
            f"Scaffolded {len(files)} file(s) for the {fw.label} app — sending to QA.",
            {"files": sorted(files), "duration_ms": duration_ms, "ifs_score": state["ifs_score"],
             "ifs_strict": state.get("ifs_strict"), "framework": fw.id},
        )
        return state
