"""Central registry of every live-event type name the pipeline emits over
the WebSocket stream (see app.events / GET /ws/projects/{id}).

Why this exists: before this module, event type strings were inline
literals scattered across graph.py — "planner_done" typed in one place,
"qa_result" typed in another. That has three real costs in a growing
codebase: (1) a typo ("qa_reslt") is a silent bug — nothing on the
frontend matches it and the mistake is invisible until a demo; here it's
an AttributeError at import time instead. (2) there was no single place
to read "what can this pipeline ever say to the frontend" — this file IS
that list. (3) a new task handler (see app.pipeline.task_handlers) that
needs a new event type had no obvious place to declare it.

frontend/src/eventTypes.js mirrors this list by hand, since JS can't
import a Python module — if you add a constant here, add the matching one
there and in EVENT_TO_NODE.
"""
from typing import Final

# --- Planner ---
PLANNER_START: Final[str] = "planner_start"
PLANNER_DONE: Final[str] = "planner_done"

# --- Developer / task handlers (app.pipeline.task_handlers) ---
DEVELOPER_START: Final[str] = "developer_start"
DEVELOPER_RETRY: Final[str] = "developer_retry"
DEVELOPER_DONE: Final[str] = "developer_done"
# Per-file event — currently only emitted by WebappTaskHandler, but any
# future multi-file handler can reuse it rather than inventing its own.
DEVELOPER_FILE_WRITTEN: Final[str] = "developer_file_written"
MEMORY_HIT: Final[str] = "memory_hit"
VENV_SETUP: Final[str] = "venv_setup"
VENV_READY: Final[str] = "venv_ready"
VENV_FAILED: Final[str] = "venv_failed"

# --- QA ---
QA_START: Final[str] = "qa_start"
QA_RESULT: Final[str] = "qa_result"

# --- Orchestration / system (app.pipeline.nodes, app.pipeline.graph) ---
RETRY_SCHEDULED: Final[str] = "retry_scheduled"
ESCALATE: Final[str] = "escalate"
TASK_COMPLETE: Final[str] = "task_complete"
PIPELINE_START: Final[str] = "pipeline_start"
PIPELINE_DONE: Final[str] = "pipeline_done"
