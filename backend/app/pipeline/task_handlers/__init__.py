"""Task handler registry.

app.pipeline.nodes looks up handlers here by Task.type — it never branches
on task.type itself. To add a new task type, write a module implementing
TaskHandler (see base.py) and add ONE line to _HANDLERS below.
"""
from app.pipeline.task_handlers.base import TaskHandler
from app.pipeline.task_handlers.code_handler import CodeTaskHandler
from app.pipeline.task_handlers.webapp_handler import WebappTaskHandler

_HANDLERS: dict[str, TaskHandler] = {
    "code": CodeTaskHandler(),
    "webapp_scaffold": WebappTaskHandler(),
}

_DEFAULT_HANDLER = _HANDLERS["code"]


def get_task_handler(task_type: str) -> TaskHandler:
    """Falls back to the code handler for an unknown/missing type rather
    than raising — this also covers task dicts created before Task.type
    existed (Phase 6 and earlier), which have no "type" key at all.
    """
    return _HANDLERS.get(task_type, _DEFAULT_HANDLER)


__all__ = ["TaskHandler", "get_task_handler"]
