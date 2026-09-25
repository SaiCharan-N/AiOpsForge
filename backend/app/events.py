"""Live agent-event bus (Phase 5: demo/visualization layer).

This is what turns the pipeline from a black box that eventually returns a
result into something you can *watch happen*: every graph node publishes a
small, human-readable event here as it works ("Planner produced 3 tasks",
"Developer writing prime_check.py", "QA running tests..."), and the
WebSocket endpoint in main.py streams those events to any connected
frontend in real time.

Design notes:
- Per-project history is kept in memory (not just fanned out to live
  subscribers) so a client that connects a moment late — or reconnects —
  gets the full story replayed instantly, then continues live. This matters
  for a demo: you don't want "missed the Planner's output because the
  WebSocket took 200ms to connect" to be a real failure mode.
- This is intentionally a single-process, in-memory bus (a list + a set of
  asyncio.Queues per project_id), not Redis pub/sub. That's a deliberate
  scope decision for a student project running as one backend process —
  worth naming explicitly if a reviewer asks "does this scale to multiple
  backend replicas" (it doesn't, without moving this to Redis pub/sub).
- Events are pure narration/telemetry for the UI. They are NOT the
  blackboard state itself (that's AgentState in schemas.py / Postgres) —
  this module only broadcasts *that something happened*, not the
  authoritative record of it.
"""
import time
from collections import defaultdict
from typing import Any, AsyncIterator
import asyncio

# project_id -> list of past events, oldest first. Unbounded growth is fine
# at student-project scale (one run = a few dozen events); if this ever
# needs to run for hours/thousands of projects, cap this with a deque.
_history: dict[str, list[dict[str, Any]]] = defaultdict(list)

# project_id -> set of live subscriber queues currently listening.
_subscribers: dict[str, set["asyncio.Queue[dict]"]] = defaultdict(set)


async def emit(
    project_id: str,
    event_type: str,
    agent: str,
    message: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publishes one event for `project_id`. Every currently-connected
    subscriber gets it immediately; it's also appended to history so future
    subscribers see it on connect.

    event_type: short machine-readable tag, e.g. "planner_done",
        "developer_writing", "qa_result", "retry", "escalate",
        "task_complete", "pipeline_done" — the frontend switches on this to
        decide which node to highlight in the flow diagram.
    agent: "planner" | "developer" | "qa" | "system" — whose chat bubble
        this renders as.
    message: human-readable narration, shown directly in the chat feed.
    data: optional structured payload (task list, filename, test output,
        attempt count, etc.) for anything the UI wants beyond the message.
    """
    event = {
        "type": event_type,
        "agent": agent,
        "message": message,
        "data": data or {},
        "ts": time.time(),
    }
    _history[project_id].append(event)
    for queue in list(_subscribers.get(project_id, ())):
        queue.put_nowait(event)
    return event


def subscribe(project_id: str) -> "asyncio.Queue[dict]":
    """Registers a new live subscriber for `project_id` and returns a queue
    already pre-loaded with every past event for that project, so the
    caller can just `await queue.get()` in a loop and see the whole story.
    """
    queue: "asyncio.Queue[dict]" = asyncio.Queue()
    for past_event in _history.get(project_id, ()):
        queue.put_nowait(past_event)
    _subscribers[project_id].add(queue)
    return queue


def unsubscribe(project_id: str, queue: "asyncio.Queue[dict]") -> None:
    _subscribers.get(project_id, set()).discard(queue)
    if not _subscribers.get(project_id):
        _subscribers.pop(project_id, None)


def clear_history(project_id: str) -> None:
    """Optional cleanup once a project is fully done and downloaded, so a
    long-running backend process doesn't accumulate history forever across
    many demo runs. Not called automatically — call it explicitly if/when
    you want that (e.g. from a "clear demo" button).
    """
    _history.pop(project_id, None)
