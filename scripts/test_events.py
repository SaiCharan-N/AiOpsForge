#!/usr/bin/env python3
"""Phase 5 checkpoint: sanity-checks app.events in isolation (no Postgres/
Redis/Ollama needed — pure asyncio) before wiring it into a live demo.

Checks:
1. A subscriber connected BEFORE any events are emitted receives them live,
   in order.
2. A subscriber connected AFTER events were already emitted (the "late
   join" / demo-reconnect case) still receives the full history first.
3. Two independent project_ids never leak events into each other.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from app import events  # noqa: E402


async def check_live_delivery():
    queue = events.subscribe("proj-A")
    await events.emit("proj-A", "planner_done", "planner", "planned 2 tasks")
    await events.emit("proj-A", "developer_done", "developer", "wrote file.py")

    first = await queue.get()
    second = await queue.get()
    assert first["type"] == "planner_done", first
    assert second["type"] == "developer_done", second
    print("[ok] live delivery, in order")


async def check_late_join_replay():
    await events.emit("proj-B", "planner_done", "planner", "planned 1 task")
    await events.emit("proj-B", "developer_done", "developer", "wrote task.py")

    # Subscriber joins AFTER both events already happened.
    queue = events.subscribe("proj-B")
    replayed = [await queue.get(), await queue.get()]
    assert [e["type"] for e in replayed] == ["planner_done", "developer_done"]
    print("[ok] late-joining subscriber gets full history replayed")


async def check_project_isolation():
    queue_a = events.subscribe("proj-iso-A")
    queue_b = events.subscribe("proj-iso-B")

    await events.emit("proj-iso-A", "qa_result", "qa", "A's event")
    await events.emit("proj-iso-B", "qa_result", "qa", "B's event")

    a_event = await queue_a.get()
    b_event = await queue_b.get()
    assert a_event["message"] == "A's event"
    assert b_event["message"] == "B's event"
    assert queue_a.empty() and queue_b.empty(), "no cross-project leakage allowed"
    print("[ok] two projects' event streams never cross")


async def main():
    await check_live_delivery()
    await check_late_join_replay()
    await check_project_isolation()
    print("\nAll app.events checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
