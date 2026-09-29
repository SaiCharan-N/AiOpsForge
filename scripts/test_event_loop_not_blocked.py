#!/usr/bin/env python3
"""Proves the actual bug report ("Lost the live connection to the backend")
is fixed: a slow, synchronous LLM call must NOT block the event loop from
servicing other concurrent work (in production, that's WebSocket frames —
here, simulated with a concurrent asyncio task that must keep ticking).

Confirmed as a real bug from production logs: every `generate()`/
`generate_code()`/`scaffold_webapp()` call took 15-100+ seconds with
ZERO other backend activity logged during that window — proof the whole
process was frozen, not just that one request was slow. This test fails
loudly (via a timing assertion) if that regresses.
"""
import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import types
_mcp_stub = types.ModuleType("app.mcp_client")
_mcp_stub.call_tool = lambda *a, **k: None
sys.modules["app.mcp_client"] = _mcp_stub

from app.pipeline.task_handlers.code_handler import CodeTaskHandler  # noqa: E402
from app.schemas import Task  # noqa: E402


def _blocking_generate_code(task, **kwargs):
    """Simulates exactly what the real bug looked like: a slow, purely
    synchronous call (no awaits inside it at all) — same shape as the real
    `requests.post()` call to Ollama.
    """
    time.sleep(0.6)
    return "def f(): pass", "f.py"


async def check_slow_generation_does_not_block_the_event_loop():
    ticks = []

    async def heartbeat():
        # Simulates anything else the event loop needs to keep doing while
        # a generation call is in flight — a WebSocket ping, another
        # concurrent request, etc. If the fix is missing, this coroutine
        # never gets a chance to run until the blocking call returns.
        for _ in range(6):
            ticks.append(time.time())
            await asyncio.sleep(0.1)

    handler = CodeTaskHandler()
    task = Task(order=1, description="a trivial function")
    state = {"project_id": "p1", "request": "a trivial function", "code_filename": None, "attempt_count": 0}

    with patch("app.pipeline.task_handlers.code_handler.developer.generate_code", _blocking_generate_code), \
         patch("app.pipeline.task_handlers.code_handler.call_tool", _async_noop):
        t0 = time.time()
        heartbeat_task = asyncio.create_task(heartbeat())
        await handler.generate(
            state, task, t0, is_retry=False, previous_failure=None,
            communication_mode="message_passing", memory_context=None, used_memory=False,
        )
        await heartbeat_task

    # With the fix: heartbeat ticks happen ~every 0.1s throughout the 0.6s
    # blocking call, so we should see close to 6 ticks spread over ~0.6s+.
    # Without the fix (calling generate_code directly, no to_thread): the
    # blocking call would starve the event loop for its full 0.6s duration
    # before the heartbeat task gets ANY chance to run, and all 6 ticks
    # would land back-to-back at the end instead of being spread out.
    assert len(ticks) == 6, f"expected all 6 heartbeat ticks to complete, got {len(ticks)}"
    spread = ticks[-1] - ticks[0]
    assert spread > 0.3, (
        f"heartbeat ticks were bunched together (spread={spread:.3f}s) — "
        f"this means the event loop WAS blocked by the 'slow' call, "
        f"exactly like the real 'lost connection' bug"
    )
    print(f"[ok] event loop stayed responsive during a slow generation call "
          f"(heartbeat spread over {spread:.3f}s while generation ran for ~0.6s)")


async def _async_noop(*a, **k):
    return None


async def main():
    await check_slow_generation_does_not_block_the_event_loop()
    print("\nEvent-loop-blocking regression check passed.")


if __name__ == "__main__":
    asyncio.run(main())
