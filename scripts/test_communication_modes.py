#!/usr/bin/env python3
"""Phase 6 checkpoint: proves the communication-mode ablation actually
changes what reaches the LLM, not just that the code runs without error.

No live Ollama needed — app.llm.generate is monkeypatched to capture the
prompt it was called with instead of hitting the network.
"""
import sys
import types
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# app.agents.developer imports app.mcp_client at module load time (for the
# graph node, which DOES call it) even though generate_code() itself never
# touches it. Stub it out so this test doesn't depend on having a working
# `mcp` package installed in whatever environment runs it — that's an
# unrelated dependency concern from the actual thing being tested here.
_stub = types.ModuleType("app.mcp_client")
_stub.call_tool = lambda *a, **k: None
sys.modules["app.mcp_client"] = _stub

from app.agents import developer  # noqa: E402
from app.schemas import Task  # noqa: E402

TASK = Task(order=1, description="Write a function that validates ISBN-13 checksums.")
ORIGINAL_REQUEST = (
    "Build an ISBN-13 validator. It must reject any input that mixes "
    "hyphens and spaces, and the checksum weight pattern is 1,3,1,3,... "
    "not the ISBN-10 pattern."
)


def _capture_prompt(monkeypatched_return="```python\ndef f(): pass\n```"):
    """Returns (fake_generate, get_last_prompt) — swap in for app.llm.generate."""
    captured = {}

    def fake_generate(prompt, system=None):
        captured["prompt"] = prompt
        return monkeypatched_return

    return fake_generate, lambda: captured["prompt"]


def check_message_passing_mode_omits_original_request():
    fake_generate, get_prompt = _capture_prompt()
    with patch("app.agents.developer.generate", fake_generate):
        developer.generate_code(TASK, original_request=None)  # message_passing arm

    prompt = get_prompt()
    assert "hyphens and spaces" not in prompt, (
        "message_passing mode leaked original_request detail into the prompt — "
        "the whole point of this arm is that the Developer ONLY sees Planner's "
        "task description"
    )
    assert TASK.description in prompt
    print("[ok] message_passing mode: Developer prompt contains only the task description")


def check_blackboard_mode_includes_original_request():
    fake_generate, get_prompt = _capture_prompt()
    with patch("app.agents.developer.generate", fake_generate):
        developer.generate_code(TASK, original_request=ORIGINAL_REQUEST)  # blackboard arm

    prompt = get_prompt()
    assert "hyphens and spaces" in prompt, (
        "blackboard mode should give the Developer the raw original request "
        "verbatim, including detail the task description compressed away"
    )
    assert TASK.description in prompt
    print("[ok] blackboard mode: Developer prompt contains task description AND original request")


def check_modes_produce_different_prompts_for_same_task():
    fake_generate, get_mp_prompt = _capture_prompt()
    with patch("app.agents.developer.generate", fake_generate):
        developer.generate_code(TASK, original_request=None)
    mp_prompt = get_mp_prompt()

    fake_generate, get_bb_prompt = _capture_prompt()
    with patch("app.agents.developer.generate", fake_generate):
        developer.generate_code(TASK, original_request=ORIGINAL_REQUEST)
    bb_prompt = get_bb_prompt()

    assert mp_prompt != bb_prompt
    assert len(bb_prompt) > len(mp_prompt), "blackboard prompt should be a strict superset in content"
    print("[ok] the two modes genuinely diverge for an identical task — this is a real ablation, not a no-op flag")


if __name__ == "__main__":
    check_message_passing_mode_omits_original_request()
    check_blackboard_mode_includes_original_request()
    check_modes_produce_different_prompts_for_same_task()
    print("\nAll communication-mode ablation checks passed.")
