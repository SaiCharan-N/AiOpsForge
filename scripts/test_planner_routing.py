#!/usr/bin/env python3
"""Phase 7 checkpoint: the Planner must route a website request to a single
webapp_scaffold task WITHOUT calling the normal task-decomposition LLM
prompt at all (it's a different, cheaper, more reliable path — not just a
post-hoc relabeling of the normal task list).
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from app.agents import planner  # noqa: E402


def check_webapp_request_skips_normal_decomposition():
    mock_generate_json = MagicMock(side_effect=AssertionError(
        "generate_json should NOT be called for a webapp request — "
        "the whole point of routing early is to skip the normal decomposition call"
    ))
    fake_summary = "A bakery site with a homepage showcasing products and a contact page."
    with patch("app.agents.planner.generate_json", mock_generate_json), \
         patch("app.agents.planner.generate", return_value=fake_summary) as mock_generate:
        tasks = planner.plan("Build a website for a bakery with a homepage and menu")

    assert len(tasks) == 1
    assert tasks[0].type == "webapp_scaffold"
    mock_generate_json.assert_not_called()
    mock_generate.assert_called_once()
    assert tasks[0].description == fake_summary, (
        "the task description should be the Planner's SUMMARY, not the raw request verbatim — "
        "otherwise the communication-mode ablation has nothing to lose in message_passing mode"
    )
    print("[ok] webapp request -> single webapp_scaffold task with a real Planner summary as its description")


def check_webapp_summarization_failure_falls_back_to_raw_request():
    with patch("app.agents.planner.generate_json") as mock_generate_json, \
         patch("app.agents.planner.generate", side_effect=ConnectionError("no LLM available")):
        tasks = planner.plan("Build a website for a bakery with a homepage and menu")

    mock_generate_json.assert_not_called()
    assert tasks[0].description == "Build a website for a bakery with a homepage and menu"
    print("[ok] summarization failure falls back to the raw request text, doesn't crash the Planner")


def check_normal_request_still_uses_llm_decomposition():
    with patch(
        "app.agents.planner.generate_json",
        return_value={"tasks": ["write a palindrome checker", "write its test"]},
    ) as mock_generate_json:
        tasks = planner.plan("a function that checks if a string is a palindrome, with a test")

    mock_generate_json.assert_called_once()
    assert len(tasks) == 2
    assert all(t.type == "code" for t in tasks)
    print("[ok] ordinary code request still goes through normal LLM decomposition, type='code'")


if __name__ == "__main__":
    check_webapp_request_skips_normal_decomposition()
    check_webapp_summarization_failure_falls_back_to_raw_request()
    check_normal_request_still_uses_llm_decomposition()
    print("\nAll planner routing checks passed.")
