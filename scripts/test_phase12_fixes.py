#!/usr/bin/env python3
"""Phase 12 checkpoint: verifies the specific bugs fixed in this pass,
mocked at the LLM/MCP/DB boundary so it runs without live infra
(matches this project's existing testing approach).

Covers:
  1. developer.generate_code actually includes the Failure Analyzer's
     root_cause/hint/affected_function in the prompt on a retry, not just
     the raw truncated QA error text.
  2. qa_node resolves the real workspace path for the deterministic
     pre-flight check (workspace_root/project_id/filename), not a bare
     filename.
  3. qa_node's deterministic pre-flight only fast-FAILS; a deterministic
     pass still falls through to the real test suite instead of skipping it.
  4. db.ensure_memory_schema emits the expected idempotent ADD COLUMN DDL.
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.agents import developer  # noqa: E402
from app.schemas import Task, TestResult  # noqa: E402


def check_failure_analysis_reaches_developer_prompt():
    task = Task(order=1, description="a function that reverses a string")
    previous_failure = TestResult(passed=False, output="", error="AssertionError: expected 'cba' got 'abc'")
    analysis = {
        "failed_file": "reverse_string.py",
        "root_cause": "the slicing step is missing [::-1]",
        "affected_function": "reverse(s)",
        "repair_strategy": "modify",
        "hint": "use s[::-1] instead of returning s unchanged",
        "similar_fixes": [
            {"problem_signature": "reverse a list", "solution_summary": "used [::-1] slicing"},
        ],
    }
    captured = {}

    def fake_generate(prompt, system=None):
        captured["prompt"] = prompt
        return "```python\ndef reverse(s): return s[::-1]\n```"

    with patch("app.agents.developer.generate", fake_generate):
        code, filename = developer.generate_code(
            task, previous_failure=previous_failure, failure_analysis=analysis,
        )

    prompt = captured["prompt"]
    assert "slicing step is missing" in prompt, "root_cause missing from developer prompt"
    assert "reverse(s)" in prompt, "affected_function missing from developer prompt"
    assert "s[::-1] instead of returning s unchanged" in prompt, "hint missing from developer prompt"
    assert "used [::-1] slicing" in prompt, "similar_fixes from memory missing from developer prompt"
    print("[ok] Failure Analyzer's diagnosis (root_cause/affected_function/hint/similar_fixes) reaches the Developer's prompt")


def check_no_failure_analysis_falls_back_gracefully():
    """A first attempt (no previous_failure, no failure_analysis) must not crash."""
    task = Task(order=1, description="a function that adds two numbers")
    with patch("app.agents.developer.generate", lambda prompt, system=None: "```python\ndef add(a, b): return a + b\n```"):
        code, filename = developer.generate_code(task)
    assert "def add" in code
    print("[ok] generate_code still works with no previous_failure/failure_analysis (first attempt, unchanged behavior)")


def check_qa_node_resolves_workspace_path_and_falls_through_on_pass():
    import types
    stub = types.ModuleType("app.mcp_client")
    stub.call_tool = AsyncMock(return_value="PASSED: true\nEXIT_CODE: 0\nOUTPUT:\nall good")
    sys.modules["app.mcp_client"] = stub

    import importlib
    import app.pipeline.nodes as nodes
    importlib.reload(nodes)

    captured_path = {}

    def fake_run_deterministic_qa(project_id, task_type, code_filepath=None, **kw):
        captured_path["path"] = code_filepath
        return True, "syntax ok"  # deterministic PASS

    with patch("app.pipeline.nodes.run_deterministic_qa", fake_run_deterministic_qa), \
         patch("app.pipeline.nodes.events.emit", AsyncMock()), \
         patch("app.pipeline.nodes._log_run_safe", MagicMock()), \
         patch("app.pipeline.nodes.memory.save_memory", MagicMock()):

        state = {
            "project_id": "proj-123",
            "current_task": {"order": 1, "description": "x", "type": "code"},
            "code_filename": "reverse_string.py",
            "attempt_count": 0,
            "request": "reverse a string",
        }
        result_state = asyncio.run(nodes.qa_node(state))

    expected_path = str(Path(nodes.settings.workspace_root) / "proj-123" / "reverse_string.py")
    assert captured_path["path"] == expected_path, (
        f"deterministic check got wrong path: {captured_path['path']!r}, expected {expected_path!r}"
    )
    print("[ok] qa_node resolves the deterministic check's file path to workspace_root/project_id/filename")

    # The critical fix: deterministic PASS must NOT short-circuit — the
    # real MCP test tool must still have been called, and the final result
    # must reflect the real test run, not just "syntax ok".
    stub.call_tool.assert_awaited_once()
    assert result_state["test_result"]["output"] == "all good", (
        "deterministic pass short-circuited QA instead of falling through to the real test run"
    )
    print("[ok] deterministic pre-flight PASS still falls through to the real test suite (no false-positive short-circuit)")


def check_qa_node_fast_fails_without_calling_real_tests():
    import types
    stub = types.ModuleType("app.mcp_client")
    stub.call_tool = AsyncMock(return_value="PASSED: true\nEXIT_CODE: 0\nOUTPUT:\nshould not be reached")
    sys.modules["app.mcp_client"] = stub

    import importlib
    import app.pipeline.nodes as nodes
    importlib.reload(nodes)

    with patch("app.pipeline.nodes.run_deterministic_qa", lambda *a, **k: (False, "SyntaxError: line 3")), \
         patch("app.pipeline.nodes.events.emit", AsyncMock()), \
         patch("app.pipeline.nodes._log_run_safe", MagicMock()):

        state = {
            "project_id": "proj-456",
            "current_task": {"order": 1, "description": "x", "type": "code"},
            "code_filename": "broken.py",
            "attempt_count": 0,
            "request": "x",
        }
        result_state = asyncio.run(nodes.qa_node(state))

    stub.call_tool.assert_not_called()
    assert result_state["_qa_decision"] == "analyze_failure"
    assert result_state["test_result"]["passed"] is False
    print("[ok] deterministic pre-flight FAIL skips the real test run (MCP not called) and routes to Failure Analyzer")


def check_migration_ddl_is_idempotent_add_column():
    import app.db as db
    captured = []

    class FakeConn:
        def execute(self, stmt):
            captured.append(str(stmt))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fake_engine = MagicMock()
    fake_engine.begin.return_value = FakeConn()

    with patch.object(db, "engine", fake_engine):
        db.ensure_memory_schema()

    assert any("ADD COLUMN IF NOT EXISTS embedding" in s for s in captured), captured
    assert any("ADD COLUMN IF NOT EXISTS execution_context" in s for s in captured), captured
    print("[ok] ensure_memory_schema issues idempotent ADD COLUMN IF NOT EXISTS for embedding + execution_context")


def main() -> int:
    check_failure_analysis_reaches_developer_prompt()
    check_no_failure_analysis_falls_back_gracefully()
    check_qa_node_resolves_workspace_path_and_falls_through_on_pass()
    check_qa_node_fast_fails_without_calling_real_tests()
    check_migration_ddl_is_idempotent_add_column()
    print("\nAll Phase 12 fix checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
