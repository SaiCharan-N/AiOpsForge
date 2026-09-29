#!/usr/bin/env python3
"""Phase 7 checkpoint: proves the scaffolder produces a usable multi-file
manifest on a good LLM response, and falls back to a guaranteed-runnable
minimal site when the model's output can't be used — never raises, never
returns an empty/unsafe file set.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from app.agents import scaffolder  # noqa: E402
from app.llm import LLMOutputError  # noqa: E402

REQUEST = "Build a website for a bakery with a homepage and a contact page"


def check_happy_path_returns_all_files_from_manifest():
    fake_manifest = {
        "files": {
            "main.py": "from fastapi import FastAPI\napp = FastAPI()\n",
            "requirements.txt": "fastapi\nuvicorn\njinja2\n",
            "templates/index.html": "<h1>Bakery</h1>",
            "templates/contact.html": "<h1>Contact</h1>",
            "static/style.css": "body { color: brown; }",
        }
    }
    with patch("app.agents.scaffolder.generate_json", return_value=fake_manifest):
        files = scaffolder.scaffold_webapp(REQUEST)

    assert set(files.keys()) == set(fake_manifest["files"].keys())
    assert files["main.py"] == fake_manifest["files"]["main.py"]
    print(f"[ok] happy path returns all {len(files)} files from the model's manifest, verbatim")


def check_path_traversal_entries_are_dropped():
    fake_manifest = {
        "files": {
            "main.py": "app = 1",
            "../../etc/passwd": "malicious",
            "/etc/shadow": "malicious",
            "ok/nested/file.py": "fine",
        }
    }
    with patch("app.agents.scaffolder.generate_json", return_value=fake_manifest):
        files = scaffolder.scaffold_webapp(REQUEST)

    assert "../../etc/passwd" not in files
    assert "/etc/shadow" not in files
    assert "main.py" in files and "ok/nested/file.py" in files
    print("[ok] path-traversal / absolute-path entries are filtered out before reaching MCP write_file")


def check_llm_output_error_falls_back_to_minimal_site():
    with patch("app.agents.scaffolder.generate_json", side_effect=LLMOutputError("model gave up")):
        files = scaffolder.scaffold_webapp(REQUEST)

    assert files == scaffolder._FALLBACK_FILES
    assert "main.py" in files and "requirements.txt" in files
    # The fallback itself must be internally consistent: templates dir
    # referenced in main.py actually exists in the file set.
    assert "templates/index.html" in files
    print("[ok] LLMOutputError falls back to the guaranteed-runnable minimal site")


def check_malformed_manifest_falls_back():
    with patch("app.agents.scaffolder.generate_json", return_value={"files": "not a dict"}):
        files = scaffolder.scaffold_webapp(REQUEST)
    assert files == scaffolder._FALLBACK_FILES
    print("[ok] malformed 'files' value also falls back cleanly (no crash)")


def check_fallback_files_are_all_nonempty():
    for path, content in scaffolder._FALLBACK_FILES.items():
        assert content.strip(), f"fallback file {path} is empty"
    print("[ok] every fallback file has real content (nothing ships empty)")


def check_previous_failure_is_included_on_retry():
    """The fix for gap #1: a retry must carry the specific failure forward,
    not regenerate blind.
    """
    captured = {}

    def fake_generate_json(prompt, system=None):
        captured["prompt"] = prompt
        return {"files": {"main.py": "app = 1", "requirements.txt": "fastapi\n"}}

    class FakeFailure:
        error = "SyntaxError: unexpected EOF while parsing in main.py"
        output = ""

    with patch("app.agents.scaffolder.generate_json", fake_generate_json):
        scaffolder.scaffold_webapp("A bakery site", previous_failure=FakeFailure())

    assert "SyntaxError: unexpected EOF" in captured["prompt"], (
        "retry prompt must include the specific previous failure, not regenerate blind"
    )
    print("[ok] retry prompt includes the specific previous check_webapp failure")


def check_original_request_only_included_in_blackboard_mode():
    """The fix for gap #2: the communication-mode ablation must actually
    apply to webapp tasks, the same way it already did for code tasks.
    """
    original_request = "Build a bakery website. Must use a warm color palette and show opening hours."

    def make_capture():
        captured = {}
        def fn(prompt, system=None):
            captured["prompt"] = prompt
            return {"files": {"main.py": "app = 1", "requirements.txt": "fastapi\n"}}
        return fn, captured

    fn_mp, captured_mp = make_capture()
    with patch("app.agents.scaffolder.generate_json", fn_mp):
        scaffolder.scaffold_webapp("A bakery site summary", original_request=None)
    assert "warm color palette" not in captured_mp["prompt"], (
        "message_passing mode leaked the original request into the scaffolder prompt"
    )

    fn_bb, captured_bb = make_capture()
    with patch("app.agents.scaffolder.generate_json", fn_bb):
        scaffolder.scaffold_webapp("A bakery site summary", original_request=original_request)
    assert "warm color palette" in captured_bb["prompt"], (
        "blackboard mode should give the Scaffolder the raw original request"
    )
    print("[ok] communication-mode ablation now applies to webapp tasks, same as code tasks")


if __name__ == "__main__":
    check_happy_path_returns_all_files_from_manifest()
    check_path_traversal_entries_are_dropped()
    check_llm_output_error_falls_back_to_minimal_site()
    check_malformed_manifest_falls_back()
    check_fallback_files_are_all_nonempty()
    check_previous_failure_is_included_on_retry()
    check_original_request_only_included_in_blackboard_mode()
    print("\nAll scaffolder checks passed.")
