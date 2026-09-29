import os
import subprocess

from workspace import project_dir

# Configurable so a slow dependency install or a genuinely large test suite
# isn't forced into the same window as a quick one — was previously a bare
# hardcoded 60, with no way to raise it without editing code.
_TIMEOUT_SECONDS = int(os.environ.get("TEST_TIMEOUT_SECONDS", "120"))

# pytest's own exit code for "the run itself was fine, it just found zero
# tests to collect" — distinct from 1 (tests ran, some failed). A "code"
# task doesn't always get a dedicated test-writing task from the Planner
# (only descriptions containing "test" get a test_*.py filename — see
# app.agents.developer._suggest_filename), so a plain "write a function
# that reverses a string" task legitimately has nothing for pytest to
# collect. That's an absence of test coverage, not a defect in the code —
# by the time run_tests is even called, qa_node's deterministic pre-flight
# has already confirmed the file compiles and imports cleanly (see
# app.pipeline.nodes.qa_node). Treating "no tests" as a hard FAIL burned a
# full retry cycle (regenerating working code against a diagnosis that had
# nothing real to point at) on every such task, every time.
_NO_TESTS_COLLECTED = 5


def run_tests(project_id: str, test_path: str = ".") -> str:
    """Run pytest inside the given project's workspace and return a
    structured PASS/FAIL summary the QA Agent can parse directly.
    """
    pd = project_dir(project_id)
    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", test_path, "-v", "--tb=short"],
            cwd=str(pd),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"PASSED: false\nEXIT_CODE: -1\nOUTPUT:\nTest run timed out after {_TIMEOUT_SECONDS}s"
    except FileNotFoundError:
        return "PASSED: false\nEXIT_CODE: -1\nOUTPUT:\npytest is not installed in this workspace"

    output = (result.stdout + "\n" + result.stderr).strip()

    if result.returncode == _NO_TESTS_COLLECTED:
        return (
            f"PASSED: true\nEXIT_CODE: {result.returncode}\nOUTPUT:\n"
            f"No tests were collected — nothing to verify beyond the syntax/import "
            f"checks already passed. Not treated as a failure.\n{output}"
        )

    passed = result.returncode == 0
    return f"PASSED: {str(passed).lower()}\nEXIT_CODE: {result.returncode}\nOUTPUT:\n{output}"
