import subprocess

from workspace import project_dir


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
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return "PASSED: false\nEXIT_CODE: -1\nOUTPUT:\nTest run timed out after 60s"
    except FileNotFoundError:
        return "PASSED: false\nEXIT_CODE: -1\nOUTPUT:\npytest is not installed in this workspace"

    passed = result.returncode == 0
    output = (result.stdout + "\n" + result.stderr).strip()
    return f"PASSED: {str(passed).lower()}\nEXIT_CODE: {result.returncode}\nOUTPUT:\n{output}"
