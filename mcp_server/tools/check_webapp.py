"""QA check for Task(type="webapp_scaffold"). Two stages: py_compile every
.py file (required to pass), then — if that succeeds and the project's own
.venv is ready — an actual boot-and-probe (best-effort, skips gracefully
rather than failing when the venv isn't ready).
"""
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from workspace import project_dir


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _probe_runtime(pd: Path, timeout: float = 8.0) -> tuple[bool, str]:
    """Actually starts the scaffolded app and makes a real HTTP request to
    it — the check py_compile alone can't do: a file can compile cleanly
    and still 500 on first request (wrong template name, missing route,
    bad Jinja2 syntax caught only at render time, etc.).

    Uses the PROJECT'S OWN .venv (set up by WebappTaskHandler via
    `python3 -m venv .venv && pip install -r requirements.txt`), not
    whatever's installed in the MCP server's own environment — this matters
    for the per-project isolation goal: two projects with different/
    conflicting dependency versions must each boot against their own venv.

    Returns (booted_ok, message). Skips gracefully (returns (True, "skipped
    — ...")) rather than failing the whole check when a runtime probe isn't
    possible — a missing .venv means dependency install didn't finish, which
    is a separate, already-reported concern (see the venv_setup/venv_failed
    events emitted by WebappTaskHandler), not evidence the CODE is broken.
    """
    venv_python = pd / ".venv" / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )
    if not venv_python.exists():
        return True, "runtime boot check skipped — no .venv found (dependency install may not have finished)"
    if not (pd / "main.py").exists():
        return True, "runtime boot check skipped — no main.py to boot"

    port = _free_port()
    proc = subprocess.Popen(
        [str(venv_python), "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(pd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.time() + timeout
        last_error = None
        while time.time() < deadline:
            if proc.poll() is not None:
                # Process died before ever serving anything — read whatever
                # output it produced as the error detail.
                output = proc.stdout.read() if proc.stdout else ""
                return False, f"app process exited immediately (code {proc.returncode}):\n{output[-1500:]}"
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1.5) as resp:
                    if resp.status < 500:
                        return True, f"app booted and responded to GET / with HTTP {resp.status}"
                    last_error = f"HTTP {resp.status} from GET /"
            except urllib.error.HTTPError as e:
                # Any non-5xx HTTPError (404 on "/", etc.) still proves the
                # server is UP and routing requests — that's what this
                # check cares about, not whether "/" specifically exists.
                if e.code < 500:
                    return True, f"app booted and responded to GET / with HTTP {e.code}"
                last_error = f"HTTP {e.code} from GET /"
            except (urllib.error.URLError, OSError) as e:
                # Covers connection-refused (server not up yet), timeouts on
                # a hung/crashing request, and reset connections — anything
                # that means "couldn't get a clean response this attempt,
                # try again until the deadline."
                last_error = str(e)
            time.sleep(0.3)
        return False, f"app did not respond within {timeout}s — last error: {last_error}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _check_webapp_impl(pd: Path) -> str:
    """Core logic, factored out from the public tool function below so it
    can be unit-tested directly (scripts/test_check_webapp.py) by pointing
    it at a temp directory, without any MCP transport machinery involved.
    """
    py_files = sorted(pd.rglob("*.py"))
    if not py_files:
        return "PASSED: false\nEXIT_CODE: 1\nOUTPUT:\nNo .py files found in the scaffolded project."

    problems: list[str] = []
    for f in py_files:
        result = subprocess.run(
            ["python3", "-m", "py_compile", str(f)],
            cwd=str(pd),
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            problems.append(f"{f.relative_to(pd)}:\n{result.stderr.strip()}")

    if not (pd / "requirements.txt").exists():
        problems.append("requirements.txt is missing.")

    runtime_note = None
    if not problems:
        booted_ok, runtime_note = _probe_runtime(pd)
        if not booted_ok:
            problems.append(f"Runtime boot check failed: {runtime_note}")

    passed = not problems
    output = f"Checked {len(py_files)} Python file(s)."
    if passed:
        output += "\nAll compiled successfully."
        if runtime_note:
            output += f"\n{runtime_note}"
    else:
        output += "\n\n" + "\n\n".join(problems)
    return f"PASSED: {str(passed).lower()}\nEXIT_CODE: {0 if passed else 1}\nOUTPUT:\n{output}"


def check_webapp(project_id: str) -> str:
    """Smoke-checks a scaffolded web app: py_compiles every .py file in the
    project workspace, confirms requirements.txt exists, and — if that
    passes and a .venv is ready — actually boots the app with uvicorn and
    makes a real HTTP request to it, catching functional bugs (wrong
    template name, a route that 500s) that syntax-checking alone would
    miss. Returns the exact same PASSED/EXIT_CODE/OUTPUT text format as
    run_tests, so app.agents.qa's existing parser handles this unmodified.
    """
    return _check_webapp_impl(project_dir(project_id))
