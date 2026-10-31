"""QA check for Task(type="webapp_scaffold"), for any supported framework.

Stages (each later stage only runs if the earlier ones found no problems):

  1. Static  — Python: py_compile every .py file. Node: `node --check` every
               server-side .js. Plus the dependency file exists / is valid
               (requirements.txt, package.json) and a few framework
               structure checks (Next.js needs pages/index.*).
  2. Tests   — the framework's own test runner, if its dependencies are
               installed:
                 fastapi  pytest, only if the project has test files
                 flask    pytest
                 django   python manage.py test
                 express  npm test           (node --test)
                 next     npm run build
  3. Boot    — start the app on a free port and make a real HTTP request:
                 fastapi  uvicorn main:app
                 flask    flask --app app run
                 django   manage.py migrate --run-syncdb, then runserver
                 express  node <main>        (PORT env)
                 next     next start

Stages 2 and 3 use the PROJECT'S OWN dependencies (.venv / node_modules, set
up by WebappTaskHandler) — never the MCP server's own — and SKIP gracefully
when those aren't installed yet: a missing install is a separate, already
reported concern (the venv_failed event), not evidence the code is broken.

The framework comes from the `framework` argument if given, else it is
detected from the project's files. Returns the exact PASSED/EXIT_CODE/OUTPUT
text run_tests returns, so app.agents.qa's parser handles it unmodified.
"""
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from workspace import project_dir

KNOWN_FRAMEWORKS = ("fastapi", "flask", "django", "express", "next")
# Directories that are installed/generated, never the model's own code.
SKIP_DIRS = {".venv", "venv", "node_modules", "__pycache__", ".git", ".next", ".pytest_cache"}
# Not server code: browser scripts / static assets are not syntax-checked with node.
NODE_SKIP_CHECK_DIRS = {"public", "static", "assets"}

TEST_TIMEOUT = 120
DJANGO_TEST_TIMEOUT = 180
NEXT_BUILD_TIMEOUT = 240
BOOT_TIMEOUT = 30.0
NEXT_BOOT_TIMEOUT = 45.0


# ----------------------------------------------------------------- helpers --

def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _iter_files(pd: Path, suffixes: tuple[str, ...], skip_dirs: set[str] = SKIP_DIRS):
    """Project files with the given suffixes, skipping installed/generated dirs
    (the old rglob("*.py") walked into .venv and py_compiled every library)."""
    out = []
    for root, dirs, files in os.walk(pd):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            if f.endswith(suffixes):
                out.append(Path(root) / f)
    return sorted(out)


def _venv_python(pd: Path) -> Optional[Path]:
    p = pd / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    return p if p.exists() else None


def _run(cmd: list[str], cwd: Path, timeout: float, env: Optional[dict] = None) -> tuple[int, str]:
    """Runs a command in its own process group (so a timeout kills npm's /
    flask's child processes too). Returns (returncode, combined output);
    returncode -9 means it timed out. FileNotFoundError propagates."""
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), env={**os.environ, **(env or {})}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out or ""
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        out, _ = proc.communicate()
        return -9, (out or "") + f"\n(timed out after {timeout:.0f}s)"


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _tail(text: str, n: int = 3000) -> str:
    text = text.strip()
    return text if len(text) <= n else "…" + text[-n:]


# --------------------------------------------------------------- detection --

def detect_framework(pd: Path, hint: Optional[str] = None) -> str:
    """The framework to check: the explicit `hint` if valid, else inferred from
    the project's files, defaulting to fastapi (the original behaviour)."""
    if hint in KNOWN_FRAMEWORKS:
        return hint
    if (pd / "manage.py").exists():
        return "django"
    if (pd / "package.json").exists():
        pkg = _read_json(pd / "package.json")
        deps = {}
        if isinstance(pkg, dict):
            for k in ("dependencies", "devDependencies"):
                if isinstance(pkg.get(k), dict):
                    deps.update(pkg[k])
        return "next" if "next" in deps else "express"   # any other Node project is checked like Express
    try:
        req = (pd / "requirements.txt").read_text().lower()
    except OSError:
        req = ""
    for name in ("django", "flask", "fastapi"):
        if re.search(rf"^\s*{name}\b", req, re.MULTILINE):
            return name
    for entry in ("app.py", "main.py"):
        try:
            src = (pd / entry).read_text()
        except OSError:
            continue
        if "Flask(" in src:
            return "flask"
        if "FastAPI(" in src:
            return "fastapi"
    return "fastapi"


def _is_node(fw: str) -> bool:
    return fw in ("express", "next")


# ------------------------------------------------------------ stage 1: static

def _static_python(pd: Path) -> tuple[int, list[str]]:
    py_files = _iter_files(pd, (".py",))
    problems: list[str] = []
    for f in py_files:
        result = subprocess.run(
            ["python3", "-m", "py_compile", str(f)], cwd=str(pd), capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            problems.append(f"{f.relative_to(pd)}:\n{result.stderr.strip()}")
    return len(py_files), problems


def _static_node(pd: Path, fw: str) -> tuple[int, list[str], Optional[str]]:
    """Returns (files checked, problems, note). Next.js is not syntax-checked
    with node (JSX) — its build is the check."""
    problems: list[str] = []
    pj = pd / "package.json"
    if not pj.exists():
        return 0, ["package.json is missing."], None
    pkg = _read_json(pj)
    if not isinstance(pkg, dict):
        try:
            json.loads(pj.read_text())
        except ValueError as e:
            problems.append(f"package.json is not valid JSON: {e}")
        else:
            problems.append("package.json must be a JSON object.")
        return 0, problems, None

    if fw == "next":
        deps = {**(pkg.get("dependencies") or {}), **(pkg.get("devDependencies") or {})}
        if "next" not in deps:
            problems.append("package.json does not list `next` as a dependency.")
        if not any((pd / "pages" / f"index{ext}").exists() for ext in (".js", ".jsx", ".ts", ".tsx")):
            problems.append("pages/index.js is missing (the Next.js home page).")
        return 0, problems, None

    js_files = [f for f in _iter_files(pd, (".js", ".cjs", ".mjs"))
                if not (set(f.relative_to(pd).parts[:-1]) & NODE_SKIP_CHECK_DIRS)]
    if not js_files:
        return 0, ["No .js files found in the scaffolded project."], None
    node = shutil.which("node")
    if not node:
        return len(js_files), [], "syntax check skipped — node is not installed in the MCP server"
    for f in js_files:
        rc, out = _run([node, "--check", str(f)], pd, 15)
        if rc != 0:
            problems.append(f"{f.relative_to(pd)}:\n{_tail(out, 1500)}")
    return len(js_files), problems, None


# ------------------------------------------------------------- stage 2: tests

def _pytest_stage(pd: Path, py: Optional[Path], required: bool) -> tuple[bool, str]:
    has_tests = bool([f for f in _iter_files(pd, (".py",)) if f.name.startswith("test_") or f.name.endswith("_test.py")])
    if not has_tests:
        return True, "tests skipped — the project has no test files"
    if py is None:
        return True, "tests skipped — no .venv found (dependency install may not have finished)"
    rc, _ = _run([str(py), "-c", "import pytest"], pd, 20)
    if rc != 0:
        return True, "tests skipped — pytest is not installed in the project's .venv"
    rc, out = _run([str(py), "-m", "pytest", "-q", "--tb=short", "--showlocals", "-p", "no:cacheprovider"],
                   pd, TEST_TIMEOUT)
    if rc == -9:
        return False, f"Test run timed out after {TEST_TIMEOUT}s\n{_tail(out, 1500)}"
    if rc in (0, 5):
        last = [l for l in out.strip().splitlines() if l.strip()][-1:] or [""]
        return True, f"tests passed (pytest): {last[0].strip('= ')}" if rc == 0 else "tests: pytest collected no tests"
    return False, f"Tests failed (pytest, exit {rc}):\n{_tail(out)}"


def _django_stage(pd: Path, py: Optional[Path]) -> tuple[bool, str]:
    if py is None:
        return True, "tests skipped — no .venv found (dependency install may not have finished)"
    rc, _ = _run([str(py), "-c", "import django"], pd, 20)
    if rc != 0:
        return True, "tests skipped — django is not installed in the project's .venv"
    rc, out = _run([str(py), "manage.py", "test", "--noinput"], pd, DJANGO_TEST_TIMEOUT)
    if rc == -9:
        return False, f"Test run timed out after {DJANGO_TEST_TIMEOUT}s\n{_tail(out, 1500)}"
    if rc == 0:
        ran = re.search(r"Ran \d+ tests? in [\d.]+s", out)
        return True, f"tests passed (manage.py test): {ran.group(0) if ran else 'ok'}"
    return False, f"Tests failed (manage.py test, exit {rc}):\n{_tail(out)}"


def _npm_test_stage(pd: Path) -> tuple[bool, str]:
    pkg = _read_json(pd / "package.json") or {}
    script = ((pkg.get("scripts") or {}).get("test") or "") if isinstance(pkg, dict) else ""
    if not script or "no test specified" in script:
        return True, "tests skipped — package.json has no test script"
    if not (pd / "node_modules").is_dir():
        return True, "tests skipped — node_modules not found (npm install may not have finished)"
    npm = shutil.which("npm")
    if not npm:
        return True, "tests skipped — npm is not installed in the MCP server"
    rc, out = _run([npm, "test", "--silent"], pd, TEST_TIMEOUT, env={"CI": "1"})
    if rc == -9:
        return False, f"Test run timed out after {TEST_TIMEOUT}s\n{_tail(out, 1500)}"
    if rc == 0:
        # node's reporters: "ℹ tests 1" (spec) or "# tests 1" (TAP, older Node)
        found = dict(re.findall(r"^(?:ℹ|#) (tests|pass|fail) (\d+)", out, re.MULTILINE))
        summary = f"{found.get('pass', '?')}/{found['tests']} passed" if "tests" in found else "ok"
        return True, f"tests passed (npm test): {summary}"
    return False, f"Tests failed (npm test, exit {rc}):\n{_tail(out)}"


def _next_build_stage(pd: Path) -> tuple[bool, str]:
    if not (pd / "node_modules" / "next").is_dir():
        return True, "build skipped — next is not installed in node_modules (npm install may not have finished)"
    npm = shutil.which("npm")
    if not npm:
        return True, "build skipped — npm is not installed in the MCP server"
    rc, out = _run([npm, "run", "build"], pd, NEXT_BUILD_TIMEOUT, env={"NEXT_TELEMETRY_DISABLED": "1", "CI": "1"})
    if rc == -9:
        return False, f"Build timed out after {NEXT_BUILD_TIMEOUT}s\n{_tail(out, 1500)}"
    if rc == 0:
        return True, "build passed (next build)"
    return False, f"Build failed (next build, exit {rc}):\n{_tail(out)}"


def _tests_stage(pd: Path, fw: str) -> tuple[bool, str]:
    py = _venv_python(pd)
    if fw == "flask":
        return _pytest_stage(pd, py, required=True)
    if fw == "fastapi":
        return _pytest_stage(pd, py, required=False)
    if fw == "django":
        return _django_stage(pd, py)
    if fw == "express":
        return _npm_test_stage(pd)
    return _next_build_stage(pd)


# -------------------------------------------------------------- stage 3: boot

def _serve_and_probe(cmd: list[str], pd: Path, port: int, timeout: float, env: Optional[dict] = None) -> tuple[bool, str]:
    """Starts `cmd`, waits for it to answer GET / on `port`, then stops it.
    Any non-5xx answer (even a 404 on "/") proves the server is up and routing."""
    proc = subprocess.Popen(
        cmd, cwd=str(pd), env={**os.environ, **(env or {})}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )
    try:
        deadline = time.time() + timeout
        last_error = None
        consecutive_5xx = 0
        while time.time() < deadline:
            if proc.poll() is not None:
                # Died before ever serving anything — its output is the error detail.
                output = proc.stdout.read() if proc.stdout else ""
                return False, f"app process exited immediately (code {proc.returncode}):\n{output[-1500:]}"
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1.5) as resp:
                    if resp.status < 500:
                        return True, f"app booted and responded to GET / with HTTP {resp.status}"
                    last_error = f"HTTP {resp.status} from GET /"
            except urllib.error.HTTPError as e:
                if e.code < 500:
                    return True, f"app booted and responded to GET / with HTTP {e.code}"
                last_error = f"HTTP {e.code} from GET /"
                consecutive_5xx += 1
                if consecutive_5xx >= 3:
                    # The server is UP but its home page crashes on every request: fail
                    # now (not after the full timeout) and hand over its traceback.
                    _kill_group(proc)
                    return False, (f"GET / returned HTTP {e.code} on {consecutive_5xx} consecutive requests — "
                                   f"the app is up but its home page crashes.{_server_output(proc)}")
            except (urllib.error.URLError, OSError) as e:
                last_error = str(e)   # not up yet / refused / reset: retry until the deadline
            time.sleep(0.3)
        _kill_group(proc)
        return False, f"app did not respond within {timeout}s — last error: {last_error}{_server_output(proc)}"
    finally:
        _kill_group(proc)


def _server_output(proc: subprocess.Popen) -> str:
    """What the (already stopped) server printed — its traceback is the best
    clue to why a request failed. Formatted for appending to a message."""
    try:
        output = proc.stdout.read() if proc.stdout else ""
    except (OSError, ValueError):
        output = ""
    return f"\nServer output:\n{output[-2500:]}" if output.strip() else ""


def _boot_stage(pd: Path, fw: str) -> tuple[bool, str]:
    """Actually starts the app and makes a real HTTP request — the check
    compile/tests alone can't do (a file can compile and still 500 on first
    request: wrong template name, missing route, bad Jinja2 syntax caught only
    at render time...). Skips (True, "skipped — ...") when the project's own
    dependencies aren't installed."""
    port = _free_port()
    if _is_node(fw):
        node, npm = shutil.which("node"), shutil.which("npm")
        if not node or (fw == "next" and not npm):
            return True, "runtime boot check skipped — node/npm is not installed in the MCP server"
        if not (pd / "node_modules").is_dir():
            return True, "runtime boot check skipped — node_modules not found (npm install may not have finished)"
        if fw == "next":
            if not (pd / "node_modules" / "next").is_dir():
                return True, "runtime boot check skipped — next is not installed in node_modules"
            return _serve_and_probe([npm, "run", "start", "--", "-p", str(port)], pd, port, NEXT_BOOT_TIMEOUT,
                                    env={"NEXT_TELEMETRY_DISABLED": "1"})
        pkg = _read_json(pd / "package.json") or {}
        entry = (pkg.get("main") if isinstance(pkg, dict) else None) or next(
            (e for e in ("server.js", "app.js", "index.js") if (pd / e).exists()), None)
        if not entry or not (pd / entry).exists():
            return True, "runtime boot check skipped — no entry file (server.js) to boot"
        return _serve_and_probe([node, entry], pd, port, BOOT_TIMEOUT, env={"PORT": str(port)})

    py = _venv_python(pd)
    if py is None:
        return True, "runtime boot check skipped — no .venv found (dependency install may not have finished)"
    if fw == "django":
        if not (pd / "manage.py").exists():
            return True, "runtime boot check skipped — no manage.py to boot"
        rc, out = _run([str(py), "manage.py", "migrate", "--run-syncdb", "--noinput"], pd, 120)
        if rc != 0:
            return False, f"database setup (manage.py migrate) failed (exit {rc}):\n{_tail(out, 1500)}"
        return _serve_and_probe([str(py), "manage.py", "runserver", f"127.0.0.1:{port}", "--noreload"], pd, port, BOOT_TIMEOUT)
    if fw == "flask":
        module = next((m for m in ("app", "main", "wsgi") if (pd / f"{m}.py").exists()), None)
        if not module:
            return True, "runtime boot check skipped — no app.py to boot"
        return _serve_and_probe([str(py), "-m", "flask", "--app", module, "run", "--host", "127.0.0.1",
                                 "--port", str(port), "--no-reload", "--no-debugger"], pd, port, BOOT_TIMEOUT)
    # fastapi (the original behaviour)
    if not (pd / "main.py").exists():
        return True, "runtime boot check skipped — no main.py to boot"
    return _serve_and_probe([str(py), "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port)],
                            pd, port, BOOT_TIMEOUT)


def _probe_runtime(pd: Path, timeout: float = 30.0, framework: Optional[str] = None) -> tuple[bool, str]:
    """Boot stage on its own (kept as a public helper: scripts/test_check_webapp.py
    and the failure-recovery tests exercise it directly)."""
    return _boot_stage(pd, detect_framework(pd, framework))


# ------------------------------------------------------------------- the tool

def _check_webapp_impl(pd: Path, framework: Optional[str] = None) -> str:
    """Core logic, factored out from the public tool function below so it
    can be unit-tested directly by pointing it at a temp directory, without
    any MCP transport machinery involved."""
    fw = detect_framework(pd, framework)
    problems: list[str] = []
    static_note = None

    if _is_node(fw):
        count, problems, static_note = _static_node(pd, fw)
        label = "JavaScript"
        if fw == "next":
            summary = "Next.js project structure OK." if not problems else ""
        else:
            summary = "All passed `node --check`." if not problems else ""
    else:
        py_files_count, problems = _static_python(pd)
        count = py_files_count
        label = "Python"
        if not py_files_count:
            return "PASSED: false\nEXIT_CODE: 1\nOUTPUT:\nNo .py files found in the scaffolded project."
        if not (pd / "requirements.txt").exists():
            problems.append("requirements.txt is missing.")
        summary = "All compiled successfully."

    notes: list[str] = []
    if static_note:
        notes.append(static_note)
    if not problems:
        tests_ok, tests_note = _tests_stage(pd, fw)
        notes.append(tests_note)
        if not tests_ok:
            problems.append(tests_note)
    if not problems:
        booted_ok, boot_note = _boot_stage(pd, fw)
        notes.append(boot_note)
        if not booted_ok:
            problems.append(f"Runtime boot check failed: {boot_note}")

    passed = not problems
    output = f"Checked {count} {label} file(s)." if count else f"Checked the {fw} project."
    if passed:
        output += f"\n{summary}"
        output += "".join(f"\n{n}" for n in notes if n)
    else:
        output += "\n\n" + "\n\n".join(problems)
    return f"PASSED: {str(passed).lower()}\nEXIT_CODE: {0 if passed else 1}\nOUTPUT:\n{output}"


def check_webapp(project_id: str, framework: str | None = None) -> str:
    """Checks a scaffolded web app (FastAPI, Flask, Django, Express or
    Next.js): static syntax/structure checks, then the framework's own test
    runner (pytest / manage.py test / npm test / next build), then a real
    boot-and-probe — catching functional bugs (wrong template name, a route
    that 500s, a failing test) that syntax-checking alone would miss. Pass
    `framework` to choose the runner explicitly; otherwise it is detected
    from the project's files. Returns the exact same PASSED/EXIT_CODE/OUTPUT
    text format as run_tests, so app.agents.qa's existing parser handles
    this unmodified.
    """
    return _check_webapp_impl(project_dir(project_id), framework)
