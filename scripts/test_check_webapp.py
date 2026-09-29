#!/usr/bin/env python3
"""Phase 7 checkpoint: exercises _check_webapp_impl against REAL files on
disk (actual subprocess py_compile calls, not mocked) — this is the one
piece of Phase 7 that's genuinely testable end-to-end without Docker/Ollama,
since it's pure filesystem + subprocess.

Post-reorg note: this now imports tools.check_webapp directly, which
(unlike the old monolithic server.py) has no dependency on the mcp package
at all — no stubbing needed anymore.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_server"))

from tools.check_webapp import _check_webapp_impl  # noqa: E402



def _make_project(files: dict[str, str]) -> Path:
    tmp = Path(tempfile.mkdtemp())
    for rel_path, content in files.items():
        full = tmp / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
    return tmp


def check_valid_app_passes():
    project = _make_project({
        "main.py": "from fastapi import FastAPI\napp = FastAPI()\n",
        "requirements.txt": "fastapi\nuvicorn\n",
        "templates/index.html": "<h1>hi</h1>",
    })
    result = _check_webapp_impl(project)
    assert "PASSED: true" in result, result
    assert "EXIT_CODE: 0" in result, result
    print("[ok] a syntactically valid multi-file app: PASSED: true")


def check_syntax_error_fails():
    project = _make_project({
        "main.py": "from fastapi import FastAPI\napp = FastAPI(\n",  # unclosed paren
        "requirements.txt": "fastapi\n",
    })
    result = _check_webapp_impl(project)
    assert "PASSED: false" in result, result
    assert "main.py" in result
    print("[ok] a real syntax error: PASSED: false, and names the offending file")


def check_missing_requirements_fails():
    project = _make_project({"main.py": "app = 1\n"})
    result = _check_webapp_impl(project)
    assert "PASSED: false" in result, result
    assert "requirements.txt" in result
    print("[ok] missing requirements.txt correctly fails the check")


def check_no_py_files_fails():
    project = _make_project({"README.md": "hello"})
    result = _check_webapp_impl(project)
    assert "PASSED: false" in result, result
    print("[ok] no .py files at all correctly fails the check")


def check_output_format_matches_run_tests_contract():
    """This is the important one: qa.parse_run_tests_output must be able to
    parse this without modification.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
    from app.agents.qa import parse_run_tests_output  # noqa: E402

    project = _make_project({
        "main.py": "app = 1\n",
        "requirements.txt": "fastapi\n",
    })
    raw = _check_webapp_impl(project)
    parsed = parse_run_tests_output(raw)
    assert parsed.passed is True
    print("[ok] check_webapp's output is parsed correctly by the EXISTING qa.parse_run_tests_output — no QA-side changes needed")


def check_valid_app_actually_boots_and_responds():
    """The real point of the Phase 7 follow-up: py_compile alone can't
    catch a template that doesn't exist, or a route that throws at
    request-time. This spins up the app for real (using the same venv as
    this test process, since no .venv is created here) is NOT what we
    test — instead we build a REAL .venv for this one project and confirm
    the probe actually boots against IT, matching production behavior.
    """
    if not shutil.which("python3"):
        print("[skip] no python3 on PATH — cannot build a venv for this check")
        return

    project = _make_project({
        "main.py": (
            "from fastapi import FastAPI\n"
            "app = FastAPI()\n\n"
            "@app.get('/')\n"
            "def index():\n"
            "    return {'status': 'ok'}\n"
        ),
        "requirements.txt": "fastapi\nuvicorn\n",
    })

    venv_dir = project / ".venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True, timeout=60)
    venv_pip = venv_dir / "bin" / "pip"
    install = subprocess.run(
        [str(venv_pip), "install", "--quiet", "fastapi", "uvicorn"],
        capture_output=True, text=True, timeout=120,
    )
    if install.returncode != 0:
        print(f"[skip] could not install deps into test venv (likely no network): {install.stderr[-300:]}")
        return

    result = _check_webapp_impl(project)
    assert "PASSED: true" in result, result
    assert "booted and responded" in result, result
    print("[ok] a genuinely working app: boots for real, responds to a real HTTP request")


def check_broken_route_is_caught_by_runtime_probe_not_just_py_compile():
    """A file that compiles fine but crashes at request time — the case
    py_compile alone would wrongly pass.
    """
    if not shutil.which("python3"):
        print("[skip] no python3 on PATH")
        return

    project = _make_project({
        "main.py": (
            "from fastapi import FastAPI\n"
            "app = FastAPI()\n\n"
            "@app.get('/')\n"
            "def index():\n"
            "    return 1 / 0  # compiles fine, crashes on every request\n"
        ),
        "requirements.txt": "fastapi\nuvicorn\n",
    })

    venv_dir = project / ".venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True, timeout=60)
    venv_pip = venv_dir / "bin" / "pip"
    install = subprocess.run(
        [str(venv_pip), "install", "--quiet", "fastapi", "uvicorn"],
        capture_output=True, text=True, timeout=120,
    )
    if install.returncode != 0:
        print(f"[skip] could not install deps into test venv (likely no network): {install.stderr[-300:]}")
        return

    # Sanity check first: py_compile alone WOULD pass this (it's valid Python).
    py_compile_only = subprocess.run(
        ["python3", "-m", "py_compile", str(project / "main.py")],
        capture_output=True, text=True, timeout=15,
    )
    assert py_compile_only.returncode == 0, "test setup issue: this file should compile fine"

    result = _check_webapp_impl(project)
    assert "PASSED: false" in result, (
        f"expected the runtime probe to catch the 500-on-every-request bug that "
        f"py_compile alone misses, got:\n{result}"
    )
    print("[ok] a route that 500s on every request IS caught — py_compile alone would have missed this")


def check_no_venv_skips_runtime_probe_gracefully():
    """No .venv (dependency install never ran/finished) shouldn't fail the
    whole check — it should skip the runtime stage and still report on
    py_compile, which IS meaningful on its own.
    """
    project = _make_project({
        "main.py": "from fastapi import FastAPI\napp = FastAPI()\n",
        "requirements.txt": "fastapi\nuvicorn\n",
    })
    result = _check_webapp_impl(project)
    assert "PASSED: true" in result, result
    assert "skipped" in result.lower(), result
    print("[ok] missing .venv skips the runtime probe gracefully instead of failing the whole check")


if __name__ == "__main__":
    check_valid_app_passes()
    check_syntax_error_fails()
    check_missing_requirements_fails()
    check_no_py_files_fails()
    check_output_format_matches_run_tests_contract()
    check_no_venv_skips_runtime_probe_gracefully()
    check_valid_app_actually_boots_and_responds()
    check_broken_route_is_caught_by_runtime_probe_not_just_py_compile()
    print("\nAll check_webapp checks passed.")
