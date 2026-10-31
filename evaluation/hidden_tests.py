"""Hidden-test evaluation of generated code (the "ground truth" correctness check).

Why this exists: the pipeline's own "passed" means the tests THE MODEL WROTE passed. A model
can write a weak test and pass its own wrong code. These tests are written by you, never shown
to any model, and run against whatever code a run produced — so `hidden_pass` is real
correctness, and (self-pass AND NOT hidden-pass) is the "false pass" rate.

    python evaluation/hidden_tests.py <project_dir> T01          # one directory, one task
    python evaluation/hidden_tests.py <project_dir> T01 --self   # + run the project's own pytest

Library use: run_hidden(), run_self_tests(), ifs_of_project(), load_tasks().
Generated code runs in a COPY of the directory (in a subprocess with a time limit, stdin closed).
It is the same kind of code the pipeline already executes for QA; still, run evaluations on a
machine/container where that is acceptable.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "backend"))

SKIP_DIRS = {"__pycache__", "node_modules", "venv", ".venv", "site-packages", ".pytest_cache"}


def load_tasks(path: Path | str | None = None) -> dict[str, dict]:
    data = json.loads(Path(path or HERE / "tasks.json").read_text(encoding="utf-8"))
    return {t["id"]: t for t in data["tasks"]}


def task_for_request(tasks: dict[str, dict], request: str) -> dict | None:
    norm = " ".join((request or "").split())
    for t in tasks.values():
        if " ".join(t["request"].split()) == norm:
            return t
    return None


def _copy(project_dir: Path) -> Path:
    dst = Path(tempfile.mkdtemp(prefix="aiops_eval_")) / "p"
    shutil.copytree(project_dir, dst, ignore=shutil.ignore_patterns(*SKIP_DIRS, ".*"))
    return dst


def _text(x) -> str:
    return x.decode("utf-8", "replace") if isinstance(x, bytes) else (x or "")


def run_hidden(project_dir: Path | str, task: dict, timeout: int = 30) -> dict:
    """-> {hidden_status, hidden_passed, hidden_total, hidden_fraction, hidden_pass, hidden_file, hidden_errors}"""
    work = _copy(Path(project_dir))
    spec = work.parent / "spec.json"
    spec.write_text(json.dumps({"entry_points": task["entry_points"], "tests": task["tests"]}), encoding="utf-8")
    total = len(task["tests"])
    out, status = "", None
    try:
        proc = subprocess.run([sys.executable, str(HERE / "_harness.py"), str(work), str(spec)],
                              capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL, cwd=str(work))
        out = _text(proc.stdout)
    except subprocess.TimeoutExpired as exc:
        out, status = _text(exc.stdout), "timeout"
    finally:
        shutil.rmtree(work.parent, ignore_errors=True)

    result, progress = None, 0
    for line in out.splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[7:])
        elif line.startswith("PROGRESS "):
            progress = json.loads(line[9:])["passed"]
    if result is None:  # hung or crashed mid-way: credit the tests that finished
        passed, st = progress, status or "crashed"
        errors, fname = [], None
    else:
        passed, st = result["passed"], result["status"]
        errors, fname = result["errors"] + result["load_errors"], result["file"]
    return {"hidden_status": st, "hidden_passed": passed, "hidden_total": total,
            "hidden_fraction": round(passed / total, 4) if total else 0.0,
            "hidden_pass": bool(st == "ok" and passed == total), "hidden_file": fname,
            "hidden_errors": " | ".join(errors)[:400]}


def run_self_tests(project_dir: Path | str, timeout: int = 60) -> dict:
    """Runs the project's OWN pytest (the tests the model wrote). Needs `pip install pytest` on the host."""
    work = _copy(Path(project_dir))
    try:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider"],
                              capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL, cwd=str(work))
        text = _text(proc.stdout) + _text(proc.stderr)
        if "No module named pytest" in text:
            return {"self_status": "pytest_missing", "self_pass": None}
        if proc.returncode == 0:
            return {"self_status": "passed", "self_pass": True}
        if proc.returncode == 5:
            return {"self_status": "no_tests", "self_pass": False}
        return {"self_status": "failed", "self_pass": False}
    except subprocess.TimeoutExpired:
        return {"self_status": "timeout", "self_pass": False}
    finally:
        shutil.rmtree(work.parent, ignore_errors=True)


def project_source(project_dir: Path | str, include_tests: bool = True) -> str:
    """All Python source of a project concatenated (file names included), for IFS."""
    parts = []
    for p in sorted(Path(project_dir).rglob("*.py")):
        if any(s in p.parts for s in SKIP_DIRS) or any(x.startswith(".") for x in p.relative_to(project_dir).parts):
            continue
        if not include_tests and p.name.startswith("test_"):
            continue
        parts.append(f"{p.name}\n{p.read_text(encoding='utf-8', errors='replace')}")
    return "\n".join(parts)


def ifs_of_project(request: str, project_dir: Path | str) -> dict:
    """IFS of the FINAL output (request vs everything the run produced), same matcher for every arm
    — pipeline arms and the raw-model arm — so the numbers are directly comparable."""
    from app import ifs  # backend/app/ifs.py (pure python)
    d = ifs.compute_ifs_detail(request, project_source(project_dir))
    return {"ifs_final": None if d is None else round(d["score"], 4),
            "ifs_final_strict": None if d is None else round(d["strict_score"], 4)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("project_dir")
    ap.add_argument("task_id")
    ap.add_argument("--self", action="store_true", help="also run the project's own pytest")
    ap.add_argument("--timeout", type=int, default=30)
    a = ap.parse_args()
    task = load_tasks()[a.task_id]
    res = run_hidden(a.project_dir, task, a.timeout)
    res.update(ifs_of_project(task["request"], a.project_dir))
    if a.self:
        res.update(run_self_tests(a.project_dir))
    print(json.dumps(res, indent=2))
