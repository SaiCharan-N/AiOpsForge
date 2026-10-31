"""Subprocess side of the hidden-test runner. Do not run by hand — see hidden_tests.py.

usage: python _harness.py <project_dir> <spec.json>

Loads the generated Python files, finds the one (or the combination) that defines every
entry point, then runs each hidden test block separately. After each test it prints a
`PROGRESS {json}` line, so if the generated code hangs the parent still knows how many
tests passed before it did. Last line printed is `RESULT {json}`.
"""
import contextlib
import io
import json
import os
import sys
import traceback

SKIP_DIRS = {"__pycache__", "node_modules", "venv", ".venv", "site-packages", "dist", "build"}


def raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    except Exception:
        return False
    return False


def collect_files(root):
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for f in filenames:
            if f.endswith(".py"):
                found.append(os.path.join(dirpath, f))
    # implementation files first, test files last; shallow before deep
    return sorted(found, key=lambda p: (os.path.basename(p).startswith("test_") or p.endswith("_test.py"),
                                        p.count(os.sep), p))


def load(path, ns):
    with open(path, encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        exec(compile(src, path, "exec"), ns)


def emit(tag, obj):
    print(tag + " " + json.dumps(obj), flush=True)


def main():
    root, spec_path = os.path.abspath(sys.argv[1]), sys.argv[2]
    with open(spec_path, encoding="utf-8") as fh:
        spec = json.load(fh)
    entry, tests = spec["entry_points"], spec["tests"]
    sys.path.insert(0, root)
    os.chdir(root)
    sys.stdin = open(os.devnull)

    files = collect_files(root)
    result = {"status": None, "file": None, "passed": 0, "total": len(tests), "errors": [], "load_errors": []}
    if not files:
        result["status"] = "no_python_files"
        emit("RESULT", result)
        return

    chosen = None
    loaded_any = False
    for path in files:                                  # 1) a single file that defines everything
        ns = {"__name__": "candidate", "__file__": path}
        try:
            load(path, ns)
            loaded_any = True
        except BaseException as exc:  # noqa: BLE001 — generated code may do anything
            result["load_errors"].append(f"{os.path.relpath(path, root)}: {type(exc).__name__}: {exc}"[:200])
            continue
        if all(callable(ns.get(n)) for n in entry):
            chosen, result["file"] = ns, os.path.relpath(path, root)
            break
    if chosen is None:                                  # 2) definitions split across files
        merged = {"__name__": "candidate"}
        for path in files:
            try:
                load(path, merged)
                loaded_any = True
            except BaseException:  # noqa: BLE001
                pass
        if all(callable(merged.get(n)) for n in entry):
            chosen, result["file"] = merged, "(merged files)"
    if chosen is None:
        result["status"] = "missing_entry_point" if loaded_any else "import_error"
        emit("RESULT", result)
        return

    result["status"] = "ok"
    for i, block in enumerate(tests):
        ns = dict(chosen)
        ns["raises"] = raises
        sink = io.StringIO()
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                exec(compile(block, f"<hidden test {i + 1}>", "exec"), ns)
            result["passed"] += 1
        except BaseException as exc:  # noqa: BLE001 — incl. SystemExit from generated code
            result["errors"].append(f"test {i + 1}: {type(exc).__name__}: {str(exc)[:120]}")
        emit("PROGRESS", {"done": i + 1, "passed": result["passed"]})
    emit("RESULT", result)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001
        emit("RESULT", {"status": "harness_error", "passed": 0, "total": 0, "errors": [traceback.format_exc()[-300:]],
                        "load_errors": [], "file": None})
