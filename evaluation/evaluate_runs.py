#!/usr/bin/env python3
"""Score finished pipeline runs against the hidden tests and write one tidy CSV.

Takes a finished batch (or experiment) from the backend, downloads every run's generated
project, runs the hidden tests for the task that run was asked to solve, computes IFS of
the final output, and merges that with the pipeline's own metrics (tokens, latency,
attempts, ...).

    python evaluation/evaluate_runs.py --batch <batch_id> --arm A3 --out results/run1/qwen_7b__A3.csv
    python evaluation/evaluate_runs.py --experiment <experiment_id> --out results/exp_cm.csv
    python evaluation/evaluate_runs.py --batch <id> --arm A2 --out x.csv --self-tests

`--self-tests` also re-runs each project's own pytest (needs `pip install pytest`); normally the
pipeline's `status == passed` is the self-test result and that flag is not needed.
Runs whose request is not one of tasks.json cannot be hidden-tested; they are kept (hidden_* empty).
"""
import argparse
import io
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hidden_tests as H  # noqa: E402
from common import RUN_COLUMNS, get_json, http, model_tag, parse_csv, to_bool, write_csv  # noqa: E402


def fetch_project(base: str, project_id: str, dest: Path) -> bool:
    try:
        data = http("GET", f"{base}/projects/{project_id}/download", timeout=120)
    except RuntimeError:
        return False
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest)
    return True


def evaluate_rows(base: str, rows: list[dict], tasks: dict, arm: str | None = None,
                  self_tests: bool = False, hidden_timeout: int = 30, log=print) -> list[dict]:
    """rows = the backend's per-run CSV rows (strings). Returns RUN_COLUMNS-shaped dicts."""
    out, seen = [], defaultdict(int)
    for r in sorted(rows, key=lambda x: int(x.get("idx") or 0)):
        task = H.task_for_request(tasks, r.get("request", ""))
        rep = r.get("rep")
        if rep in (None, ""):
            rep = seen[task["id"] if task else r.get("request")] + 1
        seen[task["id"] if task else r.get("request")] += 1
        row = {**r, "arm": arm or r.get("level") or "", "model_tag": model_tag(r.get("model") or ""),
               "task_id": task["id"] if task else "", "tier": task["tier"] if task else "", "rep": rep}
        status, pid = r.get("status", ""), r.get("project_id", "")
        row["self_pass"] = True if status == "passed" else False
        if task and pid and status in ("passed", "needs_review"):
            with tempfile.TemporaryDirectory(prefix="aiops_proj_") as tmp:
                if fetch_project(base, pid, Path(tmp)):
                    row.update(H.run_hidden(tmp, task, hidden_timeout))
                    row.update(H.ifs_of_project(task["request"], tmp))
                    if self_tests:
                        st = H.run_self_tests(tmp)
                        if st["self_pass"] is None:        # pytest not installed: keep the pipeline's own verdict
                            row["self_status"] = st["self_status"]
                        else:
                            row.update(st)
                else:
                    row.update({"hidden_status": "no_workspace", "hidden_pass": False, "hidden_fraction": 0.0})
        elif task:
            row.update({"hidden_status": "no_project", "hidden_pass": False, "hidden_passed": 0,
                        "hidden_total": len(task["tests"]), "hidden_fraction": 0.0})
        row["false_pass"] = bool(row["self_pass"] and task and not row.get("hidden_pass"))
        out.append(row)
        log(f"  {row['task_id'] or '?':4} rep{rep} {status:12} hidden={row.get('hidden_passed', '-')}/{row.get('hidden_total', '-')}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="http://localhost:8000")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--batch")
    g.add_argument("--experiment")
    ap.add_argument("--arm", help="label for this arm (default: the experiment level)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", help="path to tasks.json (default: evaluation/tasks.json)")
    ap.add_argument("--self-tests", action="store_true")
    ap.add_argument("--hidden-timeout", type=int, default=30)
    a = ap.parse_args()

    base = a.base.rstrip("/")
    tasks = H.load_tasks(a.tasks)
    if a.batch:
        text = http("GET", f"{base}/batch/{a.batch}/results?format=csv").decode("utf-8")
    else:
        text = http("GET", f"{base}/experiments/{a.experiment}/results?format=csv").decode("utf-8")
    rows = parse_csv(text)
    print(f"{len(rows)} run(s) to score")
    res = evaluate_rows(base, rows, tasks, a.arm, a.self_tests, a.hidden_timeout)
    write_csv(a.out, res)
    n = len([r for r in res if r["task_id"]])
    hp = sum(1 for r in res if to_bool(r.get("hidden_pass")))
    print(f"\nwrote {a.out}: {len(res)} rows, {n} matched a task, hidden-test pass {hp}/{n}")


if __name__ == "__main__":
    main()
