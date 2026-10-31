#!/usr/bin/env python3
"""Checkpoint for the evaluation toolkit (evaluation/).

REAL: task set + hidden tests + the subprocess harness, raw_baseline.py, run_ladder.py,
evaluate_runs.py, make_paper_assets.py, capture_environment.py.
STUBBED: one tiny local HTTP server plays both Ollama (/api/generate) and the AIOpsForge
backend (/batch..., /projects/<id>/download) with deterministic, partly-wrong "model" output.

Covers:
  1. task set: unique ids, 8 core tasks, >= 2 IFS entities each, tasks.json in sync with task_defs.py,
     reference solutions pass ALL hidden tests, a do-nothing stub fails ALL of them
  2. harness edge cases: syntax error, missing function, infinite loop (partial credit), code split
     over files, code+tests in one file, input() / sys.exit in generated code
  3. raw_baseline.py (A0) end to end; its system prompt is identical to the Developer's
  4. run_ladder.py end to end (seeds, defaults sent to the backend, A4 memory reset per repetition, resume)
  5. evaluate_runs.py, make_paper_assets.py (tables, figures, summary), capture_environment.py
"""
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "evaluation"
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(ROOT / "backend"))

import task_defs  # noqa: E402
import hidden_tests as H  # noqa: E402
from common import RUN_COLUMNS, parse_csv  # noqa: E402


def ok(msg):
    print(f"[ok] {msg}")


TASKS = {t["id"]: t for t in task_defs.TASKS}
BY_REQUEST = {" ".join(t["request"].split()): t for t in task_defs.TASKS}


# ------------------------------------------------------------- 1. task set ---

def stub_source(task):
    return "\n".join(f"def {n}(*a, **k):\n    return None" if n[0].islower() else f"class {n}:\n    pass"
                     for n in task["entry_points"])


def check_tasks():
    from app import ifs
    ids = [t["id"] for t in task_defs.TASKS]
    assert len(ids) == len(set(ids)) == 20 and sum(t["core"] for t in task_defs.TASKS) == 8
    pub = json.loads((EVAL / "tasks.json").read_text())
    assert pub["tasks"] == task_defs.public_tasks(), "tasks.json is stale: run `python evaluation/task_defs.py`"
    assert all("reference" not in t for t in pub["tasks"])
    ok("20 unique tasks, 8 core, tasks.json in sync and free of reference solutions")
    for t in task_defs.TASKS:
        assert len(ifs.extract_entities(t["request"])) >= 2, t["id"]
    ok("every request has >= 2 technical entities, so IFS is defined for all of them")
    for t in task_defs.TASKS:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "sol.py").write_text(t["reference"])
            r = H.run_hidden(d, t)
            assert r["hidden_pass"] and r["hidden_passed"] == r["hidden_total"], (t["id"], r)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "sol.py").write_text(stub_source(t))
            r = H.run_hidden(d, t)
            assert r["hidden_passed"] == 0 and not r["hidden_pass"], (t["id"], r)
    ok("reference solutions pass every hidden test; a do-nothing stub passes none (tests are neither wrong nor trivial)")


# ------------------------------------------------------------- 2. harness ----

def run_dir(files: dict, task_id="T04", timeout=30):
    with tempfile.TemporaryDirectory() as d:
        for name, src in files.items():
            p = Path(d) / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(src)
        return H.run_hidden(d, TASKS[task_id], timeout)


def check_harness():
    ref = TASKS["T04"]["reference"]
    r = run_dir({"a.py": "def chunk_list(items, size:\n  pass"})
    assert r["hidden_status"] == "import_error" and not r["hidden_pass"], r
    r = run_dir({"a.py": "def other(): pass"})
    assert r["hidden_status"] == "missing_entry_point", r
    r = run_dir({})
    assert r["hidden_status"] == "no_python_files", r
    ok("syntax error / missing function / empty project are reported with distinct statuses, never a crash")

    r = run_dir({"impl.py": ref, "test_impl.py": "from impl import chunk_list\n\ndef test_x():\n    assert chunk_list([1],1)==[[1]]\n"})
    assert r["hidden_pass"] and r["hidden_file"] == "impl.py", r
    r = run_dir({"all_in_one.py": ref + "\n\ndef test_it():\n    assert chunk_list([1,2],1)==[[1],[2]]\n"})
    assert r["hidden_pass"], r
    r = run_dir({"sub/impl.py": ref})
    assert r["hidden_pass"], r
    ok("finds the function in a separate implementation file, in a code+tests file, and in a sub-folder")

    r = run_dir({"a.py": ref.replace("return [list", "x = input()\n    return [list")})
    assert r["hidden_status"] == "ok" and r["hidden_passed"] < r["hidden_total"], r  # input() -> EOF inside the function
    r = run_dir({"a.py": "import sys\nsys.exit(3)\n" + ref})
    assert r["hidden_status"] in ("import_error", "missing_entry_point") and not r["hidden_pass"], r
    ok("input() and sys.exit() in generated code cannot hang or kill the evaluator")

    hang = ref.replace("    if size < 1:", "    if size == 5:\n        while True:\n            pass\n    if size < 1:")
    r = run_dir({"a.py": hang}, timeout=3)
    assert r["hidden_status"] == "timeout" and 0 < r["hidden_passed"] < r["hidden_total"], r
    ok(f"an infinite loop times out and keeps the credit for tests finished before it ({r['hidden_passed']}/{r['hidden_total']})")

    wrong = ref.replace("range(0, len(items), size)", "range(0, len(items), size + 1)")
    r = run_dir({"a.py": wrong})
    assert 0 < r["hidden_passed"] < r["hidden_total"] and r["hidden_errors"], r
    ok("a partly wrong solution gets partial credit and the failing assertions are listed")

    from app import ifs as app_ifs
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "a.py").write_text(ref)
        v = H.ifs_of_project(TASKS["T04"]["request"], d)
        assert v["ifs_final"] is not None and 0 <= v["ifs_final_strict"] <= v["ifs_final"] <= 1, v
    ok("IFS of the final output is computed the same way for every arm")


# --------------------------------------------------------------- 3. stubs ----

def correct_for(model: str, seed: int, task_id: str) -> bool:
    h = int(hashlib.sha256(f"{model}|{seed}|{task_id}".encode()).hexdigest(), 16)
    return h % 3 != 0 if "7b" in model else h % 2 == 0


class Stub(BaseHTTPRequestHandler):
    batches: dict = {}
    batch_bodies: list = []
    generate_calls: list = []

    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype="application/json", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        path = urlparse(self.path).path
        if path == "/api/generate":
            Stub.generate_calls.append(body)
            task = None
            for line in body.get("prompt", "").split("\n"):
                pass
            req = body.get("prompt", "").split("Task:\n", 1)[-1].split("\n\nWrite the code now.")[0]
            task = BY_REQUEST.get(" ".join(req.split()))
            seed = (body.get("options") or {}).get("seed", 0)
            if task is None:
                return self._send(json.dumps({"response": "hi"}).encode())
            good = correct_for(body["model"], seed, task["id"])
            code = task["reference"] if good else stub_source(task)
            test = f"\n\ndef test_smoke():\n    assert callable({task['entry_points'][0]})\n"
            return self._send(json.dumps({"response": f"```python\n{code}{test}```", "prompt_eval_count": 120, "eval_count": 80}).encode())
        if path == "/batch":
            Stub.batch_bodies.append(body)
            bid = f"b{len(Stub.batches) + 1}"
            Stub.batches[bid] = body
            return self._send(json.dumps({"batch_id": bid, "status": "running", "total": len(body["requests"])}).encode())
        self._send(b"{}", code=404)

    def do_GET(self):
        u = urlparse(self.path)
        parts = u.path.strip("/").split("/")
        if parts[0] == "batch" and len(parts) == 2:
            b = Stub.batches[parts[1]]
            return self._send(json.dumps({"status": "done", "finished": len(b["requests"]), "total": len(b["requests"])}).encode())
        if parts[0] == "batch" and len(parts) == 3:
            b = Stub.batches[parts[1]]
            d = b["defaults"]
            lines = ["idx,status,request,model,project_id,tokens_total,total_ms,avg_attempts,communication_mode,max_qa_attempts,"
                     "failure_recovery,use_memory,temperature,seed,num_ctx,ifs_avg,ifs_strict_avg,ifs_planner,tasks,tasks_passed,run_ok"]
            for i, it in enumerate(b["requests"]):
                task = BY_REQUEST[" ".join(it["request"].split())]
                good = correct_for(d["model"], it["seed"], task["id"])
                st = "passed" if (good or i % 2) else "needs_review"
                q = '"' + it["request"].replace('"', '""') + '"'
                lines.append(f"{i},{st},{q},{d['model']},{parts[1]}__{i},{1000 + i},{5000 + 100 * i},1.5,{d.get('communication_mode')},"
                             f"{d.get('max_qa_attempts')},{d.get('failure_recovery', '')},{d.get('use_memory')},{d.get('temperature')},"
                             f"{it['seed']},{d.get('num_ctx')},0.8,0.7,0.9,1,{1 if st == 'passed' else 0},{st == 'passed'}")
            return self._send(("\n".join(lines) + "\n").encode(), "text/csv")
        if parts[0] == "projects" and parts[-1] == "download":
            bid, idx = parts[1].split("__")
            b = Stub.batches[bid]
            it = b["requests"][int(idx)]
            task = BY_REQUEST[" ".join(it["request"].split())]
            good = correct_for(b["defaults"]["model"], it["seed"], task["id"])
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("solution.py", task["reference"] if good else stub_source(task))
                zf.writestr("test_solution.py", f"from solution import {task['entry_points'][0]}\n\ndef test_a():\n    assert True\n")
            return self._send(buf.getvalue(), "application/zip")
        if u.path == "/api/version":
            return self._send(b'{"version": "0.0-stub"}')
        self._send(b"{}", code=404)


def serve():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def run_main(module, argv):
    old = sys.argv
    sys.argv = [module.__name__] + argv
    try:
        module.main()
    finally:
        sys.argv = old


def check_raw_baseline(url, out):
    import raw_baseline
    from app.agents.developer import DEVELOPER_SYSTEM_PROMPT
    assert raw_baseline.SYSTEM_PROMPT == DEVELOPER_SYSTEM_PROMPT, "A0 must use the Developer's exact system prompt"
    ok("A0 uses the Developer's exact system prompt (any difference would make A0 vs pipeline unfair)")
    Stub.generate_calls.clear()
    run_main(raw_baseline, ["--ollama", url, "--models", "stub:3b,stub:7b", "--tasks", "T01,T04", "--reps", "2",
                            "--out-root", str(out), "--run-name", "t"])
    rows = parse_csv((out / "t" / "stub_3b__A0.csv").read_text())
    assert len(rows) == 4 and {r["arm"] for r in rows} == {"A0"} and {r["rep"] for r in rows} == {"1", "2"}
    assert all(r["tokens_total"] == "200" and r["hidden_total"] for r in rows)
    for r in rows:
        assert (r["hidden_pass"] == "True") == correct_for("stub:3b", int(r["seed"]), r["task_id"]), r
        assert r["self_pass"] in ("True", "False") and r["ifs_final"] != ""
    opts = [c["options"] for c in Stub.generate_calls]
    assert all(o["temperature"] == 0.2 and o["num_ctx"] == 8192 for o in opts) and {o["seed"] for o in opts} == {1, 2}
    ok("raw baseline: one call per task/rep with temperature, seed (1,2) and num_ctx sent; scored by the hidden tests")


def check_ladder(url, out):
    import run_ladder
    Stub.batches.clear(); Stub.batch_bodies.clear()
    marker = out / "resets.txt"
    cmd = f'{sys.executable} -c "open(r\'{marker}\',\'a\').write(\'x\')"'
    argv = ["--base", url, "--ollama", url, "--models", "stub:3b,stub:7b", "--arms", "A1,A3,A3x,A4", "--tasks", "T01,T04,T13",
            "--reps", "2", "--poll", "0", "--run-name", "t", "--out-root", str(out), "--reset-memory-cmd", cmd]
    run_main(run_ladder, argv)
    bodies = Stub.batch_bodies
    # A1/A3/A3x: one batch each per model; A4: one per repetition -> (3 + 2) * 2 models
    assert len(bodies) == 10, len(bodies)
    a1 = next(b for b in bodies if "-A1-" in b["name"])
    assert a1["defaults"]["communication_mode"] == "message_passing" and a1["defaults"]["max_qa_attempts"] == 1
    assert a1["defaults"]["use_memory"] is False and a1["max_concurrency"] == 1 and a1["defaults"]["num_ctx"] == 8192
    a3x = next(b for b in bodies if "-A3x-" in b["name"])
    assert a3x["defaults"]["failure_recovery"] is False and a3x["defaults"]["max_qa_attempts"] == 3
    assert {i["seed"] for i in a1["requests"]} == {1, 2} and len(a1["requests"]) == 6
    ok("each arm sends exactly its settings (A1 mp/1 attempt, A3x recovery off, ...), concurrency 1, seeds 1,2 per repetition")
    names = [i["project_name"] for i in a1["requests"]]
    a3 = next(b for b in bodies if "-A3-" in b["name"])
    assert names != [i["project_name"] for i in a3["requests"]] and sorted(i["request"] for i in a1["requests"]) == sorted(i["request"] for i in a3["requests"])
    ok("same tasks in every arm, but a different shuffled order per arm")
    assert marker.read_text() == "x" * (1 + 2 * 2), marker.read_text()   # once up front + before each A4 repetition, 2 models
    ok("long-term memory is wiped once up front and before every A4 repetition")
    for m in ("stub_3b", "stub_7b"):
        for arm in ("A1", "A3", "A3x", "A4"):
            rows = parse_csv((out / "t" / f"{m}__{arm}.csv").read_text())
            assert len(rows) == 6 and {r["task_id"] for r in rows} == {"T01", "T04", "T13"} and {r["rep"] for r in rows} == {"1", "2"}, (m, arm)
            for r in rows:
                assert (r["hidden_pass"] == "True") == correct_for(m.replace("_", ":"), int(r["seed"]), r["task_id"]), r
    assert json.loads((out / "t" / "manifest.json").read_text())["seeds"] == [1, 2]
    assert (out / "t" / "environment.json").exists()
    ok("one scored CSV per (model, arm) with the right tasks/reps/seeds; manifest.json and environment.json written")
    n = len(Stub.batch_bodies)
    run_main(run_ladder, argv)
    assert len(Stub.batch_bodies) == n
    ok("re-running skips finished arms (an interrupted overnight run can just be started again)")


def check_assets(out):
    import make_paper_assets
    import raw_baseline
    run_main(raw_baseline, ["--ollama", Stub_url, "--models", "stub:3b,stub:7b", "--tasks", "T01,T04,T13", "--reps", "2",
                            "--out-root", str(out), "--run-name", "t"])
    run_main(make_paper_assets, ["--results", str(out / "t")])
    pa = out / "t" / "paper_assets"
    for f in ("tables/main_table.tex", "tables/main_table.csv", "tables/comparisons.tex", "tables/cross_model.md", "tables/by_task.csv",
              "figures/fig_pass_rate.png", "figures/fig_pass_rate.pdf", "figures/fig_false_pass.png", "figures/fig_latency.png",
              "figures/fig_tokens.png", "figures/fig_ifs.png", "figures/fig_task_heatmap.png", "figures/fig_cross_model.png",
              "SUMMARY.md", "stats.json"):
        assert (pa / f).exists() and (pa / f).stat().st_size > 0, f
    summ = (pa / "SUMMARY.md").read_text()
    assert "Claims, arm by arm" in summ and "Small model + pipeline vs large model" in summ and "fewer than 5 tasks" in summ, summ[-900:]
    tex = (pa / "tables" / "main_table.tex").read_text()
    assert r"\begin{tabular}" in tex and "Hidden pass" in tex
    stats_json = json.loads((pa / "stats.json").read_text())
    assert stats_json["comparisons"] and stats_json["cross_model"]
    ok("make_paper_assets writes LaTeX/Markdown/CSV tables, PNG+PDF figures, SUMMARY.md and stats.json; warns about tiny n")


def check_evaluate_and_env(url, out):
    import evaluate_runs
    import capture_environment
    Stub.batches.clear()
    from common import post_json
    body = {"name": "x", "requests": [{"request": TASKS["T04"]["request"], "seed": 1}, {"request": TASKS["T04"]["request"], "seed": 2}],
            "defaults": {"model": "stub:7b", "communication_mode": "blackboard", "max_qa_attempts": 3}}
    bid = post_json(f"{url}/batch", body)["batch_id"]
    path = out / "ev.csv"
    run_main(evaluate_runs, ["--base", url, "--batch", bid, "--arm", "AX", "--out", str(path)])
    rows = parse_csv(path.read_text())
    assert [r["task_id"] for r in rows] == ["T04", "T04"] and [r["rep"] for r in rows] == ["1", "2"] and rows[0]["arm"] == "AX"
    ok("evaluate_runs scores a finished batch from the backend and numbers repetitions per task")
    env = out / "env.json"
    run_main(capture_environment, ["--models", "stub:7b", "--ollama", url, "--out", str(env)])
    j = json.loads(env.read_text())
    assert j["ollama"]["version"] == "0.0-stub" and j["code"]["backend_app_sha256"] and "machine" in j
    ok("capture_environment records machine, Ollama version, code fingerprint (null when unavailable, never an error)")
    assert set(RUN_COLUMNS) >= {"hidden_pass", "false_pass", "ifs_final", "ifs_final_strict", "failure_recovery", "seed"}


if __name__ == "__main__":
    check_tasks()
    check_harness()
    srv, Stub_url = serve()
    out = Path(tempfile.mkdtemp(prefix="aiops_eval_t_"))
    check_raw_baseline(Stub_url, out)
    check_ladder(Stub_url, out)
    check_assets(out)
    check_evaluate_and_env(Stub_url, out)
    print("\nAll evaluation-toolkit checks passed.")
