#!/usr/bin/env python3
"""scripts/integrate_experiments.py against a REAL server: uvicorn serving the real
FastAPI app (in this process, SQLite for the DB, simulated pipeline runs), the CLI run
as a subprocess, and then the project's original experiments/analyze_results.py run on
what the CLI wrote — proving the old analysis still works on the new experiments.
"""
import asyncio
import importlib.util
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("tb", ROOT / "scripts" / "test_batch.py")
tb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tb)
import uvicorn  # noqa: E402

from app import batch, experiments, mcp_client  # noqa: E402
from app.main import app  # noqa: E402


def ok(msg):
    print(f"[ok] {msg}")


def free_port():
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def pipeline_run(pid, request, mode, framework=None, overrides=None):
    """Memory-aware simulated run: the second time a request is seen it is warm."""
    overrides = overrides or {}
    await asyncio.sleep(0.01)
    n = pipeline_run.seen.get((request, overrides.get("use_memory")), 0)
    pipeline_run.seen[(request, overrides.get("use_memory"))] = n + 1
    warm = bool(overrides.get("use_memory")) and n >= 1
    pos = pipeline_run.pos = pipeline_run.pos + 1
    with tb.metrics.scope(pid, "pipeline"):
        tb.simulate_run(pid, request, mode, framework, overrides,
                        tokens=(100 if warm else 160) + (60 if mode == "message_passing" else 0) + tb.RND.gauss(0, 4),
                        total_ms=(1500 if warm else 3200) + tb.RND.gauss(0, 30), attempts=1 if warm else 3, memory_hit=warm)
    return tb.result_of()
pipeline_run.seen, pipeline_run.pos = {}, 0


def run_cli(base, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "integrate_experiments.py"), "--base", base, "--poll", "0.2", *args],
                          capture_output=True, text=True, timeout=180)


def main():
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    out = Path(tempfile.mkdtemp(prefix="aiops_cli_"))
    with tb.Env(run_fn=pipeline_run) as env, patch.object(batch, "manager", env.bm), patch.object(experiments, "manager", env.em):
        async def fake_call_tool(name, args):
            await asyncio.sleep(0.003)
            return "ok"
        async def fake_discover():
            await asyncio.sleep(0.01)
            return [{"name": n} for n in ("write_file", "run_shell_command", "run_tests", "check_webapp")]
        patch.object(mcp_client, "discover_tools", fake_discover).start()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        for _ in range(100):
            try:
                urllib.request.urlopen(base + "/experiments/presets", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        try:
            with patch.object(mcp_client, "call_tool", fake_call_tool):
                r = run_cli(base, "--list")
                assert r.returncode == 0 and all(p in r.stdout for p in experiments.PRESETS), r.stdout + r.stderr
                ok("--list prints every preset with its design and description")

                r = run_cli(base, "memory_speedup", "--out", str(out))
                assert r.returncode == 0, r.stdout + r.stderr
                assert "verdict:" in r.stdout and "paired p=" in r.stdout and "legacy shape" in r.stdout, r.stdout
                legacy = json.loads((out / "exp1_memory_speedup.json").read_text())
                p0 = legacy["pairs"][0]
                assert set(p0) == {"request", "first_run", "repeat_run"} and set(p0["first_run"]) == {"time_s", "attempts"}
                assert p0["repeat_run"]["memory_used"] is True and p0["first_run"]["attempts"] == 3.0 and p0["repeat_run"]["attempts"] == 1.0
                assert len(legacy["pairs"]) == 3 and 3.0 < p0["first_run"]["time_s"] < 3.4
                for f in ("memory_speedup.result.json", "memory_speedup.runs.csv", "memory_speedup.tex"):
                    assert (out / f).stat().st_size > 50, f
                ok("memory_speedup (old exp1): runs on the server, prints the verdict, writes result.json + runs.csv + .tex + the legacy exp1 file")

                r = run_cli(base, "repeat_bug_trend", "--out", str(out))
                assert r.returncode == 0, r.stdout + r.stderr
                seq = json.loads((out / "exp3_repeat_bug_trend.json").read_text())["sequence"]
                assert [s["position_in_sequence"] for s in seq] == [1, 2, 3, 4, 5] and set(seq[0]) == {"position_in_sequence", "request", "time_s", "attempts", "memory_used"}
                r = run_cli(base, "mcp_overhead", "--calls", "12", "--out", str(out))
                assert r.returncode == 0, r.stdout + r.stderr
                mcp = json.loads((out / "exp2_mcp_overhead.json").read_text())
                assert mcp["mcp"]["tools_found"] == 4 and mcp["mcp"]["discovery_time_s"] >= 0.009 and mcp["mcp"]["avg_call_time_s"] > 0.002
                assert mcp["n_calls"] == 12 and len(mcp["mcp"]["call_times_s"]) == 12 and len(mcp["direct_baseline"]["call_times_s"]) == 12 and mcp["overhead_s"] > 0.002
                ok("repeat_bug_trend (old exp3) and mcp_overhead (old exp2) also write their legacy files")

                r = run_cli(base, "communication_mode", "--repetitions", "2", "--out", str(out), "--name", "cli-ab")
                assert r.returncode == 0, r.stdout + r.stderr
                res = json.loads((out / "communication_mode.result.json").read_text())
                assert res["experiment"]["name"] == "cli-ab" and res["experiment"]["total_runs"] == 20
                assert res["result"]["comparisons"]["tokens_total"]["comparisons"][0]["significant"]
                assert "exp" not in r.stdout.split("wrote to")[1].replace("experiment", "")
                tex = (out / "communication_mode.tex").read_text()
                assert "\\toprule" in tex and "message\\_passing" in tex
                import csv
                assert len(list(csv.DictReader((out / "communication_mode.runs.csv").open()))) == 20
                ok("ab preset with --repetitions 2: 20 runs, significant result, raw CSV with 20 rows, LaTeX tables")

                r = run_cli(base, "retry_policy", "--levels", "2,4", "--repetitions", "1", "--out", str(out))
                assert r.returncode == 0 and [a["label"] for a in json.loads((out / "retry_policy.result.json").read_text())["result"]["arms"]] == ["2", "4"]
                r = run_cli(base, "memory", "--levels", "on,off", "--out", str(out))
                assert r.returncode == 0 and [a["label"] for a in json.loads((out / "memory.result.json").read_text())["result"]["arms"]] == ["on", "off"]
                ok("--levels is parsed per factor (numbers for max_qa_attempts, on/off -> booleans for use_memory)")

                r = run_cli(base, "no_such_preset")
                assert r.returncode != 0 and "unknown preset" in (r.stdout + r.stderr)
                r = run_cli(base, "retry_policy", "--levels", "1,99")
                assert r.returncode != 0 and "1 to 10" in (r.stdout + r.stderr), r.stdout + r.stderr
                r = subprocess.run([sys.executable, str(ROOT / "scripts" / "integrate_experiments.py"), "--base", "http://127.0.0.1:1", "--list"],
                                   capture_output=True, text=True, timeout=30)
                assert r.returncode != 0 and "cannot reach the backend" in (r.stdout + r.stderr)
                ok("errors are clear: unknown preset, a 422 from the backend (with its message), backend unreachable")
        finally:
            server.should_exit = True
            t.join(timeout=10)

    # the ORIGINAL analysis script on what the CLI wrote
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("[skip] analyze_results.py check (matplotlib not installed)")
        return
    work = Path(tempfile.mkdtemp(prefix="aiops_analyze_"))
    shutil.copytree(ROOT / "experiments", work / "experiments", ignore=shutil.ignore_patterns("results"))
    (work / "experiments" / "results").mkdir()
    for f in ("exp1_memory_speedup.json", "exp2_mcp_overhead.json", "exp3_repeat_bug_trend.json"):
        shutil.copy(out / f, work / "experiments" / "results" / f)
    r = subprocess.run([sys.executable, "analyze_results.py"], cwd=work / "experiments", capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    produced = sorted(p.name for p in (work / "experiments" / "results").iterdir())
    assert {"exp1_chart.png", "exp2_chart.png", "exp3_chart.png", "summary.md"} <= set(produced), produced
    ok(f"the original analyze_results.py runs on the CLI's output and produces {', '.join(n for n in produced if not n.endswith('.json'))}")
    print("\nAll integrate_experiments checks passed.")


if __name__ == "__main__":
    main()
