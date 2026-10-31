#!/usr/bin/env python3
"""Run an experiment on a running AIOpsForge backend and save everything it produces.

This replaces experiments/exp1_memory_speedup.py, exp2_mcp_overhead.py and
exp3_repeat_bug_trend.py: the experiments now run INSIDE the backend (concurrent
batches, per-run settings, statistics) and this script is just a client for them.

    python scripts/integrate_experiments.py --list
    python scripts/integrate_experiments.py memory_speedup          # old exp1
    python scripts/integrate_experiments.py mcp_overhead            # old exp2
    python scripts/integrate_experiments.py repeat_bug_trend        # old exp3
    python scripts/integrate_experiments.py communication_mode --repetitions 5
    python scripts/integrate_experiments.py retry_policy --levels 1,2,4 --out results/retry

For each run it writes into --out (default experiments/results):
    <preset>.result.json   the full result (statistics, verdict, configuration, environment)
    <preset>.runs.csv      raw per-run data — load it in R / pandas
    <preset>.tex           booktabs LaTeX tables for the paper
and, for the three original experiments, ALSO the legacy files
(exp1_memory_speedup.json, exp2_mcp_overhead.json, exp3_repeat_bug_trend.json) in the
shape experiments/analyze_results.py reads, so its charts and summary.md keep working.

Standard library only.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "experiments" / "results"


def call(base, path, body=None, raw=False):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read().decode()
            return data if raw else json.loads(data)
    except urllib.error.HTTPError as e:
        detail = e.read().decode()
        try:
            detail = json.loads(detail).get("detail", detail)
        except ValueError:
            pass
        sys.exit(f"backend said {e.code}: {detail}")
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach the backend at {base}: {e.reason}")


def legacy_exp1(res):
    """analyze_results.py: {"pairs": [{"request", "first_run": {time_s, attempts}, "repeat_run": {time_s, attempts, memory_used}}]}"""
    pairs = [{"request": p["request"],
              "first_run": {"time_s": (p["first"]["total_ms"] or 0) / 1000, "attempts": p["first"]["avg_attempts"]},
              "repeat_run": {"time_s": (p["repeat"]["total_ms"] or 0) / 1000, "attempts": p["repeat"]["avg_attempts"],
                             "memory_used": bool(p["repeat"]["memory_hits"])}} for p in res["pairs"]]
    return {"pairs": pairs}


def legacy_exp2(res):
    """{"n_calls", "mcp": {"call_times_s": [...]}, "direct_baseline": {"call_times_s": [...]}, "overhead_s"}"""
    s = res["samples"]
    disc = res.get("discovery_ms")
    return {"n_calls": res["calls"],
            "mcp": {"discovery_time_s": round(disc / 1000, 4) if disc is not None else None, "tools_found": res.get("tools_found"),
                    "avg_call_time_s": round(res["mcp_ms"]["mean"] / 1000, 4), "call_times_s": [x / 1000 for x in s["mcp_ms"]]},
            "direct_baseline": {"avg_call_time_s": round(res["direct_ms"]["mean"] / 1000, 5), "call_times_s": [x / 1000 for x in s["direct_ms"]]},
            "overhead_s": round(res["overhead_ms"] / 1000, 4)}


def legacy_exp3(res):
    """{"sequence": [{"position_in_sequence", "request", "time_s", "attempts", "memory_used"}]}"""
    return {"sequence": [{"position_in_sequence": s["position"], "request": s["request"], "time_s": (s["total_ms"] or 0) / 1000,
                          "attempts": s["avg_attempts"], "memory_used": bool(s["memory_hits"])} for s in res["series"]]}


LEGACY = {"memory_speedup": ("exp1_memory_speedup", legacy_exp1), "mcp_overhead": ("exp2_mcp_overhead", legacy_exp2),
          "repeat_bug_trend": ("exp3_repeat_bug_trend", legacy_exp3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("preset", nargs="?", help="a preset name (see --list)")
    ap.add_argument("--base", default="http://localhost:8000", help="backend URL")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--list", action="store_true", help="list the presets")
    ap.add_argument("--repetitions", type=int, help="override the preset's repetitions (ab designs)")
    ap.add_argument("--levels", help="comma-separated levels of the factor, first = baseline (ab designs)")
    ap.add_argument("--calls", type=int, help="mcp_overhead: number of tool calls")
    ap.add_argument("--concurrency", type=int, help="runs in flight at once")
    ap.add_argument("--name", help="a name for the experiment")
    ap.add_argument("--poll", type=float, default=3.0, help="seconds between status checks")
    a = ap.parse_args()

    presets = call(a.base, "/experiments/presets")
    if a.list or not a.preset:
        for p in presets["presets"]:
            print(f"{p['name']:20} {p['design']:13} {p['description']}")
        return
    preset = next((p for p in presets["presets"] if p["name"] == a.preset), None)
    if preset is None:
        sys.exit(f"unknown preset {a.preset!r}; try --list")

    body = {"preset": a.preset}
    if a.name:
        body["name"] = a.name
    if a.repetitions:
        body["repetitions"] = a.repetitions
    if a.calls:
        body["calls"] = a.calls
    if a.concurrency:
        body["max_concurrency"] = a.concurrency
    if a.levels:
        factor = preset.get("factor")
        parts = [x.strip() for x in a.levels.split(",") if x.strip()]
        body["levels"] = ([x.lower() in ("on", "true", "1", "yes") for x in parts] if factor == "use_memory"
                          else [int(x) for x in parts] if factor == "max_qa_attempts" else parts)

    exp_id = call(a.base, "/experiments", body)["experiment_id"]
    print(f"experiment {exp_id} started ({a.preset})")
    last = ""
    while True:
        st = call(a.base, f"/experiments/{exp_id}")
        line = f"  {st['status']}: {st['finished_runs']}/{st['total_runs'] or '?'} runs"
        if line != last:
            print(line)
            last = line
        if st["status"] != "running":
            break
        time.sleep(a.poll)
    if st["status"] != "done":
        sys.exit(f"experiment ended as {st['status']}" + (f": {st['error']}" if st.get("error") else ""))

    a.out.mkdir(parents=True, exist_ok=True)
    res = call(a.base, f"/experiments/{exp_id}/results")
    (a.out / f"{a.preset}.result.json").write_text(json.dumps(res, indent=2))
    (a.out / f"{a.preset}.runs.csv").write_text(call(a.base, f"/experiments/{exp_id}/results?format=csv", raw=True))
    (a.out / f"{a.preset}.tex").write_text(call(a.base, f"/experiments/{exp_id}/results?format=latex", raw=True))
    written = [f"{a.preset}.result.json", f"{a.preset}.runs.csv", f"{a.preset}.tex"]
    if a.preset in LEGACY:
        stem, convert = LEGACY[a.preset]
        (a.out / f"{stem}.json").write_text(json.dumps(convert(res["result"]), indent=2))
        written.append(f"{stem}.json  (legacy shape, for experiments/analyze_results.py)")
    print("\nverdict:")
    for v in res["result"].get("verdict", []):
        print(f"  - {v}")
    for w in res["result"].get("warnings", []):
        print(f"  ! {w}")
    print(f"\nwrote to {a.out}:\n  " + "\n  ".join(written))


if __name__ == "__main__":
    main()
