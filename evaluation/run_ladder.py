#!/usr/bin/env python3
"""Run the comparison ladder end to end: pipeline arms x models x tasks x repetitions,
score every run with the hidden tests, save one CSV per (model, arm).

The ladder (each arm changes ONE thing relative to the one before it):

    A0  raw model, one call, no pipeline            -> evaluation/raw_baseline.py (run it separately)
    A1  message passing, 1 attempt, no memory       "SE-Blackboard-style baseline inside our framework"
    A2  blackboard,       1 attempt, no memory      = SE-Blackboard's question (blackboard vs message passing)
    A3  blackboard,       3 attempts + failure recovery, no memory     = + QA feedback loop & failure analysis
    A3x same as A3 but failure_recovery OFF         = isolates the Failure Analyzer's contribution
    A4  A3 + long-term memory (wiped before every repetition so repetitions don't teach each other)

    python evaluation/run_ladder.py --dry-run                      # just print the plan and size
    python evaluation/run_ladder.py --models qwen2.5-coder:3b,qwen2.5-coder:7b --tasks core --reps 3
    python evaluation/run_ladder.py --arms A1,A3 --models qwen2.5-coder:7b --tasks T01,T02 --reps 1   # smoke test

Fairness rules built in: same tasks and same seed per (task, repetition) in every arm; concurrency 1
(so latency is not distorted by queueing); item order shuffled differently per arm (no systematic
order effect); the model is warmed up once before its arms (first-load time is not billed to an arm);
one model at a time (Ollama keeps one loaded). Re-running skips arms whose CSV already exists
(--force to redo), so an interrupted overnight run can simply be started again.
"""
import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hidden_tests as H  # noqa: E402
from common import get_json, model_tag, parse_csv, post_json, http, write_csv  # noqa: E402
from evaluate_runs import evaluate_rows  # noqa: E402
from raw_baseline import select_tasks  # noqa: E402

FINAL = ("done", "failed", "cancelled", "interrupted")

ARMS = {
    "A1": {"label": "message passing, 1 attempt, no memory (baseline)",
           "defaults": {"communication_mode": "message_passing", "max_qa_attempts": 1, "use_memory": False}},
    "A2": {"label": "blackboard, 1 attempt, no memory",
           "defaults": {"communication_mode": "blackboard", "max_qa_attempts": 1, "use_memory": False}},
    "A3": {"label": "blackboard, 3 attempts + failure recovery, no memory",
           "defaults": {"communication_mode": "blackboard", "max_qa_attempts": 3, "use_memory": False,
                        "failure_recovery": True}},
    "A3x": {"label": "A3 with failure recovery OFF",
            "defaults": {"communication_mode": "blackboard", "max_qa_attempts": 3, "use_memory": False,
                         "failure_recovery": False}},
    "A4": {"label": "A3 + long-term memory (wiped before each repetition)",
           "defaults": {"communication_mode": "blackboard", "max_qa_attempts": 3, "use_memory": True,
                        "failure_recovery": True},
           "memory": True},
}
DEFAULT_RESET_CMD = ('docker compose exec -T postgres psql -U aiops -d aiopsforge '
                     '-c "TRUNCATE memory_entries, failure_patterns"')


def plan_size(arms, models, n_tasks, reps) -> int:
    return len(arms) * len(models) * n_tasks * reps


def wait_for_batch(base: str, batch_id: str, poll: int, max_hours: float, log=print) -> dict:
    deadline = time.time() + max_hours * 3600
    last = None
    while True:
        st = get_json(f"{base}/batch/{batch_id}")
        line = f"{st.get('finished', 0)}/{st.get('total', '?')} finished, status {st.get('status')}"
        if line != last:
            log("    " + line)
            last = line
        if st.get("status") in FINAL:
            return st
        if time.time() > deadline:
            raise TimeoutError(f"batch {batch_id} still running after {max_hours} h")
        time.sleep(poll)


def build_items(tasks: list[dict], reps: list[int], base_seed: int, arm: str, model: str) -> list[dict]:
    items = [{"request": t["request"], "seed": base_seed + r - 1, "project_name": f"{arm}-{t['id']}-r{r}"}
             for r in reps for t in tasks]
    random.Random(f"{arm}|{model}|{reps[0]}").shuffle(items)
    return items


def run_batch(base, name, items, defaults, poll, max_hours, log):
    body = {"name": name, "requests": items, "defaults": defaults, "max_concurrency": 1}
    created = post_json(f"{base}/batch", body, timeout=120)
    bid = created["batch_id"]
    log(f"    batch {bid} started ({len(items)} runs)")
    wait_for_batch(base, bid, poll, max_hours, log)
    return bid, parse_csv(http("GET", f"{base}/batch/{bid}/results?format=csv").decode("utf-8"))


def reset_memory(cmd: str, log) -> None:
    r = subprocess.run(cmd, shell=True, cwd=str(H.ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"memory reset failed ({cmd!r}): {(r.stderr or r.stdout).strip()[:300]}")
    log("    long-term memory wiped")


def warmup(ollama: str, model: str, log) -> None:
    try:
        post_json(f"{ollama}/api/generate", {"model": model, "prompt": "hi", "stream": False, "keep_alive": "30m",
                                              "options": {"num_predict": 1}}, timeout=900)
        log(f"  {model} loaded (warm-up done)")
    except Exception as exc:  # noqa: BLE001
        log(f"  WARNING: could not warm up {model}: {exc} — the first run will include model load time")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://localhost:8000", help="AIOpsForge backend")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--models", default="qwen2.5-coder:3b,qwen2.5-coder:7b")
    ap.add_argument("--arms", default="A1,A2,A3,A3x,A4")
    ap.add_argument("--tasks", default="core", help="core (8) | all (20) | T01,T02,...")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1, help="repetition k uses seed (seed + k - 1) in every arm")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--run-name", default="run1")
    ap.add_argument("--out-root", default=str(H.ROOT / "results"))
    ap.add_argument("--poll", type=int, default=20, help="seconds between progress checks")
    ap.add_argument("--max-hours", type=float, default=12, help="give up on ONE batch after this long")
    ap.add_argument("--reset-memory-cmd", default=DEFAULT_RESET_CMD)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo arms whose CSV already exists")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in arms if x not in ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; choose from {list(ARMS)}")
    tasks = select_tasks(H.load_tasks(), a.tasks)
    total = plan_size(arms, models, len(tasks), a.reps)
    base, out = a.base.rstrip("/"), Path(a.out_root) / a.run_name
    log = print

    log(f"Plan: {len(arms)} arm(s) x {len(models)} model(s) x {len(tasks)} task(s) x {a.reps} repetition(s) = {total} pipeline runs")
    for k in arms:
        log(f"  {k:4} {ARMS[k]['label']}")
    log(f"  seeds per repetition: {[a.seed + r for r in range(a.reps)]}; temperature {a.temperature}; num_ctx {a.num_ctx}; concurrency 1")
    log("  Time = runs x (minutes per run on YOUR machine). Measure first: run the smoke test (1 task, 1 rep) and multiply.")
    if a.dry_run:
        return

    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({
        "started_at": datetime.now(timezone.utc).isoformat(), "models": models, "arms": {k: ARMS[k] for k in arms},
        "tasks": [t["id"] for t in tasks], "reps": a.reps, "seeds": [a.seed + r for r in range(a.reps)],
        "temperature": a.temperature, "num_ctx": a.num_ctx, "concurrency": 1, "args": vars(a)}, indent=2), encoding="utf-8")
    try:
        import capture_environment  # noqa: F401  (also run by hand; failure here must not stop the ladder)
        sys.argv = ["capture_environment", "--models", ",".join(models), "--ollama", a.ollama, "--out", str(out / "environment.json")]
        capture_environment.main()
    except Exception as exc:  # noqa: BLE001
        log(f"  (environment capture skipped: {exc})")

    if "A4" in arms and not a.dry_run:
        try:
            reset_memory(a.reset_memory_cmd, log)      # fail early, not after hours of A1-A3
        except RuntimeError as exc:
            raise SystemExit(f"{exc}\nFix --reset-memory-cmd, or drop A4 from --arms.")

    for model in models:
        log(f"\n=== model {model}")
        if not a.no_warmup:
            warmup(a.ollama, model, log)
        for arm in arms:
            path = out / f"{model_tag(model)}__{arm}.csv"
            if path.exists() and not a.force:
                log(f"  {arm}: {path.name} exists, skipping (use --force to redo)")
                continue
            log(f"  {arm}: {ARMS[arm]['label']}")
            defaults = {**ARMS[arm]["defaults"], "model": model, "temperature": a.temperature, "num_ctx": a.num_ctx}
            rows = []
            groups = [[r] for r in range(1, a.reps + 1)] if ARMS[arm].get("memory") else [list(range(1, a.reps + 1))]
            for reps in groups:
                if ARMS[arm].get("memory"):
                    reset_memory(a.reset_memory_cmd, log)
                items = build_items(tasks, reps, a.seed, arm, model)
                bid, csv_rows = run_batch(base, f"{a.run_name}-{model_tag(model)}-{arm}-r{reps[0]}", items, defaults,
                                          a.poll, a.max_hours, log)
                seed_to_rep = {a.seed + r - 1: r for r in reps}
                by_idx = {int(r["idx"]): it for r, it in zip(sorted(csv_rows, key=lambda x: int(x["idx"])), items)}
                for r in csv_rows:
                    it = by_idx[int(r["idx"])]
                    r["rep"], r["seed"], r["batch_id"] = seed_to_rep[it["seed"]], it["seed"], bid
                rows += evaluate_rows(base, csv_rows, H.load_tasks(), arm=arm, log=lambda *_: None)
            write_csv(path, rows)
            ok = sum(1 for r in rows if r.get("hidden_pass") in (True, "True"))
            log(f"  {arm}: wrote {path.name}: hidden-test pass {ok}/{len(rows)}")
    log(f"\nDone. Results in {out}\nNext: python evaluation/make_paper_assets.py --results {out}")


if __name__ == "__main__":
    main()
