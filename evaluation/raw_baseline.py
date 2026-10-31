#!/usr/bin/env python3
"""A0 — the raw-model baseline: ONE Ollama call per task, no pipeline.

Why: every claim "our pipeline is better" needs the question "better than what?". A0 is the
floor: the same model, the same request, the same sampling settings, a single attempt, no
Planner / QA / retries / memory / blackboard. If AIOpsForge cannot beat this, the pipeline
adds nothing for that model; if it does, A0 is the bar it clears.

    python evaluation/raw_baseline.py --models qwen2.5-coder:3b,qwen2.5-coder:7b --reps 3 --tasks core
    python evaluation/raw_baseline.py --models qwen2.5-coder:7b --tasks T01,T02 --reps 1

Uses the SAME system prompt the pipeline's Developer uses, the same seeds per (task, rep) as
run_ladder.py, and scores with the same hidden tests + IFS. Talks to Ollama directly (default
http://localhost:11434) — the AIOpsForge backend does not need to be running.
Output: results/<run>/<model>__A0.csv (same columns as the pipeline arms) + the generated files
under results/<run>/raw_projects/ so every number can be audited.
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hidden_tests as H  # noqa: E402
from common import model_tag, post_json, write_csv  # noqa: E402

# Identical to backend/app/agents/developer.py DEVELOPER_SYSTEM_PROMPT (kept in sync by scripts/test_evaluation.py).
SYSTEM_PROMPT = """You are the Developer Agent in an automated software development pipeline. Given one coding task, write a single, complete Python file that implements it.

Rules:
- Respond with ONLY a python code block (```python ... ```) and nothing else.
- The file must be runnable/importable on its own.
- If the task is about writing tests, use pytest-style `def test_...():` functions and `assert` statements — do not use unittest.TestCase.
- Keep it simple and correct over clever."""


def extract_code(raw: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
    return fenced.group(1).strip() if fenced else raw.strip()


def select_tasks(tasks: dict, spec: str) -> list[dict]:
    if spec == "all":
        return list(tasks.values())
    if spec == "core":
        return [t for t in tasks.values() if t["core"]]
    ids = [s.strip() for s in spec.split(",") if s.strip()]
    missing = [i for i in ids if i not in tasks]
    if missing:
        raise SystemExit(f"unknown task id(s): {missing}; known: {sorted(tasks)}")
    return [tasks[i] for i in ids]


def options(temperature, seed, num_ctx) -> dict:
    o = {}
    if temperature is not None:
        o["temperature"] = temperature
    if seed is not None:
        o["seed"] = seed
    if num_ctx is not None:
        o["num_ctx"] = num_ctx
    return o


def run_one(ollama: str, model: str, task: dict, seed: int, temperature, num_ctx, out_dir: Path,
            rate_in: float, rate_out: float, keep_alive: str = "30m") -> dict:
    prompt = f"Task:\n{task['request']}\n\nWrite the code now."
    payload = {"model": model, "prompt": prompt, "system": SYSTEM_PROMPT, "stream": False, "keep_alive": keep_alive}
    opts = options(temperature, seed, num_ctx)
    if opts:
        payload["options"] = opts
    t0 = time.time()
    try:
        body = post_json(f"{ollama}/api/generate", payload, timeout=900)
        err = ""
    except RuntimeError as exc:
        body, err = {}, str(exc)[:300]
    wall_ms = int((time.time() - t0) * 1000)
    code = extract_code(body.get("response", ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "solution.py").write_text(code + "\n", encoding="utf-8")
    tin, tout = int(body.get("prompt_eval_count") or 0), int(body.get("eval_count") or 0)
    row = {"model": model, "model_tag": model_tag(model), "arm": "A0", "task_id": task["id"], "tier": task["tier"],
           "seed": seed, "status": "error" if err else "ok", "error": err, "tasks": 1, "attempts": 1,
           "avg_attempts": 1, "llm_calls": 1, "tokens_in": tin, "tokens_out": tout, "tokens_total": tin + tout,
           "total_ms": wall_ms, "est_cost_usd": round(tin / 1000 * rate_in + tout / 1000 * rate_out, 6),
           "temperature": temperature, "num_ctx": num_ctx, "communication_mode": "none"}
    if not err:
        row.update(H.run_hidden(out_dir, task))
        row.update(H.ifs_of_project(task["request"], out_dir))
        st = H.run_self_tests(out_dir)
        row.update(st)
        row["self_pass"] = bool(st["self_pass"])
        row["status"] = "passed" if row["self_pass"] else "failed"
        row["tasks_passed"] = 1 if row["self_pass"] else 0
        row["false_pass"] = bool(row["self_pass"] and not row["hidden_pass"])
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--models", required=True, help="comma-separated, e.g. qwen2.5-coder:3b,qwen2.5-coder:7b")
    ap.add_argument("--tasks", default="core", help="core | all | T01,T02,...")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1, help="rep k uses seed (seed + k - 1), like run_ladder.py")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--run-name", default="run1", help="results/<run-name>/ output folder")
    ap.add_argument("--out-root", default=str(H.ROOT / "results"))
    ap.add_argument("--rate-in", type=float, default=0.0005, help="est. $ per 1K input tokens (match TOKEN_RATE_IN_PER_1K)")
    ap.add_argument("--rate-out", type=float, default=0.0015, help="est. $ per 1K output tokens")
    a = ap.parse_args()

    tasks = select_tasks(H.load_tasks(), a.tasks)
    out_root = Path(a.out_root) / a.run_name
    for model in [m.strip() for m in a.models.split(",") if m.strip()]:
        rows = []
        print(f"\n== A0 raw model: {model} ({len(tasks)} tasks x {a.reps} reps)")
        for rep in range(1, a.reps + 1):
            for t in tasks:
                d = out_root / "raw_projects" / model_tag(model) / f"{t['id']}_rep{rep}"
                r = run_one(a.ollama, model, t, a.seed + rep - 1, a.temperature, a.num_ctx, d, a.rate_in, a.rate_out)
                r["rep"] = rep
                rows.append(r)
                print(f"  {t['id']} rep{rep}: hidden {r.get('hidden_passed', '-')}/{r.get('hidden_total', '-')}  "
                      f"self={r.get('self_pass')}  {r['total_ms'] / 1000:.1f}s  {r['error'][:60]}")
        path = out_root / f"{model_tag(model)}__A0.csv"
        write_csv(path, rows)
        ok = sum(1 for r in rows if r.get("hidden_pass"))
        print(f"wrote {path}: hidden-test pass {ok}/{len(rows)}")


if __name__ == "__main__":
    main()
