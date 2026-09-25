#!/usr/bin/env python3
"""Experiment 1 (Week 11, Day 51): does long-term memory actually help?

Submits N different "bug-fix-shaped" tasks, each TWICE (a first-time run,
then an immediate repeat of the same kind of problem). Records wall-clock
time and attempts-to-pass for both runs. If memory is doing its job, the
second run of each pair should be faster and/or need fewer attempts.

Usage:
    python3 experiments/exp1_memory_speedup.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lib import submit_sync, save_results

TASK_PAIRS = [
    "a function that checks if a number is prime, with a test",
    "a function that flattens a nested list, with a test",
    "a function that removes duplicate items from a list while preserving order, with a test",
]


def run_pair(request_text: str) -> dict:
    result_1, time_1 = submit_sync(request_text, project_name=f"exp1-first-{hash(request_text) % 10000}")
    result_2, time_2 = submit_sync(request_text, project_name=f"exp1-repeat-{hash(request_text) % 10000}")

    attempts_1 = sum(t["attempts"] for t in result_1["task_history"])
    attempts_2 = sum(t["attempts"] for t in result_2["task_history"])
    memory_used_2 = any(t["memory_used"] for t in result_2["task_history"])

    return {
        "request": request_text,
        "first_run": {"time_s": round(time_1, 2), "attempts": attempts_1},
        "repeat_run": {"time_s": round(time_2, 2), "attempts": attempts_2, "memory_used": memory_used_2},
    }


def main():
    print(f"Running {len(TASK_PAIRS)} first/repeat pairs (this calls your live stack)...\n")
    pairs = []
    for req in TASK_PAIRS:
        print(f"  {req}")
        pairs.append(run_pair(req))
        print(f"    first: {pairs[-1]['first_run']['time_s']}s, "
              f"{pairs[-1]['first_run']['attempts']} attempt(s)")
        print(f"    repeat: {pairs[-1]['repeat_run']['time_s']}s, "
              f"{pairs[-1]['repeat_run']['attempts']} attempt(s), "
              f"memory_used={pairs[-1]['repeat_run']['memory_used']}\n")

    path = save_results("exp1_memory_speedup", {"pairs": pairs})
    print(f"Saved raw results to {path}")


if __name__ == "__main__":
    main()
