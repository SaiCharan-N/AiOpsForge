#!/usr/bin/env python3
"""Experiment 3 (Week 11, Day 53): does the system get better at a CLASS of
bug over time, not just an exact repeat?

Submits several DIFFERENT (but related-pattern) tasks in sequence — all
"off-by-one in a loop boundary"-shaped problems, phrased differently each
time — and tracks whether later ones in the sequence need fewer attempts
than earlier ones, as memory accumulates related fixes.

Usage:
    python3 experiments/exp3_repeat_bug_trend.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lib import submit_sync, save_results

# Same underlying bug PATTERN (off-by-one loop boundary), phrased differently
# each time, submitted as separate, unrelated-looking projects.
SIMILAR_PATTERN_TASKS = [
    "a function that sums all elements in a list using a manually indexed loop, with a test",
    "a function that finds the last element of a list using index arithmetic, with a test",
    "a function that reverses a list in place using a manually indexed loop, with a test",
    "a function that copies every element of an array into a new array using an indexed loop, with a test",
    "a function that computes a running total over a list using index-based iteration, with a test",
]


def main():
    print(f"Running {len(SIMILAR_PATTERN_TASKS)} related-pattern tasks in sequence...\n")
    sequence = []
    for i, req in enumerate(SIMILAR_PATTERN_TASKS, start=1):
        result, elapsed = submit_sync(req, project_name=f"exp3-seq-{i}")
        attempts = sum(t["attempts"] for t in result["task_history"])
        memory_used = any(t["memory_used"] for t in result["task_history"])
        entry = {
            "position_in_sequence": i,
            "request": req,
            "time_s": round(elapsed, 2),
            "attempts": attempts,
            "memory_used": memory_used,
        }
        sequence.append(entry)
        print(f"  [{i}/{len(SIMILAR_PATTERN_TASKS)}] attempts={attempts} "
              f"time={entry['time_s']}s memory_used={memory_used}")

    path = save_results("exp3_repeat_bug_trend", {"sequence": sequence})
    print(f"\nSaved raw results to {path}")


if __name__ == "__main__":
    main()
