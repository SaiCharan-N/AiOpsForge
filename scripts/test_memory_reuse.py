#!/usr/bin/env python3
"""Week 7 / Day 35 checkpoint: run the same type of task twice and compare.

The first run has to solve it from scratch (memory_used: false). If the
second run's task shows memory_used: true, the Developer Agent found and
reused the first run's solution — this is the long-term memory payoff.

Usage:
    python3 scripts/test_memory_reuse.py
"""
import sys

import requests

BACKEND_URL = "http://localhost:8000"
TASK_DESCRIPTION = "a function that checks if a string is a palindrome, with a test"


def submit(request_text: str) -> dict:
    resp = requests.post(f"{BACKEND_URL}/request", json={"request": request_text}, timeout=600)
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    print(f"Run 1: {TASK_DESCRIPTION!r}")
    result1 = submit(TASK_DESCRIPTION)
    memory_used_1 = any(t["memory_used"] for t in result1["task_history"])
    print(f"  memory_used: {memory_used_1}  (expected: False — nothing to reuse yet)\n")

    print(f"Run 2: same task again")
    result2 = submit(TASK_DESCRIPTION)
    memory_used_2 = any(t["memory_used"] for t in result2["task_history"])
    print(f"  memory_used: {memory_used_2}  (expected: True — reusing run 1's fix)\n")

    if not memory_used_1 and memory_used_2:
        print("CHECKPOINT PASSED: long-term memory reuse confirmed.")
        return 0
    print("CHECKPOINT NOT MET — check that the embedding model is pulled "
          "(ollama pull nomic-embed-text) and MEMORY_SIMILARITY_THRESHOLD isn't too strict.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
