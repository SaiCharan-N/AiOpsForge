#!/usr/bin/env python3
"""Week 8 / Day 40 checkpoint: kick off two unrelated projects concurrently
and confirm their task lists never cross-contaminate.

This exercises the real running stack end-to-end; the strict guarantee
itself (namespaced Redis keys, verified with a live concurrency test) is
covered in backend/app/short_term_memory.py's own test — see the Phase 3
README for how that was verified before you got this.

Usage:
    python3 scripts/test_isolation_live.py
"""
import sys
import time

import requests

BACKEND_URL = "http://localhost:8000"


def main() -> int:
    r1 = requests.post(f"{BACKEND_URL}/request/async", json={
        "request": "a function that adds two numbers, with a test",
        "project_name": "isolation-check-a",
    }, timeout=30)
    r2 = requests.post(f"{BACKEND_URL}/request/async", json={
        "request": "a function that converts Celsius to Fahrenheit, with a test",
        "project_name": "isolation-check-b",
    }, timeout=30)
    pid_a, pid_b = r1.json()["project_id"], r2.json()["project_id"]
    print(f"Started concurrently: A={pid_a}  B={pid_b}")

    for _ in range(120):
        sa = requests.get(f"{BACKEND_URL}/projects/{pid_a}/status", timeout=10).json()
        sb = requests.get(f"{BACKEND_URL}/projects/{pid_b}/status", timeout=10).json()
        if sa["status"] in ("done", "needs_human_review", "error") and \
           sb["status"] in ("done", "needs_human_review", "error"):
            break
        time.sleep(2)
    else:
        print("Timed out waiting for both projects to finish.")
        return 1

    descs_a = {t["description"] for t in sa["tasks"]}
    descs_b = {t["description"] for t in sb["tasks"]}
    print(f"Project A tasks: {descs_a}")
    print(f"Project B tasks: {descs_b}")

    if descs_a & descs_b:
        print("CHECKPOINT FAILED: task descriptions overlapped between unrelated projects.")
        return 1
    print("CHECKPOINT PASSED: two concurrent projects, zero cross-contamination.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
