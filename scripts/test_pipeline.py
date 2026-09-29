#!/usr/bin/env python3
"""Phase 2 checkpoint: sends a real request through the full running stack
(Planner -> Developer -> QA, real Ollama, real MCP server) and prints the
result. This is NOT a unit test with mocks — it's meant to be run against
`docker compose up` to prove the whole pipeline works together.

Usage:
    python3 scripts/test_pipeline.py "a function that reverses a string, with a test"
"""
import json
import sys

import requests

BACKEND_URL = "http://localhost:8000"


def main() -> int:
    request_text = " ".join(sys.argv[1:]) or "a function that adds two numbers, with a test"

    print(f"Submitting request: {request_text!r}")
    print("(this can take a couple of minutes on a local model — it's running "
          "Planner -> Developer -> QA for real, including retries on failure)\n")

    resp = requests.post(f"{BACKEND_URL}/request", json={"request": request_text}, timeout=600)
    resp.raise_for_status()
    result = resp.json()

    print(f"project_id: {result['project_id']}")
    print(f"done: {result['done']}  |  needs_human_review: {result['needs_human_review']}\n")
    print("Task outcomes:")
    for entry in result["task_history"]:
        status = "PASS" if entry["passed"] else ("ESCALATED" if entry["escalated"] else "FAIL")
        print(f"  [{status}] (attempts={entry['attempts']}) {entry['task']} -> {entry['filename']}")

    return 0 if result["done"] else 1


if __name__ == "__main__":
    sys.exit(main())
