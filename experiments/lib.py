"""Shared helpers for the Phase 4 experiment scripts (Week 11).

Every experiment writes its raw results to experiments/results/<name>.json
in the same shape, so analyze_results.py can consume all three without
per-experiment special-casing.
"""
import json
import time
from pathlib import Path

import requests

BACKEND_URL = "http://localhost:8000"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def submit_sync(request_text: str, project_name: str | None = None) -> tuple[dict, float]:
    """Submits a request via the synchronous /request endpoint and returns
    (response_json, wall_clock_seconds).
    """
    t0 = time.time()
    resp = requests.post(
        f"{BACKEND_URL}/request",
        json={"request": request_text, "project_name": project_name},
        timeout=900,
    )
    resp.raise_for_status()
    return resp.json(), time.time() - t0


def save_results(name: str, data: dict) -> Path:
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def load_results(name: str) -> dict:
    path = RESULTS_DIR / f"{name}.json"
    return json.loads(path.read_text())


def pass_at_1(task_history: list[dict]) -> float:
    """Fraction of tasks that passed on the FIRST attempt (attempts == 1)."""
    if not task_history:
        return 0.0
    first_try = sum(1 for t in task_history if t["passed"] and t["attempts"] == 1)
    return first_try / len(task_history)
