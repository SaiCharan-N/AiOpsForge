"""Small shared helpers for the evaluation scripts (standard library only)."""
import csv
import io
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

# One row per run, the same columns for every arm (pipeline arms and the raw-model arm).
RUN_COLUMNS = [
    "arm", "model_tag", "model", "task_id", "tier", "rep", "seed", "status", "run_ok",
    "hidden_pass", "hidden_passed", "hidden_total", "hidden_fraction", "hidden_status", "hidden_errors",
    "self_pass", "self_status", "false_pass",
    "tasks", "tasks_passed", "attempts", "avg_attempts", "escalations", "repeated_errors",
    "ifs_final", "ifs_final_strict", "ifs_avg", "ifs_strict_avg", "ifs_planner",
    "tokens_in", "tokens_out", "tokens_total", "est_cost_usd", "llm_calls", "total_ms",
    "communication_mode", "use_memory", "max_qa_attempts", "failure_recovery", "temperature", "num_ctx",
    "project_id", "batch_id", "error",
]


def http(method: str, url: str, body: dict | None = None, timeout: int = 60) -> bytes:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{method} {url} -> HTTP {exc.code}: {exc.read()[:300].decode('utf-8', 'replace')}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason} (is the backend running?)") from exc


def get_json(url: str, timeout: int = 60):
    return json.loads(http("GET", url, timeout=timeout))


def post_json(url: str, body: dict, timeout: int = 60):
    return json.loads(http("POST", url, body, timeout=timeout))


def parse_csv(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def write_csv(path: Path | str, rows: list[dict], columns: list[str] | None = None) -> None:
    columns = columns or RUN_COLUMNS
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in columns})


def model_tag(model: str) -> str:
    """'qwen2.5-coder:7b' -> 'qwen2.5-coder_7b' (safe in file names)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model or "unknown")


def model_size_b(model: str) -> float | None:
    """'qwen2.5-coder:7b' -> 7.0 ; '...:3b' -> 3.0 ; unknown -> None."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", (model or "").lower())
    return float(m.group(1)) if m else None


def to_bool(v):
    if isinstance(v, bool) or v is None:
        return v
    s = str(v).strip().lower()
    return True if s in ("true", "1", "yes") else False if s in ("false", "0", "no") else None


def to_num(v):
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None
