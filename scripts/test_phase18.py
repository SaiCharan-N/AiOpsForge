#!/usr/bin/env python3
"""Phase 18 checkpoint: reproducibility options, failure-recovery switch, IFS recording.

REAL here: run_config, app.llm.generate (payload sent to Ollama), app.ifs, the compiled
LangGraph via run_pipeline, node instrumentation, metrics flushing, every SQL query in
app.metrics_report (on SQLite), app.batch / app.experiments plan building.
MOCKED: the Ollama HTTP API (canned responses), the MCP session (dispatches to the real
mcp_server tool functions), Postgres (SQLite stands in), Redis, run logging.

Covers:
  1. new per-run overrides (failure_recovery, temperature, seed, num_ctx): validation + defaults
  2. Ollama `options` are sent only when configured; per-run values beat env defaults
  3. IFS: detail, strict matcher, unscorable requests, record_ifs persistence
  4. full pipeline, failure_recovery ON vs OFF: what the Developer's retry prompt contains,
     what is still measured, IFS rows recorded for planner + developer
  5. reports: list_projects columns, ifs_records, batch CSV columns
  6. experiment plans accept the new factors
"""
import asyncio
import json
import os
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))
TMP = Path(tempfile.mkdtemp(prefix="aiops_p18_"))
os.environ["WORKSPACE_ROOT"] = str(TMP)
sys.path.insert(0, str(ROOT / "mcp_server"))

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app import batch, db, experiments, ifs, llm, mcp_client, memory, metrics_report, run_config  # noqa: E402
from app.config import settings  # noqa: E402
from app.pipeline import metrics  # noqa: E402
from app.schemas import ExperimentRequest, RunSettings  # noqa: E402

settings.workspace_root = str(TMP)


def ok(msg):
    print(f"[ok] {msg}")


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    return False


class FakeResp:
    def __init__(self, body, status=200):
        self._b, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._b


def ollama_body(response, tin=100, tout=20):
    return {"response": response, "prompt_eval_count": tin, "eval_count": tout,
            "total_duration": int(5e6), "load_duration": 0, "prompt_eval_duration": int(1e6),
            "eval_duration": int(4e6)}


def make_engine():
    eng = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    with eng.begin() as c:
        c.execute(text("CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT, request TEXT, status TEXT, "
                       "created_at TEXT)"))
        c.execute(text("CREATE TABLE metrics_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, "
                       "kind TEXT, agent TEXT, node TEXT, task_order INT, attempt INT, name TEXT, "
                       "tokens_in INT DEFAULT 0, tokens_out INT DEFAULT 0, duration_ms INT DEFAULT 0, "
                       "t_start_ms BIGINT, ok BOOLEAN DEFAULT 1, meta TEXT, created_at TEXT)"))
    return eng


# ------------------------------------------------------------ 1. overrides ---

def check_overrides():
    v = run_config.validate({"failure_recovery": False, "temperature": 0, "seed": 42, "num_ctx": 8192})
    assert v == {"failure_recovery": False, "temperature": 0.0, "seed": 42, "num_ctx": 8192}, v
    for bad in ({"temperature": 2.5}, {"temperature": True}, {"seed": -1}, {"seed": 1.5},
                {"num_ctx": 100}, {"num_ctx": "big"}, {"failure_recovery": "no"}, {"topk": 3}):
        assert raises(ValueError, run_config.validate, bad), bad
    ok("new overrides validate (types and ranges); a typo'd key is rejected instead of silently ignored")

    assert run_config.failure_recovery_enabled() is True
    tok = run_config.push({"failure_recovery": False})
    try:
        assert run_config.failure_recovery_enabled() is False
    finally:
        run_config.pop(tok)
    assert run_config.failure_recovery_enabled() is True
    ok("failure_recovery defaults to ON and an override does not leak after the run")

    rs = RunSettings(failure_recovery=False, temperature=0.2, seed=7, num_ctx=4096)
    assert rs.failure_recovery is False and rs.seed == 7
    assert raises(Exception, RunSettings, temperature=3)
    r = batch.resolve_items(["x"], RunSettings(failure_recovery=False, seed=1))
    assert r[0]["overrides"] == {"failure_recovery": False, "seed": 1}, r
    ok("RunSettings / batch defaults carry the new fields into each run's overrides")


# --------------------------------------------------------- 2. ollama options --

def check_llm_options():
    sent = []

    def post(url, json=None, timeout=None):
        sent.append(json)
        return FakeResp(ollama_body("hi"))

    saved = (settings.ollama_temperature, settings.ollama_seed, settings.ollama_num_ctx)
    try:
        settings.ollama_temperature = settings.ollama_seed = settings.ollama_num_ctx = None
        with patch("requests.post", post):
            llm.generate("p")
        assert "options" not in sent[-1], sent[-1]
        ok("nothing configured -> no `options` sent (Ollama's own defaults, exactly as before)")

        settings.ollama_temperature, settings.ollama_seed, settings.ollama_num_ctx = 0.5, 11, 4096
        with patch("requests.post", post):
            llm.generate("p")
        assert sent[-1]["options"] == {"temperature": 0.5, "seed": 11, "num_ctx": 4096}, sent[-1]
        tok = run_config.push({"temperature": 0, "num_ctx": 16384})
        try:
            with patch("requests.post", post):
                llm.generate("p")
        finally:
            run_config.pop(tok)
        assert sent[-1]["options"] == {"temperature": 0.0, "seed": 11, "num_ctx": 16384}, sent[-1]
        ok("env defaults are sent; a per-run override wins for the keys it sets and inherits the rest")
    finally:
        settings.ollama_temperature, settings.ollama_seed, settings.ollama_num_ctx = saved


# ----------------------------------------------------------------- 3. IFS ----

def check_ifs():
    req = "Write `is_valid_isbn13(code)` that accepts hyphens and uses the ISBN-13 checksum. Include a test."
    d = ifs.compute_ifs_detail(req, "def is_valid_isbn13(code):\n    # ISBN-13 checksum\n    return True\n")
    assert d["entities"] >= 3 and d["score"] == 1.0 and d["strict_score"] == 1.0 and d["missing"] == [], d
    d = ifs.compute_ifs_detail(req, "def check(code):\n    return True\n")
    assert d["score"] < 0.5 and "is_valid_isbn13" in d["missing"] and d["strict_score"] <= d["score"], d
    ok("detail: entity counts, substring score, strict score and the list of lost entities")

    # strict matcher is stricter than substring: an entity buried inside a longer identifier
    d = ifs.compute_ifs_detail("Write a function named parse_date.", "def parse_date_range(s):\n    pass\n")
    assert d["score"] == 1.0 and d["strict_score"] < 1.0, d
    ok("a longer identifier containing the entity counts for the substring matcher but not the strict one")
    d = ifs.compute_ifs_detail("Write a function named parse_date.", "def parse_dates(s):\n    pass\n")
    assert d["strict_score"] == 1.0, d  # a one-letter plural is forgiven by the difflib ratio
    ok("a near-duplicate token (plural / typo) is forgiven by the strict matcher's difflib ratio")

    assert ifs.compute_ifs_detail("a function that reverses a string, with a test", "anything") is None
    assert ifs.record_ifs("developer", "a function that reverses a string", "x") is None
    ok("a request with no technical entities is unscorable (None), never a misleading 0.0")

    metrics._buffers.clear()
    st = {"project_id": "pI", "current_task": {"order": 2}, "communication_mode": "message_passing"}
    with metrics.scope("pI", "developer", 2, 1):
        det = ifs.record_ifs("developer", req, "def is_valid_isbn13(code): ...", state=st)
    ev = [e for e in metrics._buffers["pI"] if e["kind"] == "ifs"]
    assert len(ev) == 1 and ev[0]["name"] == "developer" and ev[0]["task_order"] == 2 and ev[0]["ok"], ev
    assert ev[0]["meta"]["score"] == det["score"] and ev[0]["meta"]["communication_mode"] == "message_passing"
    assert st["ifs_score"] == det["score"] and st["ifs_strict"] == det["strict_score"]
    ok("record_ifs stores a kind='ifs' row (detail + mode + task) and keeps state['ifs_score'/'ifs_strict'] in sync")
    metrics._buffers.clear()


# ------------------------------------------------------ 4. full pipeline -----

def run_pipeline_once(recovery: bool):
    """One run: planner -> developer (syntax error) -> failure analysis -> developer (fixed) -> pass."""
    from app.agents.developer import DEVELOPER_SYSTEM_PROMPT
    from app.agents.planner import PLANNER_SYSTEM_PROMPT
    from app.pipeline import graph as graph_mod, nodes
    from tools import ALL_TOOLS

    tool_fns = {f.__name__: f for f in ALL_TOOLS}
    eng = make_engine()
    pid = str(uuid.uuid4())
    dev_prompts, dev_calls, analyzer_calls = [], {"n": 0}, []
    GOOD = ("def reverse_string(s):\n    return s[::-1]\n\n\ndef test_reverse():\n"
            "    assert reverse_string('abc') == 'cba'\n")
    BAD = "def reverse_string(s:\n    return s[::-1]\n"

    def post(url, json=None, timeout=None):
        if url.endswith("/api/embeddings"):
            return FakeResp({"embedding": [0.2] * 768})
        system = json.get("system")
        if system == PLANNER_SYSTEM_PROMPT:
            return FakeResp(ollama_body('{"tasks": ["Write reverse_string with a pytest test"]}', 300, 30))
        if system == DEVELOPER_SYSTEM_PROMPT:
            dev_prompts.append(json["prompt"])
            dev_calls["n"] += 1
            return FakeResp(ollama_body(f"```python\n{BAD if dev_calls['n'] == 1 else GOOD}```", 400, 60))
        analyzer_calls.append(1)
        return FakeResp(ollama_body('{"hint": "LLM refined hint", "root_cause": "y"}', 50, 10))

    class Sess:
        async def call_tool(self, name, args):
            out = await asyncio.to_thread(tool_fns[name], **args)
            return SimpleNamespace(content=[SimpleNamespace(text=out)], isError=False)

    @asynccontextmanager
    async def fake_session():
        yield Sess()

    similar_calls = []

    def find_similar(*a, **k):
        similar_calls.append(1)
        return []

    async def scenario():
        with patch.object(db, "engine", eng), patch("requests.post", post), \
             patch.object(mcp_client, "mcp_session", fake_session), \
             patch.object(nodes, "db", MagicMock()), patch.object(nodes, "stm", MagicMock()), \
             patch.object(graph_mod, "stm", MagicMock()), \
             patch.object(memory, "retrieve_similar", lambda *a, **k: []), \
             patch.object(memory, "save_memory", lambda *a, **k: "id"), \
             patch.object(memory, "find_similar_failures", find_similar), \
             patch.object(memory, "save_failure_pattern", lambda *a, **k: "id"):
            with eng.begin() as c:
                c.execute(text("INSERT INTO projects VALUES (:i, 'rev', 'r', 'done', '2026-01-01T00:00:00')"), {"i": pid})
            result = await graph_mod.run_pipeline(
                pid, "Write a function `reverse_string` with a test", overrides={"failure_recovery": recovery})
            assert result.done and not result.needs_human_review, result.task_history
            return result, metrics_report.project_metrics(pid), metrics_report.list_projects(), \
                metrics_report.ifs_records(pid)

    out = asyncio.run(scenario())
    return out, dev_prompts, (similar_calls, analyzer_calls), eng, pid


def check_failure_recovery_switch():
    saved_mode = settings.failure_analyzer_llm_mode
    settings.failure_analyzer_llm_mode = "always"   # make the optional LLM refinement observable
    try:
        return _check_failure_recovery_switch()
    finally:
        settings.failure_analyzer_llm_mode = saved_mode


def _check_failure_recovery_switch():
    (res_on, pm_on, lp_on, ifs_on), prompts_on, (sim_on, llm_on), eng, pid = run_pipeline_once(True)
    retry_on = prompts_on[1]
    assert "Failure Analyzer diagnosis" in retry_on and "Your previous code" in retry_on, retry_on
    assert sim_on and llm_on, "ON: similar-failure lookup and LLM refinement should run"
    ok("failure_recovery ON : the retry prompt has the diagnosis AND the line-numbered previous code")

    (res_off, pm_off, lp_off, ifs_off), prompts_off, (sim_off, llm_off), eng2, pid2 = run_pipeline_once(False)
    retry_off = prompts_off[1]
    assert "Failure Analyzer diagnosis" not in retry_off and "Your previous code" not in retry_off, retry_off
    assert "Test error:" in retry_off and "previous attempt FAILED" in retry_off, retry_off
    assert not sim_off and not llm_off, "OFF: no similar-failure lookup, no LLM refinement"
    ok("failure_recovery OFF: the retry prompt has only the raw QA error (no diagnosis, no previous code)")

    # both arms are still MEASURED the same way
    for pm in (pm_on, pm_off):
        t = pm["tasks"][0]
        assert t["failures"] == 1 and t["passed"] and t["attempts"] == 2 and t["categories"], t
    ok("both arms record failures / categories / attempts, so repeated-error counts stay comparable")
    assert pm_off["tokens"]["total"] < pm_on["tokens"]["total"]
    ok("OFF spends no tokens on the (unused) LLM refinement or memory lookup: "
       f"{pm_off['tokens']['total']} vs {pm_on['tokens']['total']} tokens")

    # IFS rows: planner + developer (one per attempt)
    for ifs_rep in (ifs_on, ifs_off):
        stages = [r["stage"] for r in ifs_rep["records"]]
        assert stages.count("planner") == 1 and stages.count("developer") == 2, stages
        assert ifs_rep["planner"]["score"] == 1.0 and ifs_rep["developer"]["score"] == 1.0, ifs_rep
        assert ifs_rep["developer"]["by_task"][0]["attempt"] == 2   # the last attempt is the one kept
    ok("IFS recorded for the Planner (1 row) and the Developer (1 row per attempt); report keeps the last attempt")

    row = [r for r in lp_on if r["project_id"] == pid][0]
    assert row["ifs_avg"] == 1.0 and row["ifs_strict_avg"] == 1.0 and row["ifs_planner"] == 1.0, row
    assert pm_on["ifs"]["planner"]["score"] == 1.0
    ok("list_projects / project_metrics expose ifs_avg, ifs_strict_avg and ifs_planner")
    return row


# ------------------------------------------------------------- 5/6. exports ---

def check_exports_and_plans(row):
    for col in ("ifs_strict_avg", "ifs_planner", "failure_recovery", "temperature", "seed", "num_ctx"):
        assert col in batch.CSV_COLUMNS and col in experiments.RUN_CSV_COLUMNS, col
    assert batch.override_values({"failure_recovery": False, "seed": 3}) == {
        "use_memory": None, "max_qa_attempts": None, "failure_recovery": False, "temperature": None,
        "seed": 3, "num_ctx": None}
    assert "ifs_strict_avg" in metrics_report.CSV_COLUMNS
    keys = [m[0] for m in experiments.METRICS]
    assert "ifs_strict_avg" in keys and "ifs_planner" in keys and keys[:4] == [
        "tokens_total", "total_ms", "est_cost_usd", "avg_attempts"]
    ok("CSV exports and the experiment metric list include the new columns (old column order preserved)")

    plan = experiments.build_plan(ExperimentRequest(
        design="ab", factor="failure_recovery", levels=[False, True], requests=["a `foo_bar` function"], repetitions=2))
    assert plan["labels"] == ["off", "on"] and plan["baseline"] == "off" and plan["total_runs"] == 4
    assert plan["base"]["use_memory"] is False and plan["isolation"], plan
    plan = experiments.build_plan(ExperimentRequest(
        design="ab", factor="temperature", levels=[0, 0.8], requests=["x"], repetitions=1))
    assert plan["levels"] == [0.0, 0.8], plan
    assert raises(ValueError, experiments.build_plan, ExperimentRequest(
        design="ab", factor="num_ctx", levels=[100, 8192], requests=["x"]))
    env = experiments.environment()
    assert {"ollama_temperature", "ollama_seed", "ollama_num_ctx", "failure_analyzer_llm_mode"} <= set(env)
    ok("experiment plans accept failure_recovery / temperature / num_ctx as factors; environment records the sampling options")


if __name__ == "__main__":
    check_overrides()
    check_llm_options()
    check_ifs()
    row = check_failure_recovery_switch()
    check_exports_and_plans(row)
    print("\nAll Phase 18 checks passed.")
