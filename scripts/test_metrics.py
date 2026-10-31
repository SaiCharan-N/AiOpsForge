#!/usr/bin/env python3
"""Metrics & observability checkpoint.

REAL here: app.llm.generate / generate_json / embeddings, the MCP client's
call_tool (a fake MCP *session* dispatches to the real mcp_server tool
functions), the compiled LangGraph via run_pipeline, node instrumentation,
batched flushing, every SQL query in app.metrics_report (executed on
SQLite with the same table shapes), and the FastAPI endpoints.
MOCKED: the Ollama HTTP API (canned responses with realistic token counts
and durations), Redis, Postgres (SQLite stands in), the run-logging calls.

Covers:
  1. token capture from Ollama (real counts, estimate fallback, failures, JSON retries)
  2. timing: embed vs db (no double count), tool calls, network-overhead derivation
  3. node instrumentation, concurrent-project isolation, flush failure safety
  4. a full pipeline run (fail -> analyze -> retry -> pass) producing the expected metrics rows
  5. reports: per-project, list, compare, success/summary, models, regression, CSV
  6. HTTP endpoints
  7. overhead
"""
import asyncio
import csv
import io
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
TMP = Path(tempfile.mkdtemp(prefix="aiops_metrics_"))
os.environ["WORKSPACE_ROOT"] = str(TMP)
sys.path.insert(0, str(ROOT / "mcp_server"))

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app import db, embeddings, llm, mcp_client, memory, metrics_report  # noqa: E402
from app.config import settings  # noqa: E402
from app.pipeline import metrics  # noqa: E402

settings.workspace_root = str(TMP)


def ok(msg):
    print(f"[ok] {msg}")


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


def rows_of(pid, kind=None):
    with db.engine.connect() as c:
        q = "SELECT * FROM metrics_logs WHERE project_id = :p" + (" AND kind = :k" if kind else "") + " ORDER BY id"
        return c.execute(text(q), {"p": pid, "k": kind}).fetchall()


class FakeResp:
    def __init__(self, body, status=200):
        self._b, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._b


def ollama_body(response, tin, tout, server_ms=1000, load_ms=0, eval_ms=800):
    return {"response": response, "prompt_eval_count": tin, "eval_count": tout,
            "total_duration": int(server_ms * 1e6), "load_duration": int(load_ms * 1e6),
            "prompt_eval_duration": int(100 * 1e6), "eval_duration": int(eval_ms * 1e6)}


# ------------------------------------------------------------ 1. tokens -----

def check_token_capture():
    metrics._buffers.clear()
    calls = []

    def post(url, json=None, timeout=None):
        calls.append(url)
        time.sleep(0.05)
        return FakeResp(ollama_body("hello", 120, 40, server_ms=10, eval_ms=8))

    with patch("requests.post", post), metrics.scope("p1", "developer", 1, 1):
        assert llm.generate("say hi", system="sys") == "hello"
    ev = metrics._buffers["p1"][0]
    assert (ev["kind"], ev["tokens_in"], ev["tokens_out"], ev["agent"], ev["task_order"], ev["attempt"]) == \
           ("llm", 120, 40, "developer", 1, 1), ev
    m = ev["meta"]
    assert not m["estimated"] and m["ollama_ms"] == 10.0 and 30 < m["network_ms"] < 400, m
    assert m["tok_per_s"] == 5000.0  # 40 tokens / 8 ms
    ok("every generate() logs Ollama's real prompt/eval token counts + who made the call")
    ok(f"network overhead derived as wall - server time ({m['network_ms']:.0f} ms), tok/s from eval_duration")

    metrics._buffers.clear()
    with patch("requests.post", lambda *a, **k: FakeResp({"response": "x" * 400})), metrics.scope("p1", "planner"):
        llm.generate("p" * 80)
    ev = metrics._buffers["p1"][0]
    assert ev["meta"]["estimated"] and ev["tokens_in"] == 20 and ev["tokens_out"] == 100, ev
    ok("responses without counts fall back to a chars/4 estimate, flagged estimated=True")

    metrics._buffers.clear()

    def boom(*a, **k):
        raise ConnectionError("ollama down")
    with patch("requests.post", boom), metrics.scope("p1", "planner"):
        try:
            llm.generate("x")
            raise AssertionError("should have raised")
        except ConnectionError:
            pass
    ev = metrics._buffers["p1"][0]
    assert ev["kind"] == "llm" and ev["ok"] is False and ev["tokens_out"] == 0
    ok("a failed LLM call is recorded (ok=False) and the exception still propagates")

    metrics._buffers.clear()
    seq = iter(["not json", "still not json", '{"a": 1}'])
    with patch("requests.post", lambda *a, **k: FakeResp(ollama_body(next(seq), 10, 5))), \
         metrics.scope("p1", "planner"):
        assert llm.generate_json("q") == {"a": 1}
    evs = metrics._buffers["p1"]
    assert len(evs) == 3 and sum(e["tokens_out"] for e in evs) == 15
    ok("generate_json's internal retries are each counted (3 calls, 15 output tokens)")

    with patch("requests.post", lambda *a, **k: FakeResp(ollama_body("x", 1, 1))):
        llm.generate("no scope")  # outside any scope
    assert len(metrics._buffers["p1"]) == 3
    ok("calls made outside a scope (e.g. /demo endpoints) are ignored, not errors")


# ------------------------------------------------------------ 2. timing -----

def check_timing():
    metrics._buffers.clear()

    def post(url, json=None, timeout=None):
        time.sleep(0.04)
        return FakeResp({"embedding": [0.1] * 768})

    @metrics.timed_db("memory.fake")
    def fake_memory_op():
        time.sleep(0.02)          # real DB work
        embeddings.embed("hello world")  # +40 ms embedding inside the DB function

    with patch("requests.post", post), metrics.scope("p2", "developer") as s:
        fake_memory_op()
    by = {e["kind"]: e for e in metrics._buffers["p2"]}
    assert 35 <= by["embed"]["duration_ms"] <= 200, by["embed"]
    assert by["db"]["duration_ms"] < by["embed"]["duration_ms"], (by["db"], by["embed"])
    assert 10 <= by["db"]["duration_ms"] <= 60
    assert abs(s.acc["db"] + s.acc["embed"] - 60) < 40
    ok(f"embedding time is subtracted from the enclosing DB call (db={by['db']['duration_ms']}ms, embed={by['embed']['duration_ms']}ms, no double count)")

    metrics._buffers.clear()

    class Sess:
        async def call_tool(self, name, args):
            await asyncio.sleep(0.03)
            if name == "bad":
                return SimpleNamespace(content=[SimpleNamespace(text="nope")], isError=True)
            return SimpleNamespace(content=[SimpleNamespace(text="fine")], isError=False)

    @asynccontextmanager
    async def fake_session():
        yield Sess()

    async def run():
        with patch.object(mcp_client, "mcp_session", fake_session), metrics.scope("p2", "qa"):
            assert await mcp_client.call_tool("run_tests", {}) == "fine"
            try:
                await mcp_client.call_tool("bad", {})
            except RuntimeError:
                pass
    asyncio.run(run())
    evs = metrics._buffers["p2"]
    assert [e["name"] for e in evs] == ["run_tests", "bad"] and [e["ok"] for e in evs] == [True, False]
    assert all(e["kind"] == "tool" and e["duration_ms"] >= 28 for e in evs)
    ok("MCP tool calls timed by tool name; a tool error is recorded as ok=False")


# ----------------------------------------------------- 3. nodes / flush -----

def check_nodes_and_flush():
    eng = make_engine()
    metrics._buffers.clear()

    def keyed_post(url, json=None, timeout=None):
        # one global mock for both concurrent nodes; token count keyed by prompt
        time.sleep(0.02)
        return FakeResp(ollama_body("x", {"a": 100, "b": 900}[json["prompt"]], 7))

    async def fake_node(state):
        # an LLM call inside a worker thread (as the real agents do) ...
        await asyncio.to_thread(llm.generate, state["prompt"])
        await asyncio.sleep(0.03)  # ... and time spent in nothing measured ("other")
        return state

    async def scenario():
        with patch.object(db, "engine", eng), patch("requests.post", keyed_post):
            node = metrics.instrument_node("developer", fake_node)
            # two projects at once: the contextvar scope must keep them apart
            await asyncio.gather(
                node({"project_id": "A", "current_task": {"order": 1}, "attempt_count": 0, "prompt": "a", "tin": 100}),
                node({"project_id": "B", "current_task": {"order": 3}, "attempt_count": 1, "prompt": "b", "tin": 900}),
            )
            a, b = rows_of("A"), rows_of("B")
            assert [r.kind for r in a] == ["llm", "node"] and [r.kind for r in b] == ["llm", "node"]
            assert a[0].tokens_in == 100 and b[0].tokens_in == 900, "token attribution leaked between projects"
            assert (a[0].task_order, a[0].attempt) == (1, 1) and (b[0].task_order, b[0].attempt) == (3, 2)
            n = a[1]
            import json
            m = json.loads(n.meta)
            assert n.tokens_in == 100 and n.tokens_out == 7 and m["llm_calls"] == 1
            assert m["llm_ms"] >= 18 and m["other_ms"] >= 25 and n.duration_ms >= m["llm_ms"] + m["other_ms"] - 5
            ok("node rows carry LLM subtotal + 'other' remainder; two concurrent projects keep separate scopes/tokens")

            # exceptions: node row is still written, ok=False, exception propagates
            async def bad(state):
                raise ValueError("kaput")
            try:
                await metrics.instrument_node("qa", bad)({"project_id": "C", "attempt_count": 0})
                raise AssertionError
            except ValueError:
                pass
            assert [(r.kind, r.ok) for r in rows_of("C")] == [("node", 0)]
            ok("a crashing node still records its timing (ok=False) and re-raises")

            # flush failure: nothing lost, retried next time, pipeline unaffected
            with metrics.scope("D", "qa"):
                metrics.record("tool", "run_tests", 12)
            real = db.engine
            bad_engine = MagicMock()
            bad_engine.begin.side_effect = RuntimeError("db down")
            db.engine = bad_engine
            assert metrics.flush("D") == 0
            db.engine = real
            assert len(metrics._buffers["D"]) == 1
            assert metrics.flush("D") == 1 and len(rows_of("D")) == 1
            ok("flush with the DB down keeps the events and writes them at the next flush")

    asyncio.run(scenario())

    settings.metrics_enabled = False
    try:
        metrics._buffers.clear()
        with metrics.scope("E", "qa"):
            metrics.record("tool", "x", 1)
        assert "E" not in metrics._buffers
    finally:
        settings.metrics_enabled = True
    ok("METRICS_ENABLED=false makes recording a no-op")


# ------------------------------------------------------- 4. full pipeline ---

def check_full_pipeline():
    from app.agents.developer import DEVELOPER_SYSTEM_PROMPT
    from app.agents.planner import PLANNER_SYSTEM_PROMPT
    from app.pipeline import graph as graph_mod, nodes
    from tools import ALL_TOOLS

    tool_fns = {f.__name__: f for f in ALL_TOOLS}
    eng = make_engine()
    pid = str(uuid.uuid4())
    dev_calls = {"n": 0}
    GOOD = ("def reverse_string(s):\n    return s[::-1]\n\n\ndef test_reverse():\n"
            "    assert reverse_string('abc') == 'cba'\n")
    BAD = "def reverse_string(s:\n    return s[::-1]\n"   # SyntaxError -> deterministic pre-flight fails

    def post(url, json=None, timeout=None):
        if url.endswith("/api/embeddings"):
            return FakeResp({"embedding": [0.2] * 768})
        system = json.get("system")
        time.sleep(0.01)
        if system == PLANNER_SYSTEM_PROMPT:
            return FakeResp(ollama_body('{"tasks": ["Write reverse_string with a pytest test"]}', 300, 30))
        if system == DEVELOPER_SYSTEM_PROMPT:
            dev_calls["n"] += 1
            code = BAD if dev_calls["n"] == 1 else GOOD
            return FakeResp(ollama_body(f"```python\n{code}```", 500 + 100 * dev_calls["n"], 60))
        return FakeResp(ollama_body('{"hint": "x", "root_cause": "y"}', 50, 10))

    class Sess:
        async def call_tool(self, name, args):
            out = await asyncio.to_thread(tool_fns[name], **args)
            return SimpleNamespace(content=[SimpleNamespace(text=out)], isError=False)

    @asynccontextmanager
    async def fake_session():
        yield Sess()

    def timed_retrieve(*a, **k):
        time.sleep(0.01)
        return []

    async def scenario():
        with patch.object(db, "engine", eng), patch("requests.post", post), \
             patch.object(mcp_client, "mcp_session", fake_session), \
             patch.object(nodes, "db", MagicMock()), patch.object(nodes, "stm", MagicMock()), \
             patch.object(graph_mod, "stm", MagicMock()), \
             patch.object(memory, "retrieve_similar", metrics.timed_db("memory.retrieve_similar")(timed_retrieve)), \
             patch.object(memory, "save_memory", metrics.timed_db("memory.save_memory")(lambda *a, **k: "id")), \
             patch.object(memory, "find_similar_failures", lambda *a, **k: []), \
             patch.object(memory, "save_failure_pattern", lambda *a, **k: "id"):
            with eng.begin() as c:
                c.execute(text("INSERT INTO projects VALUES (:i, 'rev', 'reverse a string', 'done', '2026-01-01T00:00:00')"), {"i": pid})
            result = await graph_mod.run_pipeline(pid, "Write a function reverse_string with a test")
            assert result.done and not result.needs_human_review, result.task_history
            return metrics_report.project_metrics(pid)

    pm = asyncio.run(scenario())
    assert pm is not None
    t = pm["tokens"]
    assert t["llm_calls"] == 3, t                           # planner + developer x2 (analyzer stays deterministic)
    assert t["by_agent"]["planner"]["tokens_in"] == 300 and t["by_agent"]["planner"]["tokens_out"] == 30
    assert t["by_agent"]["developer"]["tokens_in"] == 600 + 700 and t["by_agent"]["developer"]["tokens_out"] == 120
    assert t["total"] == 300 + 30 + 1300 + 120
    ok("full run: exact per-agent token totals (planner 330, developer 1420) match what the fake Ollama reported")

    nodes_seen = [n["node"] for n in pm["nodes"]]
    assert nodes_seen == ["planner", "developer", "qa", "failure_analyzer", "developer", "qa", "next_task"], nodes_seen
    assert [n["start_ms"] for n in pm["nodes"]] == sorted(n["start_ms"] for n in pm["nodes"])
    assert all(n["duration_ms"] >= n["llm_ms"] + n["tool_ms"] for n in pm["nodes"])
    dev = [n for n in pm["nodes"] if n["node"] == "developer"]
    assert dev[0]["attempt"] == 1 and dev[1]["attempt"] == 2 and dev[1]["tokens_in"] == 700
    ok("per-node timeline recorded in order with attempt numbers and per-node tokens (flamegraph data)")

    assert pm["tools"]["write_file"]["calls"] == 2 and pm["tools"]["run_tests"]["calls"] == 1, pm["tools"]
    lat = pm["latency"]
    assert lat["total_ms"] >= lat["node_total_ms"] > 0 and lat["llm_ms"] >= 30 and lat["db_ms"] >= 10, lat
    assert lat["other_ms"] >= 0 and lat["embed_ms"] >= 0
    ok(f"tools by name (write_file x2, run_tests x1); latency split llm={lat['llm_ms']}ms tool={lat['tool_ms']}ms db={lat['db_ms']}ms other={lat['other_ms']}ms of {lat['total_ms']}ms total")

    s = pm["success"]
    assert (s["tasks"], s["tasks_passed"], s["pass_rate"], s["avg_attempts"], s["run_ok"]) == (1, 1, 1.0, 2.0, True), s
    assert pm["tasks"][0]["failures"] == 1 and pm["communication_mode"] == "blackboard" and pm["model"] == settings.ollama_model
    ok("task outcome row: passed on attempt 2 after 1 failure; pipeline row: ok, mode, model")
    return eng, pid


# --------------------------------------------------------------- 5. reports -

_SIM_T = [1_800_000_000_000]


def simulate_project(eng, pid, model, ms_per_tok, tout, outcomes, created, mode="blackboard",
                     escalated=None, similar=None):
    """Writes the same rows the pipeline would, without running it.
    outcomes: list of (passed, attempts) per task."""
    with eng.begin() as c:
        c.execute(text("INSERT INTO projects VALUES (:i, :n, 'req', 'done', :c)"), {"i": pid, "n": f"proj-{pid}", "c": created})
    orig = settings.ollama_model
    settings.ollama_model = model
    try:
        with metrics.scope(pid, "pipeline") as ps:
            for agent in ("planner", "developer"):
                with metrics.scope(pid, agent, 1, 1):
                    body = ollama_body("x", 100, tout, server_ms=tout * ms_per_tok, eval_ms=tout * ms_per_tok * 0.9)
                    metrics.record_llm(model, tout * ms_per_tok + 20, body, "p", "o")
            for i, (passed, attempts) in enumerate(outcomes, 1):
                st = {"_failure_history": [{"repeat": False, "category": "type", "used_similar_fix": bool(similar)}] * (attempts - 1),
                      "_last_escalated": (escalated or {}).get(i, False)}
                metrics.record_task(st, {"order": i, "type": "code"}, "passed" if passed else "needs_human_review", attempts)
            metrics.record("pipeline", mode, 5000, ok=all(p for p, _ in outcomes), scope_=ps, meta={"model": model})
        # Deterministic chronology: real calls carry true start times, but this
        # helper records synthetic durations, so stamp each project 1 minute
        # after the previous one (otherwise "start = now - duration" would make
        # slower projects look older).
        _SIM_T[0] += 60_000
        for i, ev in enumerate(metrics._buffers[pid]):
            ev["t_start_ms"] = _SIM_T[0] + i * 100
        assert metrics.flush(pid) > 0
    finally:
        settings.ollama_model = orig


def check_reports():
    eng = make_engine()
    with patch.object(db, "engine", eng):
        simulate_project(eng, "r1", "modelA", 10, 100, [(True, 1), (True, 2)], "2026-01-01T00:00:01")
        simulate_project(eng, "r2", "modelA", 10, 200, [(True, 1), (False, 3)], "2026-01-01T00:00:02",
                         mode="message_passing", escalated={2: True})
        simulate_project(eng, "r3", "modelB", 30, 100, [(True, 2)], "2026-01-01T00:00:03", similar=True)

        # list
        lst = metrics_report.list_projects()
        assert [p["project_id"] for p in lst] == ["r3", "r2", "r1"], "newest first"
        r1 = next(p for p in lst if p["project_id"] == "r1")
        assert (r1["tokens_in"], r1["tokens_out"], r1["tasks"], r1["pass_rate"], r1["avg_attempts"]) == (200, 200, 2, 1.0, 1.5)
        assert r1["est_cost_usd"] == round(0.2 * 0.0005 + 0.2 * 0.0015, 6) and r1["model"] == "modelA"
        r2 = next(p for p in lst if p["project_id"] == "r2")
        assert r2["escalations"] == 1 and r2["communication_mode"] == "message_passing" and r2["pass_rate"] == 0.5
        assert [p["project_id"] for p in metrics_report.list_projects(mode="message_passing")] == ["r2"]
        ok("list_projects: tokens, cost formula (tokens x rate/1K), pass rate, attempts, escalations; filters work")

        # compare
        cmp_ = metrics_report.compare(["r1", "r2"])
        d = cmp_["deltas"][0]["tokens_total"]
        assert d["baseline"] == 400 and d["value"] == 600 and d["pct_change"] == 50.0, d
        ok("compare: r2 vs r1 tokens +50.0% (two projects' token usage side by side)")

        # success summary
        sm = metrics_report.success_summary()
        assert sm["overall"]["tasks"] == 5 and sm["overall"]["pass_rate"] == 0.8
        assert sm["overall"]["escalation_rate"] == 0.2
        assert sm["by_model"]["modelA"]["tasks"] == 4 and sm["by_model"]["modelB"]["tasks"] == 1
        assert sm["by_task_type"]["code"]["tasks"] == 5
        assert sm["attempt_histogram"] == {"1": 2, "2": 2, "3": 1}
        fr = sm["failure_recovery"]
        assert fr["avg_attempts_with_similar_fix"] == 2.0 and fr["avg_attempts_without_similar_fix"] == 2.5, fr
        ok("summary: pass rate / attempts / escalation rate by task type and model, attempt histogram, similar-fix effect")

        # models + regression detection
        ms = metrics_report.model_stats()["models"]
        assert ms["modelA"]["ms_per_output_token"] < ms["modelB"]["ms_per_output_token"]
        assert ms["modelB"]["slower_than_fastest"] and not ms["modelA"]["slower_than_fastest"]
        assert ms["modelB"]["slowdown_vs_fastest"] > 2.5
        ok(f"models: modelB flagged {ms['modelB']['slowdown_vs_fastest']}x slower than modelA (a slower model is detectable)")

        for i in range(6):   # same model gets 2x slower halfway through
            simulate_project(eng, f"s{i}", "modelC", 10 if i < 3 else 20, 100, [(True, 1)], f"2026-02-01T00:00:0{i}")
        rg = metrics_report.regression("modelC", window=3, threshold=0.2)
        assert rg["regression"] and 90 <= rg["pct_change"] <= 110, rg
        for i in range(6):
            simulate_project(eng, f"t{i}", "modelD", 10, 100, [(True, 1)], f"2026-03-01T00:00:0{i}")
        assert not metrics_report.regression("modelD", window=3)["regression"]
        assert metrics_report.regression("nonexistent")["regression"] is False
        ok("regression: 2x slowdown on the same model flagged (+~100%); a stable model is not; unknown model is safe")

        # CSV
        rows = metrics_report.list_projects()
        parsed = list(csv.DictReader(io.StringIO(metrics_report.to_csv(rows))))
        assert list(parsed[0].keys()) == metrics_report.CSV_COLUMNS and len(parsed) == len(rows)
        assert all(parsed[0][c] != "" for c in ("project_id", "tokens_total", "est_cost_usd", "total_ms", "model"))
        ok(f"CSV export: {len(metrics_report.CSV_COLUMNS)} columns x {len(parsed)} projects, round-trips")
    return eng


# ------------------------------------------------------------- 6. HTTP ------

def check_http(eng):
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)   # not used as a context manager: startup migrations (need Postgres) don't run
    with patch.object(db, "engine", eng):
        r = client.get("/projects/r1/metrics")
        assert r.status_code == 200 and {"tokens", "cost", "latency", "tools", "nodes", "tasks", "success"} <= set(r.json())
        assert client.get("/projects/does-not-exist/metrics").status_code == 404
        assert client.get("/metrics/projects?limit=2").json()["projects"].__len__() == 2
        assert client.get("/metrics/compare?ids=r1").status_code == 422
        c = client.get("/metrics/compare?ids=r1,r2").json()
        assert len(c["projects"]) == 2 and c["deltas"]
        assert client.get("/metrics/summary").json()["overall"]["tasks"] >= 5
        assert "modelA" in client.get("/metrics/models").json()["models"]
        assert client.get("/metrics/regression?model=modelC&window=3").json()["regression"] is True
        csv_r = client.get("/metrics/export?format=csv")
        assert csv_r.headers["content-type"].startswith("text/csv") and "attachment" in csv_r.headers["content-disposition"]
        assert csv_r.text.splitlines()[0].split(",") == metrics_report.CSV_COLUMNS
        assert client.get("/metrics/export?format=json").json()["projects"]
        assert client.get("/metrics/export?format=xml").status_code == 422
    ok("HTTP: project metrics / list / compare / summary / models / regression / export(csv,json) return expected shapes; bad input -> 404/422")


# ------------------------------------------------------------ 7. overhead ---

def check_overhead():
    metrics._buffers.clear()
    n = 20000
    with metrics.scope("bench", "developer"):
        t0 = time.perf_counter()
        for _ in range(n):
            metrics.record("llm", "m", 5.0, tokens_in=10, tokens_out=5, meta={"a": 1})
        per_us = (time.perf_counter() - t0) / n * 1e6
    assert per_us < 100, per_us
    assert len(metrics._buffers["bench"]) == metrics.MAX_BUFFERED_PER_PROJECT   # bounded
    eng = make_engine()
    with patch.object(db, "engine", eng):
        t0 = time.perf_counter()
        written = metrics.flush("bench")
        flush_ms = (time.perf_counter() - t0) * 1000
    assert written == metrics.MAX_BUFFERED_PER_PROJECT and flush_ms < 2000
    ok(f"overhead: {per_us:.1f} us per recorded event; batched flush of {written} rows in {flush_ms:.0f} ms; buffer is bounded")
    print(f"     (a typical run records ~10-40 events, i.e. well under 1 ms of instrumentation vs. tens of seconds of LLM time => far below the 5% budget)")


if __name__ == "__main__":
    check_token_capture()
    check_timing()
    check_nodes_and_flush()
    check_full_pipeline()
    report_eng = check_reports()
    check_http(report_eng)
    check_overhead()
    print("\nAll Metrics checks passed.")
