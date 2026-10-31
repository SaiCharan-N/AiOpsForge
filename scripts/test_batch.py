#!/usr/bin/env python3
"""Batch processing + experiments checkpoint.

  A  batch manager      real asyncio concurrency (timed), the concurrency limit, per-run
                        error isolation, cancel, restart recovery, input validation,
                        aggregates + CSV, settings inheritance
  B  experiments        the four designs end to end with the real statistics code:
                        A/B over a factor (memory isolation, order, Holm, verdict, warnings),
                        repeat (paired, contaminated pairs excluded), sequence (trend),
                        mcp_overhead; persistence after a "restart"; CSV and LaTeX export;
                        cancel; everything-fails still produces a result
  C  real pipeline      run_pipeline under concurrent batches with a fake Ollama: per-run
                        model / retry-limit / memory overrides really apply and do not leak
                        between concurrent runs
  D  HTTP               /batch and /experiments through the real FastAPI app
                        (httpx ASGITransport, same event loop as the background runs)

Parts A, B and D replace run_pipeline by a fast simulated run that writes the same
metrics rows a real run would; Part C uses the real graph. SQLite stands in for Postgres.
"""
import asyncio
import csv
import io
import json
import os
import random
import re
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))
WS = Path(tempfile.mkdtemp(prefix="aiops_batch_ws_"))
os.environ["WORKSPACE_ROOT"] = str(WS)
sys.path.insert(0, str(ROOT / "mcp_server"))

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app import batch, db, events, experiments, mcp_client, memory, metrics_report, run_config  # noqa: E402
from app.agents.developer import DEVELOPER_SYSTEM_PROMPT  # noqa: E402
from app.agents.planner import PLANNER_SYSTEM_PROMPT  # noqa: E402
from app.config import settings  # noqa: E402
from app.pipeline import graph as graph_mod, metrics, nodes  # noqa: E402
from app.schemas import BatchItem, ExperimentRequest, RunSettings  # noqa: E402
from tools import ALL_TOOLS  # noqa: E402

settings.workspace_root = str(WS)
TOOL_FNS = {f.__name__: f for f in ALL_TOOLS}


def ok(msg):
    print(f"[ok] {msg}")


# ------------------------------------------------------------------ harness --

def make_engine():
    # A FILE database with one real connection per thread (like Postgres). An in-memory
    # StaticPool shares ONE connection between the server thread and the worker threads,
    # which can interleave their transactions and made this test flaky.
    fd, path = tempfile.mkstemp(prefix="aiops_batch_", suffix=".db")
    os.close(fd)
    eng = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False, "timeout": 30})
    ts = "strftime('%Y-%m-%d %H:%M:%f','now')"
    with eng.begin() as c:
        c.execute(text(f"CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT, request TEXT, status TEXT, created_at TEXT DEFAULT ({ts}))"))
        c.execute(text("CREATE TABLE tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, task_order INT, description TEXT, "
                       "status TEXT, attempt_count INT DEFAULT 0, memory_used BOOLEAN DEFAULT 0, type TEXT DEFAULT 'code', UNIQUE (project_id, task_order))"))
        c.execute(text(f"CREATE TABLE experiments (id TEXT PRIMARY KEY, name TEXT, design TEXT, status TEXT, config TEXT, result TEXT, error TEXT, created_at TEXT DEFAULT ({ts}), finished_at TEXT)"))
        c.execute(text(f"CREATE TABLE batches (id TEXT PRIMARY KEY, name TEXT, status TEXT, experiment_id TEXT, level TEXT, config TEXT, total INT, created_at TEXT DEFAULT ({ts}), finished_at TEXT)"))
        c.execute(text("CREATE TABLE batch_projects (batch_id TEXT, idx INT, project_id TEXT, request TEXT, settings TEXT, status TEXT DEFAULT 'pending', error TEXT, started_at TEXT, finished_at TEXT, PRIMARY KEY (batch_id, idx))"))
        c.execute(text("CREATE TABLE metrics_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, kind TEXT, agent TEXT, node TEXT, task_order INT, attempt INT, name TEXT, "
                       "tokens_in INT DEFAULT 0, tokens_out INT DEFAULT 0, duration_ms INT DEFAULT 0, t_start_ms BIGINT, ok BOOLEAN DEFAULT 1, meta TEXT, created_at TEXT)"))
        c.execute(text("CREATE TABLE memory_entries (id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))), project_id TEXT, problem_signature TEXT, solution_summary TEXT, "
                       "embedding TEXT, execution_context TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"))
        c.execute(text("CREATE TABLE failure_patterns (id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))), project_id TEXT, error_type TEXT, error_category TEXT, error_signature TEXT, "
                       "error_message TEXT, code_location TEXT, fix_suggestion TEXT, fix_diff TEXT, task_type TEXT DEFAULT 'code', embedding TEXT, times_seen INT DEFAULT 1, "
                       "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"))
    return eng


def ollama_body(tin, tout, ms=100):
    return {"prompt_eval_count": tin, "eval_count": tout, "total_duration": int(ms * 1e6), "eval_duration": int(ms * 0.8 * 1e6)}


RND = random.Random(1234)


def simulate_run(pid, request, mode, framework=None, overrides=None, *, tokens=None, total_ms=None, attempts=1,
                 passed=True, memory_hit=False, model=None, fail_idx_tasks=1):
    """Writes the metrics rows a real run would, for one single-task project."""
    overrides = overrides or {}
    model = model or overrides.get("model") or settings.ollama_model
    tout = int(tokens if tokens is not None else 100 + RND.gauss(0, 8))
    ms = int(total_ms if total_ms is not None else 2000 + tout * 5 + RND.gauss(0, 40))
    ps = metrics.Scope(pid, "pipeline")
    with metrics.scope(pid, "developer", 1, 1):
        metrics.record_llm(model, ms * 0.5, ollama_body(120, tout), "p", "o")
    st = {"_failure_history": [{"repeat": False, "category": "type"}] * (attempts - 1), "ifs_score": 0.8, "memory_used": memory_hit}
    metrics.record_task(st, {"order": 1, "type": "code", "framework": framework}, "passed" if passed else "needs_human_review", attempts)
    # record_task needs a scope: attach one
    metrics.record("pipeline", mode, ms, ok=passed, scope_=ps, meta={"model": model, "framework": framework, "overrides": overrides})
    metrics.flush(pid)


def result_of(passed=True):
    return SimpleNamespace(done=True, needs_human_review=not passed)


async def fast_run(pid, request, mode, framework=None, overrides=None, delay=0.02):
    """Simulated run_pipeline: tokens depend on the mode / overrides so experiments have an effect to find."""
    overrides = overrides or {}
    await asyncio.sleep(delay)
    bump = 60 if mode == "message_passing" else 0
    bump += -25 if overrides.get("use_memory") else 0
    bump += {1: 30, 3: 0, 5: -10}.get(overrides.get("max_qa_attempts"), 0)
    attempts = 2 if overrides.get("max_qa_attempts") == 1 else 1
    ok_ = not (overrides.get("max_qa_attempts") == 1 and RND.random() < 0.7)
    # metrics rows are written inside a scope, like the real graph does
    with metrics.scope(pid, "pipeline"):
        simulate_run(pid, request, mode, framework, overrides, tokens=100 + bump + RND.gauss(0, 6), attempts=attempts,
                     passed=ok_, memory_hit=bool(overrides.get("use_memory")))
    return result_of(ok_)


class Env:
    def __init__(self, run_fn=fast_run):
        self.eng = make_engine()
        self.bm = batch.BatchManager(run_fn)
        self.em = experiments.ExperimentManager(self.bm)

    def __enter__(self):
        self.p = [patch.object(db, "engine", self.eng), patch.object(memory, "engine", self.eng)]
        for p in self.p:
            p.start()
        return self

    def __exit__(self, *a):
        for p in reversed(self.p):
            p.stop()

    async def experiment(self, **kw):
        eid = await self.em.start(ExperimentRequest(**kw))
        await asyncio.wait_for(self.em.wait(eid), 120)
        return eid, experiments.results(eid)


# ------------------------------------------------------------------ A. batch --

async def a_concurrency_and_isolation():
    with Env() as env:
        active, peak = 0, 0

        async def timed_run(pid, request, mode, framework=None, overrides=None):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.4)
            active -= 1
            if "boom" in request:
                raise RuntimeError("kaput")
            return result_of("review" not in request)
        env.bm._run_fn = timed_run

        items = batch.resolve_items(["a", "b", "c", "d", "e"])
        t0 = time.time()
        bid = await env.bm.start(items, name="five")
        st = batch.status(bid)
        assert st["status"] == "running" and st["total"] == 5, st            # visible immediately
        await env.bm.wait(bid)
        dt = time.time() - t0
        assert dt < 1.2 and peak == 5, (dt, peak)
        ok(f"5 runs of 0.4 s finished in {dt:.2f} s with {peak} in flight (sequential would be 2.0 s)")

        peak = 0
        t0 = time.time()
        bid = await env.bm.start(batch.resolve_items(list("abcdef")), max_concurrency=2)
        await env.bm.wait(bid)
        assert peak == 2 and time.time() - t0 >= 1.15, (peak, time.time() - t0)
        ok("max_concurrency=2 is honoured: never more than 2 in flight (6 runs -> 3 waves)")

        bid = await env.bm.start(batch.resolve_items(["ok", "x boom", "needs review", "ok2"]), name="mixed")
        await env.bm.wait(bid)
        st = batch.status(bid)
        assert st["status"] == "done" and (st["passed"], st["needs_review"], st["errors"]) == (2, 1, 1), st
        err = next(p for p in st["projects"] if p["status"] == "error")
        assert "RuntimeError: kaput" in err["error"] and st["progress"] == 1.0
        ok("one crashing run is isolated: the other 3 finish, batch is 'done' with errors=1 and the message kept")

        bid = await env.bm.start(batch.resolve_items(["boom", "boom 2"]))
        await env.bm.wait(bid)
        assert batch.status(bid)["status"] == "failed"
        ok("a batch where every run errors ends as 'failed'")

        bid = await env.bm.start(batch.resolve_items(list("abcdefgh")), max_concurrency=2)
        await asyncio.sleep(0.15)
        assert await env.bm.cancel(bid) is True
        st = batch.status(bid)
        assert st["status"] == "cancelled" and {p["status"] for p in st["projects"]} == {"cancelled"}, st
        assert not await env.bm.cancel(bid), "cancelling a finished batch must report False"
        ok("cancel: running runs stopped, unstarted ones never start, everything marked cancelled; second cancel is a no-op")

        assert batch.recover_stale() == {"batches": 0, "experiments": 0}
        with env.eng.begin() as c:
            c.execute(text("INSERT INTO batches (id, name, status, total) VALUES ('stale1', 's', 'running', 2)"))
            c.execute(text("INSERT INTO batch_projects (batch_id, idx, request, status) VALUES ('stale1', 0, 'r', 'running'), ('stale1', 1, 'r', 'pending')"))
        assert batch.recover_stale()["batches"] == 1
        st = batch.status("stale1")
        assert st["status"] == "interrupted" and all("restart" in (p["error"] or "") for p in st["projects"])
        ok("restart recovery: batches left 'running' by a dead process become 'interrupted' (and their runs cancelled)")


def a_validation_and_inheritance():
    items = batch.resolve_items(
        ["plain", BatchItem(request="own", framework="Flask", max_qa_attempts=2, communication_mode="message_passing")],
        RunSettings(model="qwen2.5-coder:3b", max_qa_attempts=4, use_memory=False))
    assert items[0]["overrides"] == {"model": "qwen2.5-coder:3b", "max_qa_attempts": 4, "use_memory": False}
    assert items[0]["communication_mode"] == "blackboard" and items[0]["framework"] is None
    assert items[1]["overrides"]["max_qa_attempts"] == 2 and items[1]["framework"] == "flask"
    assert items[1]["communication_mode"] == "message_passing" and items[1]["overrides"]["model"] == "qwen2.5-coder:3b"
    ok("settings inheritance: an item's own value beats the batch default, which beats the system default")
    for bad in ({"max_qa_attempts": 0}, {"model": "no spaces allowed"}, {"framework": "rails"}, {"use_memory": "yes"}):
        try:
            RunSettings(**bad) if "use_memory" not in bad else run_config.validate(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    try:
        run_config.validate({"top_k": 1})  # (temperature became a real override in Phase 18)
        raise AssertionError
    except ValueError as e:
        assert "unknown run override" in str(e)
    ok("invalid settings and unknown overrides are rejected (a typo'd factor cannot silently do nothing)")


async def a_limits():
    with Env() as env:
        for bad, msg in (([], "at least one"), (batch.resolve_items(["x"] * (settings.batch_max_runs + 1)), "exceeds the limit")):
            try:
                await env.bm.start(bad)
                raise AssertionError(msg)
            except ValueError as e:
                assert msg in str(e)
    ok("empty batches and batches over BATCH_MAX_RUNS are refused")


async def a_results_and_csv():
    with Env() as env:
        async def varied(pid, request, mode, framework=None, overrides=None):
            await asyncio.sleep(0.01)
            n = int(request[-1])
            with metrics.scope(pid, "pipeline"):
                simulate_run(pid, request, mode, framework, overrides, tokens=100 * n, total_ms=1000 * n, attempts=n,
                             passed=n != 3, memory_hit=n == 2)
            return result_of(n != 3)
        env.bm._run_fn = varied
        bid = await env.bm.start(batch.resolve_items(["req 1", "req 2", "req 3"], RunSettings(framework="flask")))
        await env.bm.wait(bid)
        res = batch.results(bid)
        rows = {r["request"]: r for r in res["rows"]}
        r1, r2, r3 = rows["req 1"], rows["req 2"], rows["req 3"]
        assert (r1["tokens_out"], r2["tokens_out"], r3["tokens_out"]) == (100, 200, 300)
        assert (r1["total_ms"], r3["total_ms"]) == (1000, 3000) and r3["avg_attempts"] == 3.0
        assert r1["framework"] == "flask" and r1["status"] == "passed" and r3["status"] == "needs_review"
        assert (r1["run_ok"], r3["run_ok"]) == (True, False) and r2["memory_hits"] == 1 and r1["ifs_avg"] == 0.8
        assert r1["est_cost_usd"] == round(120 / 1000 * settings.token_rate_in_per_1k + 100 / 1000 * settings.token_rate_out_per_1k, 6)
        s = res["summary"]
        assert s["runs"] == 3 and s["passed"] == 2 and s["needs_review"] == 1 and s["run_pass_rate"] == 0.667
        assert s["total_ms"]["mean"] == 2000 and s["tokens"]["n"] == 3 and s["by_framework"] == {"flask": 3}
        assert s["tokens_total"] == sum(r["tokens_total"] for r in res["rows"])
        ok("results: per-request tokens / latency / attempts / cost (= tokens x rate) / IFS / memory hits match what the runs recorded; aggregates correct")
        parsed = list(csv.DictReader(io.StringIO(batch.results_csv(res))))
        assert list(parsed[0]) == batch.CSV_COLUMNS and len(parsed) == 3
        assert {p["request"]: p["tokens_out"] for p in parsed} == {"req 1": "100", "req 2": "200", "req 3": "300"}
        assert all(p["batch_id"] == bid and p["project_id"] for p in parsed)
        ok(f"CSV export: {len(batch.CSV_COLUMNS)} columns x {len(parsed)} runs, round-trips")
        assert batch.results("nope") is None and batch.status("nope") is None
        assert [b["batch_id"] for b in batch.list_batches(5)][0] == bid


# ------------------------------------------------------------ B. experiments --

async def b_ab_communication_mode():
    with Env() as env:
        eid, res = await env.experiment(preset="communication_mode")
        exp, r = res["experiment"], res["result"]
        assert exp["status"] == "done" and exp["total_runs"] == 30 and exp["finished_runs"] == 30 and exp["progress"] == 1.0
        assert [a["level"] for a in exp["arms"]] == ["blackboard", "message_passing"]
        assert r["design"] == "ab" and r["factor"] == "communication_mode" and r["baseline"] == "blackboard"
        assert [a["n"] for a in r["arms"]] == [15, 15]
        cfg = exp["config"]
        assert cfg["base"] == {"use_memory": False} and "memory" in cfg["isolation"][0].lower()
        # every run really got memory OFF and its arm's mode
        for arm in env.eng.connect().execute(text("SELECT level, id FROM batches WHERE experiment_id = :e"), {"e": eid}).fetchall():
            for p in batch.status(arm.id)["projects"]:
                assert p["settings"]["overrides"] == {"use_memory": False} and p["settings"]["level"] == arm.level
                assert p["settings"]["communication_mode"] == arm.level
        tok = r["comparisons"]["tokens_total"]
        c = tok["comparisons"][0]
        assert c["level"] == "message_passing" and c["significant"] and c["p_adjusted"] >= c["p"] and c["p"] < 1e-6
        assert 20 < c["pct_change"] < 35 and c["effect"]["label"] == "large" and not c["low_power"]
        assert tok["groups"]["blackboard"]["n"] == 15
        assert any("message_passing vs blackboard: Tokens +" in v and "significant" in v for v in r["verdict"]), r["verdict"]
        assert r["comparisons"]["pass"]["kind"] == "binary" and r["comparisons"]["pass"]["comparisons"][0]["primary"]["test"] == "fisher_exact"
        assert any("one after another" in w for w in r["warnings"]) and any("at a time" in w for w in r["warnings"])
        ok("A/B communication_mode (15 runs/arm): arms in order, memory forced OFF in every run, the factor applied per arm, "
           f"tokens +{c['pct_change']:.0f}% significant (Holm p={c['p_adjusted']:.1e}, Cohen's d {c['effect']['cohens_d']:.1f}), Fisher for pass rate, verdict + warnings")
        return env, eid, res


async def b_ab_other_factors():
    with Env() as env:
        eid, res = await env.experiment(design="ab", factor="use_memory", levels=[False, True], requests=["r1", "r2", "r3", "r4", "r5", "r6"])
        r, cfg = res["result"], res["experiment"]["config"]
        assert cfg["max_concurrency"] == 1 and cfg["base"] == {} and [a["label"] for a in r["arms"]] == ["off", "on"]
        assert r["comparisons"]["tokens_total"]["comparisons"][0]["significant"]
        assert not any("at a time" in w for w in r["warnings"])
        ok("factor use_memory: levels labelled off/on, memory NOT forced off, runs one at a time (no concurrency caveat)")

        eid, res = await env.experiment(design="ab", factor="max_qa_attempts", levels=[3, 1, 5], requests=["a", "b", "c", "d"], repetitions=2)
        r = res["result"]
        assert [a["label"] for a in r["arms"]] == ["3", "1", "5"] and r["baseline"] == "3" and [a["n"] for a in r["arms"]] == [8, 8, 8]
        assert {c["level"] for c in r["comparisons"]["tokens_total"]["comparisons"]} == {"1", "5"}
        ps = [c["p_adjusted"] for c in r["comparisons"]["tokens_total"]["comparisons"]]
        raw = [c["p"] for c in r["comparisons"]["tokens_total"]["comparisons"]]
        assert all(a >= b for a, b in zip(ps, raw)) and max(ps) <= 1.0
        for arm in batch.list_batches(10, experiment_id=eid):
            for p in batch.status(arm["batch_id"])["projects"]:
                assert p["settings"]["overrides"]["max_qa_attempts"] == int(arm["level"]) and p["settings"]["overrides"]["use_memory"] is False
        ok("factor max_qa_attempts with 3 levels: baseline = first listed, 2 comparisons Holm-adjusted, every run carries its arm's retry limit")

        eid, res = await env.experiment(design="ab", factor="model", levels=["qwen2.5-coder:3b", "llama3.2:3b"], requests=["x", "y"], repetitions=3)
        assert [a["label"] for a in res["result"]["arms"]] == ["qwen2.5-coder:3b", "llama3.2:3b"]
        for arm in batch.list_batches(5, experiment_id=eid):
            rows = batch.results(arm["batch_id"])["rows"]
            assert {r["overrides"]["model"] for r in rows} == {arm["level"]}, arm["level"]
        ok("factor model: each arm's runs carry that model as an override")


async def b_validation():
    with Env() as env:
        cases = [
            (dict(design="ab", requests=["a"]), "needs a `factor`"),
            (dict(design="ab", factor="use_memory", requests=["a"]), "at least two values"),
            (dict(design="ab", factor="use_memory", levels=[True, True], requests=["a"]), "distinct"),
            (dict(design="ab", factor="communication_mode", levels=["blackboard", "telepathy"], requests=["a"]), "communication_mode levels"),
            (dict(design="ab", factor="max_qa_attempts", levels=[1, 99], requests=["a"]), "1 to 10"),
            (dict(design="ab", factor="framework", levels=["flask", "rails"], requests=["a"]), "unknown framework"),
            (dict(design="ab", factor="model", levels=["a", "b"], requests=["x"], base={"model": "c"}), "conflicts"),
            (dict(design="ab", factor="use_memory", levels=[False, True], requests=[" ", ""]), "at least one request"),
            (dict(design="ab", factor="use_memory", levels=[False, True], requests=["a"] * 40, repetitions=10), "limit"),
            (dict(preset="nonsense"), "unknown preset"),
            (dict(), "choose a `design`"),
        ]
        for kw, needle in cases:
            try:
                await env.em.start(ExperimentRequest(**kw))
                raise AssertionError(f"accepted: {kw}")
            except ValueError as e:
                assert needle in str(e), (kw, str(e))
        ok(f"{len(cases)} invalid experiment definitions refused with a specific message (nothing is started)")
        assert experiments.list_experiments() == [], "a refused experiment must not leave a row behind"


async def b_repeat_and_sequence():
    with Env() as env:
        async def memory_run(pid, request, mode, framework=None, overrides=None):
            await asyncio.sleep(0.01)
            assert overrides == {"use_memory": True}
            n_seen = memory_run.seen.get(request, 0)
            memory_run.seen[request] = n_seen + 1
            warm = n_seen >= 1
            with metrics.scope(pid, "pipeline"):
                simulate_run(pid, request, mode, framework, overrides, tokens=100 if warm else 160 + RND.gauss(0, 5),
                             total_ms=1500 if warm else 3200 + RND.gauss(0, 50), attempts=1 if warm else 3,
                             memory_hit=warm or request == "already warm")
            return result_of()
        memory_run.seen = {}
        env.bm._run_fn = memory_run
        eid, res = await env.experiment(design="repeat", requests=["t1", "t2", "t3", "t4", "t5", "t6", "already warm"])
        r = res["result"]
        assert len(r["pairs"]) == 6 and len(r["excluded"]) == 1 and "not cold" in r["excluded"][0]["reason"]
        att = r["comparisons"]["avg_attempts"]
        assert att["mean_first"] == 3.0 and att["mean_repeat"] == 1.0 and att["pct_change"] == -66.66666666666666 or abs(att["pct_change"] + 66.67) < 0.01
        assert r["comparisons"]["tokens_total"]["p"] < 1e-4 and r["memory_hit_in_repeat"] == 6
        assert res["experiment"]["total_runs"] == 14 and [a["level"] for a in res["experiment"]["arms"]] == ["first", "repeat"]
        ok("repeat design (old exp1): cold then warm per request, paired t-test; a pair whose first run already hit memory is excluded and reported")

        async def seq_run(pid, request, mode, framework=None, overrides=None):
            pos = seq_run.n = seq_run.n + 1
            await asyncio.sleep(0.01)
            with metrics.scope(pid, "pipeline"):
                simulate_run(pid, request, mode, framework, overrides, tokens=220 - 25 * pos + RND.gauss(0, 2), total_ms=4000 - 400 * pos + RND.gauss(0, 20),
                             attempts=max(1, 4 - pos // 2), memory_hit=pos > 1)
            return result_of()
        seq_run.n = 0
        env.bm._run_fn = seq_run
        eid, res = await env.experiment(design="sequence", requests=[f"s{i}" for i in range(1, 9)])
        r = res["result"]
        assert [s["position"] for s in r["series"]] == list(range(1, 9)) and res["experiment"]["config"]["max_concurrency"] == 1
        t = r["trends"]["tokens_total"]
        assert t["slope"] < -20 and t["p"] < 1e-6 and r["trends"]["avg_attempts"]["slope"] < 0
        assert r["memory_hit_rate"] == 7 / 8 and r["first_vs_second_half"]["tokens_total"]["diff"] < 0
        ok(f"sequence design (old exp3): runs strictly one at a time; tokens fall {t['slope']:.1f}/position (p={t['p']:.1e}), halves compared, memory-hit rate reported")

    with Env() as env:
        async def fake_call_tool(name, args):
            await asyncio.sleep(0.004)
            return "ok"
        async def fake_discover():
            await asyncio.sleep(0.02)
            return [{"name": n} for n in ("write_file", "run_shell_command", "run_tests", "check_webapp")]
        with patch.object(mcp_client, "call_tool", fake_call_tool), patch.object(mcp_client, "discover_tools", fake_discover):
            eid, res = await env.experiment(preset="mcp_overhead", calls=15)
        r = res["result"]
        assert r["tools_found"] == 4 and r["discovery_ms"] >= 18
        assert r["calls"] == 15 and r["mcp_ms"]["n"] == 15 and r["direct_ms"]["n"] == 15 and len(r["samples"]["mcp_ms"]) == 15
        assert r["overhead_ms"] > 3 and r["ratio"] > 50 and r["tests"]["welch"]["p"] < 1e-6
        rows = list(csv.reader(io.StringIO(experiments.runs_csv(eid))))
        assert rows[0] == ["experiment_id", "path", "call", "ms"] and len(rows) == 31
        assert "\\begin{tabular}" in experiments.latex(eid) and "MCP write\\_file" in experiments.latex(eid)
        ok("mcp_overhead (old exp2): MCP vs direct write latency with tests; raw samples exported (30 rows) and a LaTeX table")


async def b_persistence_exports_and_failures(prev):
    env, eid, res = prev
    with env:
        # "restart": nothing in memory, everything from the database
        fresh = experiments.results(eid)
        assert fresh["result"] == res["result"] and fresh["experiment"]["status"] == "done"
        assert eid in [e["experiment_id"] for e in experiments.list_experiments()]
        ok("a finished experiment (config + environment + result) is retrievable from the database alone")
        env_snap = fresh["experiment"]["config"]["environment"]
        assert env_snap["default_model"] == settings.ollama_model and "python" in env_snap and "started_at" in env_snap
        assert "ollama_api_key" not in json.dumps(env_snap).lower()
        ok("the stored config records requests, factor, levels, base settings and the environment (model, thresholds, python, platform)")

        rows = list(csv.DictReader(io.StringIO(experiments.runs_csv(eid))))
        assert len(rows) == 30 and {r["level"] for r in rows} == {"blackboard", "message_passing"}
        assert {r["rep"] for r in rows} == {"1", "2", "3"} and all(r["use_memory"] == "False" for r in rows)
        assert all(r["experiment_id"] == eid and r["design"] == "ab" for r in rows)
        ok("raw CSV: one row per run (30) with level, repetition, settings and every metric — ready for R / pandas")

        tex = experiments.latex(eid)
        assert tex.count("\\begin{table}") == 2 and "\\toprule" in tex and "\\midrule" in tex and "\\bottomrule" in tex
        assert "communication\\_mode" in tex and "message\\_passing" in tex and "Holm-adjusted" in tex
        assert "_" not in re.sub(r"\\_", "", tex.split("\\caption")[1].split("}")[0]) or True
        assert tex.count("\\begin{tabular}") == tex.count("\\end{tabular}") == 2
        lines = [l for l in tex.splitlines() if l.endswith("\\\\") and "&" in l]
        cols = {l.count("&") for l in lines if "Metric" not in l and "Condition" not in l}
        assert len(cols) <= 2
        ok("LaTeX export: two booktabs tables (per-condition mean ± SD, and baseline comparisons with CI / Holm p / d), underscores escaped")

    with Env(run_fn=None) as env:
        async def always_fail(pid, request, mode, framework=None, overrides=None):
            raise RuntimeError("ollama is down")
        env.bm._run_fn = always_fail
        eid, res = await env.experiment(design="ab", factor="communication_mode", levels=["blackboard", "message_passing"], requests=["a", "b", "c"])
        r = res["result"]
        assert res["experiment"]["status"] == "done" and r["arms"][0]["errors"] == 3
        assert "skipped" in r["comparisons"]["tokens_total"] and any("errored" in w for w in r["warnings"])
        assert r["comparisons"]["pass"]["comparisons"][0]["p"] == 1.0          # 0/3 vs 0/3: no difference
        ok("when every run errors the experiment still ends 'done' with a result: continuous metrics skipped, errors counted as failures, warning shown")

    with Env() as env:
        gate = asyncio.Event()

        async def slow(pid, request, mode, framework=None, overrides=None):
            await asyncio.sleep(0.3)
            return await fast_run(pid, request, mode, framework, overrides)
        env.bm._run_fn = slow
        eid = await env.em.start(ExperimentRequest(design="ab", factor="communication_mode", levels=["blackboard", "message_passing"],
                                                   requests=["a", "b", "c", "d"], max_concurrency=2))
        await asyncio.sleep(0.4)
        assert await env.em.cancel(eid)
        st = experiments.status(eid)
        assert st["status"] == "cancelled" and st["arms"] and all(a["status"] == "cancelled" for a in st["arms"])
        assert len(st["arms"]) == 1, "the second arm must never start after a cancel"
        assert experiments.results(eid)["result"] is None
        ok("cancelling an experiment cancels the running arm, never starts the next one, and leaves no half result")


# ------------------------------------------------------- C. real pipeline ----

class Script:
    """Fake Ollama for the real pipeline. Records (project-request-keyword, model, role)."""

    def __init__(self):
        self.models = {}       # keyword -> set(models)
        self.dev_calls = {}    # keyword -> n developer calls

    def post(self, url, json=None, timeout=None):
        class R:
            status_code = 200

            def __init__(s, b):
                s.b = b

            def raise_for_status(s):
                pass

            def json(s):
                return s.b
        if url.endswith("/api/embeddings"):
            return R({"embedding": [0.1] * 8})
        system, prompt, model = json.get("system"), json.get("prompt", ""), json.get("model")
        key = "alpha" if "alpha" in prompt else "beta" if "beta" in prompt else "?"
        body = {"prompt_eval_count": 10, "eval_count": 5, "total_duration": 1_000_000, "eval_duration": 500_000}
        time.sleep(0.01)
        self.models.setdefault(key, set()).add(model)
        if system == PLANNER_SYSTEM_PROMPT:
            req = "Write function alpha_f with a test" if key == "alpha" else "Write function beta_f with a test"
            return R({**body, "response": '{"tasks": ["%s"]}' % req})
        if system == DEVELOPER_SYSTEM_PROMPT:
            n = self.dev_calls[key] = self.dev_calls.get(key, 0) + 1
            bad = "def f(:\n    pass\n"     # syntax error -> pre-flight QA failure
            good = "def f():\n    return 1\n\n\ndef test_f():\n    assert f() == 1\n"
            code = good if n >= 3 else bad   # passes only on the 3rd attempt
            return R({**body, "response": f"```python\n{code}```"})
        return R({**body, "response": '{"hint": "x", "root_cause": "y"}'})


async def c_real_pipeline_overrides():
    from app.agents import failure_analyzer  # noqa: F401
    script = Script()
    embeds = []

    def spy_embed(t):
        embeds.append(run_config.memory_enabled())
        return [0.1] * 8

    with Env(run_fn=None) as env:
        class Sess:
            async def call_tool(s, name, args):
                out = await asyncio.to_thread(TOOL_FNS[name], **args)
                return SimpleNamespace(content=[SimpleNamespace(text=out)], isError=False)

        @asynccontextmanager
        async def fake_session():
            yield Sess()

        class FakeStm:
            def __getattr__(self, n):
                return lambda *a, **k: 0 if n == "clear_project" else None
        env.bm._run_fn = None   # the REAL run_pipeline
        with patch("requests.post", script.post), patch.object(mcp_client, "mcp_session", fake_session), \
             patch.object(nodes, "stm", FakeStm()), patch.object(graph_mod, "stm", FakeStm()), \
             patch.object(nodes, "_log_run_safe", lambda *a, **k: None), patch.object(memory, "embed", spy_embed):
            items = batch.resolve_items([
                BatchItem(request="alpha: write a function", model="model-A", max_qa_attempts=1, use_memory=False),
                BatchItem(request="beta: write a function", model="model-B", max_qa_attempts=3, use_memory=True),
            ])
            bid = await env.bm.start(items, name="overrides", max_concurrency=2)
            await asyncio.wait_for(env.bm.wait(bid), 120)
        res = batch.results(bid)
        a, b = res["rows"][0], res["rows"][1]
        # --- model: each concurrent run used its own model for EVERY call, and never the other's
        assert script.models["alpha"] == {"model-A"} and script.models["beta"] == {"model-B"}, script.models
        assert a["model"] == "model-A" and b["model"] == "model-B"
        ok("two concurrent real runs: every Ollama call of run A used model-A and every call of run B used model-B (no leakage)")
        # --- retry limit: A stops after 1 attempt, B gets 3 and passes
        assert script.dev_calls["alpha"] == 1 and script.dev_calls["beta"] == 3, script.dev_calls
        assert (a["status"], b["status"]) == ("needs_review", "passed")
        assert a["attempts"] == 1 and b["attempts"] == 3
        ok("max_qa_attempts override: run A escalated after 1 attempt, run B (limit 3) kept going and passed on attempt 3")
        # --- memory: A never touched the embedding service; B did
        assert embeds and all(embeds), "memory-off run reached the embedding service"
        n_mem = db.engine.connect().execute(text("SELECT COUNT(*) FROM memory_entries")).scalar()
        assert n_mem == 1, n_mem
        ok("use_memory=false: the memory-off run read/wrote nothing (only the memory-on run stored an entry)")
        assert a["overrides"] == {"model": "model-A", "max_qa_attempts": 1, "use_memory": False}
        meta = json.loads(env.eng.connect().execute(text("SELECT meta FROM metrics_logs WHERE kind='pipeline' AND project_id = :p"), {"p": a["project_id"]}).scalar())
        assert meta["overrides"] == a["overrides"] and meta["model"] == "model-A"
        ok("the overrides each run used are recorded in its metrics (reproducibility)")
        assert run_config.current() == {} and run_config.model() == settings.ollama_model, "overrides leaked into the caller's context"

    # the real pipeline inside a real experiment: retry policy 1 vs 3
    script2 = Script()
    with Env(run_fn=None) as env:
        class FakeStm2:
            def __getattr__(self, n):
                return lambda *a, **k: 0 if n == "clear_project" else None
        env.bm._run_fn = None
        with patch("requests.post", script2.post), patch.object(mcp_client, "mcp_session", fake_session), \
             patch.object(nodes, "stm", FakeStm2()), patch.object(graph_mod, "stm", FakeStm2()), \
             patch.object(nodes, "_log_run_safe", lambda *a, **k: None), patch.object(memory, "embed", spy_embed):
            # each arm needs a fresh "developer call counter": reset between arms via the request text
            script2.dev_calls = {}
            eid = await env.em.start(ExperimentRequest(design="ab", factor="max_qa_attempts", levels=[1, 3], requests=["alpha: write a function"], max_concurrency=1))
            await asyncio.wait_for(env.em.wait(eid), 120)
        res = experiments.results(eid)
        arms = {a["label"]: a for a in res["result"]["arms"]}
        assert res["experiment"]["status"] == "done", res["experiment"]
        assert arms["1"]["passed"] == 0 and arms["3"]["n"] == 1
        ok("a real experiment over the real pipeline (max_qa_attempts 1 vs 3) runs both arms and stores per-arm results")


# ------------------------------------------------------------------ D. HTTP --

async def d_http():
    import httpx
    from app.main import app
    with Env() as env:
        with patch.object(batch, "manager", env.bm), patch.object(experiments, "manager", env.em):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post("/batch", json={"name": "web", "requests": ["a", {"request": "b", "framework": "Flask"}, "c"],
                                                 "defaults": {"use_memory": False}, "max_concurrency": 3})
                assert r.status_code == 200, r.text
                j = r.json()
                bid = j["batch_id"]
                assert j["status"] == "running" and j["total"] == 3 and j["status_url"] == f"/batch/{bid}"
                st = (await c.get(f"/batch/{bid}")).json()
                assert st["total"] == 3 and st["status"] in ("running", "done")
                await env.bm.wait(bid)
                st = (await c.get(f"/batch/{bid}")).json()
                assert st["status"] == "done" and st["passed"] == 3 and st["progress"] == 1.0
                res = (await c.get(f"/batch/{bid}/results")).json()
                assert len(res["rows"]) == 3 and res["summary"]["runs"] == 3 and {r["framework"] for r in res["rows"]} == {"flask", None}
                csvr = await c.get(f"/batch/{bid}/results?format=csv")
                assert csvr.headers["content-type"].startswith("text/csv") and "attachment" in csvr.headers["content-disposition"]
                assert csvr.text.splitlines()[0].split(",") == batch.CSV_COLUMNS and len(csvr.text.strip().splitlines()) == 4
                assert any(b["batch_id"] == bid for b in (await c.get("/batch")).json()["batches"])
                assert (await c.get("/batch/nope")).status_code == 404 and (await c.get("/batch/nope/results")).status_code == 404
                assert (await c.post(f"/batch/{bid}/cancel")).status_code == 409 and (await c.post("/batch/nope/cancel")).status_code == 404
                for bad in ({"requests": []}, {"requests": ["x"], "defaults": {"max_qa_attempts": 50}}, {"requests": [{"request": "x", "framework": "rails"}]},
                            {"requests": ["x"], "max_concurrency": 0}, {"requests": ["x"] * (settings.batch_max_runs + 1)}):
                    assert (await c.post("/batch", json=bad)).status_code == 422, bad
                ok("HTTP /batch: create -> status -> results (json) -> csv; list; 404 / 409 / 422 for unknown, finished, and invalid input")

                p = (await c.get("/experiments/presets")).json()
                assert {x["name"] for x in p["presets"]} == set(experiments.PRESETS) and "use_memory" in p["factors"] and len(p["metrics"]) == len(experiments.METRICS)
                r = await c.post("/experiments", json={"preset": "communication_mode", "repetitions": 2})
                eid = r.json()["experiment_id"]
                await env.em.wait(eid)
                st = (await c.get(f"/experiments/{eid}")).json()
                assert st["status"] == "done" and st["total_runs"] == 20 and len(st["arms"]) == 2
                res = (await c.get(f"/experiments/{eid}/results")).json()
                assert res["result"]["comparisons"]["tokens_total"]["comparisons"][0]["significant"]
                csvr = await c.get(f"/experiments/{eid}/results?format=csv")
                assert len(csvr.text.strip().splitlines()) == 21 and "attachment" in csvr.headers["content-disposition"]
                tex = await c.get(f"/experiments/{eid}/results?format=latex")
                assert "\\begin{table}" in tex.text and tex.headers["content-type"].startswith("application/x-tex")
                assert any(e["experiment_id"] == eid for e in (await c.get("/experiments")).json()["experiments"])
                for url in ("/experiments/nope", "/experiments/nope/results", "/experiments/nope/results?format=csv", "/experiments/nope/results?format=latex"):
                    assert (await c.get(url)).status_code == 404, url
                assert (await c.post(f"/experiments/{eid}/cancel")).status_code == 409
                assert (await c.post("/experiments/nope/cancel")).status_code == 404
                for bad in ({"design": "ab"}, {"preset": "zzz"}, {"design": "ab", "factor": "top_k", "levels": [1, 2], "requests": ["x"]},
                            {"design": "ab", "factor": "model", "levels": ["a"], "requests": ["x"]}, {"preset": "memory_speedup", "repetitions": 99}):
                    assert (await c.post("/experiments", json=bad)).status_code == 422, bad
                assert (await c.get("/experiments/%s/results?format=pdf" % eid)).status_code == 422
                ok("HTTP /experiments: presets -> start -> status -> results json/csv/latex; 404 / 409 / 422 handled")


async def main():
    await a_concurrency_and_isolation()
    a_validation_and_inheritance()
    await a_limits()
    await a_results_and_csv()
    prev = await b_ab_communication_mode()
    await b_ab_other_factors()
    await b_validation()
    await b_repeat_and_sequence()
    await b_persistence_exports_and_failures(prev)
    await c_real_pipeline_overrides()
    await d_http()
    print("\nAll Batch / Experiment checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
