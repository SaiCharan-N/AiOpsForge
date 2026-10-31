#!/usr/bin/env python3
"""User-intervention checkpoint (pause -> feedback -> resume).

REAL: the compiled LangGraph via run_pipeline, the pause gate, the controller,
every app.db helper involved (executed on SQLite with the same table shapes),
the real MCP tool functions (via a fake MCP session) incl. real pytest runs,
the FastAPI app (driven in-process with httpx's ASGITransport so the
background pipeline and the HTTP calls share one event loop, as in production).
MOCKED: the Ollama HTTP API (scripted by scenario), Redis (an in-memory
stand-in that records what is stored), Postgres (SQLite).

Scenarios
  1  plan review: pipeline really parks; edit the task list; DB plan updated; snapshot saved
  2  plan review: revise with guidance -> Planner re-runs WITH it, asks again, project-wide guidance reaches the Developer
  3  first-file review: revise -> Developer regenerates with guidance + previous version; only the FIRST file pauses
  4  qa_failure: analysis shown, guidance reaches the retry (task-scoped: NOT the next task)
  5  qa_failure: skip task -> escalated "by user", next task still runs
  6  abort at the plan / after first file: partial results kept (files on disk), status cancelled
  7  unanswered pause auto-approves after the timeout
  8  on-demand pause of a NON-interactive run ("next", and a named node)
  9  validation: wrong action / not paused / double answer / unknown project
 10  two interactive runs at once never cross feedback
 11  metrics: human wait time is excluded from latency but recorded (paused_ms)
 12  HTTP end to end: /request/async interactive -> status shows paused -> /feedback -> history
"""
import asyncio
import json
import os
import re
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
TMP = Path(tempfile.mkdtemp(prefix="aiops_intv_"))
os.environ["WORKSPACE_ROOT"] = str(TMP)
sys.path.insert(0, str(ROOT / "mcp_server"))

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app import db, events, mcp_client, memory, metrics_report  # noqa: E402
from app.agents.developer import DEVELOPER_SYSTEM_PROMPT  # noqa: E402
from app.agents.planner import PLANNER_SYSTEM_PROMPT  # noqa: E402
from app.config import settings  # noqa: E402
from app.pipeline import graph as graph_mod, intervention, nodes  # noqa: E402
from tools import ALL_TOOLS  # noqa: E402

settings.workspace_root = str(TMP)
settings.intervention_timeout_seconds = 60
TOOL_FNS = {f.__name__: f for f in ALL_TOOLS}

PASSING = "def func_x():\n    return 1\n\n\ndef test_x():\n    assert func_x() == 1\n"
BROKEN = "def func_x(:\n    return 1\n"          # SyntaxError -> deterministic pre-flight fails
VALID_NONTEST = "def helper():\n    return 1\n"


def ok(msg):
    print(f"[ok] {msg}")


# ------------------------------------------------------------- harness -------

def make_engine():
    eng = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    ts = "strftime('%Y-%m-%d %H:%M:%f','now')"
    with eng.begin() as c:
        c.execute(text(f"CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT, request TEXT, status TEXT, created_at TEXT DEFAULT ({ts}))"))
        c.execute(text("CREATE TABLE tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, task_order INT, description TEXT, "
                       "status TEXT, attempt_count INT DEFAULT 0, memory_used BOOLEAN DEFAULT 0, type TEXT DEFAULT 'code', "
                       "UNIQUE (project_id, task_order))"))
        c.execute(text(f"CREATE TABLE interventions (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, kind TEXT, point TEXT, "
                       f"node TEXT, action TEXT, guidance TEXT, clarification TEXT, tasks TEXT, source TEXT DEFAULT 'user', "
                       f"question TEXT, waited_ms INT, created_at TEXT DEFAULT ({ts}))"))
        c.execute(text("CREATE TABLE metrics_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, kind TEXT, agent TEXT, "
                       "node TEXT, task_order INT, attempt INT, name TEXT, tokens_in INT DEFAULT 0, tokens_out INT DEFAULT 0, "
                       "duration_ms INT DEFAULT 0, t_start_ms BIGINT, ok BOOLEAN DEFAULT 1, meta TEXT, created_at TEXT)"))
    return eng


class FakeStm:
    def __init__(self):
        self.store = {}

    def set_working(self, pid, key, value, ttl=None):
        self.store[(pid, key)] = value

    def get_working(self, pid, key):
        return self.store.get((pid, key))

    def list_project_keys(self, pid):
        return [k for (p, k) in self.store if p == pid]

    def clear_project(self, pid):
        keys = [k for k in self.store if k[0] == pid]
        for k in keys:
            del self.store[k]
        return len(keys)


class Script:
    """Scripted fake Ollama. `plan` / `replan` are the task lists the Planner
    returns; `dev` maps a task-description substring -> list of code per attempt."""

    def __init__(self, plan, replan=None, dev=None):
        self.plan, self.replan, self.dev = plan, replan, dev or {}
        self.planner_prompts, self.dev_prompts = [], []
        self.dev_calls = {}

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
        system, prompt = json.get("system"), json.get("prompt", "")
        body = {"prompt_eval_count": 10, "eval_count": 5, "total_duration": 1_000_000, "eval_duration": 500_000}
        time.sleep(0.005)
        if system == PLANNER_SYSTEM_PROMPT:
            self.planner_prompts.append(prompt)
            tasks = self.replan if (self.replan and "user reviewed an earlier plan" in prompt) else self.plan
            return R({**body, "response": '{"tasks": ' + __import__("json").dumps(tasks) + "}"})
        if system == DEVELOPER_SYSTEM_PROMPT:
            self.dev_prompts.append(prompt)
            desc = re.search(r"Task: (.*)", prompt).group(1)
            n = self.dev_calls[desc] = self.dev_calls.get(desc, 0) + 1
            code = PASSING
            for key, seq in self.dev.items():
                if key in desc:
                    code = seq[min(n, len(seq)) - 1]
            return R({**body, "response": f"```python\n{code}```"})
        return R({**body, "response": '{"hint": "x", "root_cause": "y"}'})


class Env:
    """Patches everything external; yields the sqlite engine + fake stm."""

    def __init__(self, script):
        self.script, self.eng, self.stm = script, make_engine(), FakeStm()

    def __enter__(self):
        class Sess:
            async def call_tool(s, name, args):
                out = await asyncio.to_thread(TOOL_FNS[name], **args)
                return SimpleNamespace(content=[SimpleNamespace(text=out)], isError=False)

        @asynccontextmanager
        async def fake_session():
            yield Sess()

        self.patches = [
            patch.object(db, "engine", self.eng), patch("requests.post", self.script.post),
            patch.object(mcp_client, "mcp_session", fake_session),
            patch.object(nodes, "stm", self.stm), patch.object(graph_mod, "stm", self.stm),
            patch.object(intervention, "stm", self.stm),
            patch.object(nodes, "_log_run_safe", lambda *a, **k: None),
            patch.object(memory, "retrieve_similar", lambda *a, **k: []),
            patch.object(memory, "save_memory", lambda *a, **k: "id"),
            patch.object(memory, "find_similar_failures", lambda *a, **k: []),
            patch.object(memory, "save_failure_pattern", lambda *a, **k: "id"),
        ]
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *a):
        for p in reversed(self.patches):
            p.stop()

    def new_project(self, request="build something"):
        pid = str(uuid.uuid4())
        db.create_project(pid, request[:40], request)
        return pid

    def rows(self, sql, **params):
        with self.eng.connect() as c:
            return c.execute(text(sql), params).fetchall()


async def wait_for(cond, timeout=20.0, what="condition"):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


async def wait_paused(pid, point=None, timeout=20.0):
    def paused():
        st = intervention.status(pid)
        return st["paused"] and (point is None or st["pause"]["point"] == point)
    await wait_for(paused, timeout, f"{pid[:6]} paused at {point}")
    return intervention.status(pid)["pause"]


def answer(pid, action="approve", **kw):
    return intervention.submit_feedback(pid, intervention.Feedback(action=action, **kw))


def types(pid):
    return [e["type"] for e in events._history[pid]]


def start(pid, **kw):
    return asyncio.create_task(graph_mod.run_pipeline(pid, kw.pop("request", "build something"), **kw))


# ------------------------------------------------------------ scenarios ------

async def s1_plan_edit():
    sc = Script(["Write func one with a test", "Write func two with a test", "Write func three with a test"])
    with Env(sc) as env:
        pid = env.new_project()
        run = start(pid, interactive=True, pause_points=["planner"])
        pause = await wait_paused(pid, "planner")
        assert pause["node"] == "planner" and [t["description"] for t in pause["payload"]["tasks"]] == sc.plan
        assert set(pause["options"]) == {"approve", "revise", "abort"}
        await asyncio.sleep(0.3)
        assert "developer_start" not in types(pid) and not run.done(), "the pipeline did not really stop"
        snap = json.loads(env.stm.get_working(pid, "paused_snapshot"))
        assert len(snap["task_list"]) == 3 and snap["paused"]["point"] == "planner"
        status = intervention.status(pid)
        assert status["paused"] and status["interactive"]
        ok("plan review: the run is genuinely parked (no developer activity while paused), state snapshot saved")

        edited = ["Write add(a, b) with a test", "Write sub(a, b) with a test"]
        answer(pid, "revise", tasks=edited, guidance="keep functions tiny")
        result = await asyncio.wait_for(run, 120)
        assert result.done and not result.needs_human_review and not result.aborted
        assert [h["task"] for h in result.task_history] == edited
        rows = env.rows("SELECT task_order, description FROM tasks WHERE project_id = :p ORDER BY task_order", p=pid)
        assert [r.description for r in rows] == edited, rows
        assert "plan_revised" in types(pid) and types(pid).count("pipeline_paused") == 1
        assert any("keep functions tiny" in p for p in sc.dev_prompts), "project-wide guidance did not reach the Developer"
        assert intervention.get(pid) is None
        ok("edited plan replaces the Planner's (3 -> 2 tasks, stale DB row pruned) and the run completes with it")
        ok("guidance given at the plan applies to every Developer prompt (project scope); control released after the run")
        return env, pid


async def s2_replan_with_guidance():
    sc = Script(["Write func one with a test", "Write func two with a test", "Write func three with a test"],
                replan=["Write the only func with a test"])
    with Env(sc) as env:
        pid = env.new_project()
        run = start(pid, interactive=True, pause_points=["planner"])
        await wait_paused(pid, "planner")
        answer(pid, "revise", guidance="make it a single task")
        pause2 = await wait_paused(pid, "planner")    # re-planned -> asks again
        assert [t["description"] for t in pause2["payload"]["tasks"]] == ["Write the only func with a test"]
        assert len(sc.planner_prompts) == 2 and "make it a single task" in sc.planner_prompts[1]
        assert "make it a single task" not in sc.planner_prompts[0]
        answer(pid, "approve")
        result = await asyncio.wait_for(run, 120)
        assert result.done and len(result.task_history) == 1
        assert len(env.rows("SELECT 1 FROM tasks WHERE project_id = :p", p=pid)) == 1
        hist = db.list_interventions(pid)
        assert [(h["kind"], h["action"]) for h in hist] == [("pause", "revise"), ("pause", "approve")]
        assert hist[0]["guidance"] == "make it a single task" and hist[0]["point"] == "planner" and hist[0]["waited_ms"] >= 0
        ok("revise with guidance: Planner re-runs WITH the guidance, asks again; shorter plan prunes old rows; history has both answers")


async def s3_first_file_review():
    sc = Script(["Write func one with a test", "Write func two with a test"])
    with Env(sc) as env:
        pid = env.new_project()
        run = start(pid, interactive=True, pause_points=["developer"])
        pause = await wait_paused(pid, "developer")
        assert pause["payload"]["filename"] and "def func_x" in pause["payload"]["code"]
        n_before = len(sc.dev_prompts)
        answer(pid, "revise", guidance="add a docstring to every function")
        pause = await wait_paused(pid, "developer")   # regenerated -> approval again
        assert len(sc.dev_prompts) == n_before + 1
        p = sc.dev_prompts[-1]
        assert "add a docstring to every function" in p and "Your previous version" in p and "def func_x" in p
        answer(pid, "approve")
        result = await asyncio.wait_for(run, 120)
        assert result.done and len(result.task_history) == 2
        pauses = [e for e in events._history[pid] if e["type"] == "pipeline_paused"]
        assert len(pauses) == 2, "only the FIRST generated file should pause (2 tasks, but 2 pauses = revise + approve)"
        ok("first-file review: revise regenerates with the guidance AND the previous version; later files don't pause")


async def s4_qa_failure_guidance_and_scope():
    sc = Script(["Write the BAD-FIRST function", "Write func two with a test"],
                dev={"BAD-FIRST": [BROKEN, VALID_NONTEST]})
    with Env(sc) as env:
        pid = env.new_project()
        run = start(pid, interactive=True, pause_points=["qa_failure"])
        pause = await wait_paused(pid, "qa_failure")
        assert pause["node"] == "failure_analyzer" and set(pause["options"]) == {"approve", "revise", "skip_task", "abort"}
        a = pause["payload"]["analysis"]
        assert a["error_type"] == "SyntaxError" and a["hint"], a
        assert pause["payload"]["attempts_failed"] == 1
        assert "Retry with the analyzer" in pause["question"]
        answer(pid, "revise", guidance="use ASCII only", clarification="the function should return 1")
        result = await asyncio.wait_for(run, 120)
        assert result.done and not result.needs_human_review
        retry_prompt = next(p for p in sc.dev_prompts if "Your previous attempt FAILED" in p)
        assert "use ASCII only" in retry_prompt and "Clarification: the function should return 1" in retry_prompt
        task2 = next(p for p in sc.dev_prompts if "Task: Write func two" in p)
        assert "use ASCII only" not in task2, "task-scoped guidance leaked into the next task"
        ok("qa_failure: shows the analysis, guidance + clarification reach the retry, and stay scoped to that task")


async def s5_skip_task():
    sc = Script(["Write the BAD-FIRST function", "Write func two with a test"],
                dev={"BAD-FIRST": [BROKEN, BROKEN, BROKEN]})
    with Env(sc) as env:
        pid = env.new_project()
        run = start(pid, interactive=True, pause_points=["qa_failure"])
        await wait_paused(pid, "qa_failure")
        answer(pid, "skip_task")
        result = await asyncio.wait_for(run, 120)
        esc = [e for e in events._history[pid] if e["type"] == "escalate"]
        assert esc and esc[0]["data"]["skipped_by_user"] and "skipped at your request" in esc[0]["message"]
        assert result.done and result.needs_human_review and len(result.task_history) == 2
        assert sc.dev_calls["Write the BAD-FIRST function"] == 1, "a skipped task must not be retried"
        assert "Write func two with a test" in sc.dev_calls
        ok("skip_task: task escalated 'by user' without further retries; the next task still runs")


async def s6_abort():
    sc = Script(["Write func one with a test", "Write func two with a test"])
    with Env(sc) as env:
        pid = env.new_project()
        run = start(pid, interactive=True, pause_points=["planner"])
        await wait_paused(pid, "planner")
        answer(pid, "abort")
        result = await asyncio.wait_for(run, 30)
        assert result.aborted and not result.done and len(result.task_list) == 2 and not sc.dev_prompts
        done = [e for e in events._history[pid] if e["type"] == "pipeline_done"][0]
        assert done["data"]["aborted"] and "cancelled" in done["message"].lower()
        assert intervention.get(pid) is None
        ok("abort at the plan: no Developer work, plan kept in the result, pipeline_done(aborted=true)")

        pid2 = env.new_project()
        run = start(pid2, interactive=True, pause_points=["developer"])
        pause = await wait_paused(pid2, "developer")
        fname = pause["payload"]["filename"]
        answer(pid2, "abort")
        result = await asyncio.wait_for(run, 30)
        assert result.aborted and (TMP / pid2 / fname).exists() and "def func_x" in (TMP / pid2 / fname).read_text()
        ok("abort after the first file: the file already written is still on disk (nothing is lost)")


async def s7_timeout():
    sc = Script(["Write func one with a test"])
    old = settings.intervention_timeout_seconds
    settings.intervention_timeout_seconds = 1
    try:
        with Env(sc) as env:
            pid = env.new_project()
            run = start(pid, interactive=True, pause_points=["planner"])
            await wait_paused(pid, "planner")
            result = await asyncio.wait_for(run, 60)       # nobody answers
            assert result.done and not result.aborted
            resumed = [e for e in events._history[pid] if e["type"] == "pipeline_resumed"][0]
            assert resumed["data"]["source"] == "timeout" and "automatically" in resumed["message"]
            assert db.list_interventions(pid)[0]["source"] == "timeout"
            ok("an unanswered pause auto-approves after the timeout (logged as source=timeout)")
    finally:
        settings.intervention_timeout_seconds = old


async def s8_on_demand_pause():
    sc = Script(["Write func one with a test", "Write func two with a test"])
    with Env(sc) as env:
        pid = env.new_project()
        run = start(pid)                      # NOT interactive
        await asyncio.sleep(0)
        assert intervention.get(pid) is not None and intervention.status(pid)["interactive"] is False
        intervention.request_pause(pid, "next")
        pause = await wait_paused(pid)
        assert pause["point"] == "manual" and pause["node"] == "planner"
        assert intervention.status(pid)["pending_pause"] == []
        answer(pid, "approve")
        intervention.request_pause(pid, "developer")
        pause = await wait_paused(pid)
        assert pause["node"] == "developer" and pause["point"] == "manual"
        n = len(sc.dev_prompts)
        answer(pid, "revise", guidance="prefer early returns")
        await wait_paused(pid)                # manual revise at the developer regenerates, then asks again
        assert len(sc.dev_prompts) == n + 1 and "prefer early returns" in sc.dev_prompts[-1]
        answer(pid, "approve")
        result = await asyncio.wait_for(run, 120)
        assert result.done and types(pid).count("pipeline_paused") == 3
        ok("on-demand pause works on a non-interactive run: 'next' pauses after the Planner; a named node pauses there; revise regenerates")

        # cancelling a pending request
        pid2 = env.new_project()
        run = start(pid2)
        await asyncio.sleep(0)
        intervention.request_pause(pid2, "qa")
        intervention.cancel_pause_requests(pid2)
        result = await asyncio.wait_for(run, 120)
        assert result.done and "pipeline_paused" not in types(pid2)
        ok("a cancelled pause request never fires")


async def s9_validation():
    sc = Script(["Write func one with a test"])
    with Env(sc) as env:
        pid = env.new_project()
        run = start(pid, interactive=True, pause_points=["planner", "qa_failure"])
        await asyncio.sleep(0)   # let the run register itself
        try:
            answer(pid, "approve")
            raise AssertionError("answered a run that isn't paused")
        except intervention.NotPaused:
            pass
        await wait_paused(pid, "planner")
        for kwargs, why in [
            (dict(action="skip_task"), "skip_task is not offered at the plan"),
            (dict(action="revise"), "revise needs content"),
            (dict(action="nonsense"), "unknown action"),
        ]:
            try:
                answer(pid, **kwargs)
                raise AssertionError(why)
            except ValueError:
                pass
        assert intervention.status(pid)["paused"], "a rejected answer must leave the pause in place"
        answer(pid, "approve")
        try:
            answer(pid, "approve")
            raise AssertionError("double answer accepted")
        except intervention.NotPaused:
            pass
        try:
            intervention.submit_feedback("no-such-project", intervention.Feedback())
            raise AssertionError
        except KeyError:
            pass
        try:
            intervention.request_pause(pid, "bogus")
            raise AssertionError
        except ValueError:
            pass
        await asyncio.wait_for(run, 120)
        ok("validation: not-paused / invalid action / empty revise / double answer / unknown project / bad pause target; a bad answer leaves the pause intact")

    from app.schemas import UserFeedback, ProjectRequest
    for bad in (dict(action="revise", tasks=[]), dict(action="revise", tasks=["  "]), dict(action="zzz")):
        try:
            UserFeedback(**bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    ProjectRequest(request="x", interactive=True, pause_points=["planner"])
    ok("request/feedback models reject empty edited plans, blank tasks, unknown actions")


async def s10_two_runs_isolated():
    sc = Script(["Write func one with a test"])
    with Env(sc) as env:
        a, b = env.new_project("A"), env.new_project("B")
        ra = start(a, interactive=True, pause_points=["planner"], request="A")
        rb = start(b, interactive=True, pause_points=["planner"], request="B")
        await wait_paused(a, "planner")
        await wait_paused(b, "planner")
        answer(b, "revise", tasks=["Write B-only func with a test"])     # answer B first
        assert intervention.status(a)["paused"], "answering B must not release A"
        answer(a, "revise", tasks=["Write A-first func with a test", "Write A-second func with a test"])
        res_a, res_b = await asyncio.wait_for(asyncio.gather(ra, rb), 180)
        assert [h["task"] for h in res_a.task_history] == ["Write A-first func with a test", "Write A-second func with a test"]
        assert [h["task"] for h in res_b.task_history] == ["Write B-only func with a test"]
        ok("two interactive runs paused at once: each gets only its own answer and keeps its own plan")


async def s11_metrics_exclude_wait(env_pid=None):
    sc = Script(["Write func one with a test"])
    with Env(sc) as env:
        pid = env.new_project()
        t0 = time.time()
        run = start(pid, interactive=True, pause_points=["planner"])
        await wait_paused(pid, "planner")
        await asyncio.sleep(1.2)                      # the "human" thinks
        answer(pid, "approve")
        await asyncio.wait_for(run, 120)
        wall_ms = (time.time() - t0) * 1000
        pm = metrics_report.project_metrics(pid)
        lat = pm["latency"]
        assert lat["paused_ms"] >= 1100, lat
        assert lat["total_ms"] < wall_ms - 1000, (lat["total_ms"], wall_ms)
        planner = next(n for n in pm["nodes"] if n["node"] == "planner")
        assert planner["duration_ms"] < 600, "human wait time leaked into the planner node's latency"
        ok(f"metrics: run waited {lat['paused_ms']} ms for the human, but total_ms={lat['total_ms']} ms and planner node={planner['duration_ms']} ms exclude it")


async def s12_http_end_to_end():
    import httpx
    from app.main import app

    sc = Script(["Write func one with a test", "Write func two with a test"])
    with Env(sc) as env:
        created = []
        real_create = db.create_project

        def capture(pid, name, req):
            created.append(pid)
            real_create(pid, name, req)

        def sqlite_status(pid):
            r = env.rows("SELECT id, name, status FROM projects WHERE id = :p", p=pid)
            return {"project_id": pid, "name": r[0].name, "status": r[0].status, "tasks": []} if r else None

        with patch.object(db, "create_project", capture), patch.object(db, "get_project_status", sqlite_status):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
                # (the response only completes when the background run does, so fire it and drive the run meanwhile)
                post = asyncio.create_task(client.post("/request/async", json={
                    "request": "two funcs", "interactive": True, "pause_points": ["planner", "qa_failure"]}))
                await wait_for(lambda: created, what="project creation")
                pid = created[0]
                await wait_paused(pid, "planner")

                st = (await client.get(f"/projects/{pid}/status")).json()
                assert st["paused"] is True and st["intervention"]["pause"]["point"] == "planner"
                assert len(st["intervention"]["pause"]["payload"]["tasks"]) == 2

                r = await client.post(f"/projects/{pid}/feedback", json={"action": "skip_task"})
                assert r.status_code == 422, r.text
                r = await client.post(f"/projects/{pid}/feedback", json={"action": "revise"})
                assert r.status_code == 422
                r = await client.post(f"/projects/{pid}/feedback", json={"action": "bogus"})
                assert r.status_code == 422
                r = await client.post(f"/projects/{pid}/feedback", json={
                    "action": "revise", "tasks": ["Write only one func with a test"], "clarification": "I meant one function"})
                assert r.status_code == 200 and r.json()["accepted"] and r.json()["point"] == "planner"
                r = await client.post(f"/projects/{pid}/feedback", json={"action": "approve"})
                assert r.status_code in (200, 409)       # 409 if the run already moved on, 200 never (nothing to answer)
                await asyncio.sleep(0.05)

                resp = await asyncio.wait_for(post, 120)
                assert resp.status_code == 200 and resp.json()["interactive"] is True

                hist = (await client.get(f"/projects/{pid}/interventions")).json()["interventions"]
                assert [(h["kind"], h["point"], h["action"]) for h in hist][0] == ("pause", "planner", "revise")
                assert hist[0]["tasks"] == ["Write only one func with a test"] and hist[0]["clarification"] == "I meant one function"
                assert env.rows("SELECT status FROM projects WHERE id = :p", p=pid)[0].status == "done"

                # finished run: pausing/answering is a clean 409, unknown project a 404
                assert (await client.post(f"/projects/{pid}/feedback", json={"action": "approve"})).status_code == 409
                assert (await client.post(f"/projects/{pid}/intervention", json={"pause_after": "next"})).status_code == 409
                assert (await client.post("/projects/nope/intervention", json={"pause_after": "next"})).status_code == 404
                assert (await client.get("/projects/nope/interventions")).status_code == 404
                assert (await client.post(f"/projects/{pid}/intervention", json={"pause_after": "bogus"})).status_code == 422
                r = await client.post("/request", json={"request": "x", "interactive": True})
                assert r.status_code == 422 and "async" in r.json()["detail"]

                # on-demand pause through HTTP, then abort -> project status 'cancelled'
                created.clear()
                post2 = asyncio.create_task(client.post("/request/async", json={"request": "again"}))
                await wait_for(lambda: created, what="second project")
                pid2 = created[0]
                await wait_for(lambda: intervention.get(pid2) is not None, what="run registered")
                r = await client.post(f"/projects/{pid2}/intervention", json={"pause_after": "next"})
                assert r.status_code == 200 and r.json()["pending_pause"] == ["next"]
                assert "pause_requested" in types(pid2)
                await wait_paused(pid2)
                r = await client.post(f"/projects/{pid2}/feedback", json={"action": "abort"})
                assert r.status_code == 200
                await asyncio.wait_for(post2, 60)
                assert env.rows("SELECT status FROM projects WHERE id = :p", p=pid2)[0].status == "cancelled"
                kinds = [h["kind"] for h in (await client.get(f"/projects/{pid2}/interventions")).json()["interventions"]]
                assert "pause_request" in kinds and "pause" in kinds
        ok("HTTP e2e: async interactive run -> status paused -> 422 on bad feedback -> edited plan accepted -> run completes -> history served")
        ok("HTTP: finished run = 409, unknown project = 404, bad target = 422, sync /request refuses interactive; on-demand pause + abort -> status 'cancelled'")


async def main():
    await s1_plan_edit()
    await s2_replan_with_guidance()
    await s3_first_file_review()
    await s4_qa_failure_guidance_and_scope()
    await s5_skip_task()
    await s6_abort()
    await s7_timeout()
    await s8_on_demand_pause()
    await s9_validation()
    await s10_two_runs_isolated()
    await s11_metrics_exclude_wait()
    await s12_http_end_to_end()
    print("\nAll Intervention checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
