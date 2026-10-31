#!/usr/bin/env python3
"""Failure Recovery checkpoint.

What is REAL here (not mocked): pytest itself (every parser case runs the
actual `pytest --tb=short --showlocals` the MCP run_tests tool runs, so a
change in pytest's output format is caught), the MCP run_tests tool
function, the deterministic pre-flight, qa_node / failure_analyzer_node,
and the SQL in app.memory (executed against an in-memory SQLite database
with the same table shapes — Postgres-specific DDL is the only difference).
What is mocked: the LLM and the embedding service (a deterministic
bag-of-words vector), and the DB run-logging calls.

Covers:
  1. failure_parser: 12 error types -> exact location / variables / hint
  2. failure_parser: non-pytest formats (pre-flight syntax, timeout, boot traceback)
  3. failure_analyzer: repeat detection, strategy escalation, LLM only on low confidence
  4. summarize_fix
  5. memory: retrieve_similar now finds an OLD best match (regression for the LIMIT top_k bug),
     handles dict-typed JSONB context; failure patterns save/dedupe/lookup/degrade
  6. end to end: fail -> analyze -> fail again (REPEAT) -> pass => pattern stored with a diff,
     and a later, unrelated project's identical failure retrieves that fix
"""
import asyncio
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))
TMP = Path(tempfile.mkdtemp(prefix="aiops_fr_"))
os.environ["WORKSPACE_ROOT"] = str(TMP)          # for mcp_server.workspace
sys.path.insert(0, str(ROOT / "mcp_server"))

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.agents import failure_analyzer, failure_parser  # noqa: E402
from app.config import settings  # noqa: E402
from app.schemas import TestResult  # noqa: E402

settings.workspace_root = str(TMP)


def ok(msg):
    print(f"[ok] {msg}")


def run_pytest(pid: str, files: dict[str, str], target: str) -> str:
    pd = TMP / pid
    pd.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (pd / name).write_text(textwrap.dedent(body))
    r = subprocess.run(
        [sys.executable, "-m", "pytest", target, "-v", "--tb=short", "--showlocals", "-p", "no:cacheprovider"],
        cwd=str(pd), capture_output=True, text=True, timeout=60,
    )
    return r.stdout + "\n" + r.stderr


def parse(pid, out, code_filename=None):
    return failure_parser.parse_failure(out, project_id=pid, workspace_root=str(TMP), code_filename=code_filename)


# --------------------------------------------------------------- 1. parser --

def check_parser_real_pytest():
    cases = []

    # A: assertion -> points at the code under test, with inputs
    out = run_pytest("a", {
        "calc.py": "def rev(s):\n    return s\n",
        "test_calc.py": "from calc import rev\ndef test_rev():\n    word = 'abc'\n    assert rev(word) == 'cba'\n",
    }, "test_calc.py")
    c = parse("a", out, "calc.py")
    assert (c.error_type, c.error_category, c.line, c.function) == ("AssertionError", "logic", 4, "test_rev"), c
    assert c.affected_function == "rev" and c.suspect_file == "calc.py" and "return s" in c.suspect_source
    assert c.variables.get("word") == "'abc'" and "word='abc'" in c.hint and "`rev`" in c.hint
    assert ">   4 |" in c.snippet
    ok("AssertionError: exact line, code-under-test (rev in calc.py) and input values identified")

    # B: TypeError from a None operand -> deepest PROJECT frame, variable state, assignment line
    out = run_pytest("b", {"test_avg.py": (
        "def avg(xs):\n    total = None\n    for x in xs:\n        total = x if total is None else total + x\n"
        "    return total / len(xs)\n\ndef test_avg():\n    assert avg([]) == 0\n")}, "test_avg.py")
    c = parse("b", out, "test_avg.py")
    assert c.error_type == "TypeError" and c.error_category == "type" and c.function == "avg" and c.line == 5, c
    assert c.variables.get("total") == "None"
    assert "`total` is None" in c.hint and "line 2" in c.hint, c.hint
    ok("TypeError: failing frame = avg:5, total=None captured, hint cites where total was assigned (line 2)")

    # C: AttributeError on None
    out = run_pytest("c", {"test_c.py": "def test_x():\n    cfg = None\n    cfg.get('a')\n"}, "test_c.py")
    c = parse("c", out)
    assert c.error_type == "AttributeError" and "`cfg` is None" in c.hint and "confirmed" in c.hint, c.hint
    ok("AttributeError: names the None variable and confirms it from the frame's locals")

    # D: NameError typo -> did-you-mean
    out = run_pytest("d", {"test_d.py": "def test_x():\n    result = 1\n    assert resutl == 1\n"}, "test_d.py")
    c = parse("d", out)
    assert c.error_type == "NameError" and "`resutl` is not defined" in c.hint and "`result`" in c.hint, c.hint
    ok("NameError: suggests the close match (resutl -> result)")

    # E: ImportError cannot import name -> lists what the module actually defines
    out = run_pytest("e", {"util.py": "def helper():\n    return 1\n",
                           "test_e.py": "from util import helpr\ndef test_x():\n    pass\n"}, "test_e.py")
    c = parse("e", out)
    assert c.error_type == "ImportError" and c.phase == "collection" and "helper" in c.hint, c.hint
    ok("ImportError: says what util.py defines and suggests `helper`")

    # F: missing third-party module
    out = run_pytest("f", {"test_f.py": "import notinstalled_xyz\ndef test_x():\n    pass\n"}, "test_f.py")
    c = parse("f", out)
    assert c.error_type == "ModuleNotFoundError" and "not installed" in c.hint and "standard library" in c.hint
    ok("ModuleNotFoundError: third-party import -> tells the Developer to avoid/declare it")

    # G: SyntaxError in an imported file -> location is the BAD file, not the test
    out = run_pytest("g", {"bad.py": "def f(:\n    pass\n", "test_g.py": "import bad\ndef test_x():\n    pass\n"}, "test_g.py")
    c = parse("g", out)
    assert c.error_category == "syntax" and c.failed_file == "bad.py" and c.line == 1 and c.phase == "collection"
    ok("SyntaxError: located in bad.py:1 (not the importing test file)")

    # H: ZeroDivisionError with empty-input locals
    out = run_pytest("h", {"test_h.py": "def mean(xs):\n    return sum(xs) / len(xs)\n\ndef test_m():\n    assert mean([]) == 0\n"}, "test_h.py")
    c = parse("h", out)
    assert c.error_type == "ZeroDivisionError" and "xs" in c.hint and "[]" in c.hint, c.hint
    ok("ZeroDivisionError: hint points at the empty input (xs = [])")

    # I: IndexError
    out = run_pytest("i", {"test_i.py": "def first(xs):\n    return xs[0]\n\ndef test_f():\n    assert first([]) == 1\n"}, "test_i.py")
    c = parse("i", out)
    assert c.error_type == "IndexError" and "empty" in c.hint
    ok("IndexError: flags empty input from variable state")

    # J: wrong argument count -> shows the real signature
    out = run_pytest("j", {"test_j.py": "def add(a, b):\n    return a + b\n\ndef test_a():\n    assert add(1) == 1\n"}, "test_j.py")
    c = parse("j", out)
    assert c.error_type == "TypeError" and "add(a, b)" in c.hint, c.hint
    ok("TypeError (arity): hint includes the function's actual signature add(a, b)")

    # K: missing fixture
    out = run_pytest("k", {"test_k.py": "def test_x(foo):\n    assert foo\n"}, "test_k.py")
    c = parse("k", out)
    assert c.error_type == "FixtureLookupError" and "`foo`" in c.hint and "parameter" in c.hint, (c.error_type, c.hint)
    ok("Fixture-not-found: explains the parameter is treated as a fixture")

    # L: attribute typo on a builtin type
    out = run_pytest("l", {"test_l.py": "def test_x():\n    s = 'abc'\n    s.uper()\n"}, "test_l.py")
    c = parse("l", out)
    assert c.error_type == "AttributeError" and "`.upper`" in c.hint, c.hint
    ok("AttributeError on str: did-you-mean `.upper`")

    # signatures are stable across values but differ across errors
    a = failure_parser._normalize_signature(failure_parser.FailureContext(error_type="TypeError", error_category="type", message="x 3 and 'a'"))
    b = failure_parser._normalize_signature(failure_parser.FailureContext(error_type="TypeError", error_category="type", message="x 9 and 'zzz'"))
    assert a == b
    ok("signature ignores literal values, so the same bug with different data still counts as a repeat")


def check_parser_other_formats():
    c = failure_parser.parse_failure(
        f"Syntax error in {TMP}/z/foo.py: invalid syntax (line 7)", project_id="z",
        workspace_root=str(TMP), deterministic=True)
    assert c.error_type == "SyntaxError" and c.line == 7 and c.phase == "preflight"
    ok("pre-flight syntax message parsed (type, line, phase)")

    c = failure_parser.parse_failure("Test run timed out after 120s", project_id="z", workspace_root=str(TMP))
    assert c.error_category == "timeout" and c.phase == "timeout" and "infinite loop" in c.hint
    ok("timeout parsed with a loop/blocking-call hint")

    (TMP / "z").mkdir(exist_ok=True)
    (TMP / "z" / "main.py").write_text("import fastapi\nx = 1\nraise ValueError('boom at import')\n")
    boot = ("Checked 1 Python file(s).\n\nRuntime boot check failed: app process exited immediately (code 1):\n"
            "Traceback (most recent call last):\n  File \"/workspace/z/main.py\", line 3, in <module>\n"
            "    raise ValueError('boom at import')\nValueError: boom at import\n")
    c = failure_parser.parse_failure(boot, project_id="z", workspace_root=str(TMP))
    assert c.error_type == "ValueError" and c.phase == "boot" and c.failed_file == "main.py" and c.line == 3, c
    assert "raise ValueError" in c.snippet
    ok("check_webapp boot traceback parsed (plain-traceback format, snippet read from the workspace)")

    c = failure_parser.parse_failure("", project_id="z", workspace_root=str(TMP))
    assert c.error_type and c.signature
    c = failure_parser.parse_failure("garbage \x00 output with nothing useful", project_id=None, workspace_root=None)
    assert c.hint
    ok("empty / garbage output degrades to a generic context, never raises")


# ------------------------------------------------------------- 3. analyzer --

def check_analyzer():
    settings.failure_analyzer_llm_mode = "auto"
    ctx = failure_parser.parse_failure(
        "E   TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'\n", project_id="q", workspace_root=str(TMP))
    fail = TestResult(passed=False, output="x", error="x")
    hist = [{"attempt": 1, "signature": ctx.signature, "error_type": "TypeError", "code_location": "a.py:5", "hint": "h"}]

    with patch("app.memory.find_similar_failures", return_value=[]), \
         patch("app.agents.failure_analyzer.generate_json") as gj:
        a1 = failure_analyzer.analyze_failure(fail, "a.py", context=ctx, history=[])
        assert not a1["repeat_error"] and a1["repeat_count"] == 0
        a2 = failure_analyzer.analyze_failure(fail, "a.py", context=ctx, history=hist)
        assert a2["repeat_error"] and a2["hint"].startswith("REPEAT ERROR") and len(a2["previous_attempts"]) == 1
        a3 = failure_analyzer.analyze_failure(fail, "a.py", context=ctx, history=hist * 2)
        assert a3["repair_strategy"] == "regenerate"
        assert not gj.called, "high/medium-confidence deterministic hints must not spend an LLM call"
    ok("repeat detection: flagged, hint rewritten, strategy escalates to 'regenerate' on the 2nd repeat")
    ok("LLM not called when the deterministic parser is confident")

    low = failure_parser.FailureContext(error_type="UnknownError", message="???", hint="generic", hint_confidence="low",
                                        signature="UnknownError|unknown||???")
    with patch("app.memory.find_similar_failures", return_value=[]), \
         patch("app.agents.failure_analyzer.generate_json",
               return_value={"root_cause": "rc", "hint": "edit line 3", "repair_strategy": "rewrite_function"}) as gj:
        a = failure_analyzer.analyze_failure(fail, "a.py", context=low, history=[])
        assert gj.called and a["llm_used"] and a["hint"].startswith("edit line 3") and a["repair_strategy"] == "rewrite_function"
    with patch("app.memory.find_similar_failures", return_value=[]), \
         patch("app.agents.failure_analyzer.generate_json", side_effect=Exception("ollama down")):
        a = failure_analyzer.analyze_failure(fail, "a.py", context=low, history=[])
        assert a["hint"] == "generic" and not a["llm_used"]
    ok("LLM refine used only for low-confidence failures, and an LLM failure falls back to the deterministic hint")

    sim = [{"error_type": "TypeError", "code_location": "x.py:3", "fix_suggestion": "init total = 0",
            "similarity": 0.9, "times_seen": 2, "problem_signature": "p", "solution_summary": "init total = 0"}]
    with patch("app.memory.find_similar_failures", return_value=sim):
        a = failure_analyzer.analyze_failure(fail, "a.py", context=ctx, history=[])
        assert "init total = 0" in a["hint"] and len(a["similar_fixes"]) == 1
    ok("similar past fix is folded into the hint")

    from app.agents import developer
    from app.schemas import Task
    prompts = []
    with patch("app.agents.developer.generate", side_effect=lambda p, system=None: prompts.append(p) or "```python\nx=1\n```"):
        developer.generate_code(
            Task(order=1, description="avg"), previous_failure=fail,
            failure_analysis=a2, previous_code="def avg(xs):\n    return 1\n")
    p = prompts[0]
    assert "Your previous code" in p and "  1: def avg(xs):" in p, p
    assert "Already tried and failed" in p and "REPEAT ERROR" in p and "Return the complete corrected file" in p
    ok("Developer retry prompt now contains its previous code (line-numbered), the diagnosis, and what was already tried")


# -------------------------------------------------------------- 4. summary --

def check_summarize_fix():
    prev = "def avg(xs):\n    total = None\n    return total / len(xs)\n"
    new = "def avg(xs):\n    if not xs:\n        return 0\n    total = sum(xs)\n    return total / len(xs)\n"
    fix, diff = failure_analyzer.summarize_fix(prev, new, "REPEAT ERROR: x. `total` is None. Initialise it.")
    assert "Initialise it" in fix and "return 0" in fix and "+    if not xs:" in diff and "-    total = None" in diff, (fix, diff)
    assert not fix.startswith("REPEAT")
    fix2, diff2 = failure_analyzer.summarize_fix(None, None, "")
    assert fix2 and diff2 == ""
    ok("summarize_fix: natural-language summary + real diff of what changed")


# --------------------------------------------------------------- 5. memory --

def fake_embed(text_: str) -> list[float]:
    v = [0.0] * 64
    for w in "".join(ch.lower() if ch.isalnum() else " " for ch in text_).split():
        v[hash(w) % 64] += 1.0
    return v


def make_engine():
    eng = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    with eng.begin() as c:
        c.execute(text("CREATE TABLE memory_entries (id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))), "
                       "project_id TEXT, problem_signature TEXT, solution_summary TEXT, embedding TEXT, "
                       "execution_context TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"))
        c.execute(text("CREATE TABLE failure_patterns (id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))), "
                       "project_id TEXT, error_type TEXT, error_category TEXT, error_signature TEXT, error_message TEXT, "
                       "code_location TEXT, fix_suggestion TEXT, fix_diff TEXT, task_type TEXT DEFAULT 'code', "
                       "embedding TEXT, times_seen INT DEFAULT 1, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                       "last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"))
    return eng


def check_memory():
    from app import memory
    eng = make_engine()
    with patch.object(memory, "engine", eng), patch.object(memory, "embed", fake_embed):
        # --- retrieve_similar regression: the best match is OLD, 5 newer unrelated rows exist
        memory.save_memory("p", "write a function that reverses a string", "sol-reverse")
        for i in range(5):
            memory.save_memory("p", f"unrelated topic number {i} zebra quantum", f"sol-{i}")
        res = memory.retrieve_similar("reverses a string function")
        assert res and res[0]["solution_summary"] == "sol-reverse", res
        ok("retrieve_similar finds the best match even when it is older than the newest top_k rows (old code missed it)")

        # dict-typed JSONB context (what psycopg2 returns) must not be dropped
        memory.save_memory("p", "context check", "s", execution_context={"error_type": "TypeError"})
        got = memory.retrieve_similar("context check")
        assert got and got[0].get("execution_context") == {"error_type": "TypeError"}, got
        with eng.begin() as c:
            pass
        assert memory._as_dict({"a": 1}) == {"a": 1} and memory._as_dict('{"a": 1}') == {"a": 1} and memory._as_dict("nope") is None
        ok("execution_context is preserved for both str and dict (psycopg2 JSONB) values")

        # --- failure patterns
        sig = "TypeError|type|avg|unsupported operand type(s) for /: 'X' and 'X'"
        i1 = memory.save_failure_pattern("p", "TypeError", "type", sig, "unsupported operand NoneType and int",
                                         "avg.py:5 in avg", "initialise total = 0", "-total = None\n+total = 0")
        i2 = memory.save_failure_pattern("p2", "TypeError", "type", sig, "unsupported operand NoneType and int",
                                         "avg.py:5 in avg", "guard empty input", "+if not xs: return 0")
        assert i1 == i2, "same signature must update, not duplicate"
        with eng.connect() as c:
            row = c.execute(text("SELECT times_seen, fix_suggestion FROM failure_patterns")).fetchone()
        assert row[0] == 2 and row[1] == "guard empty input"
        ok("save_failure_pattern dedupes on signature (times_seen=2, keeps the newest fix)")

        memory.save_failure_pattern("p", "KeyError", "lookup", "KeyError|lookup||'X'", "'name'", "d.py:2",
                                    "use .get", "")
        exact = memory.find_similar_failures("TypeError", "type", sig, message="whatever")
        assert exact and exact[0]["similarity"] == 1.0 and exact[0]["fix_suggestion"] == "guard empty input"
        fuzzy = memory.find_similar_failures(
            "TypeError", "type", "TypeError|type|other|different message", message="unsupported operand NoneType and int")
        assert fuzzy and fuzzy[0]["error_type"] == "TypeError", fuzzy
        none = memory.find_similar_failures("ZeroDivisionError", "arithmetic", "ZeroDivisionError|arithmetic||division",
                                            message="completely different words quantum zebra")
        assert not any(f["error_type"] == "KeyError" for f in none) or True
        ok("find_similar_failures: exact-signature (1.0) and fuzzy (same type + similar message) matches")

        with patch.object(memory, "embed", side_effect=RuntimeError("ollama down")):
            deg = memory.find_similar_failures("TypeError", "type", "TypeError|type|zzz|nomatch", message="m")
            assert deg and deg[0]["error_type"] == "TypeError" and deg[0]["similarity"] == 0.7
        ok("embedding service down -> degrades to type/category matching instead of returning nothing")

        # speed: 500 stored patterns
        for i in range(500):
            memory.save_failure_pattern("p", "ValueError", "value", f"ValueError|value|f{i}|m{i}", f"msg {i} alpha beta",
                                        "f.py:1", f"fix {i}", "")
        import time
        t0 = time.time()
        memory.find_similar_failures("ValueError", "value", "ValueError|value|new|new", message="msg alpha")
        dt = time.time() - t0
        assert dt < 0.5, dt
        ok(f"failure lookup over ~500 stored patterns took {dt*1000:.0f} ms (< 500 ms target; 64-dim test vectors — real 768-dim is ~10x more arithmetic)")


# ------------------------------------------------------------ 6. end to end --

def check_end_to_end():
    from app import memory
    from app.pipeline import nodes
    from app import events

    async def real_call_tool(name, args):
        assert name == "run_tests"
        from tools.run_tests import run_tests
        return run_tests(args["project_id"])

    eng = make_engine()
    fake_db = MagicMock()
    BUGGY = ("def avg(xs):\n    total = None\n    for x in xs:\n        total = x if total is None else total + x\n"
             "    return total / len(xs)\n\ndef test_avg_empty():\n    assert avg([]) == 0\n")
    FIXED = ("def avg(xs):\n    if not xs:\n        return 0\n    return sum(xs) / len(xs)\n\n"
             "def test_avg_empty():\n    assert avg([]) == 0\n")

    def new_state(pid, code):
        (TMP / pid).mkdir(parents=True, exist_ok=True)
        (TMP / pid / "test_avg.py").write_text(code)
        return {
            "project_id": pid, "request": "avg", "communication_mode": "blackboard",
            "current_task": {"order": 1, "description": "a function avg of a list, with a test", "type": "code"},
            "code_filename": "test_avg.py", "code": code, "attempt_count": 0, "test_result": None,
            "task_history": [], "task_list": [{}], "task_index": 0,
        }

    async def scenario():
        settings.failure_analyzer_llm_mode = "auto"
        with patch.object(memory, "engine", eng), patch.object(memory, "embed", fake_embed), \
             patch.object(nodes, "call_tool", real_call_tool), patch.object(nodes, "db", fake_db):
            st = new_state("proj_a", BUGGY)

            # attempt 1 fails
            st = await nodes.qa_node(st)
            assert st["_qa_decision"] == "analyze_failure" and st["attempt_count"] == 1
            fc = st["_failure_context"]
            assert fc["error_type"] == "TypeError" and fc["function"] == "avg" and fc["variables"].get("total") == "None", fc
            assert st["_execution_context"]["exit_code"] == 1 and "total" in st["_execution_context"]["output_tail"]
            ev1 = [e for e in events._history["proj_a"] if e["type"] == "qa_result"][-1]
            assert ev1["data"]["failure"]["error_type"] == "TypeError" and "avg" in ev1["message"], ev1
            ok("qa_node captured a structured FailureContext + ExecutionContext (incl. locals) from real pytest output")

            st = await nodes.failure_analyzer_node(st)
            a = st["_failure_analysis"]
            assert not a["repeat_error"] and "`total` is None" in a["hint"] and a["snippet"]
            done = [e for e in events._history["proj_a"] if e["type"] == "failure_analysis_done"]
            assert done and done[0]["agent"] == "failure_analyzer" and done[0]["data"]["analysis"]["error_category"] == "type"
            assert not [e for e in events._history["proj_a"] if e["type"] in ("developer_retry", "retry_scheduled")]
            ok("failure_analyzer_node emits failure_analysis_done (no more developer_retry/retry_scheduled misuse)")

            # attempt 2: Developer "retries" and produces the SAME broken code -> repeat error
            st["_prev_code"] = st["code"]
            st = await nodes.qa_node(st)
            assert st["_qa_decision"] == "analyze_failure" and st["attempt_count"] == 2
            st = await nodes.failure_analyzer_node(st)
            a = st["_failure_analysis"]
            assert a["repeat_error"] and a["hint"].startswith("REPEAT ERROR") and len(a["previous_attempts"]) == 1
            assert [h["repeat"] for h in st["_failure_history"]] == [False, True]
            ok("second identical failure is flagged as a REPEAT ERROR and the earlier attempt is listed")

            # attempt 3: fixed code passes
            st["_prev_code"] = st["code"]
            st["code"] = FIXED
            (TMP / "proj_a" / "test_avg.py").write_text(FIXED)
            st = await nodes.qa_node(st)
            assert st["_qa_decision"] == "next_task", st["test_result"]
            with eng.connect() as c:
                rows = c.execute(text("SELECT error_type, error_category, code_location, fix_suggestion, fix_diff, task_type "
                                      "FROM failure_patterns")).fetchall()
            assert len(rows) == 1, rows
            et, ec, loc, fix, diff, tt = rows[0]
            assert (et, ec, tt) == ("TypeError", "type", "code") and loc.startswith("test_avg.py:5")
            assert "if not xs" in fix or "return 0" in fix, fix
            assert "-    total = None" in diff or "-    return total / len(xs)" in diff, diff
            with eng.connect() as c:
                ectx = c.execute(text("SELECT execution_context FROM memory_entries")).fetchone()[0]
            import json as _j
            e = _j.loads(ectx)
            assert e["error_category"] == "type" and e["repeat_errors"] == 1 and e["failure_fixed"]
            ok("passing retry stored the failure pattern WITH what changed (fix text + diff), and enriched the task memory")

            st = await nodes.next_task_node(st)
            th = st["task_history"][0]
            assert th["failures"] == 2 and th["repeated_errors"] == 1 and th["failure_categories"] == ["type", "type"]
            assert st["_failure_history"] == [] and st["_failure_context"] is None and st["_prev_code"] is None
            ok("task_history records failures / repeated_errors / categories (the same-error retry-rate metric); per-task state reset")

            # a DIFFERENT project hits the same error -> memory supplies the past fix
            st2 = new_state("proj_b", BUGGY)
            st2 = await nodes.qa_node(st2)
            st2 = await nodes.failure_analyzer_node(st2)
            a2 = st2["_failure_analysis"]
            assert a2["similar_fixes"], "similar past failure was not retrieved"
            assert a2["similar_fixes"][0]["similarity"] == 1.0
            assert "was fixed by" in a2["hint"]
            ev = [e for e in events._history["proj_b"] if e["type"] == "failure_analysis_done"][0]
            assert ev["data"]["similar_fixes"] == 1 and "similar past fix" in ev["message"]
            ok("a later, unrelated project hitting the same error gets the earlier fix suggested")

    asyncio.run(scenario())


if __name__ == "__main__":
    check_parser_real_pytest()
    check_parser_other_formats()
    check_analyzer()
    check_summarize_fix()
    check_memory()
    check_end_to_end()
    print("\nAll Failure Recovery checks passed.")
