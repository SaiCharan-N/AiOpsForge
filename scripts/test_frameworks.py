#!/usr/bin/env python3
"""Framework-agnostic scaffolding checkpoint (FastAPI, Flask, Django, Express, Next.js).

  A  (offline logic)  detection, registry/templates, scaffolder with a fake LLM
                      (locked files, backfill, dependency ensuring, fallback),
                      planner routing, Developer note, API validation, /frameworks
  B  (REAL runtimes)  every starter project passes the real QA (compile/syntax ->
                      the framework's own tests -> boot + HTTP GET /), and each
                      realistic breakage FAILS QA and is DIAGNOSED correctly by the
                      failure parser (error type, file:line, a specific hint)
  C  (full pipeline)  run_pipeline with a fake Ollama for each framework: the
                      right files are scaffolded, QA uses the right runner, and a
                      broken first scaffold is diagnosed and repaired on retry

Part B/C need real dependencies. They are built ONCE into /tmp/aiops_fw_* (pip +
npm, needs network) and reused; if that isn't possible they are skipped with a
notice. Next.js (a ~240 MB install + real `next build`) is opt-in:
    AIOPS_TEST_NEXT=1 python scripts/test_frameworks.py
The per-project dependency *install* step (venv / npm install) is replaced in
Part C by linking the shared environment (otherwise each run would re-download
the packages); everything else — MCP tool functions, pytest, manage.py test,
node --test, uvicorn/flask/runserver/node servers, HTTP — is real.
"""
import asyncio
import json
import os
import shutil
import subprocess
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
WS = Path(tempfile.mkdtemp(prefix="aiops_fw_ws_"))
os.environ["WORKSPACE_ROOT"] = str(WS)
sys.path.insert(0, str(ROOT / "mcp_server"))

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app import db, events, mcp_client, memory  # noqa: E402
from app.agents import developer, failure_parser, planner, scaffolder  # noqa: E402
from app.config import settings  # noqa: E402
from app.llm import LLMOutputError  # noqa: E402
from app.pipeline import graph as graph_mod, nodes  # noqa: E402
from app.scaffolds import FRAMEWORKS, SUPPORTED, get_framework, prompts  # noqa: E402
from app.schemas import ProjectRequest, Task  # noqa: E402
from app.webapp_detect import detect_framework, is_webapp_request, resolve_framework, validate_framework  # noqa: E402
from tools import ALL_TOOLS  # noqa: E402
from tools.check_webapp import _check_webapp_impl  # noqa: E402

settings.workspace_root = str(WS)
TOOL_FNS = {f.__name__: f for f in ALL_TOOLS}
RUN_NEXT = os.environ.get("AIOPS_TEST_NEXT") == "1"
ENV_DIR = Path(os.environ.get("AIOPS_TEST_ENV_DIR", "/tmp"))
VENV, NODE_ENV, NEXT_ENV = ENV_DIR / "aiops_fw_venv", ENV_DIR / "aiops_fw_node", ENV_DIR / "aiops_fw_next"


def ok(msg):
    print(f"[ok] {msg}")


# =========================================================== A. offline logic ==

DETECTION = [
    ("build a Flask app with a home page and login", "flask", True),
    ("Create a Django blog with posts and comments", "django", True),
    ("Make an Express server with a /health endpoint", "express", True),
    ("Build a Node.js REST API for todos", "express", True),
    ("Make a Next.js landing page", "next", True),
    ("build a React app for tracking habits", "next", True),
    ("a fastapi app with a login page", "fastapi", True),
    ("Write a Flask route /hello returning JSON, with a test", "flask", False),
    ("Write a FastAPI endpoint that adds two numbers, with a test", "fastapi", False),
    ("Use Flask, not Django, for a bookstore app", "flask", True),
    ("Not Django — use Flask for a blog website", "flask", True),
    ("Build a bakery website without Django", None, True),
    ("a function that checks if a string is a palindrome, with a test", None, False),
    ("Write a function to express a number in binary", None, False),
    ("Implement a linked list with a Node class", None, False),
    ("Go to the next page of results", None, False),
    ("Build a todo website", None, True),
]


def a_detection():
    for text_, fw, web in DETECTION:
        assert detect_framework(text_) == fw and is_webapp_request(text_) == web, (text_, detect_framework(text_), is_webapp_request(text_))
    ok(f"detection: {len(DETECTION)} phrasings (negation, ambiguous words like express/next/node, route-vs-project)")
    assert validate_framework(None) is None and validate_framework("auto") is None and validate_framework("") is None
    assert validate_framework("Flask") == "flask" and validate_framework("Next.js") == "next" and validate_framework("node") == "express"
    try:
        validate_framework("rails")
        raise AssertionError
    except ValueError as e:
        assert "supported:" in str(e) and "flask" in str(e)
    assert resolve_framework("build a Django site", "flask") == "flask" and resolve_framework("build a Django site") == "django"
    ok("explicit choice validated/normalised (Next.js, node -> express), unknown rejected with the supported list, explicit beats detected")


def a_registry():
    assert SUPPORTED == ("fastapi", "flask", "django", "express", "next")
    for fw in FRAMEWORKS.values():
        tmpl = fw.template_files()
        missing = [r for r in fw.required if r not in tmpl]
        assert not missing, (fw.id, missing)
        # a completely empty model answer is repaired into a COMPLETE required set
        full = fw.backfill({})
        assert all(r in full for r in fw.required), fw.id
        for prefix, suffix in fw.required_any:
            assert any(p.startswith(prefix) and p.endswith(suffix) for p in full), (fw.id, prefix)
        assert fw.entry_file in tmpl and fw.dep_file in tmpl
    assert scaffolder._FALLBACK_FILES == get_framework("fastapi").template_files()
    assert prompts.FASTAPI_SYSTEM == scaffolder.SCAFFOLDER_SYSTEM_PROMPT
    ok("registry: 5 frameworks; every starter project satisfies its own required-file contract; FastAPI legacy constants intact")

    flask, django, express, nxt = (FRAMEWORKS[k] for k in ("flask", "django", "express", "next"))
    # python deps: append what's missing, keep what's there, idempotent
    out = flask.ensure_dependencies({"requirements.txt": "Flask==3.0.0\nrequests\n"})
    assert out["requirements.txt"].splitlines() == ["Flask==3.0.0", "requests", "pytest"], out
    assert flask.ensure_dependencies(out) == out
    assert flask.ensure_dependencies({})["requirements.txt"].split() == ["flask", "pytest"]
    legacy = get_framework("fastapi").ensure_dependencies({"requirements.txt": "fastapi\n"})
    assert legacy == {"requirements.txt": "fastapi\n"}, "FastAPI must stay legacy (requirements untouched)"
    # node deps
    fixed = json.loads(express.ensure_dependencies({"package.json": '{"name": "x", "dependencies": {"cors": "^2"}}'})["package.json"])
    assert fixed["dependencies"]["express"] and fixed["dependencies"]["cors"] == "^2" and fixed["scripts"]["test"] == "node --test" and fixed["main"] == "server.js"
    assert json.loads(express.ensure_dependencies({"package.json": "{not json"})["package.json"])["dependencies"]["express"]
    n = json.loads(nxt.ensure_dependencies({"package.json": '{"name": "x"}'})["package.json"])
    assert {"next", "react", "react-dom"} <= set(n["dependencies"]) and {"dev", "build", "start"} <= set(n["scripts"])
    ok("ensure_dependencies: appends missing pip packages, repairs/fills package.json (invalid JSON -> starter), FastAPI untouched")

    # locked files
    kept = django.drop_locked({"config/settings.py": "BAD", "home/views.py": "ok", "manage.py": "BAD"})
    assert kept == {"home/views.py": "ok"}
    ok("locked (platform-owned) files: the model's Django settings.py / manage.py are dropped, home/ is kept")


class FakeLLM:
    """Replaces scaffolder.generate_json; records (prompt, system)."""

    def __init__(self, files=None, error=None):
        self.files, self.error, self.calls = files, error, []

    def __call__(self, prompt, system=None):
        self.calls.append((prompt, system))
        if self.error:
            raise self.error
        return {"files": self.files}


def a_scaffolder():
    marker = {"flask": "Flask rules:", "django": "Django rules:", "express": "Express rules:", "next": "Next.js rules:"}
    for fid, mk in marker.items():
        fake = FakeLLM({"x.txt": "y"})
        with patch.object(scaffolder, "generate_json", fake):
            files = scaffolder.scaffold_webapp("a small site", framework=fid)
        prompt, system = fake.calls[0]
        fw = FRAMEWORKS[fid]
        assert mk in system and system == fw.system_prompt
        assert f"A starter {fw.label} project is already on disk" in prompt and "--- " in prompt, fid
        assert all(r in files for r in fw.required) and files["x.txt"] == "y", fid
    fake = FakeLLM({"main.py": "m", "requirements.txt": "fastapi", "templates/a.html": "a", "static/s.css": "s"})
    with patch.object(scaffolder, "generate_json", fake):
        files = scaffolder.scaffold_webapp("site")             # default = legacy FastAPI behaviour
    assert set(files) == {"main.py", "requirements.txt", "templates/a.html", "static/s.css"}
    assert "starter" not in fake.calls[0][0] and fake.calls[0][1] == scaffolder.SCAFFOLDER_SYSTEM_PROMPT
    ok("scaffolder: each framework gets ITS system prompt + a starter excerpt; required files backfilled; FastAPI default unchanged")

    fake = FakeLLM({"config/settings.py": "HALLUCINATED = 1", "home/views.py": "VIEWS", "home/urls.py": "URLS", "manage.py": "X"})
    with patch.object(scaffolder, "generate_json", fake):
        files = scaffolder.scaffold_webapp("blog", framework="django")
    assert "HALLUCINATED" not in files["config/settings.py"] and files["home/views.py"] == "VIEWS"
    assert files["manage.py"] == FRAMEWORKS["django"].template_files()["manage.py"]
    fake = FakeLLM({"app.py": "from flask import Flask\napp = Flask(__name__)\n", "requirements.txt": "flask\n"})
    with patch.object(scaffolder, "generate_json", fake):
        files = scaffolder.scaffold_webapp("site", framework="flask")
    assert "pytest" in files["requirements.txt"] and "test_app.py" in files and "templates/index.html" in files
    fake = FakeLLM({".venv/bin/python": "x", "node_modules/a.js": "x", "../evil.py": "x", "app.py": "ok"})
    with patch.object(scaffolder, "generate_json", fake):
        files = scaffolder.scaffold_webapp("site", framework="flask")
    assert not any(p.startswith((".venv", "node_modules", "..")) for p in files) and files["app.py"] == "ok"
    ok("scaffolder: platform-owned Django files restored, missing test deps added to requirements, .venv/node_modules/../ paths refused")

    for fid in SUPPORTED:
        with patch.object(scaffolder, "generate_json", FakeLLM(error=LLMOutputError("bad json"))):
            files = scaffolder.scaffold_webapp("site", framework=fid)
        assert files == FRAMEWORKS[fid].template_files(), fid
    with patch.object(scaffolder, "generate_json", FakeLLM({"../x": "y"})):
        assert scaffolder.scaffold_webapp("site", framework="express") == FRAMEWORKS["express"].template_files()
    with patch.object(scaffolder, "generate_json", FakeLLM({"a.txt": "x"})):
        assert scaffolder.scaffold_webapp("site", framework="not-a-framework").keys() >= {"main.py"}   # unknown id -> FastAPI default
    ok("unusable model output -> the framework's own complete starter project (all 5); unknown id degrades to FastAPI")


def a_planner_and_developer():
    with patch.object(planner, "generate", return_value="A site with a home page and login."):
        t = planner.plan("build a Flask app with a home page and login")
        assert len(t) == 1 and t[0].type == "webapp_scaffold" and t[0].framework == "flask"
        t = planner.plan("a todo site", None, "django")                      # explicit choice -> web app in that framework
        assert t[0].type == "webapp_scaffold" and t[0].framework == "django"
        t = planner.plan("Make a Next.js landing page")
        assert t[0].framework == "next"
        t = planner.plan("Use Flask for a notes website", None, "express")   # explicit beats the text
        assert t[0].framework == "express"
        t = planner.plan("Build a bakery website")
        assert t[0].type == "webapp_scaffold" and t[0].framework == "fastapi"   # web app naming no framework -> default
    with patch.object(planner, "generate_json", return_value={"tasks": ["Write a Flask route /hello returning JSON, with a test"]}):
        t = planner.plan("Write a Flask route /hello returning JSON, with a test")
        assert t[0].type == "code" and t[0].framework == "flask"
    with patch.object(planner, "generate_json", return_value={"tasks": ["Write is_palindrome with a test"]}):
        assert planner.plan("a palindrome checker, with a test")[0].framework is None
    ok("planner: detected / explicit / default framework on web-app tasks; code tasks keep a named framework, else None")

    prompts_seen = []
    with patch.object(developer, "generate", side_effect=lambda p, system=None: prompts_seen.append(p) or "```python\nx = 1\n```"):
        developer.generate_code(Task(order=1, description="Write a Flask route /hello, with a test", framework="flask"))
        developer.generate_code(Task(order=1, description="Write is_palindrome, with a test"))
    assert "Framework: Flask" in prompts_seen[0] and "app.test_client()" in prompts_seen[0]
    assert "Framework:" not in prompts_seen[1]
    ok("Developer prompt carries the framework's idioms + test client for a Flask task, and nothing for a framework-less task")


def a_api():
    assert ProjectRequest(request="x", framework="Flask").framework == "flask"
    assert ProjectRequest(request="x", framework="auto").framework is None
    try:
        ProjectRequest(request="x", framework="rails")
        raise AssertionError
    except ValueError:
        pass
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    j = c.get("/frameworks").json()
    assert j["default"] == "fastapi" and [f["id"] for f in j["frameworks"]] == list(SUPPORTED)
    assert all({"id", "label", "language", "entry_file", "description"} <= set(f) for f in j["frameworks"])
    for path in ("/request", "/request/async"):
        r = c.post(f"{path}?framework=rails", json={"request": "build a website"})
        assert r.status_code == 422 and "supported:" in r.json()["detail"], (path, r.text)
        r = c.post(path, json={"request": "build a website", "framework": "rails"})
        assert r.status_code == 422
    ok("API: GET /frameworks lists all 5; invalid framework (query or body) -> 422 naming the supported ones")


# =========================================================== B. real runtimes ==

def _run(cmd, cwd=None, timeout=300):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def ensure_envs() -> bool:
    """Builds the shared python venv + node_modules once. False if that isn't possible."""
    try:
        py = VENV / "bin" / "python"
        have_py = py.exists() and _run([str(py), "-c", "import flask, django, fastapi, uvicorn, jinja2, pytest, httpx"]).returncode == 0
        if not have_py:
            print("     (building the shared virtualenv: flask, django, fastapi, uvicorn, pytest, httpx ...)")
            shutil.rmtree(VENV, ignore_errors=True)
            assert _run([sys.executable, "-m", "venv", str(VENV)]).returncode == 0
            r = _run([str(py), "-m", "pip", "install", "--quiet", "flask", "pytest", "django", "fastapi", "uvicorn",
                      "jinja2", "httpx", "python-multipart"], timeout=600)
            assert r.returncode == 0, r.stderr[-500:]
        if not (NODE_ENV / "node_modules" / "express").is_dir():
            if not shutil.which("npm") or not shutil.which("node"):
                print("     (node/npm not found: Express checks will be skipped)")
                return True
            print("     (installing express into the shared node_modules ...)")
            NODE_ENV.mkdir(parents=True, exist_ok=True)
            (NODE_ENV / "package.json").write_text('{"name": "shared", "version": "1.0.0", "dependencies": {"express": "^4.19.2"}}')
            r = _run(["npm", "install", "--no-audit", "--no-fund", "--loglevel=error"], cwd=NODE_ENV, timeout=300)
            assert r.returncode == 0, r.stderr[-500:]
        if RUN_NEXT and not (NEXT_ENV / "node_modules" / "next").is_dir():
            print("     (installing next/react into the shared node_modules — a few minutes the first time ...)")
            NEXT_ENV.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / "backend/app/scaffolds/templates/next/package.json", NEXT_ENV / "package.json")
            r = _run(["npm", "install", "--no-audit", "--no-fund", "--loglevel=error"], cwd=NEXT_ENV, timeout=600)
            assert r.returncode == 0, r.stderr[-500:]
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"     (could not prepare the shared environments: {exc})")
        return False


def link_env(pd: Path, fw_id: str) -> None:
    if FRAMEWORKS[fw_id].language == "python":
        (pd / ".venv").symlink_to(VENV)
    elif fw_id == "express":
        (pd / "node_modules").symlink_to(NODE_ENV / "node_modules")
    elif fw_id == "next":
        (pd / "node_modules").symlink_to(NEXT_ENV / "node_modules")


def make_project(fw_id: str, edits: dict | None = None, files: dict | None = None) -> tuple[str, Path]:
    pid = str(uuid.uuid4())
    pd = WS / pid
    for p, c in (files or FRAMEWORKS[fw_id].template_files()).items():
        (pd / p).parent.mkdir(parents=True, exist_ok=True)
        (pd / p).write_text(c)
    for path, (old, new) in (edits or {}).items():
        t = (pd / path).read_text()
        assert old in t, (path, old)
        (pd / path).write_text(t.replace(old, new))
    link_env(pd, fw_id)
    return pid, pd


def qa(fw_id, pid, pd):
    t = time.time()
    out = _check_webapp_impl(pd, fw_id)
    return out, time.time() - t


def diagnose(out: str, pid: str, fw_id: str):
    body = out.split("OUTPUT:\n", 1)[1]
    return failure_parser.parse_failure(body, project_id=pid, workspace_root=str(WS),
                                        code_filename=FRAMEWORKS[fw_id].entry_file)


# (framework, case, edits, expected error type, expected category, expected file, hint fragments)
BREAKAGES = [
    ("flask", "undefined variable inside a view (surfaces as a 500, the test only sees assert 500 == 200)",
     {"app.py": ('title="My Site"', "title=site_title")}, "NameError", "name", "app.py", ["site_title", "not defined", "test itself only saw"]),
    ("flask", "template that does not exist",
     {"templates/index.html": ("{{ title }}</h1>", "{{ title }}</h1>{% include 'nope.html' %}")}, "TemplateNotFound", "template", "index.html", ["nope.html"]),
    ("django", "template name that does not exist",
     {"home/views.py": ('"home/index.html"', '"home/missing.html"')}, "TemplateDoesNotExist", "template", "views.py", ["home/missing.html", "was not found"]),
    ("django", "urls.py routes to a view that does not exist",
     {"home/urls.py": ("views.index", "views.homepage")}, "AttributeError", "attribute", "urls.py", ["homepage"]),
    ("express", "ReferenceError in a route (the static middleware must not mask it)",
     {"server.js": ('res.sendFile(path.join(__dirname, "public", "index.html"));', 'res.sendFile(path.join(__dirname, "public", pageName));')},
     "ReferenceError", "name", "server.js", ["pageName", "not defined"]),
    ("express", "syntax error",
     {"server.js": ("const app = express();", "const app = express(;")}, "SyntaxError", "syntax", "server.js", ["valid JavaScript"]),
    ("express", "require() of a package that is not in package.json",
     {"server.js": ('const path = require("path");', 'const path = require("path");\nconst lodash = require("lodash");')},
     "Error", "runtime", "server.js", ["lodash", "package.json"]),
    ("fastapi", "old Starlette TemplateResponse(name, {request}) signature (the fallback template used to ship it)",
     {"main.py": ('request, "index.html", {"title": "My Site"}', '"index.html", {"request": request, "title": "My Site"}')},
     "TypeError", "type", "main.py", ["TemplateResponse", "FIRST argument"]),
]


def b_real_qa():
    expectations = {
        "fastapi": ["HTTP 200"],
        "flask": ["tests passed (pytest)", "HTTP 200"],
        "django": ["tests passed (manage.py test)", "HTTP 200"],
        "express": ["tests passed (npm test)", "HTTP 200"],
    }
    for fw, needles in expectations.items():
        if fw == "express" and not (NODE_ENV / "node_modules" / "express").is_dir():
            print("[skip] express (node/npm unavailable)")
            continue
        pid, pd = make_project(fw)
        out, secs = qa(fw, pid, pd)
        assert out.startswith("PASSED: true"), out
        for n in needles:
            assert n in out, (fw, n, out)
        ok(f"{fw}: starter project passes REAL QA in {secs:.1f}s -> " + " | ".join(l for l in out.splitlines()[3:] if l))

    for fw, case, edits, etype, cat, fname, frags in BREAKAGES:
        if fw == "express" and not (NODE_ENV / "node_modules" / "express").is_dir():
            continue
        pid, pd = make_project(fw, edits)
        out, secs = qa(fw, pid, pd)
        assert out.startswith("PASSED: false"), (case, out)
        ctx = diagnose(out, pid, fw)
        assert (ctx.error_type, ctx.error_category) == (etype, cat), (case, ctx.error_type, ctx.error_category, out[-600:])
        assert fname in (ctx.failed_file or ""), (case, ctx.failed_file)
        for f in frags:
            assert f.lower() in (ctx.hint + " " + ctx.message).lower(), (case, f, ctx.hint)
        if ctx.line:
            assert ctx.snippet and f"{ctx.line:>4} |" in ctx.snippet, (case, ctx.snippet)
        ok(f"{fw}: {case} -> QA fails in {secs:.1f}s, diagnosed as {ctx.error_type} at {ctx.code_location}")
    # a status-code assertion should never hide the real exception
    pid, pd = make_project("flask", {"app.py": ('title="My Site"', "title=site_title")})
    out, _ = qa("flask", pid, pd)
    assert "assert 500 == 200" in out and "NameError" in out
    ok("flask: the raw QA output really does contain both `assert 500 == 200` and the NameError — the parser picks the root cause")

    # fail fast: a home page that 500s must not burn the whole boot timeout
    pid, pd = make_project("fastapi", {"main.py": ('request, "index.html", {"title": "My Site"}', '"index.html", {"request": request}')})
    _, secs = qa("fastapi", pid, pd)
    assert secs < 15, secs
    ok(f"boot probe fails fast on repeated HTTP 500 ({secs:.1f}s, was ~37s) and includes the server traceback")

    # the .venv is never py_compiled any more
    pid, pd = make_project("flask")
    out, _ = qa("flask", pid, pd)
    assert "Checked 2 Python file(s)" in out, out.splitlines()[:4]
    ok("static check ignores .venv/node_modules (it used to py_compile every installed library)")

    # dependencies missing -> graceful skips, not failures
    pid, pd = make_project("flask")
    (pd / ".venv").unlink()
    out, _ = qa("flask", pid, pd)
    assert out.startswith("PASSED: true") and "tests skipped" in out and "boot check skipped" in out
    ok("framework dependencies not installed yet -> tests/boot are skipped with a note (QA still passes the static checks)")

    if RUN_NEXT:
        pid, pd = make_project("next")
        out, secs = qa("next", pid, pd)
        assert out.startswith("PASSED: true") and "build passed" in out and "HTTP 200" in out, out
        ok(f"next: starter project really builds (next build) and serves in {secs:.1f}s")
        pid, pd = make_project("next", {"pages/index.js": ('import Header from "../components/Header";',
                                                           'import "../styles/globals.css";\nimport Header from "../components/Header";')})
        out, _ = qa("next", pid, pd)
        ctx = diagnose(out, pid, "next")
        assert out.startswith("PASSED: false") and ctx.error_type == "BuildError" and "pages/_app.js" in ctx.hint
        pid, pd = make_project("next", {"pages/index.js": ('import Header from "../components/Header";',
                                                           'import Header from "../components/Header";\nimport Footer from "../components/Footer";')})
        out, _ = qa("next", pid, pd)
        ctx = diagnose(out, pid, "next")
        assert ctx.error_type == "ModuleNotFoundError" and "components/Footer" in ctx.hint and ctx.failed_file == "pages/index.js"
        ok("next: a global-CSS import and a missing component are diagnosed from the real build output")
    else:
        pid, pd = make_project("next")
        nm = pd / "node_modules"
        if nm.is_symlink():          # a shared Next install may exist from an earlier AIOPS_TEST_NEXT=1 run
            nm.unlink()
        out, _ = qa("next", pid, pd)
        assert out.startswith("PASSED: true") and "build skipped" in out and "Next.js project structure OK" in out
        pid, pd = make_project("next", files={"package.json": '{"name": "x"}', "pages/index.js": "export default () => null"})
        out, _ = qa("next", pid, pd)
        assert out.startswith("PASSED: false") and "does not list `next`" in out
        ok("next (structural only; set AIOPS_TEST_NEXT=1 for a real `next build`): structure checks + graceful skip")


# ========================================================== C. full pipeline ==

def make_engine():
    eng = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    with eng.begin() as c:
        c.execute(text("CREATE TABLE metrics_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, kind TEXT, agent TEXT, "
                       "node TEXT, task_order INT, attempt INT, name TEXT, tokens_in INT DEFAULT 0, tokens_out INT DEFAULT 0, "
                       "duration_ms INT DEFAULT 0, t_start_ms BIGINT, ok BOOLEAN DEFAULT 1, meta TEXT, created_at TEXT)"))
    return eng


class FakeStm:
    def __getattr__(self, name):
        return lambda *a, **k: 0 if name == "clear_project" else None


class Ollama:
    """Fake Ollama: answers the Planner's web-app summary, and each framework's
    Scaffolder from `scaffold[fw]` (a list of file dicts: 1st answer, 2nd answer, ...)."""

    def __init__(self, scaffold):
        self.scaffold, self.calls, self.scaffold_prompts = scaffold, {}, []

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
        body = {"prompt_eval_count": 20, "eval_count": 10, "total_duration": 1_000_000, "eval_duration": 500_000}
        for fid, fw in FRAMEWORKS.items():
            if system == fw.system_prompt:
                self.scaffold_prompts.append((fid, prompt))
                n = self.calls[fid] = self.calls.get(fid, 0) + 1
                seq = self.scaffold[fid]
                files = seq[min(n, len(seq)) - 1]
                return R({**body, "response": __import__("json").dumps({"files": files})})
        return R({**body, "response": "A small website with a home page and an about page."})   # planner summary


class PipelineEnv:
    def __init__(self, ollama):
        self.ollama, self.eng = ollama, make_engine()

    def __enter__(self):
        outer = self

        class Sess:
            async def call_tool(s, name, args):
                if name == "run_shell_command":     # stand-in for the venv / npm install: link the shared environment
                    pd = WS / args["project_id"]
                    cmd = args["command"]
                    if "venv" in cmd and not (pd / ".venv").exists():
                        (pd / ".venv").symlink_to(VENV)
                    elif "npm install" in cmd and not (pd / "node_modules").exists():
                        src = NEXT_ENV if (pd / "pages").exists() else NODE_ENV
                        (pd / "node_modules").symlink_to(src / "node_modules")
                    out = "exit_code: 0\nstdout:\n"
                else:
                    out = await asyncio.to_thread(TOOL_FNS[name], **args)
                return SimpleNamespace(content=[SimpleNamespace(text=out)], isError=False)

        @asynccontextmanager
        async def fake_session():
            yield Sess()

        self.patches = [
            patch.object(db, "engine", self.eng), patch("requests.post", self.ollama.post),
            patch.object(mcp_client, "mcp_session", fake_session),
            patch.object(nodes, "db", MagicMock()), patch.object(nodes, "stm", FakeStm()), patch.object(graph_mod, "stm", FakeStm()),
            patch.object(nodes, "_log_run_safe", lambda *a, **k: None),
            patch.object(memory, "retrieve_similar", lambda *a, **k: []), patch.object(memory, "save_memory", lambda *a, **k: "id"),
            patch.object(memory, "find_similar_failures", lambda *a, **k: []),
            patch.object(memory, "save_failure_pattern", lambda *a, **k: "id"),
        ]
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *a):
        for p in reversed(self.patches):
            p.stop()


FLASK_ABOUT = {
    "app.py": ('from flask import Flask, render_template\n\napp = Flask(__name__)\n\n\n@app.route("/")\ndef index():\n'
               '    return render_template("index.html", title="Bakery")\n\n\n@app.route("/about")\ndef about():\n'
               '    return render_template("about.html", title="About us")\n'),
    "templates/about.html": "<!DOCTYPE html><html><head><title>{{ title }}</title></head><body><h1>{{ title }}</h1></body></html>\n",
    "test_app.py": ('from app import app\n\n\ndef test_home():\n    assert app.test_client().get("/").status_code == 200\n\n\n'
                    'def test_about():\n    assert app.test_client().get("/about").status_code == 200\n'),
}
FLASK_BROKEN = {**FLASK_ABOUT, "app.py": FLASK_ABOUT["app.py"].replace('title="Bakery"', "title=bakery_name")}


async def run(env, request, **kw):
    pid = str(uuid.uuid4())
    result = await graph_mod.run_pipeline(pid, request, **kw)
    return pid, result


def types(pid):
    return [e["type"] for e in events._history[pid]]


async def c_pipeline():
    scenarios = [
        # (label, request, kwargs, expected framework, entry file, files expected, QA notes expected)
        ("flask (detected)", "Build a Flask app with a home page and an about page", {}, "flask", "app.py",
         {"app.py", "test_app.py", "templates/about.html", "templates/index.html", "static/style.css"}, ["tests passed (pytest)", "HTTP 200"]),
        ("django (detected)", "Create a Django site for a small bakery", {}, "django", "manage.py",
         {"manage.py", "config/settings.py", "home/views.py", "home/tests.py"}, ["tests passed (manage.py test)", "HTTP 200"]),
        ("express (detected)", "Make an Express server for a bakery website", {}, "express", "server.js",
         {"server.js", "server.test.js", "package.json", "public/index.html"}, ["tests passed (npm test)", "HTTP 200"]),
        ("explicit override beats the text", "Build a bakery website with Django", {"framework": "flask"}, "flask", "app.py",
         {"app.py", "test_app.py"}, ["tests passed (pytest)"]),
        ("no framework named -> FastAPI default", "Build a bakery website", {}, "fastapi", "main.py",
         {"main.py", "requirements.txt", "templates/index.html"}, ["HTTP 200"]),
    ]
    for label, request, kw, fw, entry, expect_files, notes in scenarios:
        if fw == "express" and not (NODE_ENV / "node_modules" / "express").is_dir():
            print(f"[skip] {label} (node/npm unavailable)")
            continue
        scaffold = {k: [{}] for k in SUPPORTED}      # the model returns "nothing useful" -> the starter is backfilled
        scaffold["flask"] = [FLASK_ABOUT]
        with PipelineEnv(Ollama(scaffold)) as env:
            pid, result = await run(env, request, **kw)
            assert result.done and not result.needs_human_review, (label, result.task_history)
            assert result.framework == fw and result.task_list[0].framework == fw, (label, result.framework)
            on_disk = {str(p.relative_to(WS / pid)) for p in (WS / pid).rglob("*")
                       if p.is_file() and not any(x in p.parts for x in (".venv", "node_modules", "__pycache__", ".pytest_cache"))
                       and p.name != "db.sqlite3"}
            assert expect_files <= on_disk, (label, expect_files - on_disk)
            if fw != "fastapi":
                assert entry in on_disk
            evs = events._history[pid]
            planner_done = next(e for e in evs if e["type"] == "planner_done")
            assert planner_done["data"]["framework"] == fw
            qa_msg = next(e for e in evs if e["type"] == "qa_start")["message"]
            assert FRAMEWORKS[fw].label in qa_msg, qa_msg
            qa_res = [e for e in evs if e["type"] == "qa_result"][-1]
            assert qa_res["data"]["passed"], qa_res
            done = next(e for e in evs if e["type"] == "developer_done")
            assert done["data"]["framework"] == fw
            assert env.ollama.scaffold_prompts[0][0] == fw, "wrong framework's scaffolder prompt was used"
            tasks = [r for r in env.eng.connect().execute(text("SELECT meta FROM metrics_logs WHERE kind='task'"))]
            assert json.loads(tasks[0][0])["framework"] == fw
            ok(f"pipeline [{label}]: framework={fw}, scaffolded {len(on_disk)} file(s), real QA passed ({', '.join(notes)})")

    # fail -> diagnose -> repair, on a framework that isn't FastAPI
    scaffold = {k: [{}] for k in SUPPORTED}
    scaffold["flask"] = [FLASK_BROKEN, FLASK_ABOUT]
    with PipelineEnv(Ollama(scaffold)) as env:
        pid, result = await run(env, "Build a Flask app with a home page and an about page")
        assert result.done and not result.needs_human_review
        evs = events._history[pid]
        analysis = next(e for e in evs if e["type"] == "failure_analysis_done")["data"]["analysis"]
        assert analysis["error_type"] == "NameError" and "bakery_name" in analysis["hint"], analysis
        assert analysis["code_location"].startswith("app.py:")
        retry_prompt = env.ollama.scaffold_prompts[1][1]
        assert "bakery_name" in retry_prompt and "NameError" in retry_prompt and "Flask" in retry_prompt, retry_prompt[-600:]
        assert result.task_history[0]["failure_categories"] == ["name"]
        ok("pipeline [flask, broken first scaffold]: QA fails (500), root cause NameError at app.py is diagnosed, the retry prompt carries it, the 2nd scaffold passes")


async def main():
    a_detection()
    a_registry()
    a_scaffolder()
    a_planner_and_developer()
    a_api()
    print("--- Part A passed; Parts B/C use real runtimes ---")
    if not ensure_envs():
        print("[skip] Parts B and C (shared environments unavailable)")
    else:
        b_real_qa()
        await c_pipeline()
    print("\nAll Framework checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
