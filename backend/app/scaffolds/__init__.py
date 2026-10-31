"""Pluggable web-app scaffolds (FastAPI, Flask, Django, Express, Next.js).

Each framework is one `Framework` entry in FRAMEWORKS plus a directory of
real, runnable starter files under scaffolds/templates/<id>/. Everything the
pipeline needs to know about a framework lives here:

  * its Scaffolder system prompt and the Developer's framework note
  * the guaranteed-runnable starter project (used as the fallback when the
    model's output is unusable, and to backfill any REQUIRED file the model
    left out — the generalisation of the old FastAPI-only "templates/ +
    static/ were missing" fix)
  * which files the PLATFORM owns (`locked`) and the model may not overwrite
    (e.g. Django's settings.py, so a hallucinated setting can't break the skeleton)
  * which dependencies must be declared (`required_deps`), so a forgotten
    package never reaches the install step

To add a framework: drop a template directory in templates/<id>/, add a
Framework(...) below, add its detection keywords in app/webapp_detect.py and
its boot/test runner in mcp_server/tools/check_webapp.py. Nothing else
branches on the framework id.

Why this lives under backend/app/ and not at the repo root: the backend's
Docker build context is ./backend, so only files under it ship in the image.

FastAPI is deliberately the *legacy* entry: its required files, fallback and
dependency handling are exactly what the pipeline did before frameworks
existed (no smoke test is added, requirements.txt is left as the model wrote
it), so existing behaviour and tests are unchanged.
"""
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from app.scaffolds import prompts



TEMPLATES_DIR = Path(__file__).parent / "templates"
DEFAULT_FRAMEWORK = "fastapi"
PY_INSTALL = "python3 -m venv .venv && .venv/bin/pip install --quiet -r requirements.txt"
NODE_INSTALL = "npm install --no-audit --no-fund --loglevel=error"


@dataclass(frozen=True)
class Framework:
    id: str
    label: str
    language: str                       # "python" | "node"
    entry_file: str                     # primary file (shown in events; syntax-checked by QA when .py)
    dep_file: str                       # requirements.txt | package.json
    system_prompt: str
    developer_note: str
    description: str
    required: tuple[str, ...] = ()      # exact paths that must exist (backfilled from the template)
    required_any: tuple[tuple[str, str], ...] = ()   # (path prefix, suffix): at least one matching file must exist
    locked: frozenset = frozenset()     # platform-owned paths: the model's version is ignored
    required_deps: tuple[str, ...] = () # packages that must be declared in dep_file
    project_level: bool = False         # True: a request for it is always a whole project (never a single-file task)
    starter_hint: tuple[str, ...] = ()  # template files shown to the model as the starting point
    install_cmd: str = PY_INSTALL       # run in the project dir after the files are written
    install_timeout: int = 120

    # ---- template access -------------------------------------------------
    def template_files(self) -> dict[str, str]:
        """The guaranteed-runnable starter project ({relative_path: content})."""
        return dict(_load_template(self.id))

    def backfill(self, files: dict[str, str]) -> dict[str, str]:
        """Adds whatever REQUIRED file the model's output is missing, taking
        it from the starter project; keeps everything the model produced."""
        tmpl = _load_template(self.id)
        out = dict(files)
        for path in self.required:
            if path not in out and path in tmpl:
                out[path] = tmpl[path]
        for prefix, suffix in self.required_any:
            if not any(p.startswith(prefix) and p.endswith(suffix) for p in out):
                for p, c in tmpl.items():
                    if p.startswith(prefix) and p.endswith(suffix):
                        out.setdefault(p, c)
        return out

    def drop_locked(self, files: dict[str, str]) -> dict[str, str]:
        """Removes the model's versions of platform-owned files (the starter
        version is restored by backfill)."""
        return {p: c for p, c in files.items() if p not in self.locked}

    def ensure_dependencies(self, files: dict[str, str]) -> dict[str, str]:
        """Makes sure every `required_deps` package is declared. Python:
        appends missing names to requirements.txt. Node: repairs
        package.json (invalid JSON -> the starter's; missing dependencies /
        scripts -> filled in). No-op when `required_deps` is empty (FastAPI)."""
        if not self.required_deps:
            return files
        out = dict(files)
        if self.language == "python":
            current = out.get(self.dep_file, "")
            have = {_norm_pkg(line) for line in current.splitlines() if line.strip() and not line.lstrip().startswith(("#", "-"))}
            missing = [d for d in self.required_deps if _norm_pkg(d) not in have]
            if missing:
                out[self.dep_file] = current.rstrip("\n") + ("\n" if current.strip() else "") + "\n".join(missing) + "\n"
            return out
        # node
        tmpl = json.loads(_load_template(self.id)[self.dep_file])
        try:
            pkg = json.loads(out.get(self.dep_file, ""))
            if not isinstance(pkg, dict):
                raise ValueError("package.json is not an object")
        except (ValueError, TypeError):
            out[self.dep_file] = _load_template(self.id)[self.dep_file]
            return out
        deps = pkg.setdefault("dependencies", {})
        if not isinstance(deps, dict):
            deps = pkg["dependencies"] = {}
        for d in self.required_deps:
            deps.setdefault(d, tmpl["dependencies"].get(d, "latest"))
        scripts = pkg.setdefault("scripts", {})
        if not isinstance(scripts, dict):
            scripts = pkg["scripts"] = {}
        for k, v in tmpl.get("scripts", {}).items():
            scripts.setdefault(k, v)
        if "main" in tmpl:
            pkg.setdefault("main", tmpl["main"])
        out[self.dep_file] = json.dumps(pkg, indent=2) + "\n"
        return out

    def starter_excerpt(self, limit: int = 3000) -> str:
        """Short rendering of the starter project's key files, shown to the
        model so it edits a known-good skeleton instead of inventing one."""
        tmpl = _load_template(self.id)
        parts, used = [], 0
        for path in self.starter_hint:
            body = tmpl.get(path, "")
            chunk = f"--- {path} ---\n{body.strip()}\n"
            if used + len(chunk) > limit:
                break
            parts.append(chunk)
            used += len(chunk)
        return "\n".join(parts)

    def public(self) -> dict:
        """What GET /frameworks returns for this framework."""
        return {"id": self.id, "label": self.label, "language": self.language,
                "entry_file": self.entry_file, "description": self.description}


def _norm_pkg(requirement_line: str) -> str:
    name = re.split(r"[<>=!~;\[ ]", requirement_line.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


@lru_cache(maxsize=None)
def _load_template(fw_id: str) -> dict[str, str]:
    root = TEMPLATES_DIR / fw_id
    return {
        str(p.relative_to(root)).replace("\\", "/"): p.read_text()
        for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts
    }


FRAMEWORKS: dict[str, Framework] = {
    "fastapi": Framework(
        id="fastapi", label="FastAPI", language="python", entry_file="main.py", dep_file="requirements.txt",
        system_prompt=prompts.FASTAPI_SYSTEM, developer_note=prompts.FASTAPI_DEV_NOTE,
        description="FastAPI + Jinja2 (the default)",
        required=("main.py", "requirements.txt"),
        required_any=(("templates/", ".html"), ("static/", "")),
    ),
    "flask": Framework(
        id="flask", label="Flask", language="python", entry_file="app.py", dep_file="requirements.txt",
        system_prompt=prompts.FLASK_SYSTEM, developer_note=prompts.FLASK_DEV_NOTE,
        description="Flask + Jinja2, pytest smoke test",
        # templates/index.html is an EXACT requirement (not "some template exists"): "/" renders it,
        # so a model that only returns about.html would otherwise leave "/" with TemplateNotFound.
        required=("app.py", "requirements.txt", "test_app.py", "templates/index.html"),
        required_any=(("static/", ""),),
        required_deps=("flask", "pytest"),
        starter_hint=("app.py", "test_app.py"),
    ),
    "django": Framework(
        id="django", label="Django", language="python", entry_file="manage.py", dep_file="requirements.txt",
        system_prompt=prompts.DJANGO_SYSTEM, developer_note=prompts.DJANGO_DEV_NOTE,
        description="Django project (config/ + a `home` app), manage.py test",
        required=("manage.py", "requirements.txt", "config/__init__.py", "config/settings.py", "config/urls.py",
                  "config/wsgi.py", "home/__init__.py", "home/apps.py", "home/views.py", "home/urls.py", "home/tests.py",
                  "home/templates/home/index.html"),
        locked=frozenset({"manage.py", "config/__init__.py", "config/settings.py", "config/urls.py",
                          "config/wsgi.py", "home/__init__.py", "home/apps.py"}),
        required_deps=("django",),
        project_level=True,
        starter_hint=("home/views.py", "home/urls.py", "home/tests.py", "home/templates/home/index.html"),
    ),
    "express": Framework(
        id="express", label="Express (Node.js)", language="node", entry_file="server.js", dep_file="package.json",
        system_prompt=prompts.EXPRESS_SYSTEM, developer_note=prompts.EXPRESS_DEV_NOTE,
        description="Express on Node.js, node --test smoke test",
        required=("server.js", "package.json", "server.test.js", "public/index.html"),
        required_deps=("express",),
        project_level=True,
        starter_hint=("server.js", "server.test.js"),
        install_cmd=NODE_INSTALL, install_timeout=180,
    ),
    "next": Framework(
        id="next", label="Next.js", language="node", entry_file="pages/index.js", dep_file="package.json",
        system_prompt=prompts.NEXT_SYSTEM, developer_note=prompts.NEXT_DEV_NOTE,
        description="Next.js (pages router, plain JavaScript), next build",
        required=("package.json", "pages/_app.js"),
        required_any=(("pages/index.", ""),),
        required_deps=("next", "react", "react-dom"),
        project_level=True,
        starter_hint=("pages/index.js", "pages/_app.js", "components/Header.jsx"),
        install_cmd=NODE_INSTALL, install_timeout=300,
    ),
}

SUPPORTED: tuple[str, ...] = tuple(FRAMEWORKS)
# Paths the model may never write (they would be installed/generated by the platform).
FORBIDDEN_PREFIXES = (".venv/", "node_modules/", ".git/", "__pycache__/", ".next/")


def get_framework(framework_id: Optional[str]) -> Framework:
    """The Framework for an id; unknown / None -> the FastAPI default (never raises)."""
    return FRAMEWORKS.get((framework_id or "").lower(), FRAMEWORKS[DEFAULT_FRAMEWORK])


def is_supported(framework_id: Optional[str]) -> bool:
    return (framework_id or "").lower() in FRAMEWORKS
