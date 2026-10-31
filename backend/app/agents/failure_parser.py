"""Structured failure parsing.

Turns raw QA output (pytest with --showlocals, a plain Python traceback, the
deterministic pre-flight message, a check_webapp boot failure, a timeout)
into a FailureContext: error type/category, exact file+line+function, the
variable values at the failing frame, a line-numbered source snippet, the
function most likely to need the fix, and a SPECIFIC repair hint.

Deliberately deterministic (regex for the text formats + `ast` for the
source): no LLM call, so it is instant, reproducible, and never hallucinates
a line number. The Failure Analyzer only asks the LLM to refine the hint
when this module's confidence is low.

Nothing here raises to the caller for malformed input — parse_failure
always returns a FailureContext, degrading to a generic one.
"""
import ast
import builtins
import difflib
import logging
import re
import sys
from pathlib import Path
from typing import Optional

from app.schemas import FailureContext, StackFrame

logger = logging.getLogger("aiopsforge.failure_parser")

# ------------------------------------------------------------- patterns ----

# pytest --tb=short frame header:  "calc.py:5: in avg"
_FRAME_RE = re.compile(r"^(?P<file>\S+\.py):(?P<line>\d+): in (?P<func>\S+)\s*$")
# plain traceback / SyntaxError location:  'File "x.py", line 3, in f'
_TB_FRAME_RE = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+)(?:, in (?P<func>\S+))?')
# "TypeError: msg" (also dotted, e.g. json.decoder.JSONDecodeError)
# Also matches JS errors (a bare "Error: ...", "ERR_ASSERTION"-style "AssertionError [ERR_ASSERTION]: ...")
# and the Django / Jinja2 exception names that don't end in "Error".
_ERR_RE = re.compile(
    r"^(?P<type>(?:[A-Za-z_]\w*\.)*(?:[A-Z][A-Za-z0-9_]*(?:Error|Exception|Exit|Warning|NotFound|DoesNotExist)"
    r"|Error|NoReverseMatch|ImproperlyConfigured))"
    r"(?:\s*\[[A-Z_0-9]+\])?(?::\s*(?P<msg>.*))?$"
)
# Node stack frames: "at fn (/p/server.js:11:47)", "at /p/server.js:11:47", and the bare
# "fn (/p/x.js:19:12)" form node:test prints inside its YAML `stack:` block.
_NODE_FRAME_RE = re.compile(
    r"^\s*(?:at\s+)?(?:async\s+)?(?P<func>[^\s()]+(?:\s[^\s()]+)*)\s\((?P<file>[^()]+?):(?P<line>\d+):(?P<col>\d+)\)\s*$")
_NODE_ANON_RE = re.compile(
    r"^\s*at\s+(?P<file>(?:/|[A-Za-z]:\\|file://)[^\s()]+?):(?P<line>\d+):(?P<col>\d+)\s*$")
# `node --check` / uncaught SyntaxError header:  "/p/server.js:4"
_NODE_HDR_RE = re.compile(r"^(?P<file>(?:/|[A-Za-z]:\\)[^\s:]+\.(?:c|m)?jsx?):(?P<line>\d+)\s*$")
_E_LINE_RE = re.compile(r"^E {2,}")
# pytest --showlocals line:  "        total      = None"
_LOCAL_RE = re.compile(r"^ {8}(?P<name>[A-Za-z_]\w*)\s+= (?P<val>.*)$")
_SUMMARY_RE = re.compile(r"^(FAILED|ERROR) (?P<test>\S+?)(?: - (?P<msg>.*))?$")
_NOT_PROJECT = ("site-packages", "dist-packages", "/usr/lib/python", "/usr/local/lib/python",
                "<frozen", "_pytest", "importlib", "pluggy", "node_modules", "node:", "/internal/")

_CATEGORY = {
    "SyntaxError": "syntax", "IndentationError": "syntax", "TabError": "syntax",
    "ImportError": "import", "ModuleNotFoundError": "import",
    "NameError": "name", "UnboundLocalError": "name",
    "TypeError": "type", "AttributeError": "attribute",
    "KeyError": "lookup", "IndexError": "lookup",
    "ValueError": "value", "AssertionError": "logic",
    "ZeroDivisionError": "arithmetic", "OverflowError": "arithmetic",
    "RecursionError": "recursion", "TimeoutError": "timeout",
    "FileNotFoundError": "missing_file", "FixtureLookupError": "config",
    "StopIteration": "runtime",
    # web frameworks (Flask/Jinja2, Django, Node, Next.js)
    "TemplateNotFound": "template", "TemplateDoesNotExist": "template", "TemplateSyntaxError": "template",
    "UndefinedError": "template", "NoReverseMatch": "routing", "ImproperlyConfigured": "config",
    "ReferenceError": "name", "RangeError": "value", "BuildError": "build",
}
_BUILTIN_TYPES = {"str": str, "list": list, "dict": dict, "int": int, "float": float,
                  "tuple": tuple, "set": set, "bytes": bytes}


def category_for(error_type: str) -> str:
    return _CATEGORY.get(error_type.split(".")[-1], "runtime")


# --------------------------------------------------------- file helpers ----

def _is_project_file(path: str) -> bool:
    return not any(s in path for s in _NOT_PROJECT) and not path.startswith(("<", "node:"))


def _is_test_file(name: str) -> bool:
    base = Path(name).name
    return (base.startswith("test_") or base.endswith("_test.py") or base == "tests.py"
            or base.endswith((".test.js", ".spec.js", ".test.jsx", ".spec.jsx")))


def _project_dir(project_id: Optional[str], workspace_root: Optional[str]) -> Optional[Path]:
    if project_id and workspace_root:
        return Path(workspace_root) / project_id
    return None


def _resolve(pd: Optional[Path], path: str) -> Optional[Path]:
    """Maps a path from a traceback to a real file inside the project
    workspace (never outside it)."""
    if pd is None or not pd.is_dir():
        return None
    root = pd.resolve()
    p = Path(path)
    candidates = [p] if p.is_absolute() else [pd / p]
    candidates.append(pd / p.name)
    for c in candidates:
        try:
            r = c.resolve()
            if (root in r.parents or r == root) and r.is_file():
                return r
        except OSError:
            continue
    return None


def _read(path: Optional[Path]) -> Optional[str]:
    try:
        if path and path.stat().st_size <= 200_000:
            return path.read_text(errors="replace")
    except OSError:
        pass
    return None


def _numbered(lines: list[str], start: int, end: int, mark: Optional[int] = None) -> str:
    out = []
    for i in range(max(start, 1), min(end, len(lines)) + 1):
        prefix = ">" if i == mark else " "
        out.append(f"{prefix}{i:>4} | {lines[i - 1]}")
    return "\n".join(out)


# ---------------------------------------------------------- AST helpers ----

def _enclosing_function(tree: ast.AST, lineno: int):
    best = None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if n.lineno <= lineno <= (n.end_lineno or n.lineno):
                if best is None or n.lineno >= best.lineno:
                    best = n
    return best


def _find_function(tree: ast.AST, name: str):
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    return None


def _assign_lines(tree: ast.AST, name: str) -> list[int]:
    lines = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store):
            lines.add(n.lineno)
        elif isinstance(n, ast.arg) and n.arg == name:
            lines.add(n.lineno)
    return sorted(lines)


def _defined_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Import):
            names.update((a.asname or a.name).split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            names.update(a.asname or a.name for a in n.names)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)
        elif isinstance(n, ast.arg):
            names.add(n.arg)
    return names


def _called_names(line_src: str) -> list[str]:
    """Names of functions called on one source line (best effort — the
    line may be a fragment of a multi-line statement)."""
    try:
        node = ast.parse(line_src.strip())
    except SyntaxError:
        return re.findall(r"\b([A-Za-z_]\w*)\s*\(", line_src)
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                out.append(f.id)
            elif isinstance(f, ast.Attribute):
                out.append(f.attr)
    return out


# ---------------------------------------------------- block extraction -----

def _split_blocks(lines: list[str]) -> list[dict]:
    blocks: list[dict] = []
    cur: Optional[dict] = None
    for line in lines:
        if re.match(r"^={3,}\s.*\s={3,}$", line):
            if cur is not None:
                blocks.append(cur)
                cur = None
            continue
        h = re.match(r"^_{3,}\s(.+?)\s_{3,}$", line)
        if h:
            if cur is not None:
                blocks.append(cur)
            cur = {"name": h.group(1), "lines": []}
            continue
        if cur is not None:
            cur["lines"].append(line)
    if cur is not None:
        blocks.append(cur)
    return blocks or [{"name": None, "lines": lines}]


def _extract(lines: list[str]):
    """Returns (frames, err_type, err_msg, traceback_text) for one block.

    Understands three families of output: pytest (`E   ` lines, per-frame
    locals), plain Python tracebacks (also Django's unittest output and the
    server logs check_webapp appends), and Node (`at fn (file:line:col)`
    stacks, `node --check` headers, node:test's TAP YAML). Node stacks list
    the INNERMOST frame first, so each stack is reversed to keep the
    "last project frame is the failing one" rule the Python families use.
    """
    frames: list[StackFrame] = []
    cur: Optional[StackFrame] = None
    err: Optional[tuple[str, str]] = None
    saw_e = any(_E_LINE_RE.match(l) for l in lines)   # NOT a bare "E" (a unittest progress character)
    stacks: list[list[StackFrame]] = [[]]             # Node stacks; a new one starts at each error / YAML marker
    err_stack = None
    yaml_name: Optional[str] = None
    yaml_err: list[str] = []
    in_yaml_err, yaml_indent = False, 0

    def add_frame(f, ln, fn):
        nonlocal cur
        cur = StackFrame(file=f, line=int(ln), function=fn, project_frame=_is_project_file(f))
        frames.append(cur)

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if in_yaml_err:   # node:test YAML:  "error: |-" followed by an indented message
            indent = len(line) - len(line.lstrip())
            if indent > yaml_indent:
                if len(yaml_err) < 3:
                    yaml_err.append(line.strip())
                continue
            in_yaml_err = False
        m = _FRAME_RE.match(line)
        if m:
            add_frame(m["file"], m["line"], m["func"])
            continue
        is_e = bool(_E_LINE_RE.match(line))
        body = line[1:].strip() if is_e else None
        if body is not None:
            m2 = _TB_FRAME_RE.match(body)
            if m2:
                add_frame(m2["file"], m2["line"], m2["func"])
            elif err is None:
                em = _ERR_RE.match(body)
                if em:
                    err = (em["type"], (em["msg"] or "").strip())
            continue
        m3 = _TB_FRAME_RE.match(line)
        if m3:
            add_frame(m3["file"], m3["line"], m3["func"])
            continue
        mh = _NODE_HDR_RE.match(line)
        if mh:   # "/p/server.js:4" printed above a SyntaxError
            add_frame(mh["file"], mh["line"], None)
            continue
        mn = _NODE_FRAME_RE.match(line) or _NODE_ANON_RE.match(line)
        if mn:
            f = mn["file"].replace("file://", "")
            stacks[-1].append(StackFrame(file=f, line=int(mn["line"]), function=(mn.groupdict().get("func") or None),
                                         project_frame=_is_project_file(f)))
            continue
        if re.match(r"^(?:not ok|ok) \d+|^\s*stack: \|", line):
            stacks.append([])           # TAP result line / YAML `stack:` -> a fresh stack
        ym = re.match(r"^\s*name: '(\w+)'", line)
        if ym:
            yaml_name = ym.group(1)
            continue
        ye = re.match(r"^(\s*)error: \|-?\s*$", line)
        if ye:
            in_yaml_err, yaml_indent = True, len(ye.group(1))
            continue
        lm = _LOCAL_RE.match(line)
        if lm and cur is not None:
            name = lm["name"]
            if not name.startswith(("@py_", "__")) and len(cur.variables) < 12:
                cur.variables[name] = lm["val"][:120]
            continue
        if line.startswith("    ") and cur is not None and cur.source is None:
            s_ = line.strip()
            if s_ and not set(s_) <= {"^", "~"}:
                cur.source = s_
            continue
        if not saw_e and not line.startswith(" "):
            em = _ERR_RE.match(line)
            if em:
                err = (em["type"], (em["msg"] or "").strip())  # plain traceback / Node: last one wins
                stacks.append([])
                err_stack = len(stacks) - 1
    if err is None and yaml_name:   # node:test reports an assertion only in its YAML block
        err = (yaml_name, " ".join(yaml_err)[:200])
    chosen = stacks[err_stack] if (err_stack is not None and stacks[err_stack]) else next((st for st in stacks if st), [])
    frames.extend(reversed(chosen))  # Node: innermost frame first -> make it last
    tb_text = "\n".join(lines)[-2500:]
    return frames, (err[0] if err else None), (err[1] if err else ""), tb_text


def _embedded_root_cause(lines: list[str]):
    """An application exception buried in test/server output.

    A Flask/Django/Express request that raises shows up in the test as just
    `assert 500 == 200` (or node:test's `500 !== 200`); the real exception is
    in the captured log / stderr. Returns (frames, type, msg) for the first
    non-assertion exception that has a stack inside the project, else None.
    """
    # Python: "Traceback (most recent call last):" ... "<ErrType>: msg"
    for i, l in enumerate(lines):
        if l.strip() == "Traceback (most recent call last):":
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or lines[j].startswith((" ", "\t"))):
                j += 1
            seg = lines[i: j + 1]
            frames, et, msg, _ = _extract(seg)
            if et and not et.startswith("AssertionError") and any(f.project_frame for f in frames):
                return frames, et, msg
    # Node: a col-0 "XError: msg" line followed by "    at ..." frames
    for i, l in enumerate(lines):
        em = _ERR_RE.match(l) if l and not l.startswith(" ") else None
        if em and not em["type"].startswith("AssertionError"):
            j = i + 1
            while j < len(lines) and re.match(r"^\s+at\s", lines[j]):
                j += 1
            if j > i + 1:
                frames, et, msg, _ = _extract(lines[i:j])
                if et and any(f.project_frame for f in frames):
                    return frames, et, msg
    return None


def _special(text: str) -> Optional[tuple[str, str, str, Optional[str], Optional[int], str]]:
    """(type, message, phase, file, line, category_override) for failures
    that carry no traceback."""
    m = re.search(r"Test run timed out after (\d+)s", text)
    if m:
        return ("TimeoutError", f"test run exceeded {m.group(1)}s", "timeout", None, None, "timeout")
    m = re.search(r"Syntax error in (\S+): (.*?) \(line (\d+)\)", text)
    if m:
        return ("SyntaxError", m.group(2), "preflight", m.group(1), int(m.group(3)), "syntax")
    m = re.search(r"Failed to compile (\S+): (.*)", text)
    if m:
        return ("SyntaxError", m.group(2), "preflight", m.group(1), None, "syntax")
    m = re.search(r"(?:Required file|File) not found: (\S+)", text)
    if m:
        return ("FileNotFoundError", f"file not found: {m.group(1)}", "preflight", m.group(1), None, "missing_file")
    if "requirements.txt is missing" in text:
        return ("FileNotFoundError", "requirements.txt is missing", "boot", "requirements.txt", None, "missing_file")
    if "No .py files found" in text:
        return ("FileNotFoundError", "no .py files were written", "boot", None, None, "missing_file")
    m = re.search(r"Failed to compile\.\s*\n+\s*(\./\S+)\s*\n+(.+)", text)   # webpack / next build
    if m:
        msg = m.group(2).strip()
        etype = "ModuleNotFoundError" if "Module not found" in msg else "BuildError"
        return (etype, msg, "build", m.group(1).lstrip("./"), None, "import" if etype == "ModuleNotFoundError" else "build")
    m = re.search(r"app did not respond within ([\d.]+)s", text)
    if m:
        return ("TimeoutError", f"app did not respond within {m.group(1)}s", "boot", None, None, "timeout")
    if "pytest is not installed" in text:
        return ("ModuleNotFoundError", "pytest is not installed in the workspace", "test", None, None, "config")
    return None


# ------------------------------------------------------------- hints -------

def _stdlib(mod: str) -> bool:
    return mod.split(".")[0] in getattr(sys, "stdlib_module_names", set())


def _line_text(lines: list[str], n: Optional[int]) -> str:
    return lines[n - 1].strip() if n and 0 < n <= len(lines) else ""


def _templates_in(pd: Optional[Path]) -> list[str]:
    """Names of the .html templates present in the project (for did-you-mean)."""
    if pd is None or not pd.is_dir():
        return []
    out = []
    for p in sorted(pd.rglob("*.html")):
        if not ({".venv", "node_modules", ".next"} & set(p.relative_to(pd).parts)):
            out.append(str(p.relative_to(pd)))
    return out[:30]


def _web_hint(ctx: FailureContext, et: str, msg: str, where: str, text_at: str,
              src_lines: list[str], pd: Optional[Path], is_js: bool) -> Optional[tuple[str, str]]:
    """Hints for web-framework failures: JavaScript/Node/Next.js errors and
    the Jinja2 / Django template and routing errors. None = fall through to
    the generic Python rules."""
    # Node: require() of something that isn't there
    if et == "Error" and "Cannot find module" in msg:
        mm = re.search(r"Cannot find module '([^']+)'", msg)
        name = mm.group(1) if mm else "the module"
        if name.startswith("."):
            return (f"`require('{name}')` at {where} does not match any file — check the relative path and that "
                    f"the file exists.", "high")
        return (f"`{name}` is not installed: add it to package.json \"dependencies\" (only packages listed there "
                f"get installed) or remove the require at {where}.", "high")
    if et == "Error" and "EADDRINUSE" in msg:
        return ("The server tried to listen on a fixed port that is already in use. Read the port from "
                "`process.env.PORT || 3000` and call app.listen only inside `if (require.main === module)`.", "high")
    # webpack / next build: unresolved import
    if "Can't resolve" in msg:
        mm = re.search(r"Can't resolve '([^']+)'", msg)
        name = mm.group(1) if mm else "the module"
        if name.startswith("."):
            return (f"`{name}` imported from {where} doesn't exist. Create that file (components/*.jsx for "
                    f"components) or fix the import path.", "high")
        return (f"`{name}` imported from {where} is not installed: add it to package.json \"dependencies\" "
                f"or remove the import.", "high")
    if et == "BuildError":
        if "Global CSS" in msg:
            return ("Global CSS can only be imported from pages/_app.js (it already imports ../styles/globals.css). "
                    "Remove `import '...css'` from every page/component and put page-specific styles in a "
                    "CSS module (Name.module.css) or inline style objects.", "high")
        return (f"`next build` failed at {where}: {msg[:200]}. Fix that file; keep pages/ + components/ valid React.", "medium")
    if et == "ReferenceError":
        mm = re.search(r"([\w$]+) is not defined", msg)
        name = mm.group(1) if mm else None
        if name:
            declared = set(re.findall(r"\b(?:const|let|var|function|class)\s+([A-Za-z_$][\w$]*)", "\n".join(src_lines)))
            declared |= set(re.findall(r"\b(?:const|let|var)\s*\{([^}]*)\}", "\n".join(src_lines)) and
                            [n.strip() for grp in re.findall(r"\b(?:const|let|var)\s*\{([^}]*)\}", "\n".join(src_lines))
                             for n in grp.split(",")])
            close = difflib.get_close_matches(name, sorted(declared), n=1)
            extra = f" Did you mean `{close[0]}`?" if close else " Declare it (const/let), or require/import it."
            return (f"`{name}` is not defined at {where}.{extra}", "high")
    if et == "TypeError" and "unhashable type: 'dict'" in msg and (
            "TemplateResponse" in text_at or any("TemplateResponse" in (f.source or "") for f in ctx.frames)):
        return (f"At {where}: current Starlette takes `request` as the FIRST argument — write "
                f"`templates.TemplateResponse(request, \"page.html\", {{...context without 'request'...}})`, not "
                f"`TemplateResponse(\"page.html\", {{\"request\": request, ...}})`.", "high")
    if et == "TypeError" and is_js:
        mm = re.search(r"Cannot read propert(?:y|ies) of (undefined|null) \(reading '([^']+)'\)", msg)
        if mm:
            return (f"Something is {mm.group(1)} when `.{mm.group(2)}` is read at {where}. Initialise it before use, "
                    f"or guard with `?.` / an `if` check.", "high")
    # Jinja2 / Django templates
    if et in ("TemplateNotFound", "TemplateDoesNotExist"):
        name = msg.split(",")[0].strip().strip("'\"")
        present = _templates_in(pd)
        close = difflib.get_close_matches(Path(name).name, [Path(t).name for t in present], n=1)
        have = f" Templates present: {', '.join(present[:6])}." if present else ""
        did = f" Did you mean `{close[0]}`?" if close else ""
        return (f"Template `{name}` was not found (raised at {where}). Create it — Flask: templates/{Path(name).name}; "
                f"Django: home/templates/{name} — or fix the name passed to render_template/render.{did}{have}", "high")
    if et in ("TemplateSyntaxError", "UndefinedError") and ctx.error_category == "template":
        return (f"Template error ({where}): {msg[:160]}. Fix the template syntax / the variable the view passes.", "medium")
    if et == "NoReverseMatch":
        return (f"A {{% url %}} / reverse() lookup failed: {msg[:160]}. The name must match a `name=` in home/urls.py "
                f"and any arguments must match its pattern.", "high")
    if et == "ImproperlyConfigured":
        return (f"Django configuration error: {msg[:200]}. settings.py/config are managed by the platform — fix the "
                f"app code (home/) that triggered it.", "medium")
    return None


def _build_hint(ctx: FailureContext, tree, src_lines: list[str], pd: Optional[Path]) -> tuple[str, str]:
    et = ctx.error_type.split(".")[-1]
    msg = ctx.message
    loc = ctx.code_location or "the failing line"
    text_at = _line_text(src_lines, ctx.line)
    var = ctx.variables
    where = f"{loc}" + (f" (`{text_at}`)" if text_at else "")
    conf = "high" if ctx.line else "medium"
    is_js = (ctx.failed_file or "").endswith((".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"))

    # ---- JavaScript / Node / Next.js / template-engine failures ----
    web = _web_hint(ctx, et, msg, where, text_at, src_lines, pd, is_js)
    if web:
        return web

    if ctx.phase == "timeout" and et == "TimeoutError":
        src_all = "\n".join(src_lines)
        if "input(" in src_all:
            return ("The run timed out and the code calls input(), which blocks forever under "
                    "pytest. Remove input() and take values as function arguments.", "high")
        if ctx.function or ctx.phase == "boot":
            return ("The app never answered a request. Express/Node: listen on `process.env.PORT` (the checker "
                    "passes the port that way) and don't block at import. Python: define the module-level `app` "
                    "(app.py for Flask, main.py for FastAPI), no blocking call at import time, and list every "
                    "import in requirements.txt.", "medium")
        return ("The tests exceeded the time limit — look for an infinite loop (a `while` whose "
                "condition never changes), unbounded recursion, or a blocking call.", "medium")

    if et in ("SyntaxError", "IndentationError", "TabError"):
        if is_js:
            return (f"{et} at {where}: {msg}. Check brackets, braces, parentheses and quotes are balanced and "
                    f"commas/semicolons are in place. Return the whole file as valid JavaScript.", conf)
        return (f"{et} at {where}: {msg}. Check brackets/quotes are closed, colons follow "
                f"def/if/for, and indentation is consistent. Return the whole file as valid Python.", conf)

    if et == "ModuleNotFoundError" or (et == "ImportError" and "No module named" in msg):
        mm = re.search(r"No module named '([^']+)'", msg)
        mod = mm.group(1) if mm else "the module"
        top = mod.split(".")[0]
        if pd is not None and (pd / f"{top}.py").exists():
            return (f"`import {mod}` failed although {top}.py exists in the project — {top}.py itself "
                    f"has an error or the import is misspelled. Check its contents at {where}.", "medium")
        if _stdlib(mod):
            return (f"`{mod}` is standard library, so this is a typo or a wrong submodule path at {where}.", "medium")
        return (f"`{mod}` is not installed in the test environment. Use only the standard library and "
                f"pytest, or (for a web app) add `{top}` to requirements.txt. Failing import: {where}.", "high")

    if et == "ImportError":
        mm = re.search(r"cannot import name '([^']+)' from '([^']+)'", msg)
        if mm:
            name, mod = mm.groups()
            hint = f"`{mod}` has no `{name}`."
            if pd is not None and (pd / f"{Path(mod).stem}.py").exists():
                other = _read(pd / f"{Path(mod).stem}.py")
                try:
                    defined = sorted(_defined_names(ast.parse(other or "")))
                except SyntaxError:
                    defined = []
                close = difflib.get_close_matches(name, defined, n=2)
                hint += f" {Path(mod).stem}.py defines: {', '.join(defined[:10]) or 'nothing'}."
                if close:
                    hint += f" Did you mean `{close[0]}`?"
            return (f"{hint} Failing import at {where}.", "high")
        return (f"ImportError at {where}: {msg}", "medium")

    if et in ("NameError", "UnboundLocalError"):
        mm = re.search(r"name '([^']+)'", msg) or re.search(r"variable '([^']+)'", msg)
        name = mm.group(1) if mm else None
        if name:
            pool = set(dir(builtins))
            if tree is not None:
                pool |= _defined_names(tree)
            close = difflib.get_close_matches(name, sorted(pool), n=1)
            if et == "UnboundLocalError":
                return (f"`{name}` is used at {where} before it is assigned in this function — "
                        f"assign it on every path before use.", "high")
            extra = f" Did you mean `{close[0]}`?" if close else " Define it or import it before use."
            return (f"`{name}` is not defined at {where}.{extra}", "high")

    if et == "AttributeError":
        mm = re.search(r"'(\w+)' object has no attribute '(\w+)'", msg)
        mod = re.search(r"module '([^']+)' has no attribute '(\w+)'", msg)
        if mm:
            tname, attr = mm.groups()
            if tname == "NoneType":
                vm = re.search(rf"\b([A-Za-z_]\w*)(?:\[[^\]]*\])?\s*\.\s*{re.escape(attr)}\b", text_at)
                v = vm.group(1) if vm else None
                if v and tree is not None:
                    assigns = _assign_lines(tree, v)
                    a = f" `{v}` is assigned at line(s) {', '.join(map(str, assigns[:3]))}." if assigns else ""
                    conf_none = " (confirmed: it is None in the failing frame)" if var.get(v) == "None" else ""
                    return (f"`{v}` is None when `.{attr}` is accessed at {where}{conf_none}.{a} "
                            f"Make sure it is initialised, or guard with `if {v} is not None`.", "high")
                return (f"An expression at {where} evaluated to None, then `.{attr}` was accessed. "
                        f"A function on that line probably returns None on some path — make every "
                        f"path `return` a value.", "high")
            cls = _BUILTIN_TYPES.get(tname)
            close = difflib.get_close_matches(attr, dir(cls), n=1) if cls else []
            sug = f" Did you mean `.{close[0]}`?" if close else ""
            return (f"A `{tname}` has no attribute `{attr}` at {where}.{sug} Check the variable's type.", "high")
        if mod:
            return (f"Module `{mod.group(1)}` has no `{mod.group(2)}` at {where}. Check the name/spelling "
                    f"or that the file defines it.", "high")

    if et == "TypeError":
        m1 = re.search(r"unsupported operand type\(s\) for ([^:]+): '(\w+)' and '(\w+)'", msg)
        if m1:
            op, a, b = m1.groups()
            nones = [k for k, v in var.items() if v == "None"]
            if "NoneType" in (a, b):
                who = f" `{nones[0]}` is None here" if nones else " One operand is None"
                lines = ""
                if nones and tree is not None:
                    al = _assign_lines(tree, nones[0])
                    if al:
                        lines = f" (assigned at line {al[0]}: `{_line_text(src_lines, al[0])}`)"
                return (f"TypeError at {where}:{who}{lines}, so `{op}` fails. Initialise it to a real "
                        f"value (e.g. 0 or []) or handle the empty/None case before this line.", "high")
            return (f"TypeError at {where}: cannot apply `{op}` to {a} and {b}. Convert one side "
                    f"(int()/str()) so the types match.", "high")
        m2 = re.search(r"(\w+)\(\) takes (\d+) positional arguments? but (\d+) (?:was|were) given", msg)
        m3 = re.search(r"(\w+)\(\) missing (\d+) required positional arguments?: (.*)", msg)
        mm = m2 or m3
        if mm:
            fname = mm.group(1)
            sig = ""
            if tree is not None:
                fn = _find_function(tree, fname)
                if fn is not None:
                    sig = f" Its signature is `{fname}({ast.unparse(fn.args)})` (line {fn.lineno})."
            return (f"The call at {where} passes the wrong number of arguments to `{fname}`.{sig} "
                    f"Make the call and the definition agree.", "high")
        if "not iterable" in msg or "not subscriptable" in msg or "not callable" in msg:
            nones = [k for k, v in var.items() if v == "None"]
            n = f" `{nones[0]}` is None." if nones else ""
            return (f"TypeError at {where}: {msg}.{n} Check what the value actually is (a function "
                    f"that forgot to `return`, or a wrong variable).", "high" if nones else "medium")
        if "can only concatenate" in msg or "must be str" in msg:
            return (f"TypeError at {where}: {msg}. Convert with str()/int() before combining.", "high")
        return (f"TypeError at {where}: {msg}", "medium")

    if et in ("KeyError", "IndexError"):
        empties = [k for k, v in var.items() if v in ("[]", "{}", "''", '""', "()")]
        e = f" `{empties[0]}` is empty in the failing frame — handle empty input." if empties else ""
        if et == "KeyError":
            return (f"Key {msg or '?'} is missing at {where}.{e} Use `.get(key, default)` or check "
                    f"`key in d` first.", "high")
        return (f"Index out of range at {where}.{e} Check the length before indexing.", "high")

    if et == "ZeroDivisionError":
        empties = [k for k, v in var.items() if v in ("[]", "{}", "''", "0", "0.0")]
        e = f" (`{empties[0]}` = {var[empties[0]]} in the failing frame)" if empties else ""
        return (f"Division by zero at {where}{e}. Guard the divisor (e.g. return 0 or raise a clear "
                f"ValueError when the input is empty).", "high")

    if et == "RecursionError":
        return (f"Unbounded recursion at {where}. Add a base case that stops the recursion, and make "
                f"sure each call moves toward it.", "high")

    if et == "AssertionError":
        expr = ""
        try:
            n = ast.parse(text_at)
            if isinstance(n.body[0], ast.Assert):
                expr = ast.unparse(n.body[0].test)
        except (SyntaxError, IndexError):
            pass
        fn = ctx.affected_function
        varstr = ", ".join(f"{k}={v}" for k, v in list(var.items())[:4])
        target = f" The code under test is `{fn}`" + (f" in {ctx.suspect_file}" if ctx.suspect_file else "") + "." if fn else ""
        shown = f"`{expr}`" if expr else f"`{text_at}`" if text_at else "an assertion"
        return (f"Assertion failed at {where}: {shown} — the test reports: {msg or 'assert ...'}."
                f"{target}{' Inputs: ' + varstr + '.' if varstr else ''} Fix the implementation so it "
                f"returns what the test expects; do not change the test's expected value.", conf)

    if et == "FixtureLookupError" or "fixture '" in msg:
        mm = re.search(r"fixture '([^']+)'", msg)
        n = mm.group(1) if mm else "a parameter"
        return (f"pytest treats the test parameter `{n}` as a fixture that doesn't exist. Remove the "
                f"parameter (create the value inside the test) or use @pytest.mark.parametrize.", "high")

    if et == "FileNotFoundError":
        return (f"A required file is missing ({msg}). Write it, and use paths relative to the project directory.", "medium")

    if et == "ValueError":
        return (f"ValueError at {where}: {msg}. Validate the input before this line, or fix the "
                f"conversion/unpacking that raised it.", conf)

    return (f"{et} at {where}: {msg or 'see output'}. Fix the specific line above; leave working code alone.", "low")


def _normalize_signature(ctx: FailureContext) -> str:
    m = re.sub(r"'[^']*'|\"[^\"]*\"", "'X'", ctx.message)
    m = re.sub(r"\d+", "N", m)[:80]
    return f"{ctx.error_type}|{ctx.error_category}|{ctx.affected_function or ctx.function or ''}|{m}"


# -------------------------------------------------------------- public -----

def parse_failure(
    output: str,
    *,
    project_id: Optional[str] = None,
    workspace_root: Optional[str] = None,
    code_filename: Optional[str] = None,
    code: Optional[str] = None,
    deterministic: bool = False,
) -> FailureContext:
    """Parses one QA failure's raw output. Never raises."""
    try:
        return _parse(output, project_id, workspace_root, code_filename, code, deterministic)
    except Exception as exc:  # noqa: BLE001 — parsing must never break the pipeline
        logger.warning("failure parsing failed, using generic context: %s", exc)
        msg = (output or "").strip()[:200]
        ctx = FailureContext(message=msg, raw_excerpt=(output or "")[-1500:],
                             hint="Read the error output and fix the failing line.")
        ctx.signature = _normalize_signature(ctx)
        return ctx


def _parse(output, project_id, workspace_root, code_filename, code, deterministic) -> FailureContext:
    text = (output or "").strip()
    lines = text.splitlines()
    if "TAP version" in text:   # node:test's TAP reporter prefixes diagnostics (incl. stderr) with "# "
        lines = [l[2:] if l.startswith("# ") else l for l in lines]
        text = "\n".join(lines)
    pd = _project_dir(project_id, workspace_root)
    ctx = FailureContext(raw_excerpt=text[-1500:], phase="preflight" if deterministic else "test")

    ctx.failing_tests = [
        m["test"] for l in lines if (m := _SUMMARY_RE.match(l.strip()))
    ][:8]

    blocks = _split_blocks(lines)
    frames: list[StackFrame] = []
    etype = None
    msg = ""
    for b in blocks:
        frames, etype, msg, _tb = _extract(b["lines"])
        if etype:
            break
    if etype is None:
        frames, etype, msg, _tb = _extract(lines)  # whole text, in case block-splitting hid it
    if etype is None:
        # pytest reports a missing fixture as a bare "E   fixture 'x' not found" line.
        fm = re.search(r"^E\s+fixture '([^']+)' not found", text, re.MULTILINE)
        if fm:
            etype, msg = "FixtureLookupError", f"fixture '{fm.group(1)}' not found"

    # A request that raised inside the app shows up in the test as `assert 500 == 200`;
    # prefer the application exception buried in the captured log / stderr.
    surfaced_as = None
    if etype is None or etype.startswith("AssertionError"):
        root = _embedded_root_cause(lines)
        if root:
            surfaced_as = msg
            frames, etype, msg = root

    fail_file: Optional[str] = None
    fail_line: Optional[int] = None
    fail_func: Optional[str] = None
    if etype:
        ctx.error_type = etype.rsplit(".", 1)[-1]   # "jinja2.exceptions.TemplateNotFound" -> "TemplateNotFound"
        ctx.error_category = category_for(etype)
        ctx.message = msg
        if "ERROR collecting" in text:
            ctx.phase = "collection"
        elif "Runtime boot check failed" in text or "app process exited" in text:
            ctx.phase = "boot"
        proj = [f for f in frames if f.project_frame]
        if proj:
            last = proj[-1]
            fail_file, fail_line, fail_func = last.file, last.line, last.function
            ctx.frames = frames[-8:]
            ctx.variables = dict(last.variables)
    else:
        sp = _special(text)
        if sp:
            ctx.error_type, ctx.message, ctx.phase, fail_file, fail_line, ctx.error_category = sp
        else:
            # Generic fallback: an assertion-ish failure with no parseable structure.
            low = text.lower()
            ctx.error_type = "AssertionError" if ("assert" in low or "failed" in low) else "UnknownError"
            ctx.error_category = category_for(ctx.error_type) if ctx.error_type != "UnknownError" else "unknown"
            ctx.message = (lines[-1].strip() if lines else "")[:200]
            if code_filename:
                fail_file = code_filename

    # Locate the source of the failing file.
    src: Optional[str] = None
    src_path = _resolve(pd, fail_file) if fail_file else None
    if src_path is not None:
        src = _read(src_path)
    if src is None and code is not None and (not fail_file or Path(fail_file).name == Path(code_filename or "").name):
        src = code
        fail_file = fail_file or code_filename
    src_lines = src.splitlines() if src else []
    tree = None
    if src and (fail_file or "").endswith(".py"):   # `module.exports = app` is valid Python — never parse JS as Python
        try:
            tree = ast.parse(src)
        except SyntaxError:
            tree = None

    if fail_file:
        shown = Path(fail_file)
        if pd is not None and shown.is_absolute():
            try:
                shown = shown.relative_to(pd)
            except ValueError:
                shown = Path(shown.name)
        ctx.failed_file = str(shown)
    ctx.line = fail_line
    if tree is not None and fail_line and not fail_func:
        fn = _enclosing_function(tree, fail_line)
        fail_func = fn.name if fn else None
    ctx.function = None if fail_func in (None, "<module>") else fail_func
    if ctx.failed_file:
        ctx.code_location = (
            f"{ctx.failed_file}:{fail_line}" if fail_line else ctx.failed_file
        ) + (f" in {fail_func}" if fail_func else "")

    if src_lines and fail_line:
        ctx.snippet = _numbered(src_lines, fail_line - 3, fail_line + 3, mark=fail_line)

    # Which function most likely needs the fix?
    ctx.affected_function = ctx.function
    if ctx.failed_file and _is_test_file(ctx.failed_file) and pd is not None:
        frame_src = _line_text(src_lines, fail_line) or next(
            (f.source for f in reversed(ctx.frames) if f.source), "") or ""
        wanted = [n for n in _called_names(frame_src) if n not in dir(builtins)]
        candidates = sorted(pd.rglob("*.py"))[:40]
        if code_filename:
            candidates.sort(key=lambda p: p.name != Path(code_filename).name)
        found = False
        for name in wanted:
            for p in candidates:
                if _is_test_file(p.name):
                    continue
                other = _read(p)
                if not other:
                    continue
                try:
                    t2 = ast.parse(other)
                except SyntaxError:
                    continue
                fn = _find_function(t2, name)
                if fn is not None:
                    ol = other.splitlines()
                    ctx.affected_function = name
                    ctx.suspect_file = str(p.relative_to(pd))
                    ctx.suspect_source = _numbered(ol, fn.lineno, min(fn.end_lineno or fn.lineno, fn.lineno + 39))
                    found = True
                    break
            if found:
                break

    ctx.hint, ctx.hint_confidence = _build_hint(ctx, tree, src_lines, pd)
    if surfaced_as is not None:   # keep ctx.message clean (it feeds the hint rules and the signature)
        ctx.hint += f" [Raised while handling the request; the test itself only saw: {surfaced_as[:80] or 'a failed assertion'}]"
    ctx.signature = _normalize_signature(ctx)
    return ctx


def format_context_for_prompt(ctx: FailureContext) -> str:
    """Compact, model-friendly rendering of a FailureContext."""
    parts = [f"Error: {ctx.error_type} ({ctx.error_category}) — {ctx.message}".strip()]
    if ctx.code_location:
        parts.append(f"Location: {ctx.code_location}")
    if ctx.variables:
        parts.append("Variables at failure: " + ", ".join(f"{k} = {v}" for k, v in list(ctx.variables.items())[:8]))
    if ctx.snippet:
        parts.append("Failing code:\n" + ctx.snippet)
    if ctx.suspect_source:
        parts.append(f"Function under test ({ctx.suspect_file}):\n" + ctx.suspect_source)
    return "\n".join(parts)
