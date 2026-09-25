"""Scaffolder Agent (Phase 7).

Handles Task(type="webapp_scaffold") — produces a complete, minimal,
RUNNABLE multi-file FastAPI + Jinja2 web app from a plain-English request,
instead of the single file/function app.agents.developer produces for
ordinary code tasks.

Kept as its own module (not folded into developer.py) because the output
shape is fundamentally different — a dict of {relative_path: content}
covering several files, not one (code, filename) pair — and because a
website request warrants a different system prompt entirely (routes,
templates, static assets) rather than a variant of the single-function one.
"""
from app.llm import generate_json, LLMOutputError

SCAFFOLDER_SYSTEM_PROMPT = """You are the Scaffolder Agent in an automated \
software development pipeline. Given a user's request for a website or web \
app, produce a complete, minimal, RUNNABLE FastAPI + Jinja2 web application.

Respond with ONLY a JSON object of this exact shape, nothing else:
{"files": {
  "main.py": "<full file content>",
  "requirements.txt": "<full file content>",
  "templates/index.html": "<full file content>",
  "static/style.css": "<full file content>"
}}

Rules:
- main.py must be a working FastAPI app: mount /static as StaticFiles, use \
Jinja2Templates for rendering, and include at least a GET "/" route that \
renders templates/index.html.
- If the request describes multiple pages or features, add more \
templates/*.html files and matching routes in main.py — don't cram \
everything into one page if the request implies more than one.
- requirements.txt must list exact package names needed, at minimum \
fastapi, uvicorn, and jinja2.
- Keep styling in static/style.css, linked from the templates, not inline.
- Every value must be a single JSON string with newlines escaped as \\n — \
valid JSON only. No markdown code fences, no prose outside the JSON object."""

# Guaranteed-runnable minimum, used only if the model's output can't be
# parsed into a usable file set even after generate_json's internal
# retries — mirrors the same fallback philosophy as
# app.agents.planner.plan()'s single-catch-all-task fallback: a plain,
# working starter site is more useful than a hard crash on a request the
# local model happened to struggle with.
_FALLBACK_FILES = {
    "main.py": (
        "from fastapi import FastAPI, Request\n"
        "from fastapi.staticfiles import StaticFiles\n"
        "from fastapi.templating import Jinja2Templates\n\n"
        "app = FastAPI()\n"
        "app.mount(\"/static\", StaticFiles(directory=\"static\"), name=\"static\")\n"
        "templates = Jinja2Templates(directory=\"templates\")\n\n\n"
        "@app.get(\"/\")\n"
        "def index(request: Request):\n"
        "    return templates.TemplateResponse(\n"
        "        \"index.html\", {\"request\": request, \"title\": \"My Site\"}\n"
        "    )\n"
    ),
    "requirements.txt": "fastapi\nuvicorn\njinja2\npython-multipart\n",
    "templates/index.html": (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        "  <meta charset=\"UTF-8\">\n"
        "  <title>{{ title }}</title>\n"
        "  <link rel=\"stylesheet\" href=\"/static/style.css\">\n"
        "</head>\n<body>\n"
        "  <h1>{{ title }}</h1>\n"
        "  <p>This is a starter page — the Scaffolder Agent's fallback template.</p>\n"
        "</body>\n</html>\n"
    ),
    "static/style.css": (
        "body { font-family: sans-serif; max-width: 640px; margin: 3rem auto; "
        "padding: 0 1rem; color: #222; }\n"
        "h1 { color: #4f46e5; }\n"
    ),
}


def _is_safe_relative_path(path: str) -> bool:
    """Rejects anything that isn't a clean relative path — belt-and-suspenders
    on top of mcp_server's own _safe_join traversal check, so a bad path
    from the model never even reaches the MCP write_file call.
    """
    if not path or path.startswith("/") or path.startswith("\\"):
        return False
    parts = path.replace("\\", "/").split("/")
    return ".." not in parts and all(p.strip() for p in parts)


def scaffold_webapp(
    task_description: str,
    previous_failure=None,
    original_request: str | None = None,
    failure_analysis: dict | None = None,
) -> dict[str, str]:
    """Returns {relative_path: file_content} for the scaffolded app. Never
    raises — falls back to a guaranteed-runnable minimal site on any
    parsing failure, same philosophy as planner.plan()'s fallback.

    `previous_failure` (a TestResult, or None on a first attempt): when
    given, the check_webapp output from the failed previous attempt is
    included so this is a genuine fix attempt rather than a blind
    regeneration — mirrors developer.generate_code's `previous_failure`
    parameter and closes a gap in the initial Phase 7 cut, where a failed
    scaffold retried with zero memory of what broke.

    `original_request` (Phase 6 communication-mode ablation, extended to
    webapp tasks): the caller passes this only in "blackboard" mode — see
    developer.generate_code's docstring for the full C^MP/C^BB mapping.
    `task_description` is the Planner's SUMMARY of the site to build (see
    planner._summarize_webapp_request), not the raw request verbatim, so
    this distinction is now meaningful for webapp tasks the same way it
    already was for single-file code tasks.
    """
    prompt = ""
    if original_request:
        prompt += (
            f"Original user request (for full context — the task below is "
            f"the Planner's summary of this; use this if it has detail the "
            f"summary doesn't):\n{original_request}\n\n"
        )

    prompt += f"Build this: {task_description}"

    if previous_failure is not None:
        detail = previous_failure.error or previous_failure.output or "(no detail available)"
        prompt += (
            f"\n\nYour previous attempt FAILED this check:\n{detail}\n\n"
            f"Fix the specific issue above. Keep everything else that was "
            f"working intact — this is a fix, not a from-scratch rewrite."
        )
        if failure_analysis and failure_analysis.get("root_cause"):
            hint = failure_analysis.get("hint", "")
            failed_file = failure_analysis.get("failed_file", "unknown")
            prompt += (
                f"\n\nFailure Analyzer diagnosis — likely file: {failed_file}; "
                f"root cause: {failure_analysis['root_cause']}. {hint}"
            )

    prompt += "\n\nRespond with the JSON object described above."

    try:
        parsed = generate_json(prompt, system=SCAFFOLDER_SYSTEM_PROMPT)
        files = parsed["files"]
        if not isinstance(files, dict) or not files:
            raise ValueError("'files' was empty or not an object")

        cleaned = {
            path.strip(): content
            for path, content in files.items()
            if isinstance(path, str) and isinstance(content, str)
            and _is_safe_relative_path(path.strip())
        }
        if not cleaned:
            raise ValueError("no valid file entries survived path safety filtering")
        return cleaned
    except (LLMOutputError, KeyError, ValueError, TypeError):
        return dict(_FALLBACK_FILES)
