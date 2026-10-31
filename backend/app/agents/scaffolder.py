"""Scaffolder Agent (Phase 7).

Handles Task(type="webapp_scaffold") — produces a complete, minimal,
RUNNABLE multi-file web app from a plain-English request, instead of the
single file/function app.agents.developer produces for ordinary code tasks.
The framework (FastAPI by default; also Flask, Django, Express, Next.js) is
chosen per task — see app.scaffolds for the registry, starter projects and
prompts.

Kept as its own module (not folded into developer.py) because the output
shape is fundamentally different — a dict of {relative_path: content}
covering several files, not one (code, filename) pair — and because a
website request warrants a different system prompt entirely (routes,
templates, static assets) rather than a variant of the single-function one.
"""
import logging

from app.llm import generate_json, LLMOutputError
from app.scaffolds import FORBIDDEN_PREFIXES, Framework, get_framework

logger = logging.getLogger("aiopsforge.scaffolder")

# Backwards-compatible names: the FastAPI scaffold is the registry's legacy
# entry, and these two constants are exactly what they were before frameworks existed.
SCAFFOLDER_SYSTEM_PROMPT = get_framework("fastapi").system_prompt

# Guaranteed-runnable minimum for FastAPI, used only if the model's output
# can't be parsed into a usable file set even after generate_json's internal
# retries — a plain, working starter site is more useful than a hard crash on
# a request the local model happened to struggle with. (Every framework has
# its own starter project under app/scaffolds/templates/<id>/.)
_FALLBACK_FILES = get_framework("fastapi").template_files()


def _is_safe_relative_path(path: str) -> bool:
    """Rejects anything that isn't a clean relative path — belt-and-suspenders
    on top of mcp_server's own _safe_join traversal check, so a bad path
    from the model never even reaches the MCP write_file call. Also rejects
    directories the platform itself creates (.venv/, node_modules/, ...).
    """
    if not path or path.startswith("/") or path.startswith("\\"):
        return False
    parts = path.replace("\\", "/").split("/")
    if ".." in parts or not all(p.strip() for p in parts):
        return False
    return not any(path.replace("\\", "/").startswith(p) for p in FORBIDDEN_PREFIXES)


def _backfill_required_files(files: dict[str, str], framework: str | Framework | None = None) -> dict[str, str]:
    """Guarantees the minimum file set the framework's contract requires is
    actually present, backfilling only what's missing from its starter
    project rather than discarding a good partial result.

    Bug this fixes (originally FastAPI-only; now every framework): the
    model easily produces valid JSON that's missing files main.py/app.py
    references (templates/index.html, static/style.css, ...) — dropped under
    prompt/context pressure, especially on a retry with a long
    failure-analysis block appended — giving a project that compiles fine
    but crashes on boot.
    """
    fw = framework if isinstance(framework, Framework) else get_framework(framework)
    return fw.backfill(files)


def scaffold_webapp(
    task_description: str,
    previous_failure=None,
    original_request: str | None = None,
    failure_analysis: dict | None = None,
    user_guidance: list[str] | None = None,
    framework: str | None = None,
) -> dict[str, str]:
    """Returns {relative_path: file_content} for the scaffolded app. Never
    raises — falls back to the framework's guaranteed-runnable starter
    project on any parsing failure, same philosophy as planner.plan()'s fallback.

    `framework` (see app.scaffolds): which stack to build — "fastapi"
    (default, legacy behaviour), "flask", "django", "express" or "next".
    Besides the prompt, it decides which files are REQUIRED (backfilled from
    the starter project), which are platform-owned (the model's version is
    ignored) and which dependencies must be declared.

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

    fw = get_framework(framework)
    if fw.id != "fastapi":
        # A known-good skeleton to edit beats inventing the project layout from
        # nothing — especially for a small local model on a multi-file stack.
        prompt += (f"A starter {fw.label} project is already on disk. Its key files:\n"
                   f"{fw.starter_excerpt()}\n\n")

    prompt += f"Build this: {task_description}"
    if user_guidance:
        prompt += ("\n\nThe user reviewed the work so far and gave this guidance — follow it:\n"
                   + "\n".join(f"- {g}" for g in user_guidance))

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
            where = failure_analysis.get("code_location")
            prompt += (
                f"\n\nFailure Analyzer diagnosis — likely file: {failed_file}"
                f"{' (' + where + ')' if where else ''}; "
                f"root cause: {failure_analysis['root_cause']}. {hint}"
            )
            prev = failure_analysis.get("previous_attempts") or []
            if prev:
                prompt += "\nAlready tried and failed: " + "; ".join(
                    f"attempt {p.get('attempt')}: {p.get('error_type')} at {p.get('code_location')}"
                    for p in prev)

    prompt += "\n\nRespond with the JSON object described above."

    try:
        parsed = generate_json(prompt, system=fw.system_prompt)
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
        ignored = sorted(set(cleaned) & fw.locked)
        if ignored:
            logger.info("scaffolder(%s): ignoring platform-owned file(s) from the model: %s", fw.id, ignored)
        files = fw.ensure_dependencies(fw.backfill(fw.drop_locked(cleaned)))
        return files
    except (LLMOutputError, KeyError, ValueError, TypeError):
        return fw.template_files()
