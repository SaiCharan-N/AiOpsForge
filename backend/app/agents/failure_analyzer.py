"""Failure Analyzer Agent (Phase 11, rebuilt for context-aware recovery).

Takes one QA failure and produces a targeted repair plan for the Developer:

  1. Structured parsing  — app.agents.failure_parser turns the raw output
     into a FailureContext (error type/category, exact line + function,
     variables at the failing frame, source snippet, deterministic hint).
  2. Repeat detection    — compares this failure's signature with the
     failures already seen for THIS task; a repeat means the previous fix
     didn't work and the Developer must be told so (and told what was
     already tried) instead of being asked to "fix it" again.
  3. Memory lookup       — app.memory.find_similar_failures returns past
     failures with the same/similar error and HOW they were fixed.
  4. Optional LLM refine — only when the deterministic hint is low
     confidence (settings.failure_analyzer_llm_mode = auto), because a
     small local model is slower and less reliable than the parser at
     "which line, which variable".

The returned dict keeps every key the previous version returned
(failed_file, error_type, root_cause, affected_function, repair_strategy,
priority, hint, similar_fixes) so existing consumers keep working, and adds
the new structured fields.
"""
import difflib
import logging
from typing import Optional

from app.agents import failure_parser
from app import run_config
from app.config import settings
from app.llm import generate_json, LLMOutputError
from app.schemas import FailureContext, TestResult

logger = logging.getLogger("aiopsforge.failure_analyzer")


FAILURE_ANALYZER_SYSTEM_PROMPT = """You are the Failure Analyzer in an automated software development pipeline.
You are given a STRUCTURED failure report (error type, exact location, variable values, failing code) and
what earlier attempts already tried. Give the Developer the smallest specific change that fixes it.

Respond with ONLY a JSON object of this exact shape:
{
  "root_cause": "one sentence naming the exact line/variable and why it fails",
  "hint": "one or two sentences: the concrete edit to make (name the line, function, variable)",
  "repair_strategy": "modify"
}
repair_strategy is one of: "modify" (edit the failing spot), "rewrite_function" (rewrite the one
function), "regenerate" (start the file over with a different approach). Do not restate the error
message as the hint. Never suggest changing the test's expected values."""


# ------------------------------------------------------------------ main ---

def analyze_failure(
    failure: TestResult,
    code_filename: str | None = None,
    *,
    context: Optional[FailureContext] = None,
    history: Optional[list[dict]] = None,
    project_id: Optional[str] = None,
    task_type: str = "code",
    code: Optional[str] = None,
) -> dict:
    """Analyzes a QA failure and returns a repair plan (see module docstring).

    `context` is the FailureContext qa_node already captured; when omitted
    (older callers/tests) it is parsed here from `failure`.
    `history` is this task's earlier failures, oldest first — a list of
    dicts with at least `signature`, `error_type`, `code_location`, `hint`.
    """
    if failure.passed:
        return {"analysis": "No failure to analyze"}

    history = history or []
    raw = failure.output or failure.error or "Unknown error"
    ctx = context or failure_parser.parse_failure(
        raw, project_id=project_id, workspace_root=settings.workspace_root,
        code_filename=code_filename, code=code,
    )

    # Phase 18: failure_recovery=False (an ablation arm) keeps the cheap, deterministic
    # parsing + repeat detection so metrics are comparable, but skips everything that
    # would cost extra tokens or feed the Developer: memory lookup and LLM refinement.
    recovery_on = run_config.failure_recovery_enabled()

    # --- similar past failures (long-term memory) ---
    similar: list[dict] = []
    try:
        if not recovery_on:
            raise LookupError("failure recovery disabled for this run")
        from app.memory import find_similar_failures  # local: keeps import-time light

        similar = find_similar_failures(
            ctx.error_type, ctx.error_category, ctx.signature,
            message=ctx.message, task_type=task_type, top_k=2,
        )
    except LookupError:
        pass
    except Exception as exc:  # noqa: BLE001 — memory is best-effort
        logger.warning("memory lookup failed during analysis: %s", exc)

    # --- repeat detection ---
    repeats = [h for h in history if h.get("signature") == ctx.signature]
    repeat_error = bool(repeats)

    root_cause = ctx.message or ctx.error_type
    hint = ctx.hint
    strategy = "modify"
    if ctx.affected_function and ctx.error_category in ("logic", "value", "type", "arithmetic", "lookup"):
        strategy = "rewrite_function" if repeat_error else "modify"
    if len(repeats) >= 2:
        strategy = "regenerate"

    # --- optional LLM refinement ---
    mode = (settings.failure_analyzer_llm_mode or "auto").lower()
    use_llm = recovery_on and (mode == "always" or (mode == "auto" and ctx.hint_confidence == "low"))
    llm_used = False
    if use_llm:
        try:
            refined = generate_json(_build_llm_prompt(ctx, history, similar), system=FAILURE_ANALYZER_SYSTEM_PROMPT)
            if isinstance(refined, dict) and str(refined.get("hint", "")).strip():
                hint = str(refined["hint"]).strip()
                root_cause = str(refined.get("root_cause") or root_cause).strip()
                if refined.get("repair_strategy") in ("modify", "rewrite_function", "regenerate"):
                    strategy = refined["repair_strategy"]
                llm_used = True
        except (LLMOutputError, Exception) as exc:  # noqa: BLE001
            logger.warning("failure analyzer LLM refine failed, using deterministic hint: %s", exc)

    # --- fold in past fixes and repeat warnings ---
    if similar:
        top = similar[0]
        hint += (
            f" A similar past failure ({top['error_type']}"
            + (f" at {top['code_location']}" if top.get("code_location") else "")
            + f") was fixed by: {top['fix_suggestion'][:220]}"
        )
    previous_attempts = [
        {"attempt": h.get("attempt"), "error_type": h.get("error_type"),
         "code_location": h.get("code_location"), "hint": (h.get("hint") or "")[:200]}
        for h in history
    ]
    if repeat_error:
        hint = (
            "REPEAT ERROR: the previous attempt failed with this same error, so its fix did not work. "
            "Do NOT resubmit the same code — take a different approach. " + hint
        )

    analysis = {
        # legacy keys (unchanged meaning)
        "failed_file": ctx.failed_file or code_filename or "unknown",
        "error_type": ctx.error_type,
        "root_cause": root_cause,
        "affected_function": ctx.affected_function or "unknown",
        "repair_strategy": strategy,
        "priority": "high",
        "hint": hint,
        "similar_fixes": similar,
        # new structured fields
        "error_category": ctx.error_category,
        "code_location": ctx.code_location,
        "line": ctx.line,
        "message": ctx.message,
        "variables": ctx.variables,
        "snippet": ctx.snippet,
        "suspect_source": ctx.suspect_source,
        "suspect_file": ctx.suspect_file,
        "signature": ctx.signature,
        "repeat_error": repeat_error,
        "repeat_count": len(repeats),
        "previous_attempts": previous_attempts,
        "hint_confidence": ctx.hint_confidence,
        "llm_used": llm_used,
        "phase": ctx.phase,
    }
    logger.info(
        "failure analysis: type=%s cat=%s loc=%s repeat=%s similar=%d llm=%s",
        ctx.error_type, ctx.error_category, ctx.code_location, repeat_error, len(similar), llm_used,
    )
    return analysis


def _build_llm_prompt(ctx: FailureContext, history: list[dict], similar: list[dict]) -> str:
    parts = [failure_parser.format_context_for_prompt(ctx)]
    if ctx.hint:
        parts.append(f"Automatic first guess: {ctx.hint}")
    if history:
        parts.append("Earlier attempts on this task already failed with:\n" + "\n".join(
            f"- attempt {h.get('attempt')}: {h.get('error_type')} at {h.get('code_location')}" for h in history))
    if similar:
        parts.append("Past fixes for similar failures:\n" + "\n".join(
            f"- {s['error_type']}: {s['fix_suggestion'][:160]}" for s in similar))
    parts.append("Give the JSON described in the system prompt.")
    return "\n\n".join(parts)


# ------------------------------------------- what-changed summarization ----

def summarize_fix(prev_code: Optional[str], new_code: Optional[str], hint: str = "") -> tuple[str, str]:
    """Describes what changed between the failing code and the code that
    then passed QA. Returns (fix_suggestion, fix_diff): the suggestion is a
    short natural-language summary stored in long-term memory and shown to
    future analyses; the diff is the changed lines themselves.
    """
    diff_lines: list[str] = []
    if prev_code and new_code and prev_code != new_code:
        for line in difflib.unified_diff(
            prev_code.splitlines(), new_code.splitlines(), lineterm="", n=0
        ):
            if line.startswith(("+++", "---", "@@")):
                continue
            diff_lines.append(line)
    added = [l[1:].strip() for l in diff_lines if l.startswith("+") and l[1:].strip()]
    removed = [l[1:].strip() for l in diff_lines if l.startswith("-") and l[1:].strip()]

    parts = []
    if hint:
        parts.append(hint.replace("REPEAT ERROR: ", "").split(" A similar past failure")[0].strip()[:220])
    if added and removed:
        parts.append(f"Changed: {'; '.join(removed[:2])} -> {'; '.join(added[:3])}"[:260])
    elif added:
        parts.append(f"Added: {'; '.join(added[:3])}"[:260])
    suggestion = " | ".join(p for p in parts if p) or "Regenerated the code and it passed."
    return suggestion, "\n".join(diff_lines[:40])
