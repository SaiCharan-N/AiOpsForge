"""Information Fidelity Score (IFS) — Phase 6.

Directly mirrors the metric introduced in SE-Blackboard (Liu et al., IEEE
Access 2026): a stage-level score measuring how well key technical entities
from the ORIGINAL request survive into an agent's output, independent of
whether the resulting code is actually correct.

    IFS_s = |entities in agent s's output| / |entities in original request|

Their version extracts entities suited to bug-fixing an existing repo
(function names, class names, file paths, error types) from an issue
description. AIOpsForge generates fresh code from a plain-English request
rather than patching an existing codebase, so there's no file path or
traceback to extract — the entity set here is instead: quoted terms,
numbers, hyphenated/versioned identifiers (ISBN-13, UTF-8), acronyms
(JSON, CSV, API), and snake_case/camelCase/PascalCase identifiers named
directly in the request. Same idea — precise technical detail a summary
could plausibly drop — adapted to a code-generation rather than
bug-fixing pipeline.

Matching their matcher design exactly (see their Section V-B.5): an entity
counts as "preserved" on any case-insensitive SUBSTRING hit. They note this
is generous — it also motivated their robustness check with a stricter
difflib-based matcher — so treat this the same way: a useful comparative
signal between communication modes, not a precise ground truth.
"""
import difflib
import logging
import re
from typing import Optional

# Order matters only for readability when debugging; entities are deduped
# into a set before scoring.
_PATTERNS: list[re.Pattern] = [
    re.compile(r'"([^"]{2,60})"'),                    # "double quoted"
    re.compile(r"'([^']{2,60})'"),                     # 'single quoted'
    re.compile(r"`([^`]{2,60})`"),                      # `backtick quoted`
    re.compile(r"\b[A-Za-z]+-\d+(?:\.\d+)*\b"),          # ISBN-13, UTF-8, RFC-822
    re.compile(r"\b\d+\.\d+\.\d+\b"),                    # semver-style 1.0.3
    re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"),    # snake_case
    re.compile(r"\b[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+\b"),# camelCase
    re.compile(r"\b[A-Z][a-z0-9]*(?:[A-Z][a-z0-9]*)+\b"),# PascalCase
    re.compile(r"\b[A-Z]{2,6}\b"),                       # acronyms: JSON, CSV, API, ISBN
    re.compile(r"\b\w+\.\w{2,4}\b"),                     # file-like: reverse_string.py
]

# Acronym pattern above is noisy on its own (matches "THE" mid-sentence if
# capitalized oddly, sentence-starting words, etc.) — filtered separately.
_STOPWORD_ACRONYMS = {"THE", "AND", "FOR", "ARE", "NOT", "BUT", "ALL", "ANY"}


def extract_entities(text: str) -> set[str]:
    """Pulls out candidate technical entities from `text`. Deliberately
    rule-based and imprecise in the same spirit as SE-Blackboard's own
    extractor (their paper explicitly flags this as a construct-validity
    limitation, not a hidden flaw) — the goal is a consistent, cheap signal
    comparable ACROSS runs of the same pipeline, not a gold-standard NER.
    """
    if not text:
        return set()

    found: set[str] = set()
    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            token = match.group(1) if match.groups() else match.group(0)
            token = token.strip()
            if len(token) < 2:
                continue
            if token.isupper() and token in _STOPWORD_ACRONYMS:
                continue
            found.add(token)
    return found


def compute_ifs(original_request: str, agent_output: str) -> float | None:
    """Returns the fraction of entities extracted from `original_request`
    that appear (case-insensitive substring match) somewhere in
    `agent_output`. Returns None — not 0.0 — when the request yields no
    extractable entities at all, since a score of 0 there would misleadingly
    read as "total information loss" rather than "nothing precise enough to
    measure." Callers should exclude None scores from aggregate stats, the
    same way SE-Blackboard reports IFS over only non-zero pairs (n=28/50 in
    their Coder-stage result).
    """
    entities = extract_entities(original_request)
    if not entities:
        return None

    output_lower = (agent_output or "").lower()
    preserved = sum(1 for e in entities if e.lower() in output_lower)
    return preserved / len(entities)


# --------------------------------------------------------------------------
# Phase 18 — IFS *recording*: score, strict score, which entities were lost,
# and persistence into metrics_logs (kind="ifs") so every run's fidelity can be
# exported, charted and cited, per stage (planner / developer) and per attempt.
# --------------------------------------------------------------------------

logger = logging.getLogger("aiopsforge.ifs")

# The substring matcher above is generous (the original paper says so about its own).
# SE-Blackboard's robustness check uses a stricter difflib-based matcher; its exact
# threshold is not reproduced here, so STRICT_THRESHOLD is OUR choice — report it in the
# paper as "a stricter whole-token / difflib matcher (ratio >= 0.85)", not as theirs.
STRICT_THRESHOLD = 0.85
_TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-\.]*")
STAGES = ("planner", "developer")


def _strict_hit(entity: str, tokens: set[str], lowered: str) -> bool:
    """Strict match: the entity appears as a whole token (word boundaries on both
    sides, case-insensitive), or — for entities made of several tokens — as a
    whole-word phrase, or a token of the output is a near-duplicate of it
    (difflib ratio >= STRICT_THRESHOLD, which forgives a typo / pluralisation)."""
    e = entity.lower()
    if e in tokens:
        return True
    if re.search(r"(?<![A-Za-z0-9_])" + re.escape(e) + r"(?![A-Za-z0-9_])", lowered):
        return True
    for tok in tokens:
        if abs(len(tok) - len(e)) > max(2, len(e) // 5):
            continue
        if difflib.SequenceMatcher(None, e, tok).ratio() >= STRICT_THRESHOLD:
            return True
    return False


def compute_ifs_detail(original_request: str, agent_output: str) -> Optional[dict]:
    """Everything the paper needs about ONE stage output, or None when the request has
    no extractable entities (see compute_ifs for why None, not 0.0).

        {"entities": n, "preserved": k, "score": k/n,                  # substring matcher
         "strict_preserved": j, "strict_score": j/n,                    # stricter matcher
         "missing": [...], "missing_strict": [...]}                     # what was lost (<= 25 each)
    """
    entities = sorted(extract_entities(original_request))
    if not entities:
        return None
    lowered = (agent_output or "").lower()
    tokens = {t.lower().rstrip(".-") for t in _TOKEN_RE.findall(agent_output or "")}
    kept = [e for e in entities if e.lower() in lowered]
    strict = [e for e in entities if _strict_hit(e, tokens, lowered)]
    n = len(entities)
    return {
        "entities": n, "preserved": len(kept), "score": len(kept) / n,
        "strict_preserved": len(strict), "strict_score": len(strict) / n,
        "missing": [e for e in entities if e not in kept][:25],
        "missing_strict": [e for e in entities if e not in strict][:25],
    }


def record_ifs(stage: str, original_request: str, agent_output: str, *,
               state: Optional[dict] = None, communication_mode: Optional[str] = None) -> Optional[dict]:
    """Scores `agent_output` against `original_request` and RECORDS the result.

    stage  "planner" (request -> the task descriptions the Planner produced) or
           "developer" (request -> the file name + code / scaffold the Developer wrote).
    state  when given, the pipeline state is updated so existing code keeps working:
           developer -> state["ifs_score"] (substring) and state["ifs_strict"];
           planner   -> state["ifs_planner"].

    Persists one metrics_logs row (kind="ifs", name=stage) per call — for the Developer
    that is one per attempt — whose meta holds the full detail above plus the task and
    communication mode. Never raises: recording must not break a run.
    Returns the detail dict (or None when the request has no extractable entities).
    """
    try:
        detail = compute_ifs_detail(original_request, agent_output)
    except Exception as exc:  # noqa: BLE001
        logger.debug("record_ifs failed: %s", exc)
        return None
    if state is not None:
        if stage == "developer":
            state["ifs_score"] = detail["score"] if detail else None
            state["ifs_strict"] = detail["strict_score"] if detail else None
        elif stage == "planner":
            state["ifs_planner"] = detail["score"] if detail else None
    try:
        from app.pipeline import metrics  # lazy: metrics imports app.config / run_config
        task = (state or {}).get("current_task") or {}
        meta = {"stage": stage, "scorable": detail is not None,
                "communication_mode": communication_mode or (state or {}).get("communication_mode"),
                "matcher": {"substring": "case-insensitive", "strict": f"whole-token or difflib>={STRICT_THRESHOLD}"}}
        if detail:
            meta.update(detail)
        metrics.record("ifs", stage, 0, ok=detail is not None, meta=meta,
                       task_order=task.get("order") if stage == "developer" else None)
    except Exception as exc:  # noqa: BLE001
        logger.debug("record_ifs persistence failed: %s", exc)
    return detail
