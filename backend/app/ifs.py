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
import re

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
