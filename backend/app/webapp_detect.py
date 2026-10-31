"""Phase 7: detects whether a request is asking for a website/web app
rather than a single function or script.

Deliberately a simple keyword heuristic, not an LLM call — this decision
needs to happen BEFORE the Planner spends a call deciding how to break the
request down, and a wrong guess here is cheap to live with (worst case: a
plain script request gets scaffolded as a tiny one-page site, or a website
request gets treated as a single-file task and the Planner's normal
decomposition still produces something, just not a proper multi-file app).
"""
import re
from typing import Optional

from app.scaffolds import FRAMEWORKS, SUPPORTED, get_framework

_WEBAPP_PATTERNS = [
    r"\bwebsite\b",
    r"\bweb\s*app(?:lication)?\b",
    r"\bweb\s*page\b",
    r"\bwebpage\b",
    r"\bhtml\s*page\b",
    r"\blanding\s*page\b",
    r"\bflask\s*app\b",
    r"\bfastapi\s*app\b",
    r"\bsite\s*with\s*(?:pages|routes|a\s*homepage)\b",
    # "<framework> app/api/project/..." — flask & fastapi are also used for
    # single-file route tasks, so they need a project-sized noun to count.
    r"\b(?:flask|fast\s*-?\s*api|django)\s+(?:app|application|api|project|site|website|backend|server|web\s*app)\b",
]
_WEBAPP_RE = re.compile("|".join(_WEBAPP_PATTERNS), re.IGNORECASE)


# --- framework detection (Improvement #4) ---------------------------------
# Patterns are deliberately specific: "express" and "next" are ordinary
# English words, "node" is a data-structure term, so each needs a
# disambiguating companion ("express app", "next.js", "node server").
_FRAMEWORK_PATTERNS: dict[str, list[str]] = {
    "fastapi": [r"\bfast\s*-?\s*api\b"],
    "flask": [r"\bflask\b"],
    "django": [r"\bdjango\b"],
    "express": [
        r"\bexpress\.?\s?js\b",
        r"\bexpress\s+(?:app|application|server|api|backend|site|website|web\s*app)\b",
        r"\bnode\.?\s?js\b",
        r"\bnode\s+(?:app|application|server|api|backend)\b",
    ],
    "next": [
        r"\bnext\.?\s?js\b",
        r"\breact(?:\.?js)?\s+(?:app|application|site|website|frontend|web\s*app|project)\b",
    ],
}
_FRAMEWORK_RES = {k: re.compile("|".join(v), re.IGNORECASE) for k, v in _FRAMEWORK_PATTERNS.items()}
# "use Flask, NOT Django" / "without Django" / "instead of Express": a mention preceded by
# one of these is something the user does NOT want.
_NEGATION_RE = re.compile(
    r"(?:\bnot|\bno|\bwithout|\binstead\s+of|\brather\s+than|\bavoid(?:ing)?|\bnever|\bexcept)\s+(?:use\s+|using\s+|a\s+|the\s+)?$",
    re.IGNORECASE,
)


def detect_framework(request: str) -> Optional[str]:
    """The framework a request names ("flask", "django", "express", "next",
    "fastapi"), or None. When several are named the earliest NON-negated
    mention wins ("Flask, not Django" -> flask; "not Django, use Flask" -> flask).
    """
    text = request or ""
    best: Optional[tuple[int, str]] = None
    for fw_id, rx in _FRAMEWORK_RES.items():
        for m in rx.finditer(text):
            if _NEGATION_RE.search(text[max(0, m.start() - 24):m.start()]):
                continue
            if best is None or m.start() < best[0]:
                best = (m.start(), fw_id)
            break
    return best[1] if best else None


# Flask and FastAPI are also used for single-file route tasks ("write a Flask
# route that ..."), so naming one only means "web app" when the request is
# project-shaped and NOT about one route/endpoint/function.
_CODE_TASK_WORDS = re.compile(
    r"\b(?:route|endpoint|view|handler|function|decorator|blueprint|middleware|helper|class|method)\b", re.IGNORECASE)
_PROJECT_NOUNS = re.compile(
    r"\b(?:app|application|api|service|website|site|project|backend|server|dashboard|portal)\b", re.IGNORECASE)


def is_webapp_request(request: str) -> bool:
    """True if `request` should be routed to the scaffolder instead of the
    normal single-file task decomposition. Django, Express and Next.js are
    whole-project frameworks, so merely naming one makes it a web app.
    """
    if _WEBAPP_RE.search(request or ""):
        return True
    fw = detect_framework(request)
    if not fw:
        return False
    if FRAMEWORKS[fw].project_level:
        return True
    # flask / fastapi
    return not _CODE_TASK_WORDS.search(request) and bool(_PROJECT_NOUNS.search(request))


def validate_framework(framework: Optional[str]) -> Optional[str]:
    """Normalises an explicit framework choice (API param / UI select).
    None/""/"auto" -> None (auto-detect). Unknown -> ValueError listing the supported ones."""
    if framework is None or str(framework).strip().lower() in ("", "auto"):
        return None
    fw = str(framework).strip().lower().replace(".", "").replace(" ", "")
    fw = {"nextjs": "next", "expressjs": "express", "node": "express", "nodejs": "express"}.get(fw, fw)
    if fw not in SUPPORTED:
        raise ValueError(f"unknown framework {framework!r}; supported: {', '.join(SUPPORTED)} (or omit to auto-detect)")
    return fw


def resolve_framework(request: str, explicit: Optional[str] = None) -> Optional[str]:
    """Explicit choice (already validated) beats detection from the request text."""
    return explicit or detect_framework(request)


__all__ = ["detect_framework", "is_webapp_request", "validate_framework", "resolve_framework", "get_framework"]
