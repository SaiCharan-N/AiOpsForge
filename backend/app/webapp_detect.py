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
]
_WEBAPP_RE = re.compile("|".join(_WEBAPP_PATTERNS), re.IGNORECASE)


def is_webapp_request(request: str) -> bool:
    """True if `request` should be routed to the scaffolder instead of the
    normal single-file task decomposition.
    """
    return bool(_WEBAPP_RE.search(request or ""))
