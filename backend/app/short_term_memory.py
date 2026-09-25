"""Short-term / working memory (Week 8 / Days 36-40).

This is Redis-backed, per-project-namespaced state for ONE in-progress run —
the opposite of long-term memory: it must NEVER leak between two unrelated
projects, especially if they happen to run concurrently. Every key is
prefixed with the project's namespace by construction, so a bug elsewhere in
the code can't accidentally read another project's working state — there's
no key it *could* guess that would collide.

Cleared automatically via TTL, and explicitly on project completion
(clear_project) as a second, immediate safety net rather than waiting for
the TTL to expire.
"""
from app.config import settings
from app.redis_client import get_client


def _namespaced_key(project_id: str, key: str) -> str:
    return f"project:{project_id}:{key}"


def set_working(project_id: str, key: str, value: str, ttl: int | None = None) -> None:
    client = get_client()
    client.set(_namespaced_key(project_id, key), value, ex=ttl or settings.short_term_ttl_seconds)


def get_working(project_id: str, key: str) -> str | None:
    client = get_client()
    return client.get(_namespaced_key(project_id, key))


def list_project_keys(project_id: str) -> list[str]:
    """Returns every short-term key currently held for a project — used by
    the isolation test to prove no OTHER project's keys show up here.
    """
    client = get_client()
    prefix = _namespaced_key(project_id, "")
    return [k for k in client.scan_iter(match=f"{prefix}*")]


def clear_project(project_id: str) -> int:
    """Deletes every short-term key for a project immediately (Day 39) —
    called when a run finishes (success or escalation), rather than relying
    solely on the TTL. Returns the number of keys deleted.
    """
    client = get_client()
    keys = list_project_keys(project_id)
    if not keys:
        return 0
    return client.delete(*keys)
