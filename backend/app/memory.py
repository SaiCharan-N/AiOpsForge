"""Long-term memory (Phase 11: enhanced with execution context).

Deliberately organizational, not project-scoped: `retrieve_similar` searches
across every project's memory, because the entire point is that a fix
learned while building project A helps project B (a completely different,
later project) solve a similar bug faster. `save_memory` still records which
project first taught the system the fix, purely for provenance — it is
never used to filter retrieval.

(Project *isolation* — no cross-project leakage — applies to short-term
working memory instead; see short_term_memory.py.)

Phase 11: Memory now stores execution context (files changed, errors fixed,
model used) so that when a similar problem is found, not only the solution
but also the STRATEGY behind it can be reused.
"""
import json
import logging
from typing import Any

from sqlalchemy import text

from app import run_config
from app.config import settings
from app.db import engine
from app.embeddings import embed
from app.pipeline import metrics

logger = logging.getLogger("aiopsforge.memory")


@metrics.timed_db("memory.save_memory")
def save_memory(
    project_id: str,
    problem_signature: str,
    solution_summary: str,
    execution_context: dict | None = None,
) -> str:
    """Embeds and stores one long-term memory entry with optional execution
    context. Returns the new entry's id.
    
    Called after a task passes QA, summarizing (task -> fix) for future reuse.
    execution_context may include: files_changed, errors_fixed, model, timestamp, etc.

    No-op ("") when the run has long-term memory switched off (app.run_config).
    """
    if not run_config.memory_enabled():
        return ""
    vector = embed(problem_signature)
    execution_context = execution_context or {}
    
    query = text(
        "INSERT INTO memory_entries "
        "(project_id, problem_signature, solution_summary, embedding, execution_context) "
        "VALUES (:project_id, :problem_signature, :solution_summary, :embedding, :execution_context) "
        "RETURNING id"
    )
    with engine.begin() as conn:
        row = conn.execute(query, {
            "project_id": project_id,
            "problem_signature": problem_signature,
            "solution_summary": solution_summary,
            "embedding": json.dumps(vector),  # Store as JSON string
            "execution_context": json.dumps(execution_context),  # Store as JSON for JSONB
        }).fetchone()
    logger.info(
        "saved memory entry %s: problem='%s...' context=%s",
        row[0], problem_signature[:50], list(execution_context.keys()),
    )
    return str(row[0])


def _as_dict(value: Any) -> dict | None:
    """execution_context comes back as a dict from psycopg2 (JSONB is
    decoded for us) but as a str on drivers that don't decode it. The old
    code always called json.loads() on it, which raises TypeError on a dict
    — and the surrounding `except TypeError: pass` silently dropped the
    context on Postgres every time."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    try:
        loaded = json.loads(value)
        return loaded if isinstance(loaded, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


@metrics.timed_db("memory.retrieve_similar")
def retrieve_similar(
    problem_signature: str,
    top_k: int | None = None,
    return_context: bool = True,
) -> list[dict]:
    """Returns up to `top_k` past memory entries whose problem_signature is
    semantically similar to the given one, ordered by similarity
    (most similar first), filtered by settings.memory_similarity_threshold.
    Returns [] when the run has long-term memory switched off (app.run_config).

    Cosine similarity is computed in Python over JSON-encoded vectors
    (pgvector isn't installed on the alpine Postgres image).

    Bug fixed: the query used to be `ORDER BY created_at DESC LIMIT top_k`
    — i.e. it scored only the newest `top_k` (3) rows and never looked at
    anything older, so a strong match from a few tasks ago was invisible.
    It now scores the newest `settings.memory_scan_limit` rows and returns
    the best `top_k` of those.
    """
    if not run_config.memory_enabled():
        return []
    top_k = top_k or settings.memory_top_k
    vector = embed(problem_signature)

    query = text(
        "SELECT id, problem_signature, solution_summary, embedding, execution_context "
        "FROM memory_entries "
        "WHERE embedding IS NOT NULL "
        "ORDER BY created_at DESC "
        "LIMIT :scan"
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"scan": settings.memory_scan_limit}).fetchall()

    scored = []
    for r in rows:
        if not r.embedding:
            continue
        try:
            similarity = _cosine_similarity(json.loads(r.embedding), vector)
        except (json.JSONDecodeError, TypeError):
            continue
        if similarity >= settings.memory_similarity_threshold:
            entry = {
                "id": str(r.id),
                "problem_signature": r.problem_signature,
                "solution_summary": r.solution_summary,
                "similarity": float(similarity),
            }
            if return_context:
                ctx = _as_dict(r.execution_context)
                if ctx:
                    entry["execution_context"] = ctx
            scored.append(entry)

    scored.sort(key=lambda e: e["similarity"], reverse=True)
    results = scored[:top_k]
    logger.info(
        "memory retrieval: scanned %d row(s), %d above threshold %.2f, returning %d",
        len(rows), len(scored), settings.memory_similarity_threshold, len(results),
    )
    return results


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Computes cosine similarity between two vectors (0-1 scale).
    Returns a value in [0, 1] where 1 is identical.
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = sum(a * a for a in vec1) ** 0.5
    mag2 = sum(b * b for b in vec2) ** 0.5
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    # Cosine distance ranges [-1, 1]; convert to [0, 1] similarity
    cosine_dist = dot_product / (mag1 * mag2)
    return (cosine_dist + 1) / 2


# ---------------------------------------------------------------------------
# Failure patterns (Failure Recovery)
# ---------------------------------------------------------------------------

def _failure_text(error_type: str, error_category: str, message: str) -> str:
    """The text that gets embedded — same shape on save and on lookup so
    the two land near each other in embedding space."""
    return f"{error_type} ({error_category}): {message[:200]}"


def _raw_cosine(vec1: list[float], vec2: list[float]) -> float:
    """Plain cosine in [-1, 1] (unlike _cosine_similarity, which rescales
    to [0, 1] — that rescale makes even unrelated texts score ~0.7)."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    m1 = sum(a * a for a in vec1) ** 0.5
    m2 = sum(b * b for b in vec2) ** 0.5
    return 0.0 if m1 == 0 or m2 == 0 else dot / (m1 * m2)


@metrics.timed_db("memory.save_failure_pattern")
def save_failure_pattern(
    project_id: str | None,
    error_type: str,
    error_category: str,
    error_signature: str,
    error_message: str,
    code_location: str,
    fix_suggestion: str,
    fix_diff: str = "",
    task_type: str = "code",
) -> str:
    """Records that a failure with this signature was FIXED, and how.
    If the same (signature, task_type) is already stored, bumps
    `times_seen` and keeps the most recent fix instead of adding a
    near-duplicate row. Returns the row id ("" if the run has memory switched off).
    """
    if not run_config.memory_enabled():
        return ""
    try:
        vector = embed(_failure_text(error_type, error_category, error_message))
        emb = json.dumps(vector)
    except Exception as exc:  # noqa: BLE001 — an embedding outage shouldn't lose the pattern
        logger.warning("could not embed failure pattern, saving without embedding: %s", exc)
        emb = None

    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT id FROM failure_patterns WHERE error_signature = :sig AND task_type = :tt LIMIT 1"),
            {"sig": error_signature, "tt": task_type},
        ).fetchone()
        if existing:
            conn.execute(
                text(
                    "UPDATE failure_patterns SET times_seen = times_seen + 1, "
                    "fix_suggestion = :fix, fix_diff = :diff, code_location = :loc, "
                    "last_seen_at = CURRENT_TIMESTAMP, "
                    "embedding = COALESCE(:emb, embedding) WHERE id = :id"
                ),
                {"fix": fix_suggestion, "diff": fix_diff, "loc": code_location, "emb": emb, "id": existing[0]},
            )
            logger.info("failure pattern %s seen again: %s", existing[0], error_signature[:60])
            return str(existing[0])
        row = conn.execute(
            text(
                "INSERT INTO failure_patterns (project_id, error_type, error_category, error_signature, "
                "error_message, code_location, fix_suggestion, fix_diff, task_type, embedding) "
                "VALUES (:pid, :et, :ec, :sig, :msg, :loc, :fix, :diff, :tt, :emb) RETURNING id"
            ),
            {
                "pid": project_id, "et": error_type, "ec": error_category, "sig": error_signature,
                "msg": error_message[:500], "loc": code_location, "fix": fix_suggestion,
                "diff": fix_diff, "tt": task_type, "emb": emb,
            },
        ).fetchone()
    logger.info("saved failure pattern %s: %s", row[0], error_signature[:60])
    return str(row[0])


@metrics.timed_db("memory.find_similar_failures")
def find_similar_failures(
    error_type: str,
    error_category: str,
    error_signature: str,
    message: str = "",
    task_type: str | None = None,
    top_k: int = 2,
) -> list[dict]:
    """Past failures resembling this one, each with the fix that resolved it.

    Score = 1.0 for an exact signature match; otherwise raw cosine of the
    embedded "<type> (<category>): <message>" text, +0.10 if the error type
    matches, +0.05 if the category matches (capped at 1.0), kept if >=
    settings.failure_similarity_threshold. If the embedding service is
    down it degrades to type/category matching only (same type = 0.70,
    same category = 0.50) rather than returning nothing. Returns [] when the
    run has long-term memory switched off (app.run_config).
    """
    if not run_config.memory_enabled():
        return []
    try:
        qvec: list[float] | None = embed(_failure_text(error_type, error_category, message))
    except Exception as exc:  # noqa: BLE001
        logger.warning("failure lookup without embeddings (embed failed: %s)", exc)
        qvec = None

    sql = (
        "SELECT id, error_type, error_category, error_signature, error_message, code_location, "
        "fix_suggestion, fix_diff, embedding, times_seen FROM failure_patterns "
    )
    params: dict[str, Any] = {"scan": settings.memory_scan_limit}
    if task_type:
        sql += "WHERE task_type = :tt "
        params["tt"] = task_type
    sql += "ORDER BY last_seen_at DESC LIMIT :scan"
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    scored = []
    for r in rows:
        if r.error_signature == error_signature:
            score = 1.0
        else:
            same_type = r.error_type == error_type
            same_cat = r.error_category == error_category
            if qvec is not None and r.embedding:
                try:
                    cos = _raw_cosine(json.loads(r.embedding), qvec)
                except (json.JSONDecodeError, TypeError):
                    cos = 0.0
                score = min(1.0, max(cos, 0.0) + (0.10 if same_type else 0.0) + (0.05 if same_cat else 0.0))
            else:
                score = 0.70 if same_type else (0.50 if same_cat else 0.0)
        if score >= settings.failure_similarity_threshold:
            scored.append({
                "id": str(r.id),
                "error_type": r.error_type,
                "error_category": r.error_category,
                "error_signature": r.error_signature,
                "code_location": r.code_location or "",
                "fix_suggestion": r.fix_suggestion,
                "fix_diff": r.fix_diff or "",
                "times_seen": r.times_seen,
                "similarity": round(score, 3),
                # aliases so consumers written for retrieve_similar() entries work unchanged
                "problem_signature": f"{r.error_type}: {(r.error_message or '')[:80]}",
                "solution_summary": r.fix_suggestion,
            })
    scored.sort(key=lambda e: (e["similarity"], e["times_seen"]), reverse=True)
    logger.info("failure lookup: %d row(s) scanned, %d match(es)", len(rows), len(scored[:top_k]))
    return scored[:top_k]
