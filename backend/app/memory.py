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

from app.config import settings
from app.db import engine
from app.embeddings import embed

logger = logging.getLogger("aiopsforge.memory")


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
    """
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


def retrieve_similar(
    problem_signature: str,
    top_k: int | None = None,
    return_context: bool = True,
) -> list[dict]:
    """Returns up to `top_k` past memory entries whose problem_signature is
    semantically similar to the given one, ordered by similarity
    (most similar first), filtered by settings.memory_similarity_threshold.

    Cosine distance computed via JSON vector comparison (since pgvector is
    unavailable on alpine postgres).
    
    If return_context=True, includes execution_context so strategies can be reused.
    """
    top_k = top_k or settings.memory_top_k
    vector = embed(problem_signature)
    
    query = text(
        "SELECT id, problem_signature, solution_summary, embedding, execution_context "
        "FROM memory_entries "
        "WHERE embedding IS NOT NULL "
        "ORDER BY created_at DESC "
        "LIMIT :top_k"
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"top_k": top_k}).fetchall()

    results = []
    for r in rows:
        if r.embedding:
            similarity = _cosine_similarity(json.loads(r.embedding), vector)
            if similarity >= settings.memory_similarity_threshold:
                entry = {
                    "id": str(r.id),
                    "problem_signature": r.problem_signature,
                    "solution_summary": r.solution_summary,
                    "similarity": float(similarity),
                }
                if return_context and r.execution_context:
                    try:
                        entry["execution_context"] = json.loads(r.execution_context)
                    except (json.JSONDecodeError, TypeError):
                        pass
                results.append(entry)
    
    logger.info(
        "memory retrieval: %d candidate(s), %d above threshold %.2f",
        len(rows), len(results), settings.memory_similarity_threshold,
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
