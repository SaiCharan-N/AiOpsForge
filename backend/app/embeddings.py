"""Embedding generation via Ollama (Week 7).

Kept separate from llm.py since embeddings use a different Ollama endpoint
and a different (smaller, non-generative) model than the coding model used
by Planner/Developer/QA.
"""
import logging

import requests

from app.config import settings

logger = logging.getLogger("aiopsforge.embeddings")

# Verified against Ollama's own model card for the configured default
# (nomic-embed-text -> 768 dims) before touching anything schema-related,
# per the task's "verify the actual embedding dimension before changing
# the schema" requirement. Not enforced as a hard column width, since
# memory_entries.embedding is TEXT (a JSON-encoded vector), not a pgvector
# `vector(N)` column — see db.ensure_memory_schema. Used only as an early,
# loud warning if `embedding_model` is ever pointed at a model with a
# different output size, since app.memory._cosine_similarity silently
# returns 0.0 (not an error) for mismatched-length vectors.
_EXPECTED_DIMENSIONS = {"nomic-embed-text": 768}
_dimension_checked = False


def embed(text: str) -> list[float]:
    """Returns a single embedding vector for `text`."""
    response = requests.post(
        f"{settings.ollama_base_url}/api/embeddings",
        json={
            "model": settings.embedding_model,
            "prompt": text,
            # Keep the embedding model resident in Ollama between calls
            # instead of falling back to Ollama's default 5-minute
            # keep_alive — retrieve_similar/save_memory run on nearly
            # every task, interleaved with the much larger coding model's
            # generate() calls, so an explicit keep_alive avoids repeatedly
            # paying model-load latency when the two models are swapped in
            # and out of memory back to back.
            "keep_alive": "30m",
        },
        timeout=60,
    )
    response.raise_for_status()
    vector = response.json().get("embedding")
    if not vector:
        raise ValueError("Ollama returned no embedding vector")

    global _dimension_checked
    if not _dimension_checked:
        expected = _EXPECTED_DIMENSIONS.get(settings.embedding_model)
        if expected is not None and len(vector) != expected:
            logger.warning(
                "embedding_model=%s returned %d-dim vector, expected %d — "
                "existing memory_entries rows written under a different "
                "model/dimension will silently fail similarity matching",
                settings.embedding_model, len(vector), expected,
            )
        else:
            logger.info("embedding_model=%s confirmed at %d dimensions", settings.embedding_model, len(vector))
        _dimension_checked = True

    return vector
