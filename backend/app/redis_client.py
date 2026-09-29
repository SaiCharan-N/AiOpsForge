"""Redis connectivity for short-term/working memory.

Phase 1 only needs a basic SET/GET roundtrip to prove the container is wired
up correctly. The real short-term-memory design (namespacing, TTLs on
project completion) is Phase 3 / Week 8 work.
"""
import redis

from app.config import settings

_client: redis.Redis | None = None


def get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    return _client


def roundtrip_check(key: str = "aiopsforge:healthcheck", value: str = "ok") -> bool:
    """SET a value, GET it back, and confirm it matches."""
    client = get_client()
    client.set(key, value, ex=30)
    return client.get(key) == value
