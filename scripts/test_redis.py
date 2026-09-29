#!/usr/bin/env python3
"""Day 5: standalone Redis SET/GET roundtrip check.

Run from the host with the container ports exposed:
    python3 scripts/test_redis.py
Or inside the backend container:
    docker compose exec backend python -m app.redis_client   (not wired, use this script instead)
"""
import os
import sys

import redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def main() -> int:
    client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    key, value = "aiopsforge:healthcheck", "ok"
    client.set(key, value, ex=30)
    result = client.get(key)
    if result == value:
        print(f"SET/GET roundtrip OK  (key={key!r}, value={result!r})")
        return 0
    print(f"SET/GET roundtrip FAILED  (expected {value!r}, got {result!r})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
