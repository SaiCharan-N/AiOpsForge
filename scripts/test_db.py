#!/usr/bin/env python3
"""Day 4: standalone check that Postgres is reachable and db/init.sql ran.

Run from the host with port 5432 exposed:
    python3 scripts/test_db.py
"""
import os
import sys

import psycopg2

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://aiops:aiops@localhost:5432/aiopsforge"
)

EXPECTED_TABLES = {"projects", "tasks", "runs", "memory_entries"}


def main() -> int:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
                """
            )
            found = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()

    print(f"Tables found: {sorted(found)}")
    missing = EXPECTED_TABLES - found
    if missing:
        print(f"MISSING expected tables: {sorted(missing)}")
        return 1
    print("All expected Phase 1 tables are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
