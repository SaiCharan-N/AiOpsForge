"""Postgres connectivity, kept intentionally thin for Phase 1.

Connections are created lazily (never at import time) so the FastAPI app can
still start and serve /health even if Postgres isn't reachable yet — useful
while containers are still spinning up.
"""
import json

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def check_connection() -> bool:
    """Returns True if Postgres answers a trivial query."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return True


def run_schema_migrations() -> None:
    """Idempotent migrations, run once at app startup.

    Root cause these fix: db/init.sql's CREATE TABLE IF NOT EXISTS only
    executes against a brand-new, empty Postgres data volume (it's mounted
    as a docker-entrypoint-initdb.d script). On any deployment where the
    `pgdata` volume already existed from an earlier phase, a column added
    to init.sql later is silently never applied to the already-created
    table — no error at deploy time, just a column that "should" be there
    per the schema file but isn't, until something queries it.

    Two independent instances of that same problem, fixed the same way
    (idempotent ADD COLUMN, no data loss, no volume reset required):

    1. memory_entries.embedding / execution_context — the originally
       reported 'column memory_entries.embedding does not exist' error.
       Deliberately NOT a pgvector `vector(N)` column: embeddings are
       stored as JSON-in-TEXT (see app/embeddings.py + app/memory.py's
       manual cosine similarity) because the `vector` extension isn't
       installed on the plain `postgres:16-alpine` image this project
       uses, and adding it would mean adding a new technology to the stack
       for a problem the TEXT/JSON approach already solves at this
       project's scale.
    2. tasks.type — app.schemas.Task has had a `type` field ("code" or
       "webapp_scaffold") since Phase 7, but it was never added to the
       `tasks` table or to db.upsert_task's INSERT/UPDATE, so the DB record
       of a task's type was always silently absent/default even though the
       live pipeline run (which reads task.type from LangGraph's in-memory
       state, never from the DB, so routing itself was never affected)
       always had it right. This matters for anything that inspects the DB
       directly — GET /projects/{id}/status, a future dashboard, or
       reconstructing what actually ran from `runs`/`tasks` after the fact.
    """
    ddl = (
        "ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS embedding TEXT",
        "ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS execution_context JSONB",
        "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS type TEXT NOT NULL DEFAULT 'code'",
        """CREATE TABLE IF NOT EXISTS sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    messages    JSONB NOT NULL DEFAULT '[]',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)""",
        "CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at DESC)",
    )
    with engine.begin() as conn:
        for stmt in ddl:
            conn.execute(text(stmt))


# Back-compat alias — main.py's startup hook calls this name; kept as a
# thin wrapper rather than renaming call sites for a rename-only diff.
def ensure_memory_schema() -> None:
    run_schema_migrations()


def list_tables() -> list[str]:
    """Returns the table names created by db/init.sql — used by /health/deep
    and by scripts/test_db.py to confirm Day 4's schema is actually in place.
    """
    query = text(
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------
# Phase 2 persistence: one project row, one task row per planned task,
# one run row per agent invocation (planner/developer/qa), matching the
# schema in db/init.sql.
# ---------------------------------------------------------------------

def create_project(project_id: str, name: str, request: str) -> None:
    query = text(
        "INSERT INTO projects (id, name, request, status) "
        "VALUES (:id, :name, :request, 'building')"
    )
    with engine.begin() as conn:
        conn.execute(query, {"id": project_id, "name": name, "request": request})


def update_project_status(project_id: str, status: str) -> None:
    query = text("UPDATE projects SET status = :status WHERE id = :id")
    with engine.begin() as conn:
        conn.execute(query, {"id": project_id, "status": status})


def upsert_task(project_id: str, task_order: int, description: str, status: str,
                 attempt_count: int = 0, memory_used: bool = False,
                 task_type: str = "code") -> None:
    """Insert-or-update a single task's row — called incrementally as the
    graph runs (pending -> in_progress -> passed/failed/needs_human_review)
    so a polling dashboard sees live progress, not just a final summary.

    `task_type` mirrors app.schemas.Task.type ("code" | "webapp_scaffold")
    — stored for visibility/audit only; the live pipeline run always
    routes on the in-memory Task object, never on this column.
    """
    query = text(
        """
        INSERT INTO tasks (project_id, task_order, description, status, attempt_count, memory_used, type)
        VALUES (:project_id, :task_order, :description, :status, :attempt_count, :memory_used, :task_type)
        ON CONFLICT (project_id, task_order)
        DO UPDATE SET status = EXCLUDED.status,
                      attempt_count = EXCLUDED.attempt_count,
                      memory_used = EXCLUDED.memory_used,
                      type = EXCLUDED.type
        """
    )
    with engine.begin() as conn:
        conn.execute(query, {
            "project_id": project_id, "task_order": task_order, "description": description,
            "status": status, "attempt_count": attempt_count, "memory_used": memory_used,
            "task_type": task_type,
        })


def get_project_status(project_id: str) -> dict | None:
    """Everything a dashboard needs for one project: its own status plus
    every task's current status, in order. Used by GET /projects/{id}/status.
    """
    proj_query = text("SELECT id, name, request, status, created_at FROM projects WHERE id = :id")
    tasks_query = text(
        "SELECT task_order, description, status, attempt_count, memory_used, type "
        "FROM tasks WHERE project_id = :id ORDER BY task_order"
    )
    with engine.connect() as conn:
        proj_row = conn.execute(proj_query, {"id": project_id}).fetchone()
        if proj_row is None:
            return None
        task_rows = conn.execute(tasks_query, {"id": project_id}).fetchall()

    return {
        "project_id": str(proj_row.id),
        "name": proj_row.name,
        "request": proj_row.request,
        "status": proj_row.status,
        "created_at": proj_row.created_at.isoformat(),
        "tasks": [
            {
                "order": r.task_order,
                "description": r.description,
                "status": r.status,
                "attempts": r.attempt_count,
                "memory_used": r.memory_used,
                "type": r.type,
            }
            for r in task_rows
        ],
    }


def log_run(project_id: str, agent: str, input_data: dict, output_data: dict,
            status: str = "ok", duration_ms: int | None = None) -> None:
    """Logs one agent invocation to the `runs` table — this is the raw data
    Phase 4's experiments (Pass@1, timing, retry counts) will be computed from.
    """
    query = text(
        "INSERT INTO runs (project_id, agent, input, output, status, duration_ms) "
        "VALUES (:project_id, :agent, :input, :output, :status, :duration_ms)"
    )
    with engine.begin() as conn:
        conn.execute(query, {
            "project_id": project_id,
            "agent": agent,
            "input": json.dumps(input_data),
            "output": json.dumps(output_data),
            "status": status,
            "duration_ms": duration_ms,
        })
