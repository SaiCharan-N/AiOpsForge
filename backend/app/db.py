"""Postgres connectivity, kept intentionally thin for Phase 1.

Connections are created lazily (never at import time) so the FastAPI app can
still start and serve /health even if Postgres isn't reachable yet — useful
while containers are still spinning up.
"""
import json

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.pipeline import metrics

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
        # Failure Recovery: same reasoning as above — init.sql only runs on
        # a fresh volume, so the table is also created here, idempotently.
        """CREATE TABLE IF NOT EXISTS failure_patterns (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id       UUID REFERENCES projects(id) ON DELETE SET NULL,
    error_type       TEXT NOT NULL,
    error_category   TEXT NOT NULL,
    error_signature  TEXT NOT NULL,
    error_message    TEXT,
    code_location    TEXT,
    fix_suggestion   TEXT NOT NULL,
    fix_diff         TEXT,
    task_type        TEXT NOT NULL DEFAULT 'code',
    embedding        TEXT,
    times_seen       INT NOT NULL DEFAULT 1,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at     TIMESTAMPTZ NOT NULL DEFAULT now()
)""",
        "CREATE INDEX IF NOT EXISTS idx_failure_patterns_type ON failure_patterns(error_type, error_category)",
        "CREATE INDEX IF NOT EXISTS idx_failure_patterns_sig ON failure_patterns(error_signature)",
        # Batch processing & experiments (see app.batch / app.experiments).
        """CREATE TABLE IF NOT EXISTS experiments (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name         TEXT,
    design       TEXT NOT NULL,          -- ab | repeat | sequence | mcp_overhead
    status       TEXT NOT NULL DEFAULT 'running',   -- running | done | failed | cancelled | interrupted
    config       JSONB,                  -- the full request + environment snapshot (reproducibility)
    result       JSONB,                  -- computed comparison, stored when the experiment finishes
    error        TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ
)""",
        """CREATE TABLE IF NOT EXISTS batches (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name           TEXT,
    status         TEXT NOT NULL DEFAULT 'running',  -- running | done | failed | cancelled | interrupted
    experiment_id  UUID REFERENCES experiments(id) ON DELETE SET NULL,
    level          TEXT,                 -- experiment condition this batch ran (e.g. 'blackboard')
    config         JSONB,
    total          INT NOT NULL DEFAULT 0,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at    TIMESTAMPTZ
)""",
        """CREATE TABLE IF NOT EXISTS batch_projects (
    batch_id     UUID NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    idx          INT NOT NULL,
    project_id   UUID REFERENCES projects(id) ON DELETE SET NULL,
    request      TEXT NOT NULL,
    settings     JSONB,                  -- this run's resolved mode / framework / overrides / repetition
    status       TEXT NOT NULL DEFAULT 'pending',   -- pending | running | passed | needs_review | error | cancelled
    error        TEXT,
    started_at   TIMESTAMPTZ,
    finished_at  TIMESTAMPTZ,
    PRIMARY KEY (batch_id, idx)
)""",
        "CREATE INDEX IF NOT EXISTS idx_batches_experiment ON batches(experiment_id)",
        "CREATE INDEX IF NOT EXISTS idx_batch_projects_project ON batch_projects(project_id)",
        # User intervention history (see app.pipeline.intervention).
        """CREATE TABLE IF NOT EXISTS interventions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id    UUID REFERENCES projects(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL,        -- pause_request | pause
    point         TEXT,                 -- planner | developer | qa_failure | manual  (for pause_request: the node asked for)
    node          TEXT,
    action        TEXT,                 -- approve | revise | skip_task | abort (how a pause was resolved)
    guidance      TEXT,
    clarification TEXT,
    tasks         JSONB,                -- the edited plan, if the user edited it
    source        TEXT NOT NULL DEFAULT 'user',   -- user | timeout
    question      TEXT,
    waited_ms     INT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
)""",
        "CREATE INDEX IF NOT EXISTS idx_interventions_project ON interventions(project_id, created_at)",
        # Metrics & observability (see app.pipeline.metrics).
        """CREATE TABLE IF NOT EXISTS metrics_logs (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id   UUID REFERENCES projects(id) ON DELETE CASCADE,
    kind         TEXT NOT NULL,        -- llm | embed | tool | db | node | task | pipeline | ifs | pause
    agent        TEXT,                 -- planner | developer | qa | failure_analyzer | ...
    node         TEXT,
    task_order   INT,
    attempt      INT,                  -- attempt number; for kind='task' the total attempts used
    name         TEXT,                 -- model / tool / db op / node / task type / comm. mode
    tokens_in    INT NOT NULL DEFAULT 0,
    tokens_out   INT NOT NULL DEFAULT 0,
    duration_ms  INT NOT NULL DEFAULT 0,
    t_start_ms   BIGINT,               -- epoch ms; offsets in the per-node timeline are relative to the run's first event
    ok           BOOLEAN NOT NULL DEFAULT true,
    meta         JSONB,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
)""",
        "CREATE INDEX IF NOT EXISTS idx_metrics_logs_project ON metrics_logs(project_id, kind)",
        "CREATE INDEX IF NOT EXISTS idx_metrics_logs_kind_time ON metrics_logs(kind, t_start_ms DESC)",
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

@metrics.timed_db("db.create_project")
def create_project(project_id: str, name: str, request: str) -> None:
    query = text(
        "INSERT INTO projects (id, name, request, status) "
        "VALUES (:id, :name, :request, 'building')"
    )
    with engine.begin() as conn:
        conn.execute(query, {"id": project_id, "name": name, "request": request})


@metrics.timed_db("db.update_project_status")
def update_project_status(project_id: str, status: str) -> None:
    query = text("UPDATE projects SET status = :status WHERE id = :id")
    with engine.begin() as conn:
        conn.execute(query, {"id": project_id, "status": status})


@metrics.timed_db("db.upsert_task")
def upsert_task(project_id: str, task_order: int, description: str, status: str,
                 attempt_count: int = 0, memory_used: bool = False,
                 task_type: str = "code") -> None:
    """Insert-or-update a single task's row — called incrementally as the
    graph runs (pending -> in_progress -> passed/failed/needs_human_review)
    so a polling dashboard sees live progress, not just a final summary.

    The description is updated on conflict too: when the user edits or
    re-plans at a pause point, the same task_order gets new text, and the
    status endpoint must show that, not the original plan.

    `task_type` mirrors app.schemas.Task.type ("code" | "webapp_scaffold")
    — stored for visibility/audit only; the live pipeline run always
    routes on the in-memory Task object, never on this column.
    """
    query = text(
        """
        INSERT INTO tasks (project_id, task_order, description, status, attempt_count, memory_used, type)
        VALUES (:project_id, :task_order, :description, :status, :attempt_count, :memory_used, :task_type)
        ON CONFLICT (project_id, task_order)
        DO UPDATE SET description = EXCLUDED.description,
                      status = EXCLUDED.status,
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


@metrics.timed_db("db.get_project_status")
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


@metrics.timed_db("db.log_run")
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


# ---------------------------------------------------------------------------
# User intervention (app.pipeline.intervention)
# ---------------------------------------------------------------------------

@metrics.timed_db("db.delete_tasks_after")
def delete_tasks_after(project_id: str, last_order: int) -> None:
    """Removes task rows numbered above `last_order` — used when the user
    edits or re-plans to a SHORTER task list, so stale 'pending' rows from
    the old plan don't linger in the status endpoint."""
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM tasks WHERE project_id = :p AND task_order > :n"),
            {"p": project_id, "n": last_order},
        )


@metrics.timed_db("db.log_intervention")
def log_intervention(project_id: str, kind: str, point: str | None = None, node: str | None = None,
                     action: str | None = None, guidance: str | None = None,
                     clarification: str | None = None, tasks: list[str] | None = None,
                     source: str = "user", question: str | None = None,
                     waited_ms: int | None = None) -> None:
    """Appends one row to the project's intervention history."""
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO interventions (project_id, kind, point, node, action, guidance, clarification, "
                "tasks, source, question, waited_ms) VALUES (:p, :kind, :point, :node, :action, :guidance, "
                ":clar, :tasks, :source, :question, :waited)"
            ),
            {"p": project_id, "kind": kind, "point": point, "node": node, "action": action,
             "guidance": guidance, "clar": clarification,
             "tasks": json.dumps(tasks) if tasks is not None else None,
             "source": source, "question": question, "waited": waited_ms},
        )


def list_interventions(project_id: str) -> list[dict]:
    """The project's intervention history, oldest first."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT kind, point, node, action, guidance, clarification, tasks, source, question, "
                 "waited_ms, created_at FROM interventions WHERE project_id = :p ORDER BY created_at, kind"),
            {"p": project_id},
        ).fetchall()
    out = []
    for r in rows:
        tasks = r.tasks
        if isinstance(tasks, str):
            try:
                tasks = json.loads(tasks)
            except json.JSONDecodeError:
                tasks = None
        out.append({
            "kind": r.kind, "point": r.point, "node": r.node, "action": r.action, "guidance": r.guidance,
            "clarification": r.clarification, "tasks": tasks, "source": r.source, "question": r.question,
            "waited_ms": r.waited_ms, "created_at": str(r.created_at),
        })
    return out
