-- AIOpsForge — schema (Days 4, extended Week 7 for long-term memory)

CREATE EXTENSION IF NOT EXISTS "pgcrypto"; -- gives us gen_random_uuid()
-- CREATE EXTENSION IF NOT EXISTS "vector";   -- pgvector (requires pg_vector package; skipped for alpine)

-- One row per end-user request / generated project.
CREATE TABLE IF NOT EXISTS projects (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    request     TEXT NOT NULL,             -- the raw user request, e.g. "build an inventory tracker"
    status      TEXT NOT NULL DEFAULT 'created',   -- created | planning | building | testing | done | needs_human_review
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per task the Planner Agent breaks a project into.
CREATE TABLE IF NOT EXISTS tasks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id   UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    task_order   INT NOT NULL,             -- 1, 2, 3... execution order from the Planner
    description  TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',   -- pending | in_progress | passed | failed | needs_human_review
    attempt_count INT NOT NULL DEFAULT 0,
    memory_used  BOOLEAN NOT NULL DEFAULT false,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (project_id, task_order)
);

-- One row per agent invocation (Planner / Developer / QA), for debugging and
-- for the Langfuse-style analysis in Phase 4.
CREATE TABLE IF NOT EXISTS runs (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id   UUID REFERENCES projects(id) ON DELETE CASCADE,
    task_id      UUID REFERENCES tasks(id) ON DELETE CASCADE,
    agent        TEXT NOT NULL,            -- 'planner' | 'developer' | 'qa'
    input        JSONB,
    output       JSONB,
    status       TEXT NOT NULL DEFAULT 'ok',        -- ok | error
    duration_ms  INT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Long-term memory entries (Week 7 / Phase 11: improved with embeddings).
-- This is DELIBERATELY organizational, not project-scoped for retrieval:
-- the whole point is that a fix learned on one project helps a LATER,
-- unrelated project solve a similar bug faster. project_id is kept only
-- for provenance/auditing (which project first taught the system this fix),
-- never as a retrieval filter.
--
-- Project ISOLATION (no cross-project leakage) applies to SHORT-TERM /
-- working memory in Redis instead — see backend/app/short_term_memory.py —
-- because that holds live, in-progress state for a run that genuinely must
-- not bleed into a concurrently running, unrelated project.
--
-- Phase 11: embeddings are now stored as vectors for semantic similarity
-- search. The embedding is a JSON array stored as text (pgvector not
-- available in alpine, but jsonb vector comparison works for small batches).
CREATE TABLE IF NOT EXISTS memory_entries (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id         UUID REFERENCES projects(id) ON DELETE SET NULL,
    problem_signature  TEXT NOT NULL,
    solution_summary   TEXT NOT NULL,
    embedding          TEXT,                                 -- JSON vector '[0.1, 0.2, ...]' from Ollama
    execution_context  JSONB,                                -- {"files_changed": [...], "errors_fixed": [...], "model": "..."}
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Index for faster retrieval during similarity search (on creation timestamp
-- and to speed up the ordering in retrieve_similar queries).
CREATE INDEX IF NOT EXISTS idx_memory_entries_created_at ON memory_entries(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_tasks_project_id ON tasks(project_id);
CREATE INDEX IF NOT EXISTS idx_runs_project_id ON runs(project_id);
CREATE INDEX IF NOT EXISTS idx_runs_task_id ON runs(task_id);

