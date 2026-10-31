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
    type         TEXT NOT NULL DEFAULT 'code',      -- 'code' | 'webapp_scaffold' (app.schemas.Task.type)
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

-- Failure Recovery: one row per DISTINCT failure pattern that was later
-- fixed, so a future analysis can say "this TypeError was fixed by X".
-- Kept separate from memory_entries on purpose: memory_entries is keyed by
-- task description (retrieved to seed a first attempt); this is keyed by
-- error signature (retrieved to repair a failing attempt). Mixing them made
-- error text pollute task-level retrieval.
CREATE TABLE IF NOT EXISTS failure_patterns (
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
);
CREATE INDEX IF NOT EXISTS idx_failure_patterns_type ON failure_patterns(error_type, error_category);
CREATE INDEX IF NOT EXISTS idx_failure_patterns_sig ON failure_patterns(error_signature);

-- Metrics & observability: one row per LLM call / tool call / DB query /
-- node execution / finished task / pipeline run. Written in batches by
-- app.pipeline.metrics; read by app.metrics_report (GET /metrics/*).
CREATE TABLE IF NOT EXISTS metrics_logs (
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
);
CREATE INDEX IF NOT EXISTS idx_metrics_logs_project ON metrics_logs(project_id, kind);
CREATE INDEX IF NOT EXISTS idx_metrics_logs_kind_time ON metrics_logs(kind, t_start_ms DESC);

-- User intervention history: every pause request and every resolved pause
-- (what the user was asked, what they answered, how long the run waited).
CREATE TABLE IF NOT EXISTS interventions (
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
);
CREATE INDEX IF NOT EXISTS idx_interventions_project ON interventions(project_id, created_at);

-- Batch processing & experiments (app/batch.py, app/experiments.py): a batch is N projects run
-- together; an experiment is batches that differ in one factor, plus the statistics comparing them.
CREATE TABLE IF NOT EXISTS experiments (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name         TEXT,
    design       TEXT NOT NULL,          -- ab | repeat | sequence | mcp_overhead
    status       TEXT NOT NULL DEFAULT 'running',   -- running | done | failed | cancelled | interrupted
    config       JSONB,                  -- the full request + environment snapshot (reproducibility)
    result       JSONB,                  -- computed comparison, stored when the experiment finishes
    error        TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS batches (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name           TEXT,
    status         TEXT NOT NULL DEFAULT 'running',  -- running | done | failed | cancelled | interrupted
    experiment_id  UUID REFERENCES experiments(id) ON DELETE SET NULL,
    level          TEXT,                 -- experiment condition this batch ran (e.g. 'blackboard')
    config         JSONB,
    total          INT NOT NULL DEFAULT 0,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at    TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS batch_projects (
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
);
CREATE INDEX IF NOT EXISTS idx_batches_experiment ON batches(experiment_id);
CREATE INDEX IF NOT EXISTS idx_batch_projects_project ON batch_projects(project_id);

-- Sessions table for cagent (docker ai) — stores conversation history
CREATE TABLE IF NOT EXISTS sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    messages    JSONB NOT NULL DEFAULT '[]',     -- array of message objects
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at DESC);

