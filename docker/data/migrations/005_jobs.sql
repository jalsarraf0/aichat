-- 005_jobs.sql
-- Replaces jobs.db SQLite database.

CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    tool_name     TEXT NOT NULL,
    args          JSONB NOT NULL DEFAULT '{}',
    status        TEXT NOT NULL DEFAULT 'queued',
    priority      INTEGER NOT NULL DEFAULT 0,
    submitted_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at    TIMESTAMPTZ,
    finished_at   TIMESTAMPTZ,
    progress      INTEGER NOT NULL DEFAULT 0,
    input_summary TEXT NOT NULL DEFAULT '',
    result        TEXT,
    error         TEXT,
    logs          TEXT NOT NULL DEFAULT '',
    retry_count   INTEGER NOT NULL DEFAULT 0,
    max_retries   INTEGER NOT NULL DEFAULT 0,
    timeout_s     REAL NOT NULL DEFAULT 300.0,
    ttl_seconds   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_jobs_status
    ON jobs(status, priority DESC, submitted_at);
