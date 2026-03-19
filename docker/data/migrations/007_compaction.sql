-- 007_compaction.sql
-- Tables for memory compaction and summarization tracking.

CREATE TABLE IF NOT EXISTS compaction_log (
    id              SERIAL PRIMARY KEY,
    operation       TEXT NOT NULL,
    entries_before  INTEGER NOT NULL DEFAULT 0,
    entries_after   INTEGER NOT NULL DEFAULT 0,
    bytes_freed     BIGINT NOT NULL DEFAULT 0,
    ts              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS memory_summaries (
    id         SERIAL PRIMARY KEY,
    category   TEXT NOT NULL DEFAULT '',
    summary    TEXT NOT NULL,
    key_count  INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
