-- 006_embeddings.sql
-- Replaces the embeddings table that was in a separate SQLite section.

CREATE TABLE IF NOT EXISTS embeddings (
    key       TEXT PRIMARY KEY,
    content   TEXT NOT NULL,
    vector    JSONB NOT NULL DEFAULT '[]',
    model     TEXT DEFAULT '',
    topic     TEXT DEFAULT '',
    stored_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_embeddings_topic
    ON embeddings(topic) WHERE topic != '';
