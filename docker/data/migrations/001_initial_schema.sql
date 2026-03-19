-- 001_initial_schema.sql
-- Codifies the existing PostgreSQL tables that were previously created inline.

CREATE TABLE IF NOT EXISTS articles (
    id        SERIAL PRIMARY KEY,
    url       TEXT UNIQUE NOT NULL,
    title     TEXT,
    content   TEXT,
    topic     TEXT,
    stored_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS images (
    id            SERIAL PRIMARY KEY,
    url           TEXT UNIQUE NOT NULL,
    host_path     TEXT,
    alt_text      TEXT,
    subject       TEXT,
    phash         TEXT,
    quality_score REAL,
    minio_key     TEXT,
    clip_embedded BOOLEAN DEFAULT FALSE,
    source        TEXT,
    tags          TEXT,
    stored_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cache (
    key       TEXT PRIMARY KEY,
    value     TEXT NOT NULL,
    stored_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS errors (
    id        SERIAL PRIMARY KEY,
    service   TEXT,
    level     TEXT,
    message   TEXT,
    detail    TEXT,
    logged_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_articles_topic ON articles(topic);
CREATE INDEX IF NOT EXISTS idx_errors_svc ON errors(service, logged_at DESC);
