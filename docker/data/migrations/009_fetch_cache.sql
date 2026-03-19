-- 009_fetch_cache.sql
-- Web fetch cache with TTL for avoiding redundant fetches.

CREATE TABLE IF NOT EXISTS fetch_cache (
    url          TEXT PRIMARY KEY,
    content      TEXT NOT NULL,
    content_hash TEXT NOT NULL DEFAULT '',
    content_type TEXT NOT NULL DEFAULT '',
    fetched_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at   TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '1 hour')
);

CREATE INDEX IF NOT EXISTS idx_fetch_cache_expires
    ON fetch_cache(expires_at);
