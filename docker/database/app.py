"""aichat-database: PostgreSQL-backed article, image, and web-content cache service.

Replaces the old aichat-rssfeed service.  All web-related data (articles fetched
via the browser, images, cached pages) is stored here so the AI can compare
fresh results against what it previously retrieved.

Also provides a structured error-log table so other services can record and
query application-level errors.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import namedtuple
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import psycopg
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("aichat-database")

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

DB = os.environ.get("DATABASE_URL", "postgresql://aichat:aichat@localhost:5432/aichat")


def conn():
    return psycopg.connect(DB)


def _create_tables(c: psycopg.Connection) -> None:
    c.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id         SERIAL PRIMARY KEY,
            url        TEXT UNIQUE NOT NULL,
            title      TEXT,
            content    TEXT,
            topic      TEXT,
            stored_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id         SERIAL PRIMARY KEY,
            url        TEXT UNIQUE NOT NULL,
            host_path  TEXT,
            alt_text   TEXT,
            stored_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS web_cache (
            id         SERIAL PRIMARY KEY,
            url        TEXT UNIQUE NOT NULL,
            title      TEXT,
            content    TEXT,
            cached_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS error_logs (
            id         SERIAL PRIMARY KEY,
            service    TEXT NOT NULL DEFAULT 'unknown',
            level      TEXT NOT NULL DEFAULT 'ERROR',
            message    TEXT NOT NULL,
            detail     TEXT,
            logged_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    # Indexes for common query patterns on error_logs
    c.execute("CREATE INDEX IF NOT EXISTS idx_error_logs_service_ts ON error_logs (service, logged_at DESC)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_error_logs_level_ts   ON error_logs (level, logged_at DESC)")
    # Index for article topic lookups
    c.execute("CREATE INDEX IF NOT EXISTS idx_articles_topic_ts ON articles (topic, stored_at DESC)")
    # Extend images table with vision/hash columns (idempotent migration)
    for col_sql in [
        "ALTER TABLE images ADD COLUMN IF NOT EXISTS subject       TEXT DEFAULT ''",
        "ALTER TABLE images ADD COLUMN IF NOT EXISTS description   TEXT DEFAULT ''",
        "ALTER TABLE images ADD COLUMN IF NOT EXISTS phash         TEXT DEFAULT ''",
        "ALTER TABLE images ADD COLUMN IF NOT EXISTS quality_score REAL DEFAULT 0.5",
    ]:
        c.execute(col_sql)
    c.execute("CREATE INDEX IF NOT EXISTS idx_images_subject ON images (subject)")
    # Embeddings table for semantic search (cosine similarity computed in Python)
    c.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id        SERIAL PRIMARY KEY,
            key       TEXT UNIQUE NOT NULL,
            content   TEXT NOT NULL,
            embedding TEXT NOT NULL,
            model     TEXT DEFAULT '',
            topic     TEXT DEFAULT '',
            stored_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_topic ON embeddings (topic)")
    # Conversation sessions + turns (for persistent context / RAG)
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversation_sessions (
            id               SERIAL PRIMARY KEY,
            session_id       TEXT UNIQUE NOT NULL,
            title            TEXT NOT NULL DEFAULT '',
            model            TEXT NOT NULL DEFAULT '',
            created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            compact_summary  TEXT NOT NULL DEFAULT '',
            compact_from_idx INTEGER NOT NULL DEFAULT 0
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversation_turns (
            id          SERIAL PRIMARY KEY,
            session_id  TEXT NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            turn_index  INTEGER NOT NULL DEFAULT 0,
            embedding   TEXT,
            timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_conv_sessions_updated ON conversation_sessions(updated_at DESC)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_conv_turns_session    ON conversation_turns(session_id, turn_index)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_conv_turns_timestamp  ON conversation_turns(timestamp DESC)")
    # Idempotent migrations for compact state columns (safe on existing DBs)
    c.execute("ALTER TABLE conversation_sessions ADD COLUMN IF NOT EXISTS compact_summary  TEXT NOT NULL DEFAULT ''")
    c.execute("ALTER TABLE conversation_sessions ADD COLUMN IF NOT EXISTS compact_from_idx INTEGER NOT NULL DEFAULT 0")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    for attempt in range(1, 16):
        try:
            with conn() as c:
                _create_tables(c)
            log.info("Database tables ready.")
            break
        except psycopg.OperationalError as exc:
            if attempt >= 15:
                log.error("Cannot connect to PostgreSQL after 15 attempts: %s", exc)
                raise
            log.warning("PostgreSQL not ready (attempt %d/15), retrying in 1s…", attempt)
            time.sleep(1)
    yield


app = FastAPI(title="aichat-database", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Global exception handler — never return raw 500s
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exc_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error("Unhandled error on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "path": str(request.url.path)},
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    try:
        with conn() as c:
            n_articles = (c.execute("SELECT COUNT(*) FROM articles").fetchone() or [0])[0]
            n_cache    = (c.execute("SELECT COUNT(*) FROM web_cache").fetchone() or [0])[0]
        return {"ok": True, "articles": n_articles, "cached_pages": n_cache}
    except Exception as exc:
        log.error("Health check failed: %s", exc)
        return {"ok": False, "error": str(exc), "articles": 0, "cached_pages": 0}


# ---------------------------------------------------------------------------
# Error logging
# ---------------------------------------------------------------------------

class ErrorLogIn(BaseModel):
    service: str = "unknown"
    level: str = "ERROR"
    message: str
    detail: Optional[str] = None


@app.post("/errors/log")
def log_error(entry: ErrorLogIn) -> dict:
    log.log(
        logging.getLevelName(entry.level.upper()) if hasattr(logging, entry.level.upper()) else logging.ERROR,
        "[%s] %s — %s",
        entry.service,
        entry.message,
        entry.detail or "",
    )
    try:
        with conn() as c:
            c.execute(
                """
                INSERT INTO error_logs (service, level, message, detail, logged_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (entry.service, entry.level.upper(), entry.message, entry.detail,
                 datetime.now(timezone.utc)),
            )
        return {"status": "logged"}
    except Exception as exc:
        log.error("Failed to persist error log entry: %s", exc)
        return {"status": "logged_locally_only", "db_error": str(exc)}


@app.get("/errors/recent")
def recent_errors(limit: int = 50, service: Optional[str] = None) -> dict:
    try:
        with conn() as c:
            if service:
                rows = c.execute(
                    """SELECT service, level, message, detail, logged_at
                       FROM error_logs WHERE service = %s
                       ORDER BY logged_at DESC LIMIT %s""",
                    (service, limit),
                ).fetchall()
            else:
                rows = c.execute(
                    """SELECT service, level, message, detail, logged_at
                       FROM error_logs ORDER BY logged_at DESC LIMIT %s""",
                    (limit,),
                ).fetchall()
        return {
            "errors": [
                {
                    "service": r[0],
                    "level":   r[1],
                    "message": r[2],
                    "detail":  r[3],
                    "logged_at": r[4].isoformat(),
                }
                for r in rows
            ]
        }
    except Exception as exc:
        log.error("Failed to query error_logs: %s", exc)
        return {"errors": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# Articles
# ---------------------------------------------------------------------------

class ArticleIn(BaseModel):
    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    topic: Optional[str] = None


@app.post("/articles/store")
def store_article(article: ArticleIn) -> dict:
    try:
        with conn() as c:
            c.execute(
                """
                INSERT INTO articles (url, title, content, topic, stored_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (url) DO UPDATE
                  SET title     = EXCLUDED.title,
                      content   = EXCLUDED.content,
                      topic     = EXCLUDED.topic,
                      stored_at = EXCLUDED.stored_at
                """,
                (article.url, article.title, article.content, article.topic,
                 datetime.now(timezone.utc)),
            )
        log.info("Stored article: %s (topic=%s)", article.url[:80], article.topic)
        return {"status": "stored", "url": article.url}
    except Exception as exc:
        log.error("store_article failed for %s: %s", article.url[:80], exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/articles/search")
def search_articles(
    topic: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    summary_only: bool = False,
) -> dict:
    """Search stored articles.  summary_only=true truncates content to 300 chars."""
    try:
        with conn() as c:
            if topic and q:
                rows = c.execute(
                    """SELECT url, title, content, topic, stored_at FROM articles
                       WHERE topic = %s AND (title ILIKE %s OR content ILIKE %s)
                       ORDER BY stored_at DESC LIMIT %s OFFSET %s""",
                    (topic, f"%{q}%", f"%{q}%", limit, offset),
                ).fetchall()
            elif topic:
                rows = c.execute(
                    """SELECT url, title, content, topic, stored_at FROM articles
                       WHERE topic = %s
                       ORDER BY stored_at DESC LIMIT %s OFFSET %s""",
                    (topic, limit, offset),
                ).fetchall()
            elif q:
                rows = c.execute(
                    """SELECT url, title, content, topic, stored_at FROM articles
                       WHERE title ILIKE %s OR content ILIKE %s
                       ORDER BY stored_at DESC LIMIT %s OFFSET %s""",
                    (f"%{q}%", f"%{q}%", limit, offset),
                ).fetchall()
            else:
                rows = c.execute(
                    """SELECT url, title, content, topic, stored_at FROM articles
                       ORDER BY stored_at DESC LIMIT %s OFFSET %s""",
                    (limit, offset),
                ).fetchall()

        def _content(raw: Optional[str]) -> Optional[str]:
            if raw is None:
                return None
            return (raw[:300] + "…") if summary_only and len(raw) > 300 else raw

        return {
            "articles": [
                {
                    "url": r[0], "title": r[1], "content": _content(r[2]),
                    "topic": r[3], "stored_at": r[4].isoformat(),
                }
                for r in rows
            ]
        }
    except Exception as exc:
        log.error("search_articles failed (topic=%s, q=%s): %s", topic, q, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

class ImageIn(BaseModel):
    url:           str
    host_path:     Optional[str]   = None
    alt_text:      Optional[str]   = None
    subject:       Optional[str]   = None
    description:   Optional[str]   = None
    phash:         Optional[str]   = None
    quality_score: Optional[float] = None


@app.post("/images/store")
def store_image(image: ImageIn) -> dict:
    try:
        with conn() as c:
            c.execute(
                """
                INSERT INTO images
                    (url, host_path, alt_text, subject, description, phash, quality_score, stored_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (url) DO UPDATE
                  SET host_path     = COALESCE(EXCLUDED.host_path,    images.host_path),
                      alt_text      = COALESCE(EXCLUDED.alt_text,     images.alt_text),
                      subject       = COALESCE(NULLIF(EXCLUDED.subject,      ''), images.subject),
                      description   = COALESCE(NULLIF(EXCLUDED.description,  ''), images.description),
                      phash         = COALESCE(NULLIF(EXCLUDED.phash,        ''), images.phash),
                      quality_score = COALESCE(EXCLUDED.quality_score, images.quality_score),
                      stored_at     = EXCLUDED.stored_at
                """,
                (
                    image.url, image.host_path, image.alt_text,
                    image.subject or "", image.description or "", image.phash or "",
                    image.quality_score if image.quality_score is not None else 0.5,
                    datetime.now(timezone.utc),
                ),
            )
        return {"status": "stored", "url": image.url}
    except Exception as exc:
        log.error("store_image failed for %s: %s", image.url[:80], exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/images/list")
def list_images(limit: int = 20, offset: int = 0) -> dict:
    try:
        with conn() as c:
            rows = c.execute(
                """SELECT url, host_path, alt_text, stored_at FROM images
                   ORDER BY stored_at DESC LIMIT %s OFFSET %s""",
                (limit, offset),
            ).fetchall()
        return {
            "images": [
                {"url": r[0], "host_path": r[1], "alt_text": r[2], "stored_at": r[3].isoformat()}
                for r in rows
            ]
        }
    except Exception as exc:
        log.error("list_images failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/images/search")
def search_images(subject: str = "", limit: int = 20) -> dict:
    """Search confirmed images by subject or description text.  Returns highest quality first."""
    try:
        with conn() as c:
            rows = c.execute(
                """
                SELECT url, host_path, alt_text, subject, description, phash, quality_score, stored_at
                FROM images
                WHERE (subject ILIKE %s OR description ILIKE %s)
                  AND quality_score >= 0.5
                ORDER BY quality_score DESC, stored_at DESC
                LIMIT %s
                """,
                (f"%{subject}%", f"%{subject}%", limit),
            ).fetchall()
        images = [
            {
                "url": r[0], "host_path": r[1], "alt_text": r[2],
                "subject": r[3], "description": r[4], "phash": r[5],
                "quality_score": r[6],
                "stored_at": r[7].isoformat() if r[7] else "",
            }
            for r in rows
        ]
        return {"images": images, "count": len(images)}
    except Exception as exc:
        log.error("search_images failed (subject=%s): %s", subject, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Web cache  (stores full page content keyed by URL)
# ---------------------------------------------------------------------------

class CacheIn(BaseModel):
    url: str
    content: str
    title: Optional[str] = None


@app.post("/cache/store")
def cache_store(item: CacheIn) -> dict:
    try:
        with conn() as c:
            c.execute(
                """
                INSERT INTO web_cache (url, title, content, cached_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (url) DO UPDATE
                  SET title     = EXCLUDED.title,
                      content   = EXCLUDED.content,
                      cached_at = EXCLUDED.cached_at
                """,
                (item.url, item.title, item.content, datetime.now(timezone.utc)),
            )
        log.debug("Cached page: %s", item.url[:80])
        return {"status": "cached", "url": item.url}
    except Exception as exc:
        log.error("cache_store failed for %s: %s", item.url[:80], exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/cache/get")
def cache_get(url: str) -> dict:
    try:
        with conn() as c:
            row = c.execute(
                "SELECT title, content, cached_at FROM web_cache WHERE url = %s",
                (url,),
            ).fetchone()
        if not row:
            return {"found": False}
        return {
            "found": True,
            "title": row[0],
            "content": row[1],
            "cached_at": row[2].isoformat(),
        }
    except Exception as exc:
        log.error("cache_get failed for %s: %s", url[:80], exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/cache/check")
def cache_check(url: str) -> dict:
    try:
        with conn() as c:
            row = c.execute(
                "SELECT cached_at FROM web_cache WHERE url = %s",
                (url,),
            ).fetchone()
        return {"cached": bool(row), "cached_at": row[0].isoformat() if row else None}
    except Exception as exc:
        log.error("cache_check failed for %s: %s", url[:80], exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Embeddings  (semantic search via cosine similarity)
# ---------------------------------------------------------------------------

def _cosine_sim(a: list, b: list) -> float:
    """Pure-Python cosine similarity; returns 0.0 on empty/mismatched vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return 0.0 if mag_a == 0.0 or mag_b == 0.0 else dot / (mag_a * mag_b)


class EmbeddingIn(BaseModel):
    key:       str
    content:   str
    embedding: list[float]   # JSON array of floats
    model:     Optional[str] = None
    topic:     Optional[str] = None


class EmbeddingSearchIn(BaseModel):
    embedding: list[float]   # query embedding (float array)
    limit:     int = 5
    topic:     Optional[str] = None


@app.post("/embeddings/store")
def embeddings_store(item: EmbeddingIn) -> dict:
    try:
        emb_json = json.dumps(item.embedding)
        with conn() as c:
            c.execute(
                """
                INSERT INTO embeddings (key, content, embedding, model, topic, stored_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (key) DO UPDATE
                  SET content   = EXCLUDED.content,
                      embedding = EXCLUDED.embedding,
                      model     = COALESCE(NULLIF(EXCLUDED.model,  ''), embeddings.model),
                      topic     = COALESCE(NULLIF(EXCLUDED.topic,  ''), embeddings.topic),
                      stored_at = EXCLUDED.stored_at
                """,
                (
                    item.key, item.content[:4000], emb_json,
                    item.model or "", item.topic or "",
                    datetime.now(timezone.utc),
                ),
            )
        return {"status": "stored", "key": item.key}
    except Exception as exc:
        log.error("embeddings_store failed for %s: %s", item.key[:80], exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/embeddings/search")
def embeddings_search(req: EmbeddingSearchIn) -> dict:
    """Return the top-N most similar embeddings by cosine similarity."""
    try:
        limit = max(1, min(req.limit, 50))
        with conn() as c:
            if req.topic:
                rows = c.execute(
                    "SELECT key, content, embedding, model, topic, stored_at "
                    "FROM embeddings WHERE topic = %s",
                    (req.topic,),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT key, content, embedding, model, topic, stored_at "
                    "FROM embeddings"
                ).fetchall()

        query_vec = req.embedding
        scored: list[dict] = []
        for row in rows:
            try:
                vec = json.loads(row[2])
                sim = _cosine_sim(query_vec, vec)
                scored.append({
                    "key":        row[0],
                    "content":    row[1],
                    "similarity": round(sim, 6),
                    "model":      row[3],
                    "topic":      row[4],
                    "stored_at":  row[5].isoformat() if row[5] else "",
                })
            except Exception:
                continue

        scored.sort(key=lambda d: d["similarity"], reverse=True)
        return {"results": scored[:limit], "count": len(scored)}
    except Exception as exc:
        log.error("embeddings_search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/embeddings/list")
def embeddings_list(limit: int = 20, topic: Optional[str] = None) -> dict:
    try:
        with conn() as c:
            if topic:
                rows = c.execute(
                    "SELECT key, content, model, topic, stored_at FROM embeddings "
                    "WHERE topic = %s ORDER BY stored_at DESC LIMIT %s",
                    (topic, limit),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT key, content, model, topic, stored_at FROM embeddings "
                    "ORDER BY stored_at DESC LIMIT %s",
                    (limit,),
                ).fetchall()
        return {
            "embeddings": [
                {"key": r[0], "content": r[1][:200], "model": r[2],
                 "topic": r[3], "stored_at": r[4].isoformat() if r[4] else ""}
                for r in rows
            ]
        }
    except Exception as exc:
        log.error("embeddings_list failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Conversations  (persistent sessions + turns with optional embeddings)
# ---------------------------------------------------------------------------


# Named tuple matching the SELECT column order in conv_search_turns.
# Using a namedtuple instead of positional indices makes column access
# self-documenting and immune to accidental reordering of SELECT columns.
ConvRow = namedtuple("ConvRow", ["id", "session_id", "role", "content", "embedding", "timestamp"])


class ConversationSearcher:
    """Encapsulates cosine-similarity ranking of conversation turns.

    Responsibilities
    ----------------
    * Accept raw DB rows fetched outside (so the caller controls the query).
    * Score each row against a query vector using pure-Python cosine similarity.
    * Apply optional session exclusion *after* scoring (useful when the query
      already has a WHERE clause, but lets callers double-filter).
    * Return the top-N results as a sorted list of dicts.

    Usage
    -----
    searcher = ConversationSearcher(rows, query_vec)
    results  = searcher.exclude(bad_session_id).top(50)
    """

    def __init__(self, rows: list, query_vec: list[float]) -> None:
        self._query_vec = query_vec
        self._scored: list[dict] = self._score_rows(rows)

    # ── public API ───────────────────────────────────────────────────────────

    def exclude(self, session_id: str | None) -> "ConversationSearcher":
        """Return a new instance with rows from *session_id* removed."""
        if not session_id:
            return self
        filtered = [r for r in self._scored if r["session_id"] != session_id]
        copy = object.__new__(ConversationSearcher)
        copy._query_vec = self._query_vec
        copy._scored = filtered
        return copy

    def top(self, n: int) -> list[dict]:
        """Return the top *n* results sorted by similarity (highest first)."""
        return sorted(self._scored, key=lambda d: d["similarity"], reverse=True)[:n]

    # ── internal helpers ─────────────────────────────────────────────────────

    def _score_rows(self, rows: list) -> list[dict]:
        scored: list[dict] = []
        skipped = 0
        for raw in rows:
            try:
                # Accept both raw tuples and ConvRow namedtuples.
                row = raw if isinstance(raw, ConvRow) else ConvRow(*raw)
                vec = json.loads(row.embedding)
                sim = _cosine_sim(self._query_vec, vec)
                scored.append({
                    "turn_id":    row.id,
                    "session_id": row.session_id,
                    "role":       row.role,
                    "content":    row.content,
                    "similarity": round(sim, 6),
                    "timestamp":  row.timestamp.isoformat() if row.timestamp else "",
                })
            except Exception as exc:
                skipped += 1
                log.warning("ConversationSearcher: skipping corrupt row — %s", exc)
        if skipped:
            log.warning("ConversationSearcher: skipped %d corrupt row(s) out of %d", skipped, len(rows))
        return scored


class ConvSessionIn(BaseModel):
    session_id: str
    title: Optional[str] = None
    model: Optional[str] = None


class ConvTurnIn(BaseModel):
    session_id: str
    role: str
    content: str
    turn_index: Optional[int] = 0
    embedding: Optional[list[float]] = None


class ConvSearchIn(BaseModel):
    embedding: list[float]
    limit: int = 5
    exclude_session: Optional[str] = None


class ConvTitleIn(BaseModel):
    title: str


class ConvCompactStateIn(BaseModel):
    compact_summary: str
    compact_from_idx: int


@app.post("/conversations/sessions")
def conv_create_session(req: ConvSessionIn) -> dict:
    try:
        with conn() as c:
            existing = c.execute(
                "SELECT session_id FROM conversation_sessions WHERE session_id = %s",
                (req.session_id,),
            ).fetchone()
            if existing:
                return {"status": "exists", "session_id": req.session_id}
            c.execute(
                """
                INSERT INTO conversation_sessions (session_id, title, model, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    req.session_id,
                    req.title or "",
                    req.model or "",
                    datetime.now(timezone.utc),
                    datetime.now(timezone.utc),
                ),
            )
        return {"status": "created", "session_id": req.session_id}
    except Exception as exc:
        log.error("conv_create_session failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/conversations/turns")
def conv_store_turn(req: ConvTurnIn) -> dict:
    try:
        emb_json = json.dumps(req.embedding) if req.embedding is not None else None
        with conn() as c:
            row = c.execute(
                """
                INSERT INTO conversation_turns
                    (session_id, role, content, turn_index, embedding, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    req.session_id,
                    req.role,
                    req.content,
                    req.turn_index or 0,
                    emb_json,
                    datetime.now(timezone.utc),
                ),
            ).fetchone()
            turn_id = row[0] if row else None
            # Update session's updated_at
            c.execute(
                "UPDATE conversation_sessions SET updated_at = %s WHERE session_id = %s",
                (datetime.now(timezone.utc), req.session_id),
            )
        return {"status": "stored", "turn_id": turn_id}
    except Exception as exc:
        log.error("conv_store_turn failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/conversations/search")
def conv_search_turns(req: ConvSearchIn) -> dict:
    """Return the top-N conversation turns most similar to the given embedding."""
    try:
        limit = max(1, min(req.limit, 2000))
        with conn() as c:
            # Fetch all embedded turns; ConversationSearcher handles scoring +
            # exclusion in pure Python so a single query path suffices.
            rows = c.execute(
                """
                SELECT t.id, t.session_id, t.role, t.content, t.embedding, t.timestamp
                FROM conversation_turns t
                WHERE t.role IN ('user', 'assistant')
                  AND t.embedding IS NOT NULL
                """
            ).fetchall()

        results = (
            ConversationSearcher(rows, req.embedding)
            .exclude(req.exclude_session)
            .top(limit)
        )
        return {"results": results}
    except Exception as exc:
        log.error("conv_search_turns failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/conversations/sessions")
def conv_list_sessions(limit: int = 20, offset: int = 0) -> dict:
    try:
        limit  = max(1, min(limit, 500))
        offset = max(0, min(offset, 100_000))   # prevent DoS via huge offset
        with conn() as c:
            rows = c.execute(
                """
                SELECT s.session_id, s.title, s.model, s.created_at, s.updated_at,
                       COUNT(t.id) as turn_count
                FROM conversation_sessions s
                LEFT JOIN conversation_turns t ON t.session_id = s.session_id
                GROUP BY s.session_id, s.title, s.model, s.created_at, s.updated_at
                ORDER BY s.updated_at DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            ).fetchall()
        return {
            "sessions": [
                {
                    "session_id":  r[0],
                    "title":       r[1],
                    "model":       r[2],
                    "created_at":  r[3].isoformat() if r[3] else "",
                    "updated_at":  r[4].isoformat() if r[4] else "",
                    "turn_count":  r[5],
                }
                for r in rows
            ]
        }
    except Exception as exc:
        log.error("conv_list_sessions failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/conversations/sessions/{session_id}")
def conv_get_session(session_id: str, limit: int = 200) -> dict:
    try:
        with conn() as c:
            sess = c.execute(
                "SELECT session_id, title, model, created_at, compact_summary, compact_from_idx FROM conversation_sessions WHERE session_id = %s",
                (session_id,),
            ).fetchone()
            if not sess:
                raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
            turns = c.execute(
                """
                SELECT role, content, turn_index, timestamp
                FROM conversation_turns
                WHERE session_id = %s
                ORDER BY turn_index ASC, timestamp ASC
                LIMIT %s
                """,
                (session_id, limit),
            ).fetchall()
        return {
            "session_id":       sess[0],
            "title":            sess[1],
            "model":            sess[2],
            "created_at":       sess[3].isoformat() if sess[3] else "",
            "compact_summary":  sess[4] or "",
            "compact_from_idx": sess[5] or 0,
            "turns": [
                {
                    "role":       t[0],
                    "content":    t[1],
                    "turn_index": t[2],
                    "timestamp":  t[3].isoformat() if t[3] else "",
                }
                for t in turns
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.error("conv_get_session failed for %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/conversations/turns/search")
def conv_fulltext_search(q: str = "", limit: int = 10) -> dict:
    """Full-text search on conversation_turns using ILIKE (fallback when embeddings unavailable)."""
    try:
        if not q.strip():
            return {"results": []}
        with conn() as c:
            rows = c.execute(
                "SELECT t.id, t.session_id, t.role, t.content, t.timestamp "
                "FROM conversation_turns t "
                "WHERE t.content ILIKE %s AND t.role IN ('user', 'assistant') "
                "ORDER BY t.timestamp DESC LIMIT %s",
                (f"%{q}%", max(1, min(limit, 50))),
            ).fetchall()
        return {
            "results": [
                {
                    "turn_id": r[0],
                    "session_id": r[1],
                    "role": r[2],
                    "content": r[3],
                    "timestamp": r[4].isoformat() if r[4] else "",
                }
                for r in rows
            ]
        }
    except Exception as exc:
        log.error("conv_fulltext_search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.patch("/conversations/sessions/{session_id}/compact")
def conv_update_compact_state(session_id: str, req: ConvCompactStateIn) -> dict:
    try:
        with conn() as c:
            c.execute(
                "UPDATE conversation_sessions "
                "SET compact_summary = %s, compact_from_idx = %s, updated_at = %s "
                "WHERE session_id = %s",
                (req.compact_summary, req.compact_from_idx, datetime.now(timezone.utc), session_id),
            )
        return {"status": "updated"}
    except Exception as exc:
        log.error("conv_update_compact_state failed for %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.patch("/conversations/sessions/{session_id}/title")
def conv_update_title(session_id: str, req: ConvTitleIn) -> dict:
    try:
        with conn() as c:
            c.execute(
                "UPDATE conversation_sessions SET title = %s, updated_at = %s WHERE session_id = %s",
                (req.title, datetime.now(timezone.utc), session_id),
            )
        return {"status": "updated"}
    except Exception as exc:
        log.error("conv_update_title failed for %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))
