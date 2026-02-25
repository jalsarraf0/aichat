"""aichat-database: PostgreSQL-backed article, image, and web-content cache service.

Replaces the old aichat-rssfeed service.  All web-related data (articles fetched
via the browser, images, cached pages) is stored here so the AI can compare
fresh results against what it previously retrieved.

Also provides a structured error-log table so other services can record and
query application-level errors.
"""
from __future__ import annotations

import logging
import os
import time
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
            n_articles = c.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
            n_cache    = c.execute("SELECT COUNT(*) FROM web_cache").fetchone()[0]
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
    url: str
    host_path: Optional[str] = None
    alt_text: Optional[str] = None


@app.post("/images/store")
def store_image(image: ImageIn) -> dict:
    try:
        with conn() as c:
            c.execute(
                """
                INSERT INTO images (url, host_path, alt_text, stored_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (url) DO UPDATE
                  SET host_path = EXCLUDED.host_path,
                      alt_text  = EXCLUDED.alt_text,
                      stored_at = EXCLUDED.stored_at
                """,
                (image.url, image.host_path, image.alt_text, datetime.now(timezone.utc)),
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
