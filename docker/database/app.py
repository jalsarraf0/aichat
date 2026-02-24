"""aichat-database: PostgreSQL-backed article, image, and web-content cache service.

Replaces the old aichat-rssfeed service.  All web-related data (articles fetched
via the browser, images, cached pages) is stored here so the AI can compare
fresh results against what it previously retrieved.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Optional

import psycopg
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="aichat-database")

DB = os.environ.get("DATABASE_URL", "postgresql://aichat:aichat@localhost:5432/aichat")


def conn():
    return psycopg.connect(DB)


@app.on_event("startup")
def startup() -> None:
    for attempt in range(1, 16):
        try:
            with conn() as c:
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
            break
        except psycopg.OperationalError:
            if attempt >= 15:
                raise
            time.sleep(1)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    with conn() as c:
        n_articles = c.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        n_cache    = c.execute("SELECT COUNT(*) FROM web_cache").fetchone()[0]
    return {"ok": True, "articles": n_articles, "cached_pages": n_cache}


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
    return {"status": "stored", "url": article.url}


@app.get("/articles/search")
def search_articles(
    topic: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 20,
) -> dict:
    with conn() as c:
        if topic and q:
            rows = c.execute(
                """SELECT url, title, content, topic, stored_at FROM articles
                   WHERE topic = %s AND (title ILIKE %s OR content ILIKE %s)
                   ORDER BY stored_at DESC LIMIT %s""",
                (topic, f"%{q}%", f"%{q}%", limit),
            ).fetchall()
        elif topic:
            rows = c.execute(
                """SELECT url, title, content, topic, stored_at FROM articles
                   WHERE topic = %s
                   ORDER BY stored_at DESC LIMIT %s""",
                (topic, limit),
            ).fetchall()
        elif q:
            rows = c.execute(
                """SELECT url, title, content, topic, stored_at FROM articles
                   WHERE title ILIKE %s OR content ILIKE %s
                   ORDER BY stored_at DESC LIMIT %s""",
                (f"%{q}%", f"%{q}%", limit),
            ).fetchall()
        else:
            rows = c.execute(
                """SELECT url, title, content, topic, stored_at FROM articles
                   ORDER BY stored_at DESC LIMIT %s""",
                (limit,),
            ).fetchall()
    return {
        "articles": [
            {
                "url": r[0], "title": r[1], "content": r[2],
                "topic": r[3], "stored_at": r[4].isoformat(),
            }
            for r in rows
        ]
    }


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

class ImageIn(BaseModel):
    url: str
    host_path: Optional[str] = None
    alt_text: Optional[str] = None


@app.post("/images/store")
def store_image(image: ImageIn) -> dict:
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


@app.get("/images/list")
def list_images(limit: int = 20) -> dict:
    with conn() as c:
        rows = c.execute(
            """SELECT url, host_path, alt_text, stored_at FROM images
               ORDER BY stored_at DESC LIMIT %s""",
            (limit,),
        ).fetchall()
    return {
        "images": [
            {"url": r[0], "host_path": r[1], "alt_text": r[2], "stored_at": r[3].isoformat()}
            for r in rows
        ]
    }


# ---------------------------------------------------------------------------
# Web cache  (stores full page content keyed by URL)
# ---------------------------------------------------------------------------

class CacheIn(BaseModel):
    url: str
    content: str
    title: Optional[str] = None


@app.post("/cache/store")
def cache_store(item: CacheIn) -> dict:
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
    return {"status": "cached", "url": item.url}


@app.get("/cache/get")
def cache_get(url: str) -> dict:
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


@app.get("/cache/check")
def cache_check(url: str) -> dict:
    with conn() as c:
        row = c.execute(
            "SELECT cached_at FROM web_cache WHERE url = %s",
            (url,),
        ).fetchone()
    return {"cached": bool(row), "cached_at": row[0].isoformat() if row else None}
