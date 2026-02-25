"""aichat-memory: persistent key-value memory service.

Provides a simple SQLite-backed store so the AI can remember facts,
notes, and context across sessions.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Generator, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("aichat-memory")

# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------

_DATABASE_URL = os.environ.get("DATABASE_URL", "http://aichat-database:8091")
_SERVICE_NAME = "aichat-memory"


async def _report_error(message: str, detail: str | None = None) -> None:
    """Fire-and-forget: send an error entry to aichat-database."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"{_DATABASE_URL}/errors/log",
                json={"service": _SERVICE_NAME, "level": "ERROR",
                      "message": message, "detail": detail},
            )
    except Exception:
        pass  # never let error reporting crash the service

# ---------------------------------------------------------------------------
# SQLite database helpers
# ---------------------------------------------------------------------------

DB_PATH = Path("/data/memory.db")


@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _db() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                ts         INTEGER NOT NULL,
                expires_at INTEGER          -- Unix timestamp; NULL = never expires
            )
            """
        )
        # Add expires_at column to existing databases that don't have it yet
        try:
            con.execute("ALTER TABLE memory ADD COLUMN expires_at INTEGER")
        except Exception:
            pass  # column already exists
        # Purge any already-expired entries on startup
        con.execute("DELETE FROM memory WHERE expires_at IS NOT NULL AND expires_at < ?", (int(time.time()),))
    yield


app = FastAPI(title="aichat-memory", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Global exception handler — logs to aichat-database, never returns raw 500s
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    message = str(exc)
    detail = f"{request.method} {request.url.path}"
    log.error("Unhandled error [%s %s]: %s", request.method, request.url.path, exc, exc_info=True)
    asyncio.create_task(_report_error(message, detail))
    return JSONResponse(status_code=500, content={"error": message})


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class StoreRequest(BaseModel):
    key: str
    value: str
    ttl_seconds: Optional[int] = None  # if set, entry expires after this many seconds


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/store")
async def store(req: StoreRequest) -> dict:
    key = req.key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="key must not be empty")
    now = int(time.time())
    expires_at = (now + req.ttl_seconds) if req.ttl_seconds and req.ttl_seconds > 0 else None
    with _db() as con:
        con.execute(
            "INSERT INTO memory (key, value, ts, expires_at) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value, ts=excluded.ts, expires_at=excluded.expires_at",
            (key, req.value, now, expires_at),
        )
    return {"stored": key, **({"expires_at": expires_at} if expires_at else {})}


@app.get("/recall")
async def recall(key: str = Query(default=""), pattern: str = Query(default="")) -> dict:
    """Recall memory entries.  Use 'key' for exact match, 'pattern' for SQL LIKE (e.g. 'whatsapp:%')."""
    now = int(time.time())
    key = key.strip()
    pattern = pattern.strip()
    with _db() as con:
        if key:
            row = con.execute(
                "SELECT key, value, ts FROM memory WHERE key=? AND (expires_at IS NULL OR expires_at > ?)",
                (key, now),
            ).fetchone()
            if row is None:
                return {"key": key, "found": False, "entries": []}
            return {
                "key": key,
                "found": True,
                "entries": [{"key": row["key"], "value": row["value"], "ts": row["ts"]}],
            }
        if pattern:
            rows = con.execute(
                "SELECT key, value, ts FROM memory WHERE key LIKE ? AND (expires_at IS NULL OR expires_at > ?)"
                " ORDER BY ts DESC LIMIT 50",
                (pattern, now),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT key, value, ts FROM memory WHERE (expires_at IS NULL OR expires_at > ?)"
                " ORDER BY ts DESC LIMIT 50",
                (now,),
            ).fetchall()
        return {
            "key": key or pattern,
            "found": bool(rows),
            "entries": [{"key": r["key"], "value": r["value"], "ts": r["ts"]} for r in rows],
        }


@app.delete("/delete")
async def delete(key: str = Query(...)) -> dict:
    key = key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="key must not be empty")
    with _db() as con:
        cur = con.execute("DELETE FROM memory WHERE key=?", (key,))
    return {"deleted": key, "found": cur.rowcount > 0}


@app.delete("/clear")
async def clear() -> dict:
    with _db() as con:
        cur = con.execute("DELETE FROM memory")
    return {"cleared": cur.rowcount}
