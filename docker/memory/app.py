"""aichat-memory: persistent key-value memory service.

Provides a simple SQLite-backed store so the AI can remember facts,
notes, and context across sessions.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

DB_PATH = Path("/data/memory.db")
app = FastAPI(title="aichat-memory")


@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


@app.on_event("startup")
def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _db() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                ts    INTEGER NOT NULL
            )
            """
        )


class StoreRequest(BaseModel):
    key: str
    value: str


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/store")
async def store(req: StoreRequest) -> dict:
    key = req.key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="key must not be empty")
    with _db() as con:
        con.execute(
            "INSERT INTO memory (key, value, ts) VALUES (?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value, ts=excluded.ts",
            (key, req.value, int(time.time())),
        )
    return {"stored": key}


@app.get("/recall")
async def recall(key: str = Query(default="")) -> dict:
    key = key.strip()
    with _db() as con:
        if key:
            row = con.execute("SELECT key, value, ts FROM memory WHERE key=?", (key,)).fetchone()
            if row is None:
                return {"key": key, "found": False, "entries": []}
            return {
                "key": key,
                "found": True,
                "entries": [{"key": row["key"], "value": row["value"], "ts": row["ts"]}],
            }
        rows = con.execute("SELECT key, value, ts FROM memory ORDER BY ts DESC LIMIT 50").fetchall()
        return {
            "key": "",
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
