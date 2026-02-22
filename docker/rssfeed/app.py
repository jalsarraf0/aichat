import asyncio
import os
from datetime import datetime, timedelta, timezone

import psycopg
from fastapi import FastAPI

app = FastAPI()
DB = os.environ.get("DATABASE_URL", "postgresql://rss:rss@localhost:5432/rssfeed")
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "30"))
PURGE_INTERVAL = int(os.environ.get("PURGE_INTERVAL_SECONDS", "21600"))
last_purge_at = None


def conn():
    return psycopg.connect(DB)


@app.on_event("startup")
def startup():
    with conn() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS items (
          id SERIAL PRIMARY KEY,
          topic TEXT NOT NULL,
          title TEXT NOT NULL,
          url TEXT NOT NULL,
          published_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """)
    asyncio.create_task(purge_loop())


async def purge_loop():
    global last_purge_at
    while True:
        with conn() as c:
            c.execute(
                "DELETE FROM items WHERE published_at < %s",
                (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS),),
            )
        last_purge_at = datetime.now(timezone.utc).isoformat()
        await asyncio.sleep(PURGE_INTERVAL)


@app.get("/health")
def health():
    with conn() as c:
        count = c.execute("SELECT COUNT(*) FROM items WHERE published_at >= NOW() - INTERVAL '30 days'").fetchone()[0]
    return {"ok": True, "last_purge_at": last_purge_at, "items_last_30_days": count}


@app.get("/sources")
def sources(topic: str):
    return {"topic": topic, "sources": []}


@app.post("/refresh")
def refresh(payload: dict):
    return {"topic": payload.get("topic", ""), "refreshed": True}


@app.get("/latest")
def latest(topic: str):
    with conn() as c:
        rows = c.execute(
            "SELECT title,url,published_at FROM items WHERE topic=%s ORDER BY published_at DESC LIMIT 20",
            (topic,),
        ).fetchall()
    return {"topic": topic, "items": [{"title": r[0], "url": r[1], "published_at": r[2].isoformat()} for r in rows]}


@app.post("/ingest")
def ingest(payload: dict):
    topic = payload.get("topic", "general")
    with conn() as c:
        for item in payload.get("items", []):
            c.execute(
                "INSERT INTO items(topic,title,url,published_at) VALUES (%s,%s,%s,%s)",
                (topic, item["title"], item["url"], datetime.now(timezone.utc)),
            )
    return {"inserted": len(payload.get("items", []))}


@app.post("/purge")
def purge(payload: dict):
    days = int(payload.get("retention_days", 30))
    with conn() as c:
        cur = c.execute("DELETE FROM items WHERE published_at < %s", (datetime.now(timezone.utc) - timedelta(days=days),))
    return {"purged": cur.rowcount, "retention_days": days}
