"""aichat-data: Consolidated data, memory, graph, planning, research, and job-queue service.

All persistence is PostgreSQL-backed (no more SQLite).  Migrations are applied
automatically at startup via ``migrate.run_migrations``.

Route prefixes (backward compatible with existing MCP env-var URLs):
  /                     -- PostgreSQL-backed article/image/cache/error store
  /memory/*             -- key-value memory with TTL + compaction
  /research/*           -- RSS feed discovery and ingest
  /graph/*              -- NetworkX knowledge graph (PostgreSQL-backed)
  /planner/*            -- dependency-aware task queue
  /jobs/*               -- Durable async job system
  /embeddings/*         -- Embedding store with cosine similarity
  /batch/*              -- Batch write operations

Health: GET /health -- aggregates all sub-service health.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

import feedparser
import networkx as nx
import psycopg
from psycopg.rows import dict_row
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("aichat-data")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_POSTGRES = os.environ.get(
    "DATABASE_URL",
    "postgresql://aichat:aichat@aichat-db:5432/aichat",
)
_SERVICE = "aichat-data"

# Job status constants
JOB_QUEUED    = "queued"
JOB_RUNNING   = "running"
JOB_SUCCEEDED = "succeeded"
JOB_FAILED    = "failed"
JOB_CANCELLED = "cancelled"
VALID_JOB_STATUSES = {JOB_QUEUED, JOB_RUNNING, JOB_SUCCEEDED, JOB_FAILED, JOB_CANCELLED}

VALID_TASK_STATUSES = {"pending", "ready", "in_progress", "done", "failed", "cancelled"}

# ---------------------------------------------------------------------------
# Postgres helpers
# ---------------------------------------------------------------------------

@contextmanager
def _pg() -> Generator[psycopg.Connection, None, None]:
    con = psycopg.connect(DB_POSTGRES, row_factory=dict_row)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run migrations
    from migrate import run_migrations
    from sqlite_to_pg import transfer_sqlite_to_pg
    try:
        run_migrations(DB_POSTGRES, Path("migrations"))
        log.info("PostgreSQL migrations applied")
    except Exception as exc:
        log.error("Migration error: %s", exc, exc_info=True)

    # One-time SQLite → PostgreSQL data transfer (idempotent)
    try:
        transfer_sqlite_to_pg(DB_POSTGRES, Path("/data"))
    except Exception as exc:
        log.warning("SQLite transfer skipped: %s", exc)

    # Start background purge task
    purge_task = asyncio.create_task(_auto_purge_loop())
    log.info("aichat-data ready (all-PostgreSQL)")
    yield
    purge_task.cancel()


async def _auto_purge_loop() -> None:
    """Background task: purge expired memory entries and old terminal jobs."""
    while True:
        await asyncio.sleep(3600)  # every hour
        try:
            with _pg() as pg:
                # Purge expired memory
                cur = pg.execute(
                    "DELETE FROM memory WHERE expires_at IS NOT NULL AND expires_at < NOW()"
                )
                expired = cur.rowcount
                # Purge terminal jobs older than 7 days
                cur2 = pg.execute(
                    "DELETE FROM jobs WHERE status IN ('succeeded', 'failed', 'cancelled') "
                    "AND finished_at < NOW() - INTERVAL '7 days'"
                )
                purged_jobs = cur2.rowcount
                if expired or purged_jobs:
                    log.info("auto-purge: %d expired memory, %d old jobs", expired, purged_jobs)
        except Exception as exc:
            log.warning("auto-purge error: %s", exc)


app = FastAPI(title="aichat-data", lifespan=lifespan)


@app.exception_handler(Exception)
async def _global_exc(request: Request, exc: Exception) -> JSONResponse:
    msg = str(exc)
    log.error("Unhandled [%s %s]: %s", request.method, request.url.path, msg, exc_info=True)
    return JSONResponse(status_code=500, content={"error": msg})

# ===========================================================================
# ROOT - PostgreSQL database routes (backward compat: was aichat-database:8091)
# ===========================================================================

@app.get("/health")
def health() -> dict:
    """Aggregate health — single PostgreSQL check."""
    subs: dict[str, str] = {}
    try:
        with _pg() as pg:
            pg.execute("SELECT 1")
        subs["postgres"] = "ok"
    except Exception as exc:
        subs["postgres"] = f"error: {exc}"

    # Report sub-service health using backward-compatible names
    # (tests expect "memory", "graph", "planner", "jobs")
    _table_to_name = {
        "memory": "memory",
        "graph_nodes": "graph",
        "tasks": "planner",
        "jobs": "jobs",
        "embeddings": "embeddings",
    }
    for tbl, name in _table_to_name.items():
        try:
            with _pg() as pg:
                pg.execute(f"SELECT 1 FROM {tbl} LIMIT 0")  # noqa: S608
            subs[name] = "ok"
        except Exception:
            subs[name] = "not_initialized"

    overall = "ok" if subs.get("postgres") == "ok" else "degraded"
    return {"status": overall, "services": subs}


@app.post("/articles/store")
def article_store(payload: dict) -> dict:
    url     = str(payload.get("url", "")).strip()
    title   = str(payload.get("title", "")).strip()
    content = str(payload.get("content", "")).strip()
    topic   = str(payload.get("topic", "")).strip()
    if not url:
        raise HTTPException(status_code=422, detail="'url' is required")
    with _pg() as pg:
        pg.execute(
            "INSERT INTO articles(url, title, content, topic) VALUES(%s,%s,%s,%s) "
            "ON CONFLICT(url) DO UPDATE SET title=EXCLUDED.title, content=EXCLUDED.content, "
            "topic=EXCLUDED.topic, stored_at=NOW()",
            (url, title or None, content or None, topic or None),
        )
    return {"stored": url}


@app.get("/articles/search")
def article_search(
    q: str = Query(default=""),
    topic: str = Query(default=""),
    limit: int = Query(default=20),
    offset: int = Query(default=0),
    summary_only: bool = Query(default=False),
) -> dict:
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    conditions: list[str] = []
    params: list[Any] = []
    if q:
        conditions.append("(title ILIKE %s OR content ILIKE %s)")
        params += [f"%{q}%", f"%{q}%"]
    if topic:
        conditions.append("topic ILIKE %s")
        params.append(f"%{topic}%")
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params += [limit, offset]
    sql = (
        f"SELECT id, url, title, content, topic, stored_at FROM articles {where} "  # noqa: S608
        "ORDER BY stored_at DESC LIMIT %s OFFSET %s"
    )
    with _pg() as pg:
        rows = pg.execute(sql, params).fetchall()

    def _content(raw: Any) -> str:
        text = str(raw or "")
        if summary_only and len(text) > 300:
            return text[:300] + "\u2026"
        return text

    return {
        "results": [
            {"id": r["id"], "url": r["url"], "title": r["title"],
             "content": _content(r["content"]), "topic": r["topic"],
             "stored_at": str(r["stored_at"])}
            for r in rows
        ],
        "count": len(rows),
    }


@app.post("/images/store")
def image_store(payload: dict) -> dict:
    url           = str(payload.get("url", "")).strip()
    host_path     = str(payload.get("host_path", "")).strip()
    alt_text      = str(payload.get("alt_text", "") or payload.get("description", "")).strip()
    subject       = str(payload.get("subject", "")).strip() or None
    phash         = str(payload.get("phash", "")).strip() or None
    quality_score = payload.get("quality_score")
    if quality_score is not None:
        try:
            quality_score = float(quality_score)
        except (TypeError, ValueError):
            quality_score = None
    if not url:
        raise HTTPException(status_code=422, detail="'url' is required")
    with _pg() as pg:
        pg.execute(
            "INSERT INTO images(url, host_path, alt_text, subject, phash, quality_score) "
            "VALUES(%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT(url) DO UPDATE SET host_path=EXCLUDED.host_path, "
            "alt_text=EXCLUDED.alt_text, subject=EXCLUDED.subject, "
            "phash=EXCLUDED.phash, quality_score=EXCLUDED.quality_score, stored_at=NOW()",
            (url, host_path or None, alt_text or None, subject, phash, quality_score),
        )
    return {"stored": url}


@app.get("/images/list")
def image_list(
    limit: int = Query(default=50),
    offset: int = Query(default=0),
) -> dict:
    limit  = max(1, min(limit, 500))
    offset = max(0, offset)
    with _pg() as pg:
        rows = pg.execute(
            "SELECT url, host_path, alt_text, subject, phash, quality_score, stored_at "
            "FROM images ORDER BY stored_at DESC LIMIT %s OFFSET %s",
            (limit, offset),
        ).fetchall()
    return {
        "images": [
            {"url": r["url"], "host_path": r["host_path"], "alt_text": r["alt_text"],
             "subject": r["subject"], "phash": r["phash"],
             "quality_score": r["quality_score"], "stored_at": str(r["stored_at"])}
            for r in rows
        ],
        "count": len(rows),
    }


@app.get("/images/search")
def image_search(
    subject: str = Query(default=""),
    limit: int = Query(default=20),
    offset: int = Query(default=0),
) -> dict:
    limit  = max(1, min(limit, 200))
    offset = max(0, offset)
    with _pg() as pg:
        if subject.strip():
            rows = pg.execute(
                "SELECT url, host_path, alt_text, subject, phash, quality_score, stored_at "
                "FROM images WHERE subject ILIKE %s "
                "ORDER BY stored_at DESC LIMIT %s OFFSET %s",
                (f"%{subject.strip()}%", limit, offset),
            ).fetchall()
        else:
            rows = pg.execute(
                "SELECT url, host_path, alt_text, subject, phash, quality_score, stored_at "
                "FROM images ORDER BY stored_at DESC LIMIT %s OFFSET %s",
                (limit, offset),
            ).fetchall()
    return {
        "images": [
            {"url": r["url"], "host_path": r["host_path"], "alt_text": r["alt_text"],
             "subject": r["subject"], "phash": r["phash"],
             "quality_score": r["quality_score"], "stored_at": str(r["stored_at"])}
            for r in rows
        ],
        "count": len(rows),
    }


@app.post("/cache/store")
def cache_store(payload: dict) -> dict:
    key   = str(payload.get("key") or payload.get("url", "")).strip()
    value = str(payload.get("value") or payload.get("content", "")).strip()
    if not key:
        raise HTTPException(status_code=422, detail="'key' (or 'url') is required")
    with _pg() as pg:
        pg.execute(
            "INSERT INTO cache(key, value) VALUES(%s,%s) "
            "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value, stored_at=NOW()",
            (key, value),
        )
    return {"stored": key}


@app.get("/cache/get")
def cache_get(key: str = Query(...)) -> dict:
    key = key.strip()
    if not key:
        raise HTTPException(status_code=422, detail="'key' is required")
    with _pg() as pg:
        row = pg.execute("SELECT value, stored_at FROM cache WHERE key=%s", (key,)).fetchone()
    if not row:
        return {"key": key, "found": False, "value": None}
    return {"key": key, "found": True, "value": row["value"], "stored_at": str(row["stored_at"])}


@app.post("/errors/log")
def error_log(payload: dict) -> dict:
    service = str(payload.get("service", "unknown")).strip()
    level   = str(payload.get("level", "ERROR")).strip()
    message = str(payload.get("message", "")).strip()
    detail  = str(payload.get("detail", "")).strip()
    with _pg() as pg:
        pg.execute(
            "INSERT INTO errors(service, level, message, detail) VALUES(%s,%s,%s,%s)",
            (service, level, message[:2000], detail[:2000] or None),
        )
    return {"logged": True}


@app.get("/errors/recent")
def errors_recent(service: str = Query(default=""), limit: int = Query(default=50)) -> dict:
    limit = max(1, min(limit, 500))
    with _pg() as pg:
        if service:
            rows = pg.execute(
                "SELECT service, level, message, detail, logged_at FROM errors "
                "WHERE service=%s ORDER BY logged_at DESC LIMIT %s",
                (service, limit),
            ).fetchall()
        else:
            rows = pg.execute(
                "SELECT service, level, message, detail, logged_at FROM errors "
                "ORDER BY logged_at DESC LIMIT %s",
                (limit,),
            ).fetchall()
    return {
        "errors": [
            {"service": r["service"], "level": r["level"], "message": r["message"],
             "detail": r["detail"], "logged_at": str(r["logged_at"])}
            for r in rows
        ],
        "count": len(rows),
    }

# ===========================================================================
# /memory - PostgreSQL key-value persistent memory
# ===========================================================================

memory_router = APIRouter(prefix="/memory", tags=["memory"])


class _StoreReq(BaseModel):
    key: str
    value: str
    ttl_seconds: Optional[int] = None


@memory_router.get("/health")
def memory_health() -> dict:
    with _pg() as pg:
        row = pg.execute("SELECT COUNT(*) AS cnt FROM memory").fetchone()
    return {"status": "ok", "entries": row["cnt"] if row else 0}


@memory_router.post("/store")
def memory_store(req: _StoreReq) -> dict:
    key = req.key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="key must not be empty")
    expires_at = None
    if req.ttl_seconds and req.ttl_seconds > 0:
        expires_at = f"NOW() + INTERVAL '{int(req.ttl_seconds)} seconds'"
    with _pg() as pg:
        if expires_at:
            pg.execute(
                f"INSERT INTO memory(key, value, ts, expires_at) "
                f"VALUES(%s, %s, NOW(), {expires_at}) "
                "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value, ts=NOW(), "
                f"expires_at={expires_at}",
                (key, req.value),
            )
        else:
            pg.execute(
                "INSERT INTO memory(key, value, ts) VALUES(%s, %s, NOW()) "
                "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value, ts=NOW(), "
                "expires_at=NULL",
                (key, req.value),
            )
    return {"stored": key}


@memory_router.get("/recall")
def memory_recall(
    key: str = Query(default=""),
    pattern: str = Query(default=""),
) -> dict:
    key = key.strip()
    pattern = pattern.strip()
    with _pg() as pg:
        if key:
            row = pg.execute(
                "SELECT key, value, ts FROM memory "
                "WHERE key=%s AND (expires_at IS NULL OR expires_at > NOW())",
                (key,),
            ).fetchone()
            if row is None:
                return {"key": key, "found": False, "entries": []}
            return {
                "key": key, "found": True,
                "entries": [{"key": row["key"], "value": row["value"],
                             "ts": str(row["ts"])}],
            }
        if pattern:
            rows = pg.execute(
                "SELECT key, value, ts FROM memory "
                "WHERE key LIKE %s AND (expires_at IS NULL OR expires_at > NOW()) "
                "ORDER BY ts DESC LIMIT 50",
                (pattern,),
            ).fetchall()
        else:
            rows = pg.execute(
                "SELECT key, value, ts FROM memory "
                "WHERE (expires_at IS NULL OR expires_at > NOW()) "
                "ORDER BY ts DESC LIMIT 50",
            ).fetchall()
    return {
        "key": key or pattern, "found": bool(rows),
        "entries": [{"key": r["key"], "value": r["value"], "ts": str(r["ts"])} for r in rows],
    }


@memory_router.delete("/delete")
def memory_delete(key: str = Query(...)) -> dict:
    key = key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="key must not be empty")
    with _pg() as pg:
        cur = pg.execute("DELETE FROM memory WHERE key=%s", (key,))
    return {"deleted": key, "found": cur.rowcount > 0}


@memory_router.delete("/clear")
def memory_clear() -> dict:
    with _pg() as pg:
        cur = pg.execute("DELETE FROM memory")
    return {"cleared": cur.rowcount}


@memory_router.post("/compact")
def memory_compact() -> dict:
    """Purge expired entries and return stats."""
    with _pg() as pg:
        before = pg.execute("SELECT COUNT(*) AS cnt FROM memory").fetchone()["cnt"]
        pg.execute(
            "DELETE FROM memory WHERE expires_at IS NOT NULL AND expires_at < NOW()"
        )
        after = pg.execute("SELECT COUNT(*) AS cnt FROM memory").fetchone()["cnt"]
        pg.execute(
            "INSERT INTO compaction_log(operation, entries_before, entries_after) "
            "VALUES('memory_compact', %s, %s)",
            (before, after),
        )
    return {"before": before, "after": after, "purged": before - after}


@memory_router.get("/stats")
def memory_stats() -> dict:
    """Memory usage statistics."""
    with _pg() as pg:
        total = pg.execute("SELECT COUNT(*) AS cnt FROM memory").fetchone()["cnt"]
        with_ttl = pg.execute(
            "SELECT COUNT(*) AS cnt FROM memory WHERE expires_at IS NOT NULL"
        ).fetchone()["cnt"]
        expired = pg.execute(
            "SELECT COUNT(*) AS cnt FROM memory "
            "WHERE expires_at IS NOT NULL AND expires_at < NOW()"
        ).fetchone()["cnt"]
    return {"total": total, "with_ttl": with_ttl, "expired_pending_purge": expired}


app.include_router(memory_router)

# ===========================================================================
# /research - RSS feed discovery and ingest
# ===========================================================================

research_router = APIRouter(prefix="/research", tags=["research"])


@research_router.get("/health")
def research_health() -> dict:
    return {"status": "ok"}


@research_router.get("/search-feeds")
def research_search_feeds(topic: str) -> dict:
    from urllib.parse import quote_plus
    safe = quote_plus(topic.strip())
    return {
        "topic": topic,
        "feeds": [
            f"https://news.google.com/rss/search?q={safe}",
            f"https://hnrss.org/newest?q={safe}",
        ],
    }


@research_router.post("/push-feed")
async def research_push_feed(payload: dict) -> dict:
    topic    = str(payload.get("topic", "")).strip()
    feed_url = str(payload.get("feed_url", "")).strip()
    if not topic or not feed_url:
        return {"error": "topic and feed_url are required", "inserted": 0, "failed": 0}
    from urllib.parse import urlparse as _urlp
    if _urlp(feed_url).scheme not in ("http", "https"):
        return {"error": "feed_url must use http or https", "inserted": 0, "failed": 0}

    loop = asyncio.get_running_loop()
    try:
        parsed = await asyncio.wait_for(
            loop.run_in_executor(None, feedparser.parse, feed_url),
            timeout=20.0,
        )
    except asyncio.TimeoutError:
        return {"topic": topic, "feed_url": feed_url, "inserted": 0, "failed": 0,
                "errors": [{"url": feed_url, "error": "feed fetch timed out after 20s"}]}

    items = [
        {"title": e.get("title", "untitled"), "url": e.get("link", feed_url)}
        for e in getattr(parsed, "entries", [])[:20]
    ]

    stored = failed = 0
    errors: list[dict] = []
    for item in items:
        try:
            article_store({"url": item["url"], "title": item["title"], "topic": topic})
            stored += 1
        except Exception as exc:
            failed += 1
            errors.append({"url": item["url"], "error": str(exc)})

    result: dict = {"topic": topic, "feed_url": feed_url, "inserted": stored, "failed": failed}
    if errors:
        result["errors"] = errors
    return result


app.include_router(research_router)

# ===========================================================================
# /graph - PostgreSQL knowledge graph with NetworkX
# ===========================================================================

graph_router = APIRouter(prefix="/graph", tags=["graph"])


def _build_nx_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    with _pg() as pg:
        for row in pg.execute("SELECT id FROM graph_nodes"):
            G.add_node(row["id"])
        for row in pg.execute("SELECT from_id, to_id, label FROM graph_edges"):
            G.add_edge(row["from_id"], row["to_id"], type=row["label"])
    return G


@graph_router.get("/health")
def graph_health() -> dict:
    with _pg() as pg:
        n = pg.execute("SELECT COUNT(*) AS cnt FROM graph_nodes").fetchone()["cnt"]
        e = pg.execute("SELECT COUNT(*) AS cnt FROM graph_edges").fetchone()["cnt"]
    return {"status": "ok", "nodes": n, "edges": e}


@graph_router.post("/nodes/add")
def graph_add_node(payload: dict) -> dict:
    node_id    = str(payload.get("id", "")).strip()
    labels     = list(payload.get("labels", []))
    properties = dict(payload.get("properties", {}))
    if not node_id:
        raise HTTPException(status_code=422, detail="'id' is required")
    # Determine label and node_type from labels list
    label = labels[0] if labels else ""
    node_type = payload.get("type", "")
    with _pg() as pg:
        pg.execute(
            "INSERT INTO graph_nodes(id, label, node_type, data) VALUES(%s, %s, %s, %s::jsonb) "
            "ON CONFLICT(id) DO UPDATE SET label=EXCLUDED.label, "
            "node_type=EXCLUDED.node_type, data=EXCLUDED.data",
            (node_id, label, node_type, json.dumps({"labels": labels, "properties": properties})),
        )
    return {"added": node_id, "labels": labels, "properties": properties}


@graph_router.post("/edges/add")
def graph_add_edge(payload: dict) -> dict:
    from_id    = str(payload.get("from_id", "")).strip()
    to_id      = str(payload.get("to_id",   "")).strip()
    etype      = str(payload.get("type", "related")).strip() or "related"
    properties = dict(payload.get("properties", {}))
    weight     = float(payload.get("weight", 1.0))
    if not from_id or not to_id:
        raise HTTPException(status_code=422, detail="'from_id' and 'to_id' are required")
    with _pg() as pg:
        row = pg.execute(
            "INSERT INTO graph_edges(from_id, to_id, label, weight, data) "
            "VALUES(%s, %s, %s, %s, %s::jsonb) RETURNING id",
            (from_id, to_id, etype, weight, json.dumps(properties)),
        ).fetchone()
        edge_id = row["id"] if row else None
    return {"added": edge_id, "from_id": from_id, "to_id": to_id, "type": etype}


@graph_router.get("/nodes/{node_id}")
def graph_get_node(node_id: str) -> dict:
    with _pg() as pg:
        row = pg.execute(
            "SELECT id, label, node_type, data, created_at FROM graph_nodes WHERE id=%s",
            (node_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
        data = row["data"] if isinstance(row["data"], dict) else json.loads(row["data"] or "{}")
        node = {
            "id": row["id"],
            "labels": data.get("labels", [row["label"]] if row["label"] else []),
            "properties": data.get("properties", {}),
            "created_at": str(row["created_at"]),
        }
        out_edges = pg.execute(
            "SELECT id, from_id, to_id, label AS type, data, created_at "
            "FROM graph_edges WHERE from_id=%s", (node_id,),
        ).fetchall()
        in_edges = pg.execute(
            "SELECT id, from_id, to_id, label AS type, data, created_at "
            "FROM graph_edges WHERE to_id=%s", (node_id,),
        ).fetchall()

    def _edge(r: dict) -> dict:
        d = r["data"] if isinstance(r["data"], dict) else json.loads(r["data"] or "{}")
        return {"id": r["id"], "from_id": r["from_id"], "to_id": r["to_id"],
                "type": r["type"], "properties": d, "created_at": str(r["created_at"])}
    return {"node": node, "out_edges": [_edge(e) for e in out_edges],
            "in_edges": [_edge(e) for e in in_edges]}


@graph_router.get("/nodes/{node_id}/neighbors")
def graph_get_neighbors(node_id: str) -> dict:
    with _pg() as pg:
        row = pg.execute("SELECT 1 FROM graph_nodes WHERE id=%s", (node_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
        edges = pg.execute(
            "SELECT e.to_id, e.label AS edge_type, n.data AS node_data "
            "FROM graph_edges e LEFT JOIN graph_nodes n ON n.id=e.to_id "
            "WHERE e.from_id=%s",
            (node_id,),
        ).fetchall()
    neighbors = []
    for r in edges:
        nd = r["node_data"] if isinstance(r["node_data"], dict) else json.loads(r["node_data"] or "{}")
        neighbors.append({
            "id": r["to_id"], "edge_type": r["edge_type"],
            "properties": nd.get("properties", {}),
        })
    return {"node_id": node_id, "neighbors": neighbors, "count": len(neighbors)}


@graph_router.post("/path")
def graph_find_path(payload: dict) -> dict:
    from_id = str(payload.get("from_id", "")).strip()
    to_id   = str(payload.get("to_id",   "")).strip()
    if not from_id or not to_id:
        raise HTTPException(status_code=422, detail="'from_id' and 'to_id' are required")
    G = _build_nx_graph()
    if not G.has_node(from_id):
        raise HTTPException(status_code=404, detail=f"Node '{from_id}' not in graph")
    if not G.has_node(to_id):
        raise HTTPException(status_code=404, detail=f"Node '{to_id}' not in graph")
    try:
        path = nx.shortest_path(G, from_id, to_id)
    except nx.NetworkXNoPath:
        return {"from_id": from_id, "to_id": to_id, "path": None, "length": -1,
                "message": "No path found"}
    return {"from_id": from_id, "to_id": to_id, "path": path, "length": len(path) - 1}


@graph_router.post("/search")
def graph_search(payload: dict) -> dict:
    label = str(payload.get("label", "")).strip()
    props = dict(payload.get("properties", {}))
    limit = max(1, min(int(payload.get("limit", 50)), 500))
    with _pg() as pg:
        if label:
            rows = pg.execute(
                "SELECT id, label, node_type, data, created_at FROM graph_nodes "
                "WHERE label ILIKE %s OR data::text ILIKE %s LIMIT %s",
                (f"%{label}%", f"%{label}%", limit * 3),
            ).fetchall()
        else:
            rows = pg.execute(
                "SELECT id, label, node_type, data, created_at FROM graph_nodes LIMIT %s",
                (limit * 3,),
            ).fetchall()
    results = []
    for row in rows:
        data = row["data"] if isinstance(row["data"], dict) else json.loads(row["data"] or "{}")
        node = {
            "id": row["id"],
            "labels": data.get("labels", [row["label"]] if row["label"] else []),
            "properties": data.get("properties", {}),
            "created_at": str(row["created_at"]),
        }
        if props:
            np_ = node["properties"]
            if not all(str(np_.get(k)) == str(v) for k, v in props.items()):
                continue
        results.append(node)
        if len(results) >= limit:
            break
    return {"results": results, "count": len(results), "label": label}


@graph_router.delete("/nodes/{node_id}")
def graph_delete_node(node_id: str) -> dict:
    with _pg() as pg:
        if not pg.execute("SELECT 1 FROM graph_nodes WHERE id=%s", (node_id,)).fetchone():
            raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
        pg.execute("DELETE FROM graph_edges WHERE from_id=%s OR to_id=%s", (node_id, node_id))
        pg.execute("DELETE FROM graph_nodes WHERE id=%s", (node_id,))
    return {"deleted": node_id}


app.include_router(graph_router)

# ===========================================================================
# /planner - Dependency-aware task queue
# ===========================================================================

planner_router = APIRouter(prefix="/planner", tags=["planner"])


def _task_is_ready(task_id: str, pg: psycopg.Connection) -> bool:
    row = pg.execute("SELECT depends_on FROM tasks WHERE id=%s", (task_id,)).fetchone()
    if not row:
        return False
    deps = row["depends_on"] if isinstance(row["depends_on"], list) else json.loads(row["depends_on"] or "[]")
    if not deps:
        return True
    for dep_id in deps:
        dep = pg.execute("SELECT status FROM tasks WHERE id=%s", (dep_id,)).fetchone()
        if not dep or dep["status"] != "done":
            return False
    return True


@planner_router.get("/health")
def planner_health() -> dict:
    with _pg() as pg:
        total = pg.execute("SELECT COUNT(*) AS cnt FROM tasks").fetchone()["cnt"]
        ready = pg.execute(
            "SELECT COUNT(*) AS cnt FROM tasks WHERE status IN ('pending','ready')"
        ).fetchone()["cnt"]
    return {"status": "ok", "tasks_total": total, "tasks_pending_or_ready": ready}


@planner_router.post("/tasks")
def planner_create_task(payload: dict) -> dict:
    title       = str(payload.get("title", "")).strip()
    description = str(payload.get("description", "")).strip()
    depends_on  = list(payload.get("depends_on", []))
    priority    = int(payload.get("priority", 0))
    metadata    = dict(payload.get("metadata", {}))
    if not title:
        raise HTTPException(status_code=422, detail="'title' is required")
    task_id = uuid.uuid4().hex[:12]
    with _pg() as pg:
        for dep in depends_on:
            if not pg.execute("SELECT 1 FROM tasks WHERE id=%s", (dep,)).fetchone():
                raise HTTPException(status_code=422,
                                    detail=f"depends_on references unknown task: '{dep}'")
        pg.execute(
            "INSERT INTO tasks(id, title, description, status, priority, depends_on, "
            "result, metadata) VALUES(%s,%s,%s,'pending',%s,%s::jsonb,NULL,%s::jsonb)",
            (task_id, title, description, priority,
             json.dumps(depends_on), json.dumps(metadata)),
        )
    return {"id": task_id, "title": title, "status": "pending",
            "depends_on": depends_on, "priority": priority}


@planner_router.get("/tasks/ready")
def planner_list_ready() -> dict:
    with _pg() as pg:
        rows = pg.execute(
            "SELECT * FROM tasks WHERE status IN ('pending','ready') "
            "ORDER BY priority DESC, created_at"
        ).fetchall()
        tasks = [r for r in rows if _task_is_ready(r["id"], pg)]
    return {"tasks": tasks, "count": len(tasks)}


@planner_router.get("/tasks")
def planner_list_tasks(
    status: str = Query(default=""),
    limit: int = Query(default=50),
    offset: int = Query(default=0),
) -> dict:
    limit  = max(1, min(limit, 500))
    offset = max(0, offset)
    with _pg() as pg:
        if status and status in VALID_TASK_STATUSES:
            rows = pg.execute(
                "SELECT * FROM tasks WHERE status=%s ORDER BY priority DESC, created_at "
                "LIMIT %s OFFSET %s", (status, limit, offset),
            ).fetchall()
            total = pg.execute(
                "SELECT COUNT(*) AS cnt FROM tasks WHERE status=%s", (status,)
            ).fetchone()["cnt"]
        else:
            rows = pg.execute(
                "SELECT * FROM tasks ORDER BY priority DESC, created_at LIMIT %s OFFSET %s",
                (limit, offset),
            ).fetchall()
            total = pg.execute("SELECT COUNT(*) AS cnt FROM tasks").fetchone()["cnt"]
    # Serialize JSONB fields
    for r in rows:
        for jcol in ("depends_on", "metadata"):
            if jcol in r and isinstance(r[jcol], str):
                try:
                    r[jcol] = json.loads(r[jcol])
                except Exception:
                    pass
        for tcol in ("created_at", "updated_at"):
            if tcol in r:
                r[tcol] = str(r[tcol])
    return {"tasks": rows, "total": total, "limit": limit, "offset": offset}


@planner_router.get("/tasks/{task_id}")
def planner_get_task(task_id: str) -> dict:
    with _pg() as pg:
        row = pg.execute("SELECT * FROM tasks WHERE id=%s", (task_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    for jcol in ("depends_on", "metadata"):
        if jcol in row and isinstance(row[jcol], str):
            try:
                row[jcol] = json.loads(row[jcol])
            except Exception:
                pass
    for tcol in ("created_at", "updated_at"):
        if tcol in row:
            row[tcol] = str(row[tcol])
    return dict(row)


@planner_router.patch("/tasks/{task_id}")
def planner_update_task(task_id: str, payload: dict) -> dict:
    with _pg() as pg:
        row = pg.execute("SELECT * FROM tasks WHERE id=%s", (task_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        sets: list[str] = []
        params: list[Any] = []
        if "status" in payload:
            s = payload["status"]
            if s not in VALID_TASK_STATUSES:
                raise HTTPException(status_code=422, detail=f"Invalid status: {s}")
            sets.append("status=%s"); params.append(s)
        if "metadata" in payload:
            sets.append("metadata=%s::jsonb"); params.append(json.dumps(dict(payload["metadata"])))
        if "priority" in payload:
            sets.append("priority=%s"); params.append(int(payload["priority"]))
        if "description" in payload:
            sets.append("description=%s"); params.append(str(payload["description"]))
        if sets:
            sets.append("updated_at=NOW()")
            params.append(task_id)
            pg.execute(
                f"UPDATE tasks SET {', '.join(sets)} WHERE id=%s", params  # noqa: S608
            )
    return planner_get_task(task_id)


@planner_router.delete("/tasks/{task_id}")
def planner_delete_task(task_id: str) -> dict:
    with _pg() as pg:
        if not pg.execute("SELECT 1 FROM tasks WHERE id=%s", (task_id,)).fetchone():
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        pg.execute("DELETE FROM tasks WHERE id=%s", (task_id,))
    return {"deleted": task_id}


@planner_router.post("/tasks/{task_id}/complete")
def planner_complete_task(task_id: str) -> dict:
    with _pg() as pg:
        if not pg.execute("SELECT 1 FROM tasks WHERE id=%s", (task_id,)).fetchone():
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        pg.execute("UPDATE tasks SET status='done', updated_at=NOW() WHERE id=%s", (task_id,))
    return {"id": task_id, "status": "done"}


@planner_router.post("/tasks/{task_id}/fail")
def planner_fail_task(task_id: str, payload: dict | None = None) -> dict:
    detail = str((payload or {}).get("detail", "")).strip()
    with _pg() as pg:
        row = pg.execute("SELECT metadata FROM tasks WHERE id=%s", (task_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        meta = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
        if detail:
            meta["fail_reason"] = detail
        pg.execute(
            "UPDATE tasks SET status='failed', metadata=%s::jsonb, updated_at=NOW() WHERE id=%s",
            (json.dumps(meta), task_id),
        )
    return {"id": task_id, "status": "failed", "detail": detail}


@planner_router.get("/graph")
def planner_dependency_graph() -> dict:
    with _pg() as pg:
        rows = pg.execute("SELECT id, title, status, depends_on FROM tasks").fetchall()
    nodes = [{"id": r["id"], "title": r["title"], "status": r["status"]} for r in rows]
    edges = []
    for row in rows:
        deps = row["depends_on"] if isinstance(row["depends_on"], list) else json.loads(row["depends_on"] or "[]")
        for dep in deps:
            edges.append({"from": dep, "to": row["id"]})
    return {"nodes": nodes, "edges": edges}


app.include_router(planner_router)

# ===========================================================================
# /jobs - Durable async job system
# ===========================================================================

jobs_router = APIRouter(prefix="/jobs", tags=["jobs"])


@jobs_router.get("/health")
def jobs_health() -> dict:
    with _pg() as pg:
        total   = pg.execute("SELECT COUNT(*) AS cnt FROM jobs").fetchone()["cnt"]
        queued  = pg.execute("SELECT COUNT(*) AS cnt FROM jobs WHERE status='queued'").fetchone()["cnt"]
        running = pg.execute("SELECT COUNT(*) AS cnt FROM jobs WHERE status='running'").fetchone()["cnt"]
    return {"status": "ok", "total": total, "queued": queued, "running": running}


@jobs_router.post("")
def jobs_create(payload: dict) -> dict:
    tool_name     = str(payload.get("tool_name", "")).strip()
    args          = dict(payload.get("args", {}))
    priority      = int(payload.get("priority", 0))
    timeout_s     = float(payload.get("timeout_s", 300.0))
    max_retries   = int(payload.get("max_retries", 0))
    input_summary = str(payload.get("input_summary", json.dumps(args)[:200]))
    ttl_seconds   = payload.get("ttl_seconds")
    if not tool_name:
        raise HTTPException(status_code=422, detail="'tool_name' is required")
    job_id = uuid.uuid4().hex[:16]
    with _pg() as pg:
        pg.execute(
            "INSERT INTO jobs(id, tool_name, args, status, priority, "
            "input_summary, timeout_s, max_retries, ttl_seconds) "
            "VALUES(%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s)",
            (job_id, tool_name, json.dumps(args), JOB_QUEUED, priority,
             input_summary, timeout_s, max_retries, ttl_seconds),
        )
    log.info("job_create %s tool=%s", job_id, tool_name)
    return {"id": job_id, "tool_name": tool_name, "status": JOB_QUEUED}


@jobs_router.get("")
def jobs_list(
    status: str = Query(default=""),
    tool_name: str = Query(default=""),
    limit: int = Query(default=50),
    offset: int = Query(default=0),
) -> dict:
    limit  = max(1, min(limit, 500))
    offset = max(0, offset)
    conditions: list[str] = []
    params: list[Any] = []
    if status and status in VALID_JOB_STATUSES:
        conditions.append("status=%s"); params.append(status)
    if tool_name:
        conditions.append("tool_name=%s"); params.append(tool_name)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    count_params = list(params)
    params += [limit, offset]
    sql = f"SELECT * FROM jobs {where} ORDER BY priority DESC, submitted_at DESC LIMIT %s OFFSET %s"  # noqa: S608
    sql_count = f"SELECT COUNT(*) AS cnt FROM jobs {where}"  # noqa: S608
    with _pg() as pg:
        rows = pg.execute(sql, params).fetchall()
        total = pg.execute(sql_count, count_params).fetchone()["cnt"]
    # Serialize timestamps and JSONB
    for r in rows:
        if "args" in r and isinstance(r["args"], str):
            try:
                r["args"] = json.loads(r["args"])
            except Exception:
                pass
        for tcol in ("submitted_at", "started_at", "finished_at"):
            if tcol in r and r[tcol] is not None:
                r[tcol] = str(r[tcol])
    return {"jobs": rows, "total": total, "limit": limit, "offset": offset}


@jobs_router.get("/{job_id}")
def jobs_get(job_id: str) -> dict:
    with _pg() as pg:
        row = pg.execute("SELECT * FROM jobs WHERE id=%s", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if "args" in row and isinstance(row["args"], str):
        try:
            row["args"] = json.loads(row["args"])
        except Exception:
            pass
    for tcol in ("submitted_at", "started_at", "finished_at"):
        if tcol in row and row[tcol] is not None:
            row[tcol] = str(row[tcol])
    return dict(row)


@jobs_router.patch("/{job_id}")
def jobs_update(job_id: str, payload: dict) -> dict:
    with _pg() as pg:
        row = pg.execute("SELECT * FROM jobs WHERE id=%s", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        fields: list[str] = []
        params: list[Any] = []
        for col in ("status", "started_at", "finished_at", "progress",
                    "result", "error", "retry_count"):
            if col in payload:
                val = payload[col]
                if col == "status" and val not in VALID_JOB_STATUSES:
                    raise HTTPException(status_code=422, detail=f"Invalid status: {val}")
                fields.append(f"{col}=%s")
                params.append(val)
        if "logs" in payload:
            existing = row.get("logs") or ""
            new_logs = existing + payload["logs"]
            fields.append("logs=%s")
            params.append(new_logs[-50000:])
        if not fields:
            return jobs_get(job_id)
        params.append(job_id)
        pg.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id=%s", params)  # noqa: S608
    return jobs_get(job_id)


@jobs_router.post("/{job_id}/cancel")
def jobs_cancel(job_id: str) -> dict:
    with _pg() as pg:
        row = pg.execute("SELECT status FROM jobs WHERE id=%s", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        if row["status"] in (JOB_SUCCEEDED, JOB_FAILED, JOB_CANCELLED):
            return {"id": job_id, "status": row["status"], "note": "already terminal"}
        pg.execute(
            "UPDATE jobs SET status=%s, finished_at=NOW() WHERE id=%s",
            (JOB_CANCELLED, job_id),
        )
    return {"id": job_id, "status": JOB_CANCELLED}


@jobs_router.post("/batch")
def jobs_batch_create(payload: dict) -> dict:
    items = list(payload.get("items", []))
    if not items:
        raise HTTPException(status_code=422,
                            detail="'items' list is required and must not be empty")
    if len(items) > 100:
        raise HTTPException(status_code=422, detail="Maximum 100 items per batch")
    created: list[dict] = []
    for item in items:
        job = jobs_create(item)
        created.append({"id": job["id"], "tool_name": job["tool_name"]})
    return {"job_ids": created, "count": len(created)}


app.include_router(jobs_router)

# ===========================================================================
# /embeddings - PostgreSQL embedding store with cosine similarity
# ===========================================================================

embed_router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@embed_router.post("/store")
def embed_store(payload: dict) -> dict:
    key     = str(payload.get("key", "")).strip()
    content = str(payload.get("content", "")).strip()
    vector  = payload.get("embedding", [])
    model   = str(payload.get("model", "")).strip()
    topic   = str(payload.get("topic", "")).strip()
    if not key:
        raise HTTPException(status_code=422, detail="'key' is required")
    if not vector:
        raise HTTPException(status_code=422, detail="'embedding' vector is required")
    with _pg() as pg:
        pg.execute(
            "INSERT INTO embeddings (key, content, vector, model, topic) "
            "VALUES (%s, %s, %s::jsonb, %s, %s) "
            "ON CONFLICT (key) DO UPDATE SET content=EXCLUDED.content, "
            "vector=EXCLUDED.vector, model=EXCLUDED.model, topic=EXCLUDED.topic, "
            "stored_at=NOW()",
            (key, content, json.dumps(vector), model, topic),
        )
    return {"stored": key, "dims": len(vector)}


@embed_router.post("/search")
def embed_search(payload: dict) -> dict:
    query_vec = payload.get("embedding", [])
    limit     = max(1, min(50, int(payload.get("limit", 5))))
    topic     = str(payload.get("topic", "")).strip()
    if not query_vec:
        raise HTTPException(status_code=422, detail="'embedding' vector is required")
    q_norm = math.sqrt(sum(v * v for v in query_vec)) or 1.0

    with _pg() as pg:
        if topic:
            rows = pg.execute(
                "SELECT key, content, vector, model, topic FROM embeddings WHERE topic=%s",
                (topic,),
            ).fetchall()
        else:
            rows = pg.execute(
                "SELECT key, content, vector, model, topic FROM embeddings"
            ).fetchall()

    scored = []
    for r in rows:
        try:
            vec = r["vector"] if isinstance(r["vector"], list) else json.loads(r["vector"])
            dot = sum(a * b for a, b in zip(query_vec, vec))
            v_norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            score = dot / (q_norm * v_norm)
            scored.append({"key": r["key"], "content": r["content"][:500],
                           "score": round(score, 4),
                           "model": r["model"], "topic": r["topic"]})
        except Exception:
            continue

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"results": scored[:limit], "total": len(scored)}


app.include_router(embed_router)

# ===========================================================================
# /batch - Batch write operations
# ===========================================================================

batch_router = APIRouter(prefix="/batch", tags=["batch"])


@batch_router.post("/store")
def batch_store(payload: dict) -> dict:
    """Batch insert articles, images, or memory entries."""
    articles = payload.get("articles", [])
    images   = payload.get("images", [])
    memories = payload.get("memories", [])
    results = {"articles_stored": 0, "images_stored": 0, "memories_stored": 0, "errors": []}

    with _pg() as pg:
        for a in articles:
            try:
                url = str(a.get("url", "")).strip()
                if not url:
                    continue
                pg.execute(
                    "INSERT INTO articles(url, title, content, topic) VALUES(%s,%s,%s,%s) "
                    "ON CONFLICT(url) DO UPDATE SET title=EXCLUDED.title, "
                    "content=EXCLUDED.content, topic=EXCLUDED.topic, stored_at=NOW()",
                    (url, a.get("title"), a.get("content"), a.get("topic")),
                )
                results["articles_stored"] += 1
            except Exception as exc:
                results["errors"].append({"type": "article", "error": str(exc)})

        for img in images:
            try:
                url = str(img.get("url", "")).strip()
                if not url:
                    continue
                pg.execute(
                    "INSERT INTO images(url, host_path, alt_text, subject) VALUES(%s,%s,%s,%s) "
                    "ON CONFLICT(url) DO UPDATE SET host_path=EXCLUDED.host_path, "
                    "alt_text=EXCLUDED.alt_text, subject=EXCLUDED.subject, stored_at=NOW()",
                    (url, img.get("host_path"), img.get("alt_text"), img.get("subject")),
                )
                results["images_stored"] += 1
            except Exception as exc:
                results["errors"].append({"type": "image", "error": str(exc)})

        for m in memories:
            try:
                key = str(m.get("key", "")).strip()
                if not key:
                    continue
                pg.execute(
                    "INSERT INTO memory(key, value, ts) VALUES(%s, %s, NOW()) "
                    "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value, ts=NOW()",
                    (key, str(m.get("value", ""))),
                )
                results["memories_stored"] += 1
            except Exception as exc:
                results["errors"].append({"type": "memory", "error": str(exc)})

    return results


app.include_router(batch_router)
