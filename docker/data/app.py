"""aichat-data: Consolidated data, memory, graph, planning, research, and job-queue service.

Replaces five separate microservices (database, memory, graph, planner, researchbox)
with one FastAPI application using APIRouter sub-mounts.

Route prefixes (backward compatible with existing MCP env-var URLs):
  /                     -- PostgreSQL-backed article/image/cache/error store
  /memory/*             -- SQLite key-value memory        (was aichat-memory:8094)
  /research/*           -- RSS feed discovery and ingest  (was aichat-researchbox:8092)
  /graph/*              -- NetworkX knowledge graph       (was aichat-graph:8098)
  /planner/*            -- dependency-aware task queue    (was aichat-planner:8102)
  /jobs/*               -- Durable async job system       (NEW)

Health: GET /health -- aggregates all sub-service health.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

import feedparser
import networkx as nx
import psycopg
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

DB_POSTGRES = os.environ.get("DATABASE_URL", "postgresql://aichat:aichat@aichat-db:5432/aichat")
_SQLITE_ROOT = Path(os.environ.get("DATA_DIR", "/data"))
_SQLITE_ROOT.mkdir(parents=True, exist_ok=True)

MEMORY_DB  = _SQLITE_ROOT / "memory.db"
GRAPH_DB   = _SQLITE_ROOT / "graph.db"
PLANNER_DB = _SQLITE_ROOT / "planner.db"
JOBS_DB    = _SQLITE_ROOT / "jobs.db"

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
    con = psycopg.connect(DB_POSTGRES)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _create_pg_tables(pg: psycopg.Connection) -> None:
    pg.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id        SERIAL PRIMARY KEY,
            url       TEXT UNIQUE NOT NULL,
            title     TEXT,
            content   TEXT,
            topic     TEXT,
            stored_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    pg.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id        SERIAL PRIMARY KEY,
            url       TEXT UNIQUE NOT NULL,
            host_path TEXT,
            alt_text  TEXT,
            stored_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    pg.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key       TEXT PRIMARY KEY,
            value     TEXT NOT NULL,
            stored_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    pg.execute("""
        CREATE TABLE IF NOT EXISTS errors (
            id        SERIAL PRIMARY KEY,
            service   TEXT,
            level     TEXT,
            message   TEXT,
            detail    TEXT,
            logged_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    pg.execute("CREATE INDEX IF NOT EXISTS idx_articles_topic ON articles(topic)")
    pg.execute("CREATE INDEX IF NOT EXISTS idx_errors_svc ON errors(service, logged_at DESC)")

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _sqlite(path: Path, timeout: float = 5.0) -> sqlite3.Connection:
    con = sqlite3.connect(str(path), check_same_thread=False, timeout=timeout)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ts() -> int:
    return int(time.time())


# ---------------------------------------------------------------------------
# SQLite schema creation
# ---------------------------------------------------------------------------

def _init_memory_db() -> None:
    with _sqlite(MEMORY_DB) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                ts         INTEGER NOT NULL,
                expires_at INTEGER
            )
        """)
        try:
            con.execute("ALTER TABLE memory ADD COLUMN expires_at INTEGER")
        except Exception:
            pass
        con.execute(
            "DELETE FROM memory WHERE expires_at IS NOT NULL AND expires_at < ?",
            (_now_ts(),),
        )
        con.commit()


def _init_graph_db() -> None:
    with _sqlite(GRAPH_DB) as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id         TEXT PRIMARY KEY,
                labels     TEXT NOT NULL DEFAULT '[]',
                properties TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS edges (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                from_id    TEXT NOT NULL,
                to_id      TEXT NOT NULL,
                type       TEXT NOT NULL DEFAULT 'related',
                properties TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id);
            CREATE INDEX IF NOT EXISTS idx_edges_to   ON edges(to_id);
        """)
        con.commit()


def _init_planner_db() -> None:
    with _sqlite(PLANNER_DB) as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                status      TEXT NOT NULL DEFAULT 'pending',
                priority    INTEGER NOT NULL DEFAULT 0,
                depends_on  TEXT NOT NULL DEFAULT '[]',
                metadata    TEXT NOT NULL DEFAULT '{}',
                due_at      TEXT,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
        """)
        con.commit()


def _init_jobs_db() -> None:
    with _sqlite(JOBS_DB) as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id            TEXT PRIMARY KEY,
                tool_name     TEXT NOT NULL,
                args          TEXT NOT NULL DEFAULT '{}',
                status        TEXT NOT NULL DEFAULT 'queued',
                priority      INTEGER NOT NULL DEFAULT 0,
                submitted_at  TEXT NOT NULL,
                started_at    TEXT,
                finished_at   TEXT,
                progress      INTEGER NOT NULL DEFAULT 0,
                input_summary TEXT NOT NULL DEFAULT '',
                result        TEXT,
                error         TEXT,
                logs          TEXT NOT NULL DEFAULT '',
                retry_count   INTEGER NOT NULL DEFAULT 0,
                max_retries   INTEGER NOT NULL DEFAULT 0,
                timeout_s     REAL NOT NULL DEFAULT 300.0
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, priority DESC, submitted_at);
        """)
        con.commit()

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        with _pg() as pg:
            _create_pg_tables(pg)
        log.info("PostgreSQL tables ready")
    except Exception as exc:
        log.error("PostgreSQL init error: %s", exc)

    _init_memory_db()
    _init_graph_db()
    _init_planner_db()
    _init_jobs_db()
    log.info("aichat-data ready")
    yield


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
    """Aggregate health across all sub-services."""
    subs: dict[str, str] = {}
    try:
        with _pg() as pg:
            pg.execute("SELECT 1")
        subs["postgres"] = "ok"
    except Exception as exc:
        subs["postgres"] = f"error: {exc}"
    for name, path in [("memory", MEMORY_DB), ("graph", GRAPH_DB),
                       ("planner", PLANNER_DB), ("jobs", JOBS_DB)]:
        try:
            with _sqlite(path) as con:
                con.execute("SELECT 1")
            subs[name] = "ok"
        except Exception as exc:
            subs[name] = f"error: {exc}"
    overall = "ok" if all(v == "ok" for v in subs.values()) else "degraded"
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
) -> dict:
    limit = max(1, min(limit, 200))
    conditions: list[str] = []
    params: list[Any] = []
    if q:
        conditions.append("(title ILIKE %s OR content ILIKE %s)")
        params += [f"%{q}%", f"%{q}%"]
    if topic:
        conditions.append("topic ILIKE %s")
        params.append(f"%{topic}%")
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)
    with _pg() as pg:
        rows = pg.execute(  # nosec B608
            f"SELECT id, url, title, topic, stored_at FROM articles {where} "
            "ORDER BY stored_at DESC LIMIT %s",
            params,
        ).fetchall()
    return {
        "results": [
            {"id": r[0], "url": r[1], "title": r[2], "topic": r[3], "stored_at": str(r[4])}
            for r in rows
        ],
        "count": len(rows),
    }


@app.post("/images/store")
def image_store(payload: dict) -> dict:
    url       = str(payload.get("url", "")).strip()
    host_path = str(payload.get("host_path", "")).strip()
    alt_text  = str(payload.get("alt_text", "")).strip()
    if not url:
        raise HTTPException(status_code=422, detail="'url' is required")
    with _pg() as pg:
        pg.execute(
            "INSERT INTO images(url, host_path, alt_text) VALUES(%s,%s,%s) "
            "ON CONFLICT(url) DO UPDATE SET host_path=EXCLUDED.host_path, "
            "alt_text=EXCLUDED.alt_text, stored_at=NOW()",
            (url, host_path or None, alt_text or None),
        )
    return {"stored": url}


@app.get("/images/list")
def image_list(limit: int = Query(default=50)) -> dict:
    limit = max(1, min(limit, 500))
    with _pg() as pg:
        rows = pg.execute(
            "SELECT url, host_path, alt_text, stored_at FROM images "
            "ORDER BY stored_at DESC LIMIT %s",
            (limit,),
        ).fetchall()
    return {
        "images": [
            {"url": r[0], "host_path": r[1], "alt_text": r[2], "stored_at": str(r[3])}
            for r in rows
        ],
        "count": len(rows),
    }


@app.post("/cache/store")
def cache_store(payload: dict) -> dict:
    key   = str(payload.get("key", "")).strip()
    value = str(payload.get("value", "")).strip()
    if not key:
        raise HTTPException(status_code=422, detail="'key' is required")
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
    return {"key": key, "found": True, "value": row[0], "stored_at": str(row[1])}


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
            {"service": r[0], "level": r[1], "message": r[2],
             "detail": r[3], "logged_at": str(r[4])}
            for r in rows
        ],
        "count": len(rows),
    }

# ===========================================================================
# /memory - SQLite key-value persistent memory  (was aichat-memory:8094)
# ===========================================================================

memory_router = APIRouter(prefix="/memory", tags=["memory"])


class _StoreReq(BaseModel):
    key: str
    value: str
    ttl_seconds: Optional[int] = None


@memory_router.get("/health")
def memory_health() -> dict:
    return {"status": "ok"}


@memory_router.post("/store")
def memory_store(req: _StoreReq) -> dict:
    key = req.key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="key must not be empty")
    now = _now_ts()
    expires_at = (now + req.ttl_seconds) if req.ttl_seconds and req.ttl_seconds > 0 else None
    with _sqlite(MEMORY_DB) as con:
        con.execute(
            "INSERT INTO memory(key, value, ts, expires_at) VALUES(?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, ts=excluded.ts, "
            "expires_at=excluded.expires_at",
            (key, req.value, now, expires_at),
        )
        con.commit()
    return {"stored": key, **({"expires_at": expires_at} if expires_at else {})}


@memory_router.get("/recall")
def memory_recall(
    key: str = Query(default=""),
    pattern: str = Query(default=""),
) -> dict:
    now = _now_ts()
    key = key.strip()
    pattern = pattern.strip()
    with _sqlite(MEMORY_DB) as con:
        if key:
            row = con.execute(
                "SELECT key, value, ts FROM memory "
                "WHERE key=? AND (expires_at IS NULL OR expires_at > ?)",
                (key, now),
            ).fetchone()
            if row is None:
                return {"key": key, "found": False, "entries": []}
            return {
                "key": key, "found": True,
                "entries": [{"key": row["key"], "value": row["value"], "ts": row["ts"]}],
            }
        if pattern:
            rows = con.execute(
                "SELECT key, value, ts FROM memory "
                "WHERE key LIKE ? AND (expires_at IS NULL OR expires_at > ?) "
                "ORDER BY ts DESC LIMIT 50",
                (pattern, now),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT key, value, ts FROM memory "
                "WHERE (expires_at IS NULL OR expires_at > ?) "
                "ORDER BY ts DESC LIMIT 50",
                (now,),
            ).fetchall()
    return {
        "key": key or pattern, "found": bool(rows),
        "entries": [{"key": r["key"], "value": r["value"], "ts": r["ts"]} for r in rows],
    }


@memory_router.delete("/delete")
def memory_delete(key: str = Query(...)) -> dict:
    key = key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="key must not be empty")
    with _sqlite(MEMORY_DB) as con:
        cur = con.execute("DELETE FROM memory WHERE key=?", (key,))
        con.commit()
    return {"deleted": key, "found": cur.rowcount > 0}


@memory_router.delete("/clear")
def memory_clear() -> dict:
    with _sqlite(MEMORY_DB) as con:
        cur = con.execute("DELETE FROM memory")
        con.commit()
    return {"cleared": cur.rowcount}


app.include_router(memory_router)

# ===========================================================================
# /research - RSS feed discovery and ingest  (was aichat-researchbox:8092)
# ===========================================================================

research_router = APIRouter(prefix="/research", tags=["research"])


@research_router.get("/health")
def research_health() -> dict:
    return {"status": "ok"}


@research_router.get("/search-feeds")
def research_search_feeds(topic: str) -> dict:
    return {
        "topic": topic,
        "feeds": [
            f"https://news.google.com/rss/search?q={topic}",
            f"https://hnrss.org/newest?q={topic}",
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
# /graph - SQLite knowledge graph with NetworkX  (was aichat-graph:8098)
# ===========================================================================

graph_router = APIRouter(prefix="/graph", tags=["graph"])


def _graph_node_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"], "labels": json.loads(row["labels"]),
        "properties": json.loads(row["properties"]), "created_at": row["created_at"],
    }


def _graph_edge_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"], "from_id": row["from_id"], "to_id": row["to_id"],
        "type": row["type"], "properties": json.loads(row["properties"]),
        "created_at": row["created_at"],
    }


def _build_nx_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    with _sqlite(GRAPH_DB) as con:
        for row in con.execute("SELECT from_id, to_id, type FROM edges"):
            G.add_edge(row["from_id"], row["to_id"], type=row["type"])
    return G


@graph_router.get("/health")
def graph_health() -> dict:
    with _sqlite(GRAPH_DB) as con:
        n = con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        e = con.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    return {"status": "ok", "nodes": n, "edges": e}


@graph_router.post("/nodes/add")
def graph_add_node(payload: dict) -> dict:
    node_id    = str(payload.get("id", "")).strip()
    labels     = list(payload.get("labels", []))
    properties = dict(payload.get("properties", {}))
    if not node_id:
        raise HTTPException(status_code=422, detail="'id' is required")
    with _sqlite(GRAPH_DB) as con:
        con.execute(
            "INSERT INTO nodes(id, labels, properties, created_at) VALUES(?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET labels=excluded.labels, "
            "properties=excluded.properties",
            (node_id, json.dumps(labels), json.dumps(properties), _now_iso()),
        )
        con.commit()
    return {"added": node_id, "labels": labels, "properties": properties}


@graph_router.post("/edges/add")
def graph_add_edge(payload: dict) -> dict:
    from_id    = str(payload.get("from_id", "")).strip()
    to_id      = str(payload.get("to_id",   "")).strip()
    etype      = str(payload.get("type", "related")).strip() or "related"
    properties = dict(payload.get("properties", {}))
    if not from_id or not to_id:
        raise HTTPException(status_code=422, detail="'from_id' and 'to_id' are required")
    with _sqlite(GRAPH_DB) as con:
        cur = con.execute(
            "INSERT INTO edges(from_id, to_id, type, properties, created_at) "
            "VALUES(?,?,?,?,?)",
            (from_id, to_id, etype, json.dumps(properties), _now_iso()),
        )
        edge_id = cur.lastrowid
        con.commit()
    return {"added": edge_id, "from_id": from_id, "to_id": to_id, "type": etype}


@graph_router.get("/nodes/{node_id}")
def graph_get_node(node_id: str) -> dict:
    with _sqlite(GRAPH_DB) as con:
        row = con.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
        node = _graph_node_row(row)
        out_edges = [_graph_edge_row(r) for r in
                     con.execute("SELECT * FROM edges WHERE from_id=?", (node_id,))]
        in_edges  = [_graph_edge_row(r) for r in
                     con.execute("SELECT * FROM edges WHERE to_id=?", (node_id,))]
    return {"node": node, "out_edges": out_edges, "in_edges": in_edges}


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
    with _sqlite(GRAPH_DB) as con:
        if label:
            rows = con.execute(
                "SELECT * FROM nodes WHERE labels LIKE ? LIMIT ?",
                (f"%{label}%", limit * 3),
            ).fetchall()
        else:
            rows = con.execute("SELECT * FROM nodes LIMIT ?", (limit * 3,)).fetchall()
    results = []
    for row in rows:
        node = _graph_node_row(row)
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
    with _sqlite(GRAPH_DB) as con:
        if not con.execute("SELECT 1 FROM nodes WHERE id=?", (node_id,)).fetchone():
            raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
        con.execute("DELETE FROM edges WHERE from_id=? OR to_id=?", (node_id, node_id))
        con.execute("DELETE FROM nodes WHERE id=?", (node_id,))
        con.commit()
    return {"deleted": node_id}


app.include_router(graph_router)

# ===========================================================================
# /planner - Dependency-aware task queue  (was aichat-planner:8102)
# ===========================================================================

planner_router = APIRouter(prefix="/planner", tags=["planner"])


def _task_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"], "title": row["title"], "description": row["description"],
        "status": row["status"], "priority": row["priority"],
        "depends_on": json.loads(row["depends_on"]), "metadata": json.loads(row["metadata"]),
        "due_at": row["due_at"], "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _task_is_ready(task_id: str, con: sqlite3.Connection) -> bool:
    row = con.execute("SELECT depends_on FROM tasks WHERE id=?", (task_id,)).fetchone()
    if not row:
        return False
    deps = json.loads(row["depends_on"])
    if not deps:
        return True
    for dep_id in deps:
        dep = con.execute("SELECT status FROM tasks WHERE id=?", (dep_id,)).fetchone()
        if not dep or dep["status"] != "done":
            return False
    return True


@planner_router.get("/health")
def planner_health() -> dict:
    with _sqlite(PLANNER_DB) as con:
        total = con.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        ready = con.execute(
            "SELECT COUNT(*) FROM tasks WHERE status='pending' OR status='ready'"
        ).fetchone()[0]
    return {"status": "ok", "tasks_total": total, "tasks_pending_or_ready": ready}


@planner_router.post("/tasks")
def planner_create_task(payload: dict) -> dict:
    title       = str(payload.get("title", "")).strip()
    description = str(payload.get("description", "")).strip()
    depends_on  = list(payload.get("depends_on", []))
    priority    = int(payload.get("priority", 0))
    metadata    = dict(payload.get("metadata", {}))
    due_at      = payload.get("due_at")
    if not title:
        raise HTTPException(status_code=422, detail="'title' is required")
    task_id = uuid.uuid4().hex[:12]
    now = _now_iso()
    with _sqlite(PLANNER_DB) as con:
        for dep in depends_on:
            if not con.execute("SELECT 1 FROM tasks WHERE id=?", (dep,)).fetchone():
                raise HTTPException(status_code=422,
                                    detail=f"depends_on references unknown task: '{dep}'")
        con.execute(
            "INSERT INTO tasks(id, title, description, status, priority, depends_on, "
            "metadata, due_at, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (task_id, title, description, "pending", priority,
             json.dumps(depends_on), json.dumps(metadata), due_at, now, now),
        )
        con.commit()
    return {"id": task_id, "title": title, "status": "pending",
            "depends_on": depends_on, "priority": priority}


@planner_router.get("/tasks/ready")
def planner_list_ready() -> dict:
    with _sqlite(PLANNER_DB) as con:
        rows = con.execute(
            "SELECT * FROM tasks WHERE status IN ('pending','ready') "
            "ORDER BY priority DESC, created_at"
        ).fetchall()
        tasks = [_task_row(r) for r in rows if _task_is_ready(r["id"], con)]
    return {"tasks": tasks, "count": len(tasks)}


@planner_router.get("/tasks")
def planner_list_tasks(
    status: str = Query(default=""),
    limit: int = Query(default=50),
    offset: int = Query(default=0),
) -> dict:
    limit  = max(1, min(limit, 500))
    offset = max(0, offset)
    with _sqlite(PLANNER_DB) as con:
        if status and status in VALID_TASK_STATUSES:
            rows = con.execute(
                "SELECT * FROM tasks WHERE status=? ORDER BY priority DESC, created_at "
                "LIMIT ? OFFSET ?", (status, limit, offset),
            ).fetchall()
            total = con.execute(
                "SELECT COUNT(*) FROM tasks WHERE status=?", (status,)
            ).fetchone()[0]
        else:
            rows = con.execute(
                "SELECT * FROM tasks ORDER BY priority DESC, created_at LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            total = con.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    return {"tasks": [_task_row(r) for r in rows], "total": total,
            "limit": limit, "offset": offset}


@planner_router.get("/tasks/{task_id}")
def planner_get_task(task_id: str) -> dict:
    with _sqlite(PLANNER_DB) as con:
        row = con.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return _task_row(row)


@planner_router.patch("/tasks/{task_id}")
def planner_update_task(task_id: str, payload: dict) -> dict:
    with _sqlite(PLANNER_DB) as con:
        row = con.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        status = row["status"]; meta = row["metadata"]
        priority = row["priority"]; desc = row["description"]
        changed = False
        if "status" in payload:
            s = payload["status"]
            if s not in VALID_TASK_STATUSES:
                raise HTTPException(status_code=422, detail=f"Invalid status: {s}")
            status = s; changed = True
        if "metadata" in payload:
            meta = json.dumps(dict(payload["metadata"])); changed = True
        if "priority" in payload:
            priority = int(payload["priority"]); changed = True
        if "description" in payload:
            desc = str(payload["description"]); changed = True
        if changed:
            con.execute(
                "UPDATE tasks SET status=?, metadata=?, priority=?, description=?, "
                "updated_at=? WHERE id=?",
                (status, meta, priority, desc, _now_iso(), task_id),
            )
            con.commit()
    return planner_get_task(task_id)


@planner_router.delete("/tasks/{task_id}")
def planner_delete_task(task_id: str) -> dict:
    with _sqlite(PLANNER_DB) as con:
        if not con.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone():
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        con.execute("DELETE FROM tasks WHERE id=?", (task_id,))
        con.commit()
    return {"deleted": task_id}


@planner_router.post("/tasks/{task_id}/complete")
def planner_complete_task(task_id: str) -> dict:
    with _sqlite(PLANNER_DB) as con:
        if not con.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone():
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        con.execute(
            "UPDATE tasks SET status='done', updated_at=? WHERE id=?",
            (_now_iso(), task_id),
        )
        con.commit()
    return {"id": task_id, "status": "done"}


@planner_router.post("/tasks/{task_id}/fail")
def planner_fail_task(task_id: str, payload: dict | None = None) -> dict:
    detail = str((payload or {}).get("detail", "")).strip()
    with _sqlite(PLANNER_DB) as con:
        row = con.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        meta = json.loads(row["metadata"])
        if detail:
            meta["fail_reason"] = detail
        con.execute(
            "UPDATE tasks SET status='failed', metadata=?, updated_at=? WHERE id=?",
            (json.dumps(meta), _now_iso(), task_id),
        )
        con.commit()
    return {"id": task_id, "status": "failed", "detail": detail}


@planner_router.get("/graph")
def planner_dependency_graph() -> dict:
    with _sqlite(PLANNER_DB) as con:
        rows = con.execute("SELECT id, title, status, depends_on FROM tasks").fetchall()
    nodes = [{"id": r["id"], "title": r["title"], "status": r["status"]} for r in rows]
    edges = []
    for row in rows:
        for dep in json.loads(row["depends_on"]):
            edges.append({"from": dep, "to": row["id"]})
    return {"nodes": nodes, "edges": edges}


app.include_router(planner_router)

# ===========================================================================
# /jobs - Durable async job system (NEW)
# ===========================================================================

jobs_router = APIRouter(prefix="/jobs", tags=["jobs"])


def _job_row(row: sqlite3.Row) -> dict:
    return {
        "id":            row["id"],
        "tool_name":     row["tool_name"],
        "args":          json.loads(row["args"]),
        "status":        row["status"],
        "priority":      row["priority"],
        "submitted_at":  row["submitted_at"],
        "started_at":    row["started_at"],
        "finished_at":   row["finished_at"],
        "progress":      row["progress"],
        "input_summary": row["input_summary"],
        "result":        row["result"],
        "error":         row["error"],
        "logs":          row["logs"],
        "retry_count":   row["retry_count"],
        "max_retries":   row["max_retries"],
        "timeout_s":     row["timeout_s"],
    }


@jobs_router.get("/health")
def jobs_health() -> dict:
    with _sqlite(JOBS_DB) as con:
        total   = con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        queued  = con.execute(
            "SELECT COUNT(*) FROM jobs WHERE status='queued'"
        ).fetchone()[0]
        running = con.execute(
            "SELECT COUNT(*) FROM jobs WHERE status='running'"
        ).fetchone()[0]
    return {"status": "ok", "total": total, "queued": queued, "running": running}


@jobs_router.post("")
def jobs_create(payload: dict) -> dict:
    """Create a new async job. Execution is handled by the aichat-mcp background worker."""
    tool_name     = str(payload.get("tool_name", "")).strip()
    args          = dict(payload.get("args", {}))
    priority      = int(payload.get("priority", 0))
    timeout_s     = float(payload.get("timeout_s", 300.0))
    max_retries   = int(payload.get("max_retries", 0))
    input_summary = str(payload.get("input_summary", json.dumps(args)[:200]))
    if not tool_name:
        raise HTTPException(status_code=422, detail="'tool_name' is required")
    job_id = uuid.uuid4().hex[:16]
    now = _now_iso()
    with _sqlite(JOBS_DB) as con:
        con.execute(
            "INSERT INTO jobs(id, tool_name, args, status, priority, submitted_at, "
            "progress, input_summary, logs, retry_count, max_retries, timeout_s) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (job_id, tool_name, json.dumps(args), JOB_QUEUED, priority, now,
             0, input_summary, "", 0, max_retries, timeout_s),
        )
        con.commit()
    log.info("job_create %s tool=%s", job_id, tool_name)
    return {"id": job_id, "tool_name": tool_name, "status": JOB_QUEUED, "submitted_at": now}


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
        conditions.append("status=?"); params.append(status)
    if tool_name:
        conditions.append("tool_name=?"); params.append(tool_name)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    count_params = list(params)
    params += [limit, offset]
    sql_jobs = f"SELECT * FROM jobs {where} ORDER BY priority DESC, submitted_at DESC LIMIT ? OFFSET ?"  # nosec B608
    sql_count = f"SELECT COUNT(*) FROM jobs {where}"  # nosec B608
    with _sqlite(JOBS_DB) as con:
        rows = con.execute(sql_jobs, params).fetchall()
        total = con.execute(sql_count, count_params).fetchone()[0]
    return {"jobs": [_job_row(r) for r in rows], "total": total,
            "limit": limit, "offset": offset}


@jobs_router.get("/{job_id}")
def jobs_get(job_id: str) -> dict:
    with _sqlite(JOBS_DB) as con:
        row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return _job_row(row)


@jobs_router.patch("/{job_id}")
def jobs_update(job_id: str, payload: dict) -> dict:
    """Update job state -- called by the MCP worker as execution progresses."""
    with _sqlite(JOBS_DB) as con:
        row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
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
                fields.append(f"{col}=?")
                params.append(val)
        if "logs" in payload:
            existing = row["logs"] or ""
            new_logs = existing + payload["logs"]
            fields.append("logs=?")
            params.append(new_logs[-50000:])
        if not fields:
            return _job_row(row)
        params.append(job_id)
        con.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id=?", params)  # nosec B608
        con.commit()
    return jobs_get(job_id)


@jobs_router.post("/{job_id}/cancel")
def jobs_cancel(job_id: str) -> dict:
    with _sqlite(JOBS_DB) as con:
        row = con.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        if row["status"] in (JOB_SUCCEEDED, JOB_FAILED, JOB_CANCELLED):
            return {"id": job_id, "status": row["status"], "note": "already terminal"}
        con.execute(
            "UPDATE jobs SET status=?, finished_at=? WHERE id=?",
            (JOB_CANCELLED, _now_iso(), job_id),
        )
        con.commit()
    return {"id": job_id, "status": JOB_CANCELLED}


@jobs_router.post("/batch")
def jobs_batch_create(payload: dict) -> dict:
    """Submit multiple jobs atomically. Returns list of job_ids."""
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
