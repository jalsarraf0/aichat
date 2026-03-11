"""aichat-graph: SQLite-backed knowledge graph with NetworkX path queries.

Endpoints:
  POST /nodes/add             — add or update a node
  POST /edges/add             — add an edge between nodes
  GET  /nodes/{id}            — get node with its edges
  GET  /nodes/{id}/neighbors  — get direct neighbors
  POST /path                  — BFS shortest path between two nodes
  POST /search                — search nodes by label and/or property
  DELETE /nodes/{id}          — delete node and all connected edges
  GET  /health                — service health + counts
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("aichat-graph")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH      = Path(os.environ.get("GRAPH_DB", "/data/graph.db"))
DB_API       = os.environ.get("DATABASE_URL", "http://aichat-database:8091")
_SERVICE     = "aichat-graph"

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    return c


def _create_tables() -> None:
    with _conn() as c:
        c.executescript("""
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


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------

async def _report_error(message: str, detail: str | None = None) -> None:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"{DB_API}/errors/log",
                json={"service": _SERVICE, "level": "ERROR",
                      "message": message, "detail": detail},
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _create_tables()
    log.info("aichat-graph ready at %s", DB_PATH)
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(Exception)
async def _global_exc(request: Request, exc: Exception) -> JSONResponse:
    msg = str(exc)
    log.error("Unhandled error [%s %s]: %s", request.method, request.url.path, msg, exc_info=True)
    asyncio.create_task(_report_error(msg, f"{request.method} {request.url.path}"))
    return JSONResponse(status_code=500, content={"error": msg})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _node_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id":         row["id"],
        "labels":     json.loads(row["labels"]),
        "properties": json.loads(row["properties"]),
        "created_at": row["created_at"],
    }


def _edge_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id":         row["id"],
        "from_id":    row["from_id"],
        "to_id":      row["to_id"],
        "type":       row["type"],
        "properties": json.loads(row["properties"]),
        "created_at": row["created_at"],
    }


def _build_graph() -> nx.DiGraph:
    """Load all edges into a NetworkX DiGraph for path queries."""
    G = nx.DiGraph()
    with _conn() as c:
        for row in c.execute("SELECT from_id, to_id, type FROM edges"):
            G.add_edge(row["from_id"], row["to_id"], type=row["type"])
    return G


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    with _conn() as c:
        n = c.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        e = c.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    return {"status": "ok", "nodes": n, "edges": e}


@app.post("/nodes/add")
def add_node(payload: dict) -> dict:
    node_id    = str(payload.get("id", "")).strip()
    labels     = list(payload.get("labels", []))
    properties = dict(payload.get("properties", {}))
    if not node_id:
        raise HTTPException(status_code=422, detail="'id' is required")
    with _conn() as c:
        c.execute(
            """INSERT INTO nodes(id, labels, properties, created_at)
               VALUES(?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 labels=excluded.labels,
                 properties=excluded.properties""",
            (node_id, json.dumps(labels), json.dumps(properties), _now()),
        )
    log.info("add_node %s labels=%s", node_id, labels)
    return {"added": node_id, "labels": labels, "properties": properties}


@app.post("/edges/add")
def add_edge(payload: dict) -> dict:
    from_id    = str(payload.get("from_id", "")).strip()
    to_id      = str(payload.get("to_id",   "")).strip()
    etype      = str(payload.get("type",    "related")).strip() or "related"
    properties = dict(payload.get("properties", {}))
    if not from_id or not to_id:
        raise HTTPException(status_code=422, detail="'from_id' and 'to_id' are required")
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO edges(from_id, to_id, type, properties, created_at) VALUES(?,?,?,?,?)",
            (from_id, to_id, etype, json.dumps(properties), _now()),
        )
        edge_id = cur.lastrowid
    log.info("add_edge %s -[%s]-> %s", from_id, etype, to_id)
    return {"added": edge_id, "from_id": from_id, "to_id": to_id, "type": etype}


@app.get("/nodes/{node_id}")
def get_node(node_id: str) -> dict:
    with _conn() as c:
        row = c.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
        node = _node_row(row)
        out_edges = [_edge_row(r) for r in
                     c.execute("SELECT * FROM edges WHERE from_id=?", (node_id,))]
        in_edges  = [_edge_row(r) for r in
                     c.execute("SELECT * FROM edges WHERE to_id=?",   (node_id,))]
    return {"node": node, "out_edges": out_edges, "in_edges": in_edges}


@app.get("/nodes/{node_id}/neighbors")
def get_neighbors(node_id: str) -> dict:
    with _conn() as c:
        if not c.execute("SELECT 1 FROM nodes WHERE id=?", (node_id,)).fetchone():
            raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
        # outgoing neighbors
        rows = c.execute("""
            SELECT n.*, e.type as edge_type FROM nodes n
            JOIN edges e ON n.id = e.to_id
            WHERE e.from_id = ?
        """, (node_id,)).fetchall()
        neighbors = [
            {**_node_row(r), "edge_type": r["edge_type"]} for r in rows
        ]
    return {"node_id": node_id, "neighbors": neighbors, "count": len(neighbors)}


@app.post("/path")
def find_path(payload: dict) -> dict:
    from_id = str(payload.get("from_id", "")).strip()
    to_id   = str(payload.get("to_id",   "")).strip()
    if not from_id or not to_id:
        raise HTTPException(status_code=422, detail="'from_id' and 'to_id' are required")
    G = _build_graph()
    if not G.has_node(from_id):
        raise HTTPException(status_code=404, detail=f"Node '{from_id}' not found in graph")
    if not G.has_node(to_id):
        raise HTTPException(status_code=404, detail=f"Node '{to_id}' not found in graph")
    try:
        path = nx.shortest_path(G, from_id, to_id)
    except nx.NetworkXNoPath:
        return {"from_id": from_id, "to_id": to_id, "path": None, "length": -1,
                "message": "No path found"}
    return {"from_id": from_id, "to_id": to_id, "path": path, "length": len(path) - 1}


@app.post("/search")
def search_nodes(payload: dict) -> dict:
    label      = str(payload.get("label", "")).strip()
    props      = dict(payload.get("properties", {}))
    limit      = max(1, min(int(payload.get("limit", 50)), 500))
    with _conn() as c:
        if label:
            rows = c.execute(
                "SELECT * FROM nodes WHERE labels LIKE ? LIMIT ?",
                (f"%{label}%", limit * 3),  # over-fetch for prop filtering
            ).fetchall()
        else:
            rows = c.execute("SELECT * FROM nodes LIMIT ?", (limit * 3,)).fetchall()
    results = []
    for row in rows:
        node = _node_row(row)
        if props:
            node_props = node["properties"]
            if not all(str(node_props.get(k)) == str(v) for k, v in props.items()):
                continue
        results.append(node)
        if len(results) >= limit:
            break
    return {"results": results, "count": len(results), "label": label}


@app.delete("/nodes/{node_id}")
def delete_node(node_id: str) -> dict:
    with _conn() as c:
        if not c.execute("SELECT 1 FROM nodes WHERE id=?", (node_id,)).fetchone():
            raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
        c.execute("DELETE FROM edges WHERE from_id=? OR to_id=?", (node_id, node_id))
        c.execute("DELETE FROM nodes WHERE id=?", (node_id,))
    log.info("delete_node %s", node_id)
    return {"deleted": node_id}
