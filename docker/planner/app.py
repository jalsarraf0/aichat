"""aichat-planner: SQLite-backed async task queue with dependency graph.

Status flow: pending → ready → in_progress → done | failed | cancelled
A task is 'ready' when all its depends_on tasks have status 'done'.

Endpoints:
  POST   /tasks                   — create task
  GET    /tasks/{id}              — get task
  PATCH  /tasks/{id}              — update task
  DELETE /tasks/{id}              — delete task
  GET    /tasks                   — list tasks (?status=, ?limit=, ?offset=)
  GET    /tasks/ready             — tasks ready to run (all deps done)
  POST   /tasks/{id}/complete     — mark task done
  POST   /tasks/{id}/fail         — mark task failed
  GET    /graph                   — full dependency graph
  GET    /health
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("aichat-planner")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH  = Path(os.environ.get("PLANNER_DB", "/data/planner.db"))
DB_API   = os.environ.get("DATABASE_URL", "http://aichat-database:8091")
_SERVICE = "aichat-planner"

VALID_STATUSES = {"pending", "ready", "in_progress", "done", "failed", "cancelled"}

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def _create_tables() -> None:
    with _conn() as c:
        c.executescript("""
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
    log.info("aichat-planner ready at %s", DB_PATH)
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(Exception)
async def _global_exc(request: Request, exc: Exception) -> JSONResponse:
    msg = str(exc)
    log.error("Unhandled [%s %s]: %s", request.method, request.url.path, msg, exc_info=True)
    asyncio.create_task(_report_error(msg, f"{request.method} {request.url.path}"))
    return JSONResponse(status_code=500, content={"error": msg})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_row(row: sqlite3.Row) -> dict:
    return {
        "id":          row["id"],
        "title":       row["title"],
        "description": row["description"],
        "status":      row["status"],
        "priority":    row["priority"],
        "depends_on":  json.loads(row["depends_on"]),
        "metadata":    json.loads(row["metadata"]),
        "due_at":      row["due_at"],
        "created_at":  row["created_at"],
        "updated_at":  row["updated_at"],
    }


def _is_ready(task_id: str, c: sqlite3.Connection) -> bool:
    """Check if all depends_on tasks are done."""
    row = c.execute("SELECT depends_on FROM tasks WHERE id=?", (task_id,)).fetchone()
    if not row:
        return False
    deps = json.loads(row["depends_on"])
    if not deps:
        return True
    for dep_id in deps:
        dep = c.execute("SELECT status FROM tasks WHERE id=?", (dep_id,)).fetchone()
        if not dep or dep["status"] != "done":
            return False
    return True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        ready = c.execute(
            "SELECT COUNT(*) FROM tasks WHERE status='pending' OR status='ready'"
        ).fetchone()[0]
    return {"status": "ok", "tasks_total": total, "tasks_pending_or_ready": ready}


@app.post("/tasks")
def create_task(payload: dict) -> dict:
    title       = str(payload.get("title", "")).strip()
    description = str(payload.get("description", "")).strip()
    depends_on  = list(payload.get("depends_on", []))
    priority    = int(payload.get("priority", 0))
    metadata    = dict(payload.get("metadata", {}))
    due_at      = payload.get("due_at")

    if not title:
        raise HTTPException(status_code=422, detail="'title' is required")

    task_id = uuid.uuid4().hex[:12]
    now     = _now()

    with _conn() as c:
        # Validate depends_on IDs exist
        for dep in depends_on:
            if not c.execute("SELECT 1 FROM tasks WHERE id=?", (dep,)).fetchone():
                raise HTTPException(status_code=422,
                                    detail=f"depends_on references unknown task id: '{dep}'")
        c.execute(
            """INSERT INTO tasks(id, title, description, status, priority,
               depends_on, metadata, due_at, created_at, updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (task_id, title, description, "pending", priority,
             json.dumps(depends_on), json.dumps(metadata), due_at, now, now),
        )

    log.info("create_task %s %r", task_id, title)
    return {"id": task_id, "title": title, "status": "pending",
            "depends_on": depends_on, "priority": priority}


@app.get("/tasks/ready")
def list_ready() -> dict:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM tasks WHERE status IN ('pending','ready') ORDER BY priority DESC, created_at"
        ).fetchall()
        tasks = []
        for row in rows:
            if _is_ready(row["id"], c):
                tasks.append(_task_row(row))
    return {"tasks": tasks, "count": len(tasks)}


@app.get("/tasks")
def list_tasks(status: str = "", limit: int = 50, offset: int = 0) -> dict:
    limit  = max(1, min(limit, 500))
    offset = max(0, offset)
    with _conn() as c:
        if status and status in VALID_STATUSES:
            rows = c.execute(
                "SELECT * FROM tasks WHERE status=? ORDER BY priority DESC, created_at LIMIT ? OFFSET ?",
                (status, limit, offset),
            ).fetchall()
            total = c.execute("SELECT COUNT(*) FROM tasks WHERE status=?", (status,)).fetchone()[0]
        else:
            rows = c.execute(
                "SELECT * FROM tasks ORDER BY priority DESC, created_at LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            total = c.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    return {"tasks": [_task_row(r) for r in rows], "total": total, "limit": limit, "offset": offset}


@app.get("/tasks/{task_id}")
def get_task(task_id: str) -> dict:
    with _conn() as c:
        row = c.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return _task_row(row)


@app.patch("/tasks/{task_id}")
def update_task(task_id: str, payload: dict) -> dict:
    with _conn() as c:
        row = c.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        updates = {}
        if "status" in payload:
            s = payload["status"]
            if s not in VALID_STATUSES:
                raise HTTPException(status_code=422, detail=f"Invalid status: {s}")
            updates["status"] = s
        if "metadata" in payload:
            updates["metadata"] = json.dumps(dict(payload["metadata"]))
        if "priority" in payload:
            updates["priority"] = int(payload["priority"])
        if "description" in payload:
            updates["description"] = str(payload["description"])
        if not updates:
            return _task_row(row)
        updates["updated_at"] = _now()
        set_clause = ", ".join(f"{k}=?" for k in updates)
        c.execute(f"UPDATE tasks SET {set_clause} WHERE id=?",
                  (*updates.values(), task_id))
    return get_task(task_id)


@app.delete("/tasks/{task_id}")
def delete_task(task_id: str) -> dict:
    with _conn() as c:
        if not c.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone():
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        c.execute("DELETE FROM tasks WHERE id=?", (task_id,))
    log.info("delete_task %s", task_id)
    return {"deleted": task_id}


@app.post("/tasks/{task_id}/complete")
def complete_task(task_id: str) -> dict:
    with _conn() as c:
        if not c.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone():
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        c.execute(
            "UPDATE tasks SET status='done', updated_at=? WHERE id=?",
            (_now(), task_id),
        )
    log.info("complete_task %s", task_id)
    return {"id": task_id, "status": "done"}


@app.post("/tasks/{task_id}/fail")
def fail_task(task_id: str, payload: dict | None = None) -> dict:
    detail = str((payload or {}).get("detail", "")).strip()
    with _conn() as c:
        row = c.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        meta = json.loads(row["metadata"])
        if detail:
            meta["fail_reason"] = detail
        c.execute(
            "UPDATE tasks SET status='failed', metadata=?, updated_at=? WHERE id=?",
            (json.dumps(meta), _now(), task_id),
        )
    log.info("fail_task %s detail=%r", task_id, detail)
    return {"id": task_id, "status": "failed", "detail": detail}


@app.get("/graph")
def dependency_graph() -> dict:
    with _conn() as c:
        rows = c.execute("SELECT id, title, status, depends_on FROM tasks").fetchall()
    nodes = [{"id": r["id"], "title": r["title"], "status": r["status"]} for r in rows]
    edges = []
    for row in rows:
        deps = json.loads(row["depends_on"])
        for dep in deps:
            edges.append({"from": dep, "to": row["id"]})
    return {"nodes": nodes, "edges": edges}
