"""One-time data transfer from SQLite databases to PostgreSQL.

Called once at startup after migrations have been applied.  If a SQLite file
exists and contains data, its rows are inserted into the corresponding
PostgreSQL tables using ``INSERT ... ON CONFLICT DO NOTHING`` (idempotent).

After a successful transfer the SQLite file is renamed to ``*.migrated``
so the transfer does not repeat on subsequent startups.  The ``.migrated``
files are kept as a safety net for manual rollback if needed.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import psycopg

log = logging.getLogger("aichat-data.sqlite-to-pg")


def _sqlite_rows(db_path: Path, table: str) -> list[dict]:
    """Read all rows from a SQLite table as dicts."""
    if not db_path.is_file():
        return []
    try:
        con = sqlite3.connect(str(db_path), timeout=5)
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(f"SELECT * FROM {table}")]  # noqa: S608
        con.close()
        return rows
    except Exception as exc:
        log.warning("Could not read %s from %s: %s", table, db_path, exc)
        return []


def _rename_migrated(db_path: Path) -> None:
    """Rename SQLite file to .migrated."""
    if db_path.is_file():
        dest = db_path.with_suffix(".migrated")
        db_path.rename(dest)
        log.info("Renamed %s → %s", db_path, dest)


def transfer_sqlite_to_pg(dsn: str, data_dir: Path) -> None:
    """Transfer data from SQLite databases to PostgreSQL."""
    memory_db  = data_dir / "memory.db"
    graph_db   = data_dir / "graph.db"
    planner_db = data_dir / "planner.db"
    jobs_db    = data_dir / "jobs.db"

    any_work = any(p.is_file() for p in [memory_db, graph_db, planner_db, jobs_db])
    if not any_work:
        log.info("No SQLite databases found — nothing to transfer")
        return

    log.info("Starting SQLite → PostgreSQL data transfer")

    with psycopg.connect(dsn) as conn:
        # Memory
        rows = _sqlite_rows(memory_db, "memory")
        if rows:
            log.info("Transferring %d memory entries", len(rows))
            for r in rows:
                try:
                    conn.execute(
                        "INSERT INTO memory (key, value, ts) VALUES (%s, %s, TO_TIMESTAMP(%s)) "
                        "ON CONFLICT (key) DO NOTHING",
                        (r["key"], r["value"], r.get("ts", 0)),
                    )
                except Exception as exc:
                    log.warning("memory row skip key=%s: %s", r.get("key"), exc)
            conn.commit()
            _rename_migrated(memory_db)

        # Graph nodes
        nodes = _sqlite_rows(graph_db, "nodes")
        if nodes:
            log.info("Transferring %d graph nodes", len(nodes))
            for r in nodes:
                data_json = r.get("data", "{}")
                if isinstance(data_json, str):
                    try:
                        json.loads(data_json)
                    except Exception:
                        data_json = "{}"
                conn.execute(
                    "INSERT INTO graph_nodes (id, label, node_type, data) "
                    "VALUES (%s, %s, %s, %s::jsonb) ON CONFLICT (id) DO NOTHING",
                    (r["id"], r.get("label", ""), r.get("type", ""), data_json),
                )
            conn.commit()

        # Graph edges
        edges = _sqlite_rows(graph_db, "edges")
        if edges:
            log.info("Transferring %d graph edges", len(edges))
            for r in edges:
                data_json = r.get("data", "{}")
                if isinstance(data_json, str):
                    try:
                        json.loads(data_json)
                    except Exception:
                        data_json = "{}"
                try:
                    conn.execute(
                        "INSERT INTO graph_edges (from_id, to_id, label, weight, data) "
                        "VALUES (%s, %s, %s, %s, %s::jsonb)",
                        (r["from_id"], r["to_id"], r.get("label", "related"),
                         r.get("weight", 1.0), data_json),
                    )
                except Exception as exc:
                    log.warning("edge skip %s→%s: %s", r.get("from_id"), r.get("to_id"), exc)
            conn.commit()
            _rename_migrated(graph_db)

        # Planner tasks
        tasks = _sqlite_rows(planner_db, "tasks")
        if tasks:
            log.info("Transferring %d planner tasks", len(tasks))
            for r in tasks:
                depends = r.get("depends_on", "[]")
                if isinstance(depends, str):
                    try:
                        json.loads(depends)
                    except Exception:
                        depends = "[]"
                conn.execute(
                    "INSERT INTO tasks (id, title, description, status, priority, depends_on, result) "
                    "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s) ON CONFLICT (id) DO NOTHING",
                    (r["id"], r.get("title", ""), r.get("description", ""),
                     r.get("status", "pending"), r.get("priority", 0),
                     depends, r.get("result")),
                )
            conn.commit()
            _rename_migrated(planner_db)

        # Jobs
        job_rows = _sqlite_rows(jobs_db, "jobs")
        if job_rows:
            log.info("Transferring %d jobs", len(job_rows))
            for r in job_rows:
                params = r.get("params", "{}")
                if isinstance(params, str):
                    try:
                        json.loads(params)
                    except Exception:
                        params = "{}"
                try:
                    conn.execute(
                        "INSERT INTO jobs (id, tool_name, args, status, priority, "
                        "input_summary, result, error) "
                        "VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s, %s) "
                        "ON CONFLICT (id) DO NOTHING",
                        (r["id"], r.get("tool", r.get("tool_name", "")), params,
                         r.get("status", "queued"), r.get("priority", 0),
                         r.get("input_summary", ""), r.get("result"), r.get("error")),
                    )
                except Exception as exc:
                    log.warning("job skip id=%s: %s", r.get("id"), exc)
            conn.commit()
            _rename_migrated(jobs_db)

    log.info("SQLite → PostgreSQL transfer complete")
