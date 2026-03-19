"""Forward-only PostgreSQL migration runner.

Tracks applied migrations in a ``_migrations`` table.  Called at container
startup (via the app lifespan) before FastAPI begins serving traffic.

Usage::

    from migrate import run_migrations
    run_migrations(pg_dsn, Path("migrations"))
"""
from __future__ import annotations

import logging
from pathlib import Path

import psycopg

log = logging.getLogger("aichat-data.migrate")


def run_migrations(dsn: str, migrations_dir: Path) -> int:
    """Apply all pending migrations and return the count of newly applied ones."""
    applied_count = 0
    with psycopg.connect(dsn) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                version  INTEGER PRIMARY KEY,
                name     TEXT NOT NULL,
                applied  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        conn.commit()

        rows = conn.execute("SELECT version FROM _migrations").fetchall()
        already_applied: set[int] = {r[0] for r in rows}

        for sql_file in sorted(migrations_dir.glob("*.sql")):
            try:
                version = int(sql_file.name.split("_", 1)[0])
            except (ValueError, IndexError):
                log.warning("Skipping non-numbered migration file: %s", sql_file.name)
                continue

            if version in already_applied:
                continue

            log.info("Applying migration %03d: %s", version, sql_file.name)
            sql = sql_file.read_text(encoding="utf-8")
            try:
                conn.execute(sql)
                conn.execute(
                    "INSERT INTO _migrations (version, name) VALUES (%s, %s)",
                    (version, sql_file.name),
                )
                conn.commit()
                applied_count += 1
                log.info("Migration %03d applied successfully", version)
            except Exception:
                conn.rollback()
                log.exception("Migration %03d FAILED — rolling back", version)
                raise

    if applied_count:
        log.info("Applied %d new migration(s)", applied_count)
    else:
        log.info("Database schema is up to date")
    return applied_count
