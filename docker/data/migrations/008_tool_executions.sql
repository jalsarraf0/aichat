-- 008_tool_executions.sql
-- Tool execution log for observability.

CREATE TABLE IF NOT EXISTS tool_executions (
    id             SERIAL PRIMARY KEY,
    tool_name      TEXT NOT NULL,
    args_summary   TEXT NOT NULL DEFAULT '',
    result_summary TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL DEFAULT 'ok',
    duration_ms    INTEGER NOT NULL DEFAULT 0,
    ts             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tool_exec_ts
    ON tool_executions(ts DESC);
CREATE INDEX IF NOT EXISTS idx_tool_exec_tool
    ON tool_executions(tool_name, ts DESC);
