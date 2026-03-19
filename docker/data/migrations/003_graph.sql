-- 003_graph.sql
-- Replaces graph.db SQLite database (nodes + edges for NetworkX graph).

CREATE TABLE IF NOT EXISTS graph_nodes (
    id         TEXT PRIMARY KEY,
    label      TEXT NOT NULL DEFAULT '',
    node_type  TEXT NOT NULL DEFAULT '',
    data       JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS graph_edges (
    id         SERIAL PRIMARY KEY,
    from_id    TEXT NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
    to_id      TEXT NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
    label      TEXT NOT NULL DEFAULT 'related',
    weight     REAL NOT NULL DEFAULT 1.0,
    data       JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_graph_edges_from ON graph_edges(from_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_to   ON graph_edges(to_id);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_type  ON graph_nodes(node_type) WHERE node_type != '';
