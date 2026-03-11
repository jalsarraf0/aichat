"""Root-level pytest configuration for the vision stack.

Sets up sys.path so that the mcp-server and vision-router packages are
importable. Because both have an ``app/`` package, we default to exposing
mcp-server first (where most tests live). Tests that need the router's
``app.preprocessing`` module use the ``load_router_module`` fixture.
"""
from __future__ import annotations

import sys
from pathlib import Path

_vision_root = Path(__file__).parent
_repo_root = _vision_root.parent

MCP_SERVER_ROOT = str(_repo_root / "vision" / "mcp-server")
ROUTER_ROOT = str(_repo_root / "vision" / "services" / "vision-router")

# Insert mcp-server first so ``from app.config import Settings`` etc. resolve
# to the MCP server package by default.
if MCP_SERVER_ROOT not in sys.path:
    sys.path.insert(0, MCP_SERVER_ROOT)
