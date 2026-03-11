"""Unit test configuration.

The vision-router's preprocessing module lives in its own ``app`` package.
This conftest makes a ``router_app`` fixture that temporarily adds the
router root to sys.path so that ``from app.preprocessing import ...`` works
without conflicting with the mcp-server ``app`` package.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_repo_root = Path(__file__).parent.parent.parent.parent  # vision/tests/unit/ → aichat/
_router_root = str(_repo_root / "vision" / "services" / "vision-router")
_mcp_root = str(_repo_root / "vision" / "mcp-server")
