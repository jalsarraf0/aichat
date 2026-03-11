"""Pytest configuration and shared fixtures for vision stack tests."""
from __future__ import annotations

import sys
from pathlib import Path

# Add vision/mcp-server and vision/services/vision-router to sys.path so that
# tests can import from those packages without installing them.
_repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_repo_root / "vision" / "mcp-server"))
sys.path.insert(0, str(_repo_root / "vision" / "services" / "vision-router"))

import pytest


@pytest.fixture
def test_jpeg_bytes():
    from vision.tests.fixtures.images import make_test_jpeg
    return make_test_jpeg()


@pytest.fixture
def test_jpeg_b64():
    from vision.tests.fixtures.images import make_test_b64_jpeg
    return make_test_b64_jpeg()


@pytest.fixture
def test_png_bytes():
    from vision.tests.fixtures.images import make_test_png
    return make_test_png()


@pytest.fixture
def test_png_b64():
    from vision.tests.fixtures.images import make_test_b64_png
    return make_test_b64_png()
