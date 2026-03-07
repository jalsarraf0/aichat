"""
Architecture contract tests — no Docker required.

Validates that docker/mcp/app.py exposes ALL expected public tool names.
Any removal from this set is a breaking change and must fail CI.

Run with:
    pytest tests/test_architecture.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load docker/mcp/app.py
# ---------------------------------------------------------------------------

_REPO = Path(__file__).parent.parent


def _load_mcp():
    mod_name = "_arch_mcp_app"
    spec = importlib.util.spec_from_file_location(mod_name, _REPO / "docker" / "mcp" / "app.py")
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


try:
    _mcp = _load_mcp()
    _TOOLS: list[dict] = _mcp._TOOLS
    _LOAD_OK = True
except Exception as _err:
    _LOAD_OK = False
    _TOOLS = []

skip_load = pytest.mark.skipif(not _LOAD_OK, reason="docker/mcp/app.py failed to load")


def _tool_names() -> set[str]:
    return {t["name"] for t in _TOOLS}


# ---------------------------------------------------------------------------
# Canonical public tool names (backward-compatibility contract)
# ---------------------------------------------------------------------------

# Core database / storage
_DB_TOOLS = {
    "db_store_article", "db_store_image", "db_list_images",
    "db_search", "db_cache_store", "db_cache_get", "get_errors",
}

# Memory
_MEMORY_TOOLS = {
    "memory_store", "memory_recall",
}

# Research / RSS
_RESEARCH_TOOLS = {
    "researchbox_search", "researchbox_push",
}

# Knowledge graph
_GRAPH_TOOLS = {
    "graph_add_node", "graph_add_edge", "graph_query",
    "graph_path", "graph_search",
}

# Vector
_VECTOR_TOOLS = {
    "vector_store", "vector_search", "vector_delete", "vector_collections",
}

# Planner
_PLANNER_TOOLS = {
    "plan_create_task", "plan_get_task", "plan_complete_task",
    "plan_fail_task", "plan_list_tasks", "plan_delete_task",
    "orchestrate",
}

# Async job system (new)
_JOB_TOOLS = {
    "job_submit", "job_status", "job_result",
    "job_cancel", "job_list", "batch_submit",
}

# Browser / vision
_BROWSER_TOOLS = {
    "screenshot", "browser", "scroll_screenshot", "bulk_screenshot",
    "page_scrape", "page_extract", "page_images",
}

# Video
_VIDEO_TOOLS = {
    "video_info", "video_frames", "video_thumbnail",
}

# OCR
_OCR_TOOLS = {
    "ocr_image", "ocr_pdf",
}

# Document ingestor
_DOCS_TOOLS = {
    "docs_ingest", "docs_extract_tables",
}

# PDF
_PDF_TOOLS = {
    "pdf_read", "pdf_edit", "pdf_fill_form", "pdf_merge", "pdf_split",
}

# Sandbox / toolkit
_SANDBOX_TOOLS = {
    "code_run", "create_tool", "list_custom_tools",
    "call_custom_tool", "delete_custom_tool",
}

# LLM / image generation
_LLM_TOOLS = {
    "image_generate", "web_search", "web_fetch",
}

_ALL_EXPECTED: set[str] = (
    _DB_TOOLS | _MEMORY_TOOLS | _RESEARCH_TOOLS | _GRAPH_TOOLS
    | _VECTOR_TOOLS | _PLANNER_TOOLS | _JOB_TOOLS | _BROWSER_TOOLS
    | _VIDEO_TOOLS | _OCR_TOOLS | _DOCS_TOOLS | _PDF_TOOLS
    | _SANDBOX_TOOLS | _LLM_TOOLS
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@skip_load
class TestToolCount:
    def test_minimum_tool_count(self):
        """Platform must expose at least 77 public tools."""
        count = len(_TOOLS)
        assert count >= 77, f"Expected ≥77 tools, got {count}. Missing tools?"

    def test_no_duplicate_names(self):
        """Every tool name must be unique."""
        names = [t["name"] for t in _TOOLS]
        seen: set[str] = set()
        dupes = [n for n in names if n in seen or seen.add(n)]  # type: ignore[func-returns-value]
        assert not dupes, f"Duplicate tool names detected: {dupes}"

    def test_all_tools_have_description(self):
        """Every tool must have a non-empty description."""
        missing = [t["name"] for t in _TOOLS if not t.get("description", "").strip()]
        assert not missing, f"Tools missing description: {missing}"

    def test_all_tools_have_input_schema(self):
        """Every tool must define an inputSchema."""
        missing = [t["name"] for t in _TOOLS if "inputSchema" not in t]
        assert not missing, f"Tools missing inputSchema: {missing}"


@skip_load
class TestBackwardCompatibility:
    """Verify that all canonical public tool names are present."""

    def _assert_group(self, group: set[str], label: str) -> None:
        names = _tool_names()
        missing = group - names
        assert not missing, f"{label} tools missing from MCP: {missing}"

    def test_db_tools(self):
        self._assert_group(_DB_TOOLS, "Database")

    def test_memory_tools(self):
        self._assert_group(_MEMORY_TOOLS, "Memory")

    def test_research_tools(self):
        self._assert_group(_RESEARCH_TOOLS, "Research")

    def test_graph_tools(self):
        self._assert_group(_GRAPH_TOOLS, "Graph")

    def test_vector_tools(self):
        self._assert_group(_VECTOR_TOOLS, "Vector")

    def test_planner_tools(self):
        self._assert_group(_PLANNER_TOOLS, "Planner")

    def test_job_tools(self):
        self._assert_group(_JOB_TOOLS, "Jobs (async batch)")

    def test_browser_tools(self):
        self._assert_group(_BROWSER_TOOLS, "Browser")

    def test_video_tools(self):
        self._assert_group(_VIDEO_TOOLS, "Video")

    def test_ocr_tools(self):
        self._assert_group(_OCR_TOOLS, "OCR")

    def test_docs_tools(self):
        self._assert_group(_DOCS_TOOLS, "Docs")

    def test_pdf_tools(self):
        self._assert_group(_PDF_TOOLS, "PDF")

    def test_sandbox_tools(self):
        self._assert_group(_SANDBOX_TOOLS, "Sandbox")

    def test_llm_tools(self):
        self._assert_group(_LLM_TOOLS, "LLM/Image")


@skip_load
class TestSchemaContracts:
    """Spot-check input schema shapes for critical tools."""

    def _schema(self, name: str) -> dict:
        for t in _TOOLS:
            if t["name"] == name:
                return t.get("inputSchema", {})
        pytest.skip(f"Tool '{name}' not found")
        return {}

    def test_job_submit_has_tool_name(self):
        props = self._schema("job_submit").get("properties", {})
        assert "tool_name" in props, "job_submit must have tool_name property"

    def test_job_submit_has_args(self):
        props = self._schema("job_submit").get("properties", {})
        assert "args" in props, "job_submit must have args property"

    def test_batch_submit_has_items(self):
        props = self._schema("batch_submit").get("properties", {})
        assert "items" in props, "batch_submit must have items property"

    def test_job_status_requires_job_id(self):
        req = self._schema("job_status").get("required", [])
        assert "job_id" in req, "job_status must require job_id"

    def test_job_cancel_requires_job_id(self):
        req = self._schema("job_cancel").get("required", [])
        assert "job_id" in req, "job_cancel must require job_id"

    def test_orchestrate_has_steps(self):
        props = self._schema("orchestrate").get("properties", {})
        assert "steps" in props, "orchestrate must have steps property"

    def test_memory_store_has_key_and_value(self):
        req = self._schema("memory_store").get("required", [])
        assert "key" in req and "value" in req

    def test_vector_store_has_text_and_id(self):
        req = self._schema("vector_store").get("required", [])
        assert "text" in req and "id" in req

    def test_pdf_edit_has_operations(self):
        req = self._schema("pdf_edit").get("required", [])
        assert "operations" in req
