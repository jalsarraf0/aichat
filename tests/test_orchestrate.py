"""
Orchestration tool tests for docker/mcp/app.py.

Structure
---------
TestWorkflowStep          — unit tests for WorkflowStep dataclass
TestWorkflowExecutor      — unit tests for WorkflowExecutor (wave, interpolate, format)
TestOrchestrateSchema     — schema introspection (no Docker)
TestOrchestrateE2E        — end-to-end with AsyncMock (no Docker)
TestOrchestrateSmoke      — @pytest.mark.smoke: live MCP endpoint
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Load docker/mcp/app.py via importlib
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent


def _load_mcp_mod():
    spec = importlib.util.spec_from_file_location(
        "mcp_app_orchestrate",
        _REPO_ROOT / "docker" / "mcp" / "app.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # Required in Python 3.14+: @dataclass inspects sys.modules[cls.__module__]
    sys.modules["mcp_app_orchestrate"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


try:
    _mcp = _load_mcp_mod()
    WorkflowStep     = _mcp.WorkflowStep
    WorkflowResult   = _mcp.WorkflowResult
    WorkflowExecutor = _mcp.WorkflowExecutor
    _TOOLS           = _mcp._TOOLS
    _LOAD_OK         = True
except Exception as _load_err:
    _LOAD_OK     = False
    _LOAD_ERR    = str(_load_err)

skip_load = pytest.mark.skipif(
    not _LOAD_OK,
    reason=f"docker/mcp/app.py load failed: {_LOAD_ERR if not _LOAD_OK else ''}",
)

# Live-service check
_MCP_URL = "http://localhost:8096"
_MCP_UP  = False
try:
    _MCP_UP = httpx.get(f"{_MCP_URL}/health", timeout=2).status_code == 200
except Exception:
    pass

skip_mcp_svc = pytest.mark.skipif(not _MCP_UP, reason="aichat-mcp not reachable")


# ===========================================================================
# 1. WorkflowStep — dataclass unit tests
# ===========================================================================

@skip_load
class TestWorkflowStep:
    """Unit tests for the WorkflowStep dataclass."""

    def test_fields_set_correctly(self):
        s = WorkflowStep(id="a", tool="screenshot", args={"url": "https://example.com"},
                         depends_on=["b"], label="Take screenshot")
        assert s.id == "a"
        assert s.tool == "screenshot"
        assert s.args == {"url": "https://example.com"}
        assert s.depends_on == ["b"]
        assert s.label == "Take screenshot"

    def test_depends_on_defaults_empty(self):
        s = WorkflowStep(id="x", tool="web_search", args={"query": "test"})
        assert s.depends_on == []

    def test_label_defaults_empty_string(self):
        s = WorkflowStep(id="x", tool="web_search", args={})
        assert s.label == ""

    def test_args_stored_as_dict(self):
        args = {"url": "https://example.com", "timeout": 5}
        s = WorkflowStep(id="x", tool="screenshot", args=args)
        assert isinstance(s.args, dict)
        assert s.args["url"] == "https://example.com"

    def test_id_is_string(self):
        s = WorkflowStep(id="step_1", tool="web_search", args={})
        assert isinstance(s.id, str)


# ===========================================================================
# 2. WorkflowExecutor — unit tests
# ===========================================================================

@skip_load
class TestWorkflowExecutor:
    """Unit tests for WorkflowExecutor logic (no network calls)."""

    # Helpers ----------------------------------------------------------------

    def _make_mock_tool(self, return_text: str = "ok result"):
        """Return an AsyncMock that returns a text content block."""
        mock = AsyncMock(return_value=[{"type": "text", "text": return_text}])
        return mock

    def _run(self, coro):
        return asyncio.run(coro)

    # Wave building ----------------------------------------------------------

    def test_build_waves_single_step(self):
        steps = [WorkflowStep(id="a", tool="t", args={})]
        ex = WorkflowExecutor(steps)
        waves = ex._build_waves()
        assert len(waves) == 1
        assert waves[0][0].id == "a"

    def test_build_waves_parallel_steps(self):
        steps = [
            WorkflowStep(id="a", tool="t", args={}),
            WorkflowStep(id="b", tool="t", args={}),
        ]
        ex = WorkflowExecutor(steps)
        waves = ex._build_waves()
        assert len(waves) == 1
        assert {s.id for s in waves[0]} == {"a", "b"}

    def test_build_waves_sequential_order(self):
        steps = [
            WorkflowStep(id="a", tool="t", args={}),
            WorkflowStep(id="b", tool="t", args={}, depends_on=["a"]),
            WorkflowStep(id="c", tool="t", args={}, depends_on=["b"]),
        ]
        ex = WorkflowExecutor(steps)
        waves = ex._build_waves()
        assert len(waves) == 3
        assert waves[0][0].id == "a"
        assert waves[1][0].id == "b"
        assert waves[2][0].id == "c"

    def test_build_waves_diamond_dependency(self):
        """A→B, A→C, B+C→D should give 3 waves."""
        steps = [
            WorkflowStep(id="a", tool="t", args={}),
            WorkflowStep(id="b", tool="t", args={}, depends_on=["a"]),
            WorkflowStep(id="c", tool="t", args={}, depends_on=["a"]),
            WorkflowStep(id="d", tool="t", args={}, depends_on=["b", "c"]),
        ]
        ex = WorkflowExecutor(steps)
        waves = ex._build_waves()
        assert len(waves) == 3
        assert waves[0][0].id == "a"
        assert {s.id for s in waves[1]} == {"b", "c"}
        assert waves[2][0].id == "d"

    def test_cycle_raises_value_error(self):
        steps = [
            WorkflowStep(id="a", tool="t", args={}, depends_on=["b"]),
            WorkflowStep(id="b", tool="t", args={}, depends_on=["a"]),
        ]
        ex = WorkflowExecutor(steps)
        with pytest.raises(ValueError, match="cycle"):
            ex._build_waves()

    def test_unknown_dep_raises_value_error(self):
        steps = [WorkflowStep(id="a", tool="t", args={}, depends_on=["nonexistent"])]
        ex = WorkflowExecutor(steps)
        with pytest.raises(ValueError, match="unknown step"):
            ex._build_waves()

    # Interpolation ----------------------------------------------------------

    def test_interpolate_replaces_placeholder(self):
        completed = {
            "search": WorkflowResult("search", "Search", "web_search",
                                     "AI news from 2025", True, 100),
        }
        ex = WorkflowExecutor([])
        result = ex._interpolate({"text": "{search.result}"}, completed)
        assert result["text"] == "AI news from 2025"

    def test_interpolate_multiple_placeholders(self):
        completed = {
            "a": WorkflowResult("a", "A", "t", "hello", True, 10),
            "b": WorkflowResult("b", "B", "t", "world", True, 10),
        }
        ex = WorkflowExecutor([])
        result = ex._interpolate({"text": "{a.result} {b.result}"}, completed)
        assert result["text"] == "hello world"

    def test_interpolate_leaves_non_string_unchanged(self):
        completed = {"a": WorkflowResult("a", "A", "t", "x", True, 10)}
        ex = WorkflowExecutor([])
        result = ex._interpolate({"count": 42, "flag": True}, completed)
        assert result["count"] == 42
        assert result["flag"] is True

    # Format report ----------------------------------------------------------

    def test_format_report_has_labels(self):
        results = [
            WorkflowResult("s1", "My Step One", "web_search", "found stuff", True, 150),
            WorkflowResult("s2", "My Step Two", "screenshot", "image saved", True, 300),
        ]
        report = WorkflowExecutor._format_report(results)
        assert "My Step One" in report
        assert "My Step Two" in report

    def test_format_report_has_timing(self):
        results = [WorkflowResult("s1", "Step", "t", "result", True, 250)]
        report = WorkflowExecutor._format_report(results)
        assert "ms" in report
        assert "250" in report

    def test_format_report_starts_with_header(self):
        results = [WorkflowResult("s1", "Step", "t", "result", True, 10)]
        report = WorkflowExecutor._format_report(results)
        assert report.startswith("## Workflow Results")

    def test_format_report_shows_failed_status(self):
        results = [WorkflowResult("s1", "Bad Step", "t", "Error: something", False, 10)]
        report = WorkflowExecutor._format_report(results)
        assert "FAILED" in report

    # Execution with mocked _call_tool ---------------------------------------

    def test_single_step_runs(self):
        steps = [WorkflowStep(id="a", tool="web_search", args={"query": "test"}, label="Search")]
        ex = WorkflowExecutor(steps)
        mock = self._make_mock_tool("search result text")
        with patch.object(_mcp, "_call_tool", mock):
            results = self._run(ex.run())
        assert len(results) == 1
        assert results[0].step_id == "a"
        assert results[0].result == "search result text"
        assert results[0].ok is True

    def test_parallel_steps_both_run(self):
        steps = [
            WorkflowStep(id="a", tool="web_search", args={"query": "A"}, label="Search A"),
            WorkflowStep(id="b", tool="web_search", args={"query": "B"}, label="Search B"),
        ]
        ex = WorkflowExecutor(steps)
        mock = self._make_mock_tool("parallel result")
        with patch.object(_mcp, "_call_tool", mock):
            results = self._run(ex.run())
        assert len(results) == 2
        assert mock.call_count == 2

    def test_stop_on_error_true_halts(self):
        steps = [
            WorkflowStep(id="a", tool="t", args={}, label="OK"),
            WorkflowStep(id="b", tool="t", args={}, depends_on=["a"], label="FAIL"),
            WorkflowStep(id="c", tool="t", args={}, depends_on=["b"], label="Should skip"),
        ]
        ex = WorkflowExecutor(steps, stop_on_error=True)

        call_count = 0

        async def _mock_tool(name, args):
            nonlocal call_count
            call_count += 1
            if name == "t" and call_count == 2:
                raise RuntimeError("step b failed")  # raises → ok=False
            return [{"type": "text", "text": "ok"}]

        with patch.object(_mcp, "_call_tool", _mock_tool):
            results = self._run(ex.run())
        # step c should be skipped due to stop_on_error
        assert len(results) == 2
        assert results[1].ok is False

    def test_stop_on_error_false_continues(self):
        steps = [
            WorkflowStep(id="a", tool="t", args={}, label="OK"),
            WorkflowStep(id="b", tool="t", args={}, label="FAIL"),
            WorkflowStep(id="c", tool="t", args={}, depends_on=["a", "b"], label="Dep on both"),
        ]
        ex = WorkflowExecutor(steps, stop_on_error=False)

        async def _mock_tool(name, args):
            # b will be detected as error by ok-check
            if "b" in str(args):
                return [{"type": "text", "text": "Error: failed"}]
            return [{"type": "text", "text": "success"}]

        with patch.object(_mcp, "_call_tool", _mock_tool):
            results = self._run(ex.run())
        # all 3 steps should run
        assert len(results) == 3


# ===========================================================================
# 3. TestOrchestrateSchema — _TOOLS list introspection
# ===========================================================================

@skip_load
class TestOrchestrateSchema:
    """Verify the orchestrate tool schema is correctly registered in _TOOLS."""

    def _get_schema(self):
        return next((t for t in _TOOLS if t["name"] == "orchestrate"), None)

    def test_orchestrate_in_tools_list(self):
        names = {t["name"] for t in _TOOLS}
        assert "orchestrate" in names, f"orchestrate not in _TOOLS (found: {sorted(names)})"

    def test_steps_param_required(self):
        schema = self._get_schema()
        assert schema is not None
        assert "steps" in schema["inputSchema"]["required"]

    def test_step_item_has_id_tool_args_required(self):
        schema = self._get_schema()
        item_schema = (
            schema["inputSchema"]["properties"]["steps"]["items"]
        )
        required = item_schema.get("required", [])
        assert "id" in required
        assert "tool" in required
        assert "args" in required

    def test_stop_on_error_is_boolean(self):
        schema = self._get_schema()
        soe = schema["inputSchema"]["properties"].get("stop_on_error", {})
        assert soe.get("type") == "boolean"

    def test_total_tools_count(self):
        """Tool count should be 49 (48 original + orchestrate)."""
        assert len(_TOOLS) == 49, f"Expected 49 tools, got {len(_TOOLS)}"


# ===========================================================================
# 4. TestOrchestrateE2E — full pipeline with mocked _call_tool
# ===========================================================================

@skip_load
class TestOrchestrateE2E:
    """End-to-end tests: WorkflowExecutor.run() → _format_report() with AsyncMocks."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_two_parallel_steps_report_has_both_labels(self):
        steps = [
            WorkflowStep(id="s1", tool="web_search", args={"query": "q1"}, label="First Search"),
            WorkflowStep(id="s2", tool="web_search", args={"query": "q2"}, label="Second Search"),
        ]
        ex = WorkflowExecutor(steps)
        mock = AsyncMock(return_value=[{"type": "text", "text": "result text"}])
        with patch.object(_mcp, "_call_tool", mock):
            results = self._run(ex.run())
        report = WorkflowExecutor._format_report(results)
        assert "First Search" in report
        assert "Second Search" in report
        assert mock.call_count == 2

    def test_interpolation_passes_prior_result_to_next_step(self):
        """Step B should receive step A's result via {a.result} interpolation."""
        received_args = {}

        async def _mock_tool(name, args):
            received_args.update(args)
            if name == "web_search":
                return [{"type": "text", "text": "SEARCH_OUTPUT"}]
            return [{"type": "text", "text": "SUMMARY_OUTPUT"}]

        steps = [
            WorkflowStep(id="a", tool="web_search", args={"query": "test"}, label="Search"),
            WorkflowStep(id="b", tool="smart_summarize", args={"text": "{a.result}"},
                         depends_on=["a"], label="Summarise"),
        ]
        ex = WorkflowExecutor(steps)
        with patch.object(_mcp, "_call_tool", _mock_tool):
            results = self._run(ex.run())
        assert results[1].step_id == "b"
        assert "SEARCH_OUTPUT" in results[1].result or received_args.get("text") == "SEARCH_OUTPUT"

    def test_error_step_marked_not_ok(self):
        steps = [WorkflowStep(id="bad", tool="t", args={}, label="Bad Step")]
        ex = WorkflowExecutor(steps)
        mock = AsyncMock(return_value=[{"type": "text", "text": "Error calling t: something bad"}])
        with patch.object(_mcp, "_call_tool", mock):
            results = self._run(ex.run())
        assert results[0].ok is False

    def test_report_format_structure(self):
        steps = [
            WorkflowStep(id="x", tool="web_search", args={}, label="The Search Step"),
        ]
        ex = WorkflowExecutor(steps)
        mock = AsyncMock(return_value=[{"type": "text", "text": "found: AI news"}])
        with patch.object(_mcp, "_call_tool", mock):
            results = self._run(ex.run())
        report = WorkflowExecutor._format_report(results)
        assert report.startswith("## Workflow Results")
        assert "The Search Step" in report
        assert "web_search" in report
        assert "OK" in report


# ===========================================================================
# 5. TestOrchestrateSmoke — live MCP endpoint
# ===========================================================================

@skip_mcp_svc
@skip_load
@pytest.mark.smoke
class TestOrchestrateSmoke:
    """Smoke tests against the live aichat-mcp service at localhost:8096."""

    def _mcp_call(self, name: str, arguments: dict, timeout: float = 30) -> dict:
        r = httpx.post(
            f"{_MCP_URL}/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": name, "arguments": arguments}},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()

    def test_orchestrate_appears_in_tool_list(self):
        """tools/list must include orchestrate with correct schema."""
        r = httpx.post(
            f"{_MCP_URL}/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            timeout=10,
        )
        r.raise_for_status()
        tools = r.json().get("result", {}).get("tools", [])
        names = {t["name"] for t in tools}
        assert "orchestrate" in names, f"orchestrate missing from live tools/list: {sorted(names)}"

    def test_orchestrate_empty_steps_returns_error(self):
        """Calling orchestrate with empty steps should return an error message, not crash."""
        resp = self._mcp_call("orchestrate", {"steps": []})
        content = resp.get("result", {}).get("content", [])
        texts = [b["text"] for b in content if b.get("type") == "text"]
        assert texts, "No text content in response"
        assert any("steps" in t.lower() or "empty" in t.lower() or "non-empty" in t.lower()
                   for t in texts), f"Unexpected error message: {texts}"
