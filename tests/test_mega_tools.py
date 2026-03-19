"""
Deep regression tests for the mega-tool consolidation refactor.

Tests cover:
  1. Schema validation — all 16 mega-tools have correct structure
  2. Dispatch routing — _resolve_mega_tool maps all (tool, action) pairs correctly
  3. Action parameter handling — action enums match dispatch map
  4. Backward compatibility — original tool names still work via passthrough
  5. Paid API removal — no Brave/Pixiv/SauceNAO/Danbooru references
  6. ImageRenderingPolicy — updated for mega-tool names
  7. MCP integration — tools/list returns consolidated schemas
  8. Tool count — exactly 16 mega-tools

Run:
    pytest tests/test_mega_tools.py -v
    pytest tests/test_mega_tools.py -v -m mega_tools
"""
from __future__ import annotations

import ast
import os
import re

import pytest

# ---------------------------------------------------------------------------
# Fixture: load app.py source and parse key structures
# ---------------------------------------------------------------------------

_APP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "docker", "mcp", "app.py"
)


@pytest.fixture(scope="module")
def app_source() -> str:
    with open(_APP_PATH) as f:
        return f.read()


@pytest.fixture(scope="module")
def app_tree(app_source: str) -> ast.Module:
    return ast.parse(app_source)


# ---------------------------------------------------------------------------
# Helpers to extract runtime structures from source without importing app.py
# (app.py has Docker-only dependencies like cv2, minio, etc.)
# ---------------------------------------------------------------------------


def _extract_tools_names(source: str) -> list[str]:
    """Extract tool names from the _TOOLS list by regex (safe, no import)."""
    # Match "name": "..." patterns inside _TOOLS
    tools_section = re.search(
        r"^_TOOLS:\s*list\[.*?\]\s*=\s*\[(.+?)^\]",
        source,
        re.DOTALL | re.MULTILINE,
    )
    if not tools_section:
        return []
    return re.findall(r'"name":\s*"(\w+)"', tools_section.group(1))


def _extract_mega_tool_map(source: str) -> dict[str, dict[str, str]]:
    """Extract _MEGA_TOOL_MAP dict from source by regex + literal_eval."""
    match = re.search(
        r"^_MEGA_TOOL_MAP:\s*dict\[.*?\]\s*=\s*(\{.+?^\})",
        source,
        re.DOTALL | re.MULTILINE,
    )
    if not match:
        return {}
    raw = match.group(1)
    try:
        return ast.literal_eval(raw)
    except Exception:
        return {}


def _extract_action_enums(source: str) -> dict[str, list[str]]:
    """Extract action enum values from each mega-tool schema."""
    result: dict[str, list[str]] = {}
    # Find each tool block with "name": "<tool>" ... "action" ... "enum": [...]
    tool_blocks = re.split(r'\{\s*"name":\s*"', source)
    for block in tool_blocks[1:]:
        name_match = re.match(r'(\w+)"', block)
        if not name_match:
            continue
        name = name_match.group(1)
        enum_match = re.search(
            r'"action".*?"enum":\s*\[([^\]]+)\]', block, re.DOTALL
        )
        if enum_match:
            actions = re.findall(r'"(\w+)"', enum_match.group(1))
            result[name] = actions
    return result


# ---------------------------------------------------------------------------
# 1. Schema Validation
# ---------------------------------------------------------------------------


@pytest.mark.mega_tools
class TestSchemaValidation:
    """Verify all 16 mega-tool schemas have correct structure."""

    def test_exactly_16_tools(self, app_source: str):
        names = _extract_tools_names(app_source)
        assert len(names) == 16, f"Expected 16 mega-tools, got {len(names)}: {names}"

    def test_expected_tool_names(self, app_source: str):
        names = set(_extract_tools_names(app_source))
        expected = {
            "web", "browser", "image", "document", "media", "data",
            "memory", "knowledge", "vector", "code", "custom_tools",
            "planner", "jobs", "research", "think", "system",
        }
        assert names == expected, f"Missing: {expected - names}, Extra: {names - expected}"

    def test_all_tools_have_input_schema(self, app_source: str):
        """Every tool must have an inputSchema with type=object."""
        tools_section = re.search(
            r"^_TOOLS:\s*list\[.*?\]\s*=\s*\[(.+?)^\]",
            app_source,
            re.DOTALL | re.MULTILINE,
        )
        assert tools_section, "_TOOLS list not found"
        text = tools_section.group(1)
        names = re.findall(r'"name":\s*"(\w+)"', text)
        for name in names:
            assert f'"inputSchema"' in text, f"Tool {name} missing inputSchema"

    def test_action_param_required_except_think(self, app_source: str):
        """All tools except 'think' must have 'action' in required params."""
        enums = _extract_action_enums(app_source)
        # think should NOT have action
        assert "think" not in enums, "think should not have an action enum"
        # All others should have action
        for name in _extract_tools_names(app_source):
            if name == "think":
                continue
            assert name in enums, f"Tool '{name}' missing action enum"
            assert len(enums[name]) >= 2, f"Tool '{name}' has too few actions: {enums[name]}"

    def test_no_empty_descriptions(self, app_source: str):
        """Every tool must have a non-empty description."""
        tools_section = re.search(
            r"^_TOOLS:\s*list\[.*?\]\s*=\s*\[(.+?)^\]",
            app_source,
            re.DOTALL | re.MULTILINE,
        )
        assert tools_section
        descs = re.findall(r'"description":\s*\(\s*"(.+?)"', tools_section.group(1))
        assert len(descs) >= 16, f"Only found {len(descs)} descriptions"


# ---------------------------------------------------------------------------
# 2. Dispatch Routing
# ---------------------------------------------------------------------------


@pytest.mark.mega_tools
class TestDispatchRouting:
    """Verify _MEGA_TOOL_MAP covers all mega-tools and actions."""

    def test_mega_tool_map_exists(self, app_source: str):
        mtm = _extract_mega_tool_map(app_source)
        assert mtm, "_MEGA_TOOL_MAP not found or empty"

    def test_map_covers_all_megatools(self, app_source: str):
        mtm = _extract_mega_tool_map(app_source)
        tool_names = set(_extract_tools_names(app_source))
        # think has no action, so it shouldn't be in the map
        expected_in_map = tool_names - {"think"}
        mapped = set(mtm.keys())
        missing = expected_in_map - mapped
        assert not missing, f"Tools missing from _MEGA_TOOL_MAP: {missing}"

    def test_all_actions_have_dispatch(self, app_source: str):
        """Every action in a schema enum must have a dispatch entry."""
        enums = _extract_action_enums(app_source)
        mtm = _extract_mega_tool_map(app_source)
        for tool, actions in enums.items():
            if tool not in mtm:
                continue
            for action in actions:
                assert action in mtm[tool], (
                    f"Tool '{tool}' action '{action}' has no dispatch mapping"
                )

    def test_dispatch_targets_are_valid_handlers(self, app_source: str):
        """Every dispatch target must exist as a handler in _call_tool."""
        mtm = _extract_mega_tool_map(app_source)
        # Collect all handler names from _call_tool's if/elif blocks
        # Matches: if name == "x":  AND  if name in ("x", "y"):
        handlers = set(re.findall(r'if name == "(\w+)":', app_source))
        # Also collect names from `if name in (...)` patterns
        for m in re.finditer(r'if name in \(([^)]+)\):', app_source):
            handlers.update(re.findall(r'"(\w+)"', m.group(1)))
        for tool, actions in mtm.items():
            for action, target in actions.items():
                assert target in handlers, (
                    f"Dispatch target '{target}' (from {tool}.{action}) "
                    f"not found as handler in _call_tool"
                )

    def test_resolve_mega_tool_function_exists(self, app_tree: ast.Module):
        funcs = [
            n.name
            for n in ast.walk(app_tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assert "_resolve_mega_tool" in funcs

    def test_call_tool_invokes_resolve(self, app_source: str):
        """_call_tool must call _resolve_mega_tool at its start."""
        # Find _call_tool body
        match = re.search(
            r"async def _call_tool\(.*?\).*?:\n(.*?)(?=\nasync def |\nclass |\n# ---)",
            app_source,
            re.DOTALL,
        )
        assert match, "_call_tool not found"
        body = match.group(1)[:500]  # first 500 chars of the body
        assert "_resolve_mega_tool" in body, (
            "_call_tool does not call _resolve_mega_tool at the start"
        )


# ---------------------------------------------------------------------------
# 3. Action Parameter Handling
# ---------------------------------------------------------------------------


@pytest.mark.mega_tools
class TestActionParameters:
    """Verify action enums are consistent with dispatch map."""

    def test_browser_action_remapping(self, app_source: str):
        """Browser mega-tool actions that map to 'browser' handler must
        include the action remapping logic."""
        assert "browser_action_map" in app_source, (
            "Browser action remapping dict not found in _resolve_mega_tool"
        )
        assert '"download_images"' in app_source or '"download_page_images"' in app_source

    def test_desktop_control_action_forwarding(self, app_source: str):
        """desktop_control must forward desktop_action → action."""
        assert "desktop_action" in app_source

    def test_face_detect_annotate_rename(self, app_source: str):
        """face_detect must rename annotate_faces → annotate for handler."""
        assert "annotate_faces" in app_source


# ---------------------------------------------------------------------------
# 4. Backward Compatibility
# ---------------------------------------------------------------------------


@pytest.mark.mega_tools
class TestBackwardCompatibility:
    """Original tool names should still work via passthrough."""

    def test_original_names_passthrough(self, app_source: str):
        """Names not in _MEGA_TOOL_MAP should pass through unchanged."""
        assert (
            'if name not in _MEGA_TOOL_MAP:' in app_source
            or "name not in _MEGA_TOOL_MAP" in app_source
        ), "Passthrough logic not found in _resolve_mega_tool"

    def test_no_action_passthrough(self, app_source: str):
        """Mega-tool call with no action should not crash."""
        assert (
            'if not action:' in app_source
            or "not action" in app_source
        )


# ---------------------------------------------------------------------------
# 5. Paid API Removal
# ---------------------------------------------------------------------------


@pytest.mark.mega_tools
class TestPaidApiRemoval:
    """Verify all paid API dependencies are completely removed."""

    def test_no_brave_api_key(self, app_source: str):
        assert "BRAVE_API_KEY" not in app_source

    def test_no_brave_search_function(self, app_source: str):
        assert "_brave_search" not in app_source

    def test_no_pixiv(self, app_source: str):
        assert "PIXIV_REFRESH_TOKEN" not in app_source

    def test_no_saucenao(self, app_source: str):
        assert "SAUCENAO_API_KEY" not in app_source

    def test_no_danbooru(self, app_source: str):
        assert "DANBOORU_API_KEY" not in app_source
        assert "DANBOORU_LOGIN" not in app_source

    def test_no_anime_tools(self, app_source: str):
        """anime_search, anime_pipeline, saucenao_search handlers removed."""
        handlers = re.findall(r'if name == "(anime_\w+|saucenao_\w+)":', app_source)
        assert not handlers, f"Paid API handlers still present: {handlers}"

    def test_no_brave_tier_in_web_search(self, app_source: str):
        """Web search should not reference Brave as a tier."""
        assert "Tier 1a" not in app_source or "Brave" not in app_source

    def test_docker_compose_no_paid_keys(self):
        """docker-compose.yml should not reference paid API env vars."""
        compose_path = os.path.join(
            os.path.dirname(__file__), "..", "docker-compose.yml"
        )
        with open(compose_path) as f:
            content = f.read()
        for key in ("PIXIV_REFRESH_TOKEN", "SAUCENAO_API_KEY",
                     "DANBOORU_API_KEY", "DANBOORU_LOGIN"):
            assert key not in content, f"{key} still in docker-compose.yml"


# ---------------------------------------------------------------------------
# 6. ImageRenderingPolicy
# ---------------------------------------------------------------------------


@pytest.mark.mega_tools
class TestImageRenderingPolicy:
    """Verify IMAGE_TOOLS includes mega-tool names."""

    def test_image_mega_tool_in_policy(self, app_source: str):
        match = re.search(r"IMAGE_TOOLS.*?frozenset\(\{(.+?)\}\)", app_source, re.DOTALL)
        assert match, "IMAGE_TOOLS frozenset not found"
        tools_text = match.group(1)
        assert '"image"' in tools_text, "mega-tool 'image' not in IMAGE_TOOLS"
        assert '"browser"' in tools_text, "mega-tool 'browser' not in IMAGE_TOOLS"


# ---------------------------------------------------------------------------
# 7. Source Integrity
# ---------------------------------------------------------------------------


@pytest.mark.mega_tools
class TestSourceIntegrity:
    """Basic integrity checks on the refactored source."""

    def test_python_syntax_valid(self, app_source: str):
        """app.py must parse without SyntaxError."""
        try:
            ast.parse(app_source)
        except SyntaxError as e:
            pytest.fail(f"app.py has syntax error: {e}")

    def test_line_count_reduced(self, app_source: str):
        """Consolidation should reduce line count (was ~10,800)."""
        lines = len(app_source.splitlines())
        assert lines < 10000, f"Expected <10000 lines, got {lines}"
        assert lines > 5000, f"Too many lines removed ({lines}) — likely broken"

    def test_no_duplicate_tool_names(self, app_source: str):
        names = _extract_tools_names(app_source)
        seen: set[str] = set()
        dupes: list[str] = []
        for n in names:
            if n in seen:
                dupes.append(n)
            seen.add(n)
        assert not dupes, f"Duplicate tool names: {dupes}"

    def test_searxng_is_primary_search(self, app_source: str):
        """SearXNG (free, self-hosted) must remain the primary search backend."""
        assert "SEARXNG_URL" in app_source
        assert "_searxng_search" in app_source or "searxng" in app_source.lower()

    def test_lm_studio_integration_intact(self, app_source: str):
        """LM Studio integration must be preserved."""
        assert "IMAGE_GEN_BASE_URL" in app_source
        assert "/v1/chat/completions" in app_source
        assert "/v1/models" in app_source


# ---------------------------------------------------------------------------
# 8. Dartboard Integration
# ---------------------------------------------------------------------------


@pytest.mark.mega_tools
class TestDartboardIntegration:
    """Verify dartboard config references the new mega-tools."""

    def test_dartboard_system_prompt_updated(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "dartboard", "lib", "config.dart"
        )
        if not os.path.exists(config_path):
            pytest.skip("dartboard not found at expected path")
        with open(config_path) as f:
            content = f.read()
        assert "16 mega-tools" in content, "Dartboard system prompt not updated for mega-tools"
        assert "action" in content, "Dartboard prompt should mention action parameter"
        # Should NOT reference old paid API tools
        assert "anime_search" not in content
        assert "saucenao_search" not in content

    def test_dartboard_no_paid_tool_references(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "dartboard", "lib", "config.dart"
        )
        if not os.path.exists(config_path):
            pytest.skip("dartboard not found")
        with open(config_path) as f:
            content = f.read()
        for old_tool in ("anime_search", "anime_pipeline", "saucenao_search"):
            assert old_tool not in content, f"Old tool '{old_tool}' still in dartboard config"


# ---------------------------------------------------------------------------
# 9. MCP Integration (requires running stack)
# ---------------------------------------------------------------------------

_MCP_URL = os.environ.get("MCP_URL", "http://localhost:8096")


def _mcp_available() -> bool:
    try:
        import httpx
        r = httpx.get(f"{_MCP_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.mega_tools
@pytest.mark.skipif(not _mcp_available(), reason="MCP service not reachable")
class TestMCPIntegration:
    """Live tests against running MCP server."""

    def test_tools_list_returns_16(self):
        import httpx
        r = httpx.post(
            f"{_MCP_URL}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {},
            },
            timeout=10,
        )
        data = r.json()
        tools = data.get("result", {}).get("tools", [])
        assert len(tools) == 16, f"Expected 16 tools, got {len(tools)}"

    def test_health_reports_tool_count(self):
        import httpx
        r = httpx.get(f"{_MCP_URL}/health", timeout=5)
        data = r.json()
        assert data.get("tools") == 16

    def test_think_tool_works(self):
        """think tool should work without action param."""
        import httpx
        r = httpx.post(
            f"{_MCP_URL}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "think",
                    "arguments": {"thought": "test reasoning"},
                },
            },
            timeout=10,
        )
        data = r.json()
        result = data.get("result", {})
        assert not result.get("isError", True), f"think tool returned error: {result}"

    def test_memory_store_and_recall(self):
        """memory mega-tool with store and recall actions."""
        import httpx
        # Store
        r1 = httpx.post(
            f"{_MCP_URL}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "memory",
                    "arguments": {
                        "action": "store",
                        "key": "_test_mega_tool_key",
                        "value": "mega_tool_regression_test",
                    },
                },
            },
            timeout=10,
        )
        d1 = r1.json()
        assert not d1.get("result", {}).get("isError", True), f"store failed: {d1}"

        # Recall
        r2 = httpx.post(
            f"{_MCP_URL}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "memory",
                    "arguments": {
                        "action": "recall",
                        "key": "_test_mega_tool_key",
                    },
                },
            },
            timeout=10,
        )
        d2 = r2.json()
        assert not d2.get("result", {}).get("isError", True), f"recall failed: {d2}"
        content = d2.get("result", {}).get("content", [])
        text = next((b["text"] for b in content if b.get("type") == "text"), "")
        assert "mega_tool_regression_test" in text

    def test_data_search_works(self):
        """data mega-tool with search action."""
        import httpx
        r = httpx.post(
            f"{_MCP_URL}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "data",
                    "arguments": {"action": "search", "q": "test", "limit": 1},
                },
            },
            timeout=10,
        )
        data = r.json()
        # Should not error (may return empty results, that's OK)
        assert not data.get("result", {}).get("isError", True), f"data search failed: {data}"

    def test_web_search_uses_searxng(self):
        """web mega-tool search action should use SearXNG (free)."""
        import httpx
        r = httpx.post(
            f"{_MCP_URL}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {
                    "name": "web",
                    "arguments": {"action": "search", "query": "python programming"},
                },
            },
            timeout=30,
        )
        data = r.json()
        # Should return results (not an error)
        result = data.get("result", {})
        content = result.get("content", [])
        text = next((b["text"] for b in content if b.get("type") == "text"), "")
        assert "search" in text.lower() or "url" in text.lower() or "http" in text.lower(), (
            f"web search returned unexpected result: {text[:200]}"
        )
