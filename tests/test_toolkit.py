import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aichat.tools.manager import ToolManager
from aichat.tools.errors import ToolRequestError
from aichat.state import ApprovalMode


class TestToolkitManager(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.manager = ToolManager(max_tool_calls_per_turn=10)

    async def test_refresh_custom_tools_success(self) -> None:
        fake_tools = [
            {"name": "weather", "description": "Get weather", "parameters": {"type": "object", "properties": {}}},
        ]
        self.manager.toolkit.list_tools = AsyncMock(return_value=fake_tools)
        await self.manager.refresh_custom_tools()
        self.assertIn("weather", self.manager._custom_tools)
        self.assertEqual(self.manager._custom_tools["weather"]["description"], "Get weather")

    async def test_refresh_custom_tools_graceful_failure(self) -> None:
        self.manager.toolkit.list_tools = AsyncMock(side_effect=Exception("connection refused"))
        # Should not raise — toolkit unavailable is a soft failure
        await self.manager.refresh_custom_tools()
        self.assertEqual(self.manager._custom_tools, {})

    def test_is_custom_tool(self) -> None:
        self.manager._custom_tools["my_tool"] = {"description": "test", "parameters": {}}
        self.assertTrue(self.manager.is_custom_tool("my_tool"))
        self.assertFalse(self.manager.is_custom_tool("nonexistent"))

    async def test_run_create_tool(self) -> None:
        self.manager.toolkit.register_tool = AsyncMock(return_value={"registered": "search_web"})
        fake_tools = [
            {"name": "search_web", "description": "Search web", "parameters": {}},
        ]
        self.manager.toolkit.list_tools = AsyncMock(return_value=fake_tools)

        result = await self.manager.run_create_tool(
            "search_web",
            "Search web",
            {"type": "object", "properties": {"query": {"type": "string"}}},
            "return await httpx.get(f'https://google.com?q={kwargs[\"query\"]}')",
            ApprovalMode.AUTO,
            None,
        )
        self.assertEqual(result["registered"], "search_web")
        self.assertIn("search_web", self.manager._custom_tools)

    async def test_run_list_custom_tools(self) -> None:
        fake_tools = [{"name": "t1", "description": "T1", "parameters": {}, "file": "t1.py"}]
        self.manager.toolkit.list_tools = AsyncMock(return_value=fake_tools)

        result = await self.manager.run_list_custom_tools(ApprovalMode.AUTO, None)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "t1")

    async def test_run_delete_custom_tool(self) -> None:
        self.manager._custom_tools["old_tool"] = {"description": "Old", "parameters": {}}
        self.manager.toolkit.delete_tool = AsyncMock(return_value={"deleted": "old_tool"})
        self.manager.toolkit.list_tools = AsyncMock(return_value=[])

        result = await self.manager.run_delete_custom_tool("old_tool", ApprovalMode.AUTO, None)
        self.assertEqual(result["deleted"], "old_tool")
        self.assertNotIn("old_tool", self.manager._custom_tools)

    async def test_run_custom_tool(self) -> None:
        self.manager._custom_tools["greet"] = {
            "description": "Say hello",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}},
        }
        self.manager.toolkit.call_tool = AsyncMock(return_value={"tool": "greet", "result": "Hello, Alice!"})

        result = await self.manager.run_custom_tool("greet", {"name": "Alice"}, ApprovalMode.AUTO, None)
        self.assertEqual(result["result"], "Hello, Alice!")

    async def test_tool_denied_when_mode_deny(self) -> None:
        from aichat.tools.manager import ToolDeniedError
        with self.assertRaises(ToolDeniedError):
            await self.manager.run_list_custom_tools(ApprovalMode.DENY, None)

    def test_tool_definitions_includes_meta_tools(self) -> None:
        defs = self.manager.tool_definitions(shell_enabled=False)
        names = [d["function"]["name"] for d in defs]
        self.assertIn("create_tool", names)
        self.assertIn("list_custom_tools", names)
        self.assertIn("delete_custom_tool", names)

    def test_tool_definitions_includes_custom_tools(self) -> None:
        self.manager._custom_tools["my_custom"] = {
            "description": "My custom tool",
            "parameters": {"type": "object", "properties": {}},
        }
        defs = self.manager.tool_definitions(shell_enabled=False)
        names = [d["function"]["name"] for d in defs]
        self.assertIn("my_custom", names)

    def test_tool_definitions_shell_conditional(self) -> None:
        defs_no_shell = self.manager.tool_definitions(shell_enabled=False)
        defs_shell = self.manager.tool_definitions(shell_enabled=True)
        names_no = [d["function"]["name"] for d in defs_no_shell]
        names_yes = [d["function"]["name"] for d in defs_shell]
        self.assertNotIn("shell_exec", names_no)
        self.assertIn("shell_exec", names_yes)
