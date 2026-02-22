from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import Enum

from ..state import ApprovalMode
from .researchbox import ResearchboxTool
from .rss import RSSTool
from .shell import ShellTool


class ToolDeniedError(RuntimeError):
    pass


class ToolName(str, Enum):
    SHELL = "shell"
    RSS = "rss"
    RESEARCHBOX = "researchbox"


class ToolManager:
    def __init__(self, max_tool_calls_per_turn: int = 1) -> None:
        self.shell = ShellTool()
        self.rss = RSSTool()
        self.researchbox = ResearchboxTool()
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self._calls_this_turn = 0

    def reset_turn(self) -> None:
        self._calls_this_turn = 0

    async def _check_approval(
        self,
        mode: ApprovalMode,
        tool: str,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> None:
        if self._calls_this_turn >= self.max_tool_calls_per_turn:
            raise ToolDeniedError("Tool call limit reached for current turn")
        if mode == ApprovalMode.DENY:
            raise ToolDeniedError("Tool execution denied by current approval mode")
        if mode == ApprovalMode.ASK:
            if confirmer is None or not await confirmer(tool):
                raise ToolDeniedError(f"Tool '{tool}' rejected by user")
        self._calls_this_turn += 1

    async def run_shell(
        self,
        command: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
        cwd: str | None = None,
    ) -> str:
        await self._check_approval(mode, ToolName.SHELL.value, confirmer)
        session_id = await self.shell.sh_start(cwd=cwd)
        await self.shell.sh_send(session_id, command + "\n")
        output = await self.shell.sh_read(session_id, timeout_ms=1200)
        await self.shell.sh_close(session_id)
        return output.strip()

    async def run_rss(
        self,
        topic: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.RSS.value, confirmer)
        return await self.rss.news_latest(topic)

    async def run_researchbox(
        self,
        topic: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.RESEARCHBOX.value, confirmer)
        return await self.researchbox.rb_search_feeds(topic)

    def active_sessions(self) -> list[str]:
        return [f"shell:{sid}" for sid in self.shell.sessions]

    def tool_definitions(self) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "rss_latest",
                    "description": "Fetch the latest RSS items for a topic from the docker rssfeed service.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "Topic to query for recent items."}
                        },
                        "required": ["topic"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "researchbox_search",
                    "description": "Search for RSS feed sources for a topic via the docker researchbox service.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "Topic to search for feeds."}
                        },
                        "required": ["topic"],
                    },
                },
            },
        ]
