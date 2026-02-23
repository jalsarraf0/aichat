from __future__ import annotations

import asyncio
import os
import shlex
from pathlib import Path
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
    RESEARCHBOX_PUSH = "researchbox_push"


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
        exit_code, output = await self._run_shell_process(command, cwd=cwd)
        trimmed = output.strip()
        if exit_code != 0:
            if trimmed:
                return f"{trimmed}\n(exit {exit_code})"
            return f"(exit {exit_code})"
        return trimmed

    async def run_shell_stream(
        self,
        command: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
        *,
        cwd: str | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> tuple[int, str]:
        await self._check_approval(mode, ToolName.SHELL.value, confirmer)
        return await self._run_shell_process(command, cwd=cwd, on_output=on_output)

    async def _run_shell_process(
        self,
        command: str,
        *,
        cwd: str | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> tuple[int, str]:
        ensure_project_dirs(command, cwd)
        command = self._ensure_non_interactive_sudo(command)
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-lc",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=os.environ.copy(),
        )
        output_chunks: list[str] = []
        assert proc.stdout is not None
        while True:
            data = await proc.stdout.read(1024)
            if not data:
                break
            text = data.decode(errors="replace")
            output_chunks.append(text)
            if on_output:
                on_output(text)
        exit_code = await proc.wait()
        return exit_code, "".join(output_chunks).strip()

    def _ensure_non_interactive_sudo(self, command: str) -> str:
        stripped = command.lstrip()
        if not stripped.startswith("sudo "):
            return command
        try:
            parts = shlex.split(command)
        except ValueError:
            return command
        if not parts or parts[0] != "sudo":
            return command
        if "-n" in parts or "--non-interactive" in parts:
            return command
        parts.insert(1, "-n")
        return shlex.join(parts)

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

    async def run_researchbox_push(
        self,
        feed_url: str,
        topic: str,
        mode: ApprovalMode,
        confirmer: Callable[[str], Awaitable[bool]] | None,
    ) -> dict:
        await self._check_approval(mode, ToolName.RESEARCHBOX_PUSH.value, confirmer)
        return await self.researchbox.rb_push_feeds(feed_url, topic)

    def active_sessions(self) -> list[str]:
        return [f"shell:{sid}" for sid in self.shell.sessions]

    def tool_definitions(self, shell_enabled: bool) -> list[dict[str, object]]:
        tools: list[dict[str, object]] = [
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
            {
                "type": "function",
                "function": {
                    "name": "researchbox_push",
                    "description": "Fetch an RSS feed and store items in the docker rssfeed service.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "feed_url": {"type": "string", "description": "RSS feed URL to ingest."},
                            "topic": {"type": "string", "description": "Topic label to store items under."},
                        },
                        "required": ["feed_url", "topic"],
                    },
                },
            },
        ]
        if shell_enabled:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "shell_exec",
                        "description": "Run a shell command on the host machine.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {"type": "string", "description": "Shell command to execute."}
                            },
                            "required": ["command"],
                        },
                    },
                }
            )
        return tools


def ensure_project_dirs(command: str, cwd: str | None) -> None:
    current = Path(cwd) if cwd else Path.cwd()
    _ensure_dir(current)
    for target in _iter_cd_targets(command, current):
        _ensure_dir(target)
        current = target


def _ensure_dir(path: Path) -> None:
    candidate = path.expanduser()
    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError:
        return


def _iter_cd_targets(command: str, start_dir: Path) -> list[Path]:
    if not command:
        return []
    targets: list[Path] = []
    current = start_dir
    for line in command.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        cleaned = stripped.replace("&&", ";").replace("||", ";")
        for segment in cleaned.split(";"):
            try:
                parts = shlex.split(segment)
            except ValueError:
                continue
            for idx, token in enumerate(parts):
                if token != "cd":
                    continue
                j = idx + 1
                while j < len(parts) and parts[j].startswith("-"):
                    if parts[j] == "--":
                        j += 1
                        break
                    j += 1
                if j >= len(parts):
                    continue
                target = parts[j].rstrip(";")
                if target == "-":
                    continue
                if target in {"~", "~/"}:
                    target_path = Path.home()
                else:
                    target_path = Path(target)
                if not target_path.is_absolute():
                    target_path = current / target_path
                target_path = target_path.expanduser()
                targets.append(target_path)
                current = target_path
    return targets
