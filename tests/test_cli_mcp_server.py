from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aichat import cli, mcp_server


def test_repo_subcommand_required() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["repo"])
    assert exc.value.code == 2


def test_github_subcommand_required() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["github"])
    assert exc.value.code == 2


def test_main_routes_to_app(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"app": 0}

    def _app_main() -> None:
        calls["app"] += 1

    monkeypatch.setattr(cli, "app_main", _app_main)
    monkeypatch.setattr(sys, "argv", ["aichat"])
    cli.main()
    assert calls["app"] == 1


def test_main_routes_to_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"mcp": 0}

    def _mcp_main() -> None:
        calls["mcp"] += 1

    monkeypatch.setattr(cli, "mcp_main", _mcp_main)
    monkeypatch.setattr(sys, "argv", ["aichat", "mcp"])
    cli.main()
    assert calls["mcp"] == 1


def test_main_fallthrough_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyParser:
        def __init__(self) -> None:
            self.print_help_called = False

        def parse_args(self) -> Namespace:
            return Namespace(command="repo")

        def print_help(self) -> None:
            self.print_help_called = True

    dummy = _DummyParser()
    monkeypatch.setattr(cli, "build_parser", lambda: dummy)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert dummy.print_help_called is True


async def _run_handle(
    monkeypatch: pytest.MonkeyPatch,
    req: dict,
    *,
    tool_blocks: list[dict] | None = None,
) -> list[dict]:
    writes: list[dict] = []

    if tool_blocks is not None:
        async def _fake_call_tool(name: str, arguments: dict) -> list[dict]:
            return tool_blocks

        monkeypatch.setattr(mcp_server, "_call_tool", _fake_call_tool)

    monkeypatch.setattr(mcp_server, "_write", lambda obj: writes.append(obj))
    await mcp_server._handle(json.dumps(req))
    return writes


@pytest.mark.asyncio
async def test_initialized_notification_has_no_response(monkeypatch: pytest.MonkeyPatch) -> None:
    writes = await _run_handle(
        monkeypatch,
        {"jsonrpc": "2.0", "id": 1, "method": "initialized", "params": {}},
    )
    assert writes == []


@pytest.mark.asyncio
async def test_tools_call_unknown_tool_sets_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    writes = await _run_handle(
        monkeypatch,
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        },
        tool_blocks=[{"type": "text", "text": "Unknown tool: nonexistent_tool"}],
    )
    assert len(writes) == 1
    payload = writes[0]["result"]
    assert payload["isError"] is True


@pytest.mark.asyncio
async def test_image_tool_enforces_inline_image_block(monkeypatch: pytest.MonkeyPatch) -> None:
    writes = await _run_handle(
        monkeypatch,
        {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {"name": "screenshot", "arguments": {"url": "https://example.com"}},
        },
        tool_blocks=[{"type": "text", "text": "Screenshot failed: timeout"}],
    )
    assert len(writes) == 1
    payload = writes[0]["result"]
    assert payload["isError"] is True
    assert any(block.get("type") == "image" for block in payload["content"])


@pytest.mark.asyncio
async def test_browser_screenshot_enforces_inline_image_block(monkeypatch: pytest.MonkeyPatch) -> None:
    writes = await _run_handle(
        monkeypatch,
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": "browser", "arguments": {"action": "screenshot"}},
        },
        tool_blocks=[{"type": "text", "text": "Screenshot failed: browser offline"}],
    )
    payload = writes[0]["result"]
    assert any(block.get("type") == "image" for block in payload["content"])


@pytest.mark.asyncio
async def test_non_image_tool_success_not_error(monkeypatch: pytest.MonkeyPatch) -> None:
    writes = await _run_handle(
        monkeypatch,
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {"name": "web_fetch", "arguments": {"url": "https://example.com"}},
        },
        tool_blocks=[{"type": "text", "text": "Example Domain"}],
    )
    payload = writes[0]["result"]
    assert payload["isError"] is False
    assert all(block.get("type") != "image" for block in payload["content"])
