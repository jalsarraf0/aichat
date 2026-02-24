from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import Callable
from datetime import datetime
from importlib.resources import as_file, files
from pathlib import Path

from .tool_args import parse_tool_args
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Header, Log, Markdown, Static, TextArea


class PromptInput(TextArea):
    async def _on_key(self, event) -> None:
        self._restart_blink()
        if self.read_only:
            return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            asyncio.create_task(self.app.action_send())
            return
        if event.key == "shift+enter":
            event.stop()
            event.prevent_default()
            start, end = self.selection
            self._replace_via_keyboard("\n", start, end)
            return
        await super()._on_key(event)

from .client import LLMClient, LLMClientError
from .config import load_config, save_config
from .model_labels import model_options
from .personalities import DEFAULT_PERSONALITY_ID, merge_personalities
from .sanitizer import format_structured, sanitize_response
from .state import AppState, ApprovalMode, Message
from .themes import THEMES
from .tool_scheduler import ToolCall, ToolResult, ToolScheduler
from .tools.manager import ToolDeniedError, ToolManager
from .transcript import TranscriptStore
from .ui.keybind_bar import KeybindBar
from .ui.keybinds import binding_list, render_keybinds
from .ui.modals import ChoiceModal, PersonalityAddModal, RssIngestModal, SearchModal, SettingsModal


class ChatMessage(Widget):
    """A single chat message bubble with a speaker label and markdown body."""

    def __init__(self, speaker: str, content: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._speaker = speaker
        self._content = content

    def compose(self) -> ComposeResult:
        yield Static(self._speaker, classes="msg-speaker")
        yield Markdown(self._content or " ", classes="msg-body")

    def update_content(self, content: str) -> None:
        """Replace the rendered body (used during streaming)."""
        self._content = content
        try:
            self.query_one(".msg-body", Markdown).update(content or " ")
        except Exception:
            pass


class AIChatApp(App):
    BINDINGS = binding_list() + [
        Binding("pageup", "scroll_up", "ScrollUp", priority=True),
        Binding("pagedown", "scroll_down", "ScrollDown", priority=True),
        Binding("ctrl+;", "personality", "Persona", show=False, priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        cfg = load_config()
        self._project_root = Path(cfg.get("project_root", str(Path.home() / "git")))
        self.personalities: list[dict[str, str]] = merge_personalities(cfg.get("personalities", []))
        self.state = AppState(
            model=cfg["model"],
            base_url=cfg["base_url"],
            theme=cfg["theme"],
            approval=ApprovalMode(cfg["approval"]),
            concise_mode=cfg.get("concise_mode", False),
            shell_enabled=cfg.get("shell_enabled", False),
            personality_id=cfg.get("active_personality", DEFAULT_PERSONALITY_ID),
            cwd=str(Path.cwd()),
        )
        self.client = LLMClient(self.state.base_url)
        self.tools = ToolManager(max_tool_calls_per_turn=self.state.max_tool_calls_per_turn)
        self.transcript_store = TranscriptStore()
        self.messages = self.transcript_store.load_messages()
        self.active_task: asyncio.Task[None] | None = None
        self._loaded_theme_sources: set[str] = set()
        self.system_prompt = self._build_system_prompt()
        self._session_notes: list[str] = []
        self._pending_tools: int = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="app"):
            with Horizontal(id="status-bar"):
                yield Static("", id="model-line")
                yield Static("", id="status")
            with Horizontal(id="body"):
                with Vertical(id="chat-pane"):
                    yield Static("Transcript", classes="pane-title")
                    yield VerticalScroll(id="transcript")
                with Vertical(id="tool-pane"):
                    yield Static("Tools", classes="pane-title")
                    yield Log(id="toolpane", auto_scroll=True)
                    yield Static("Sessions", classes="pane-title")
                    yield Log(id="sessionpane", auto_scroll=True)
            with Vertical(id="input-pane"):
                yield Static("Prompt", classes="pane-title")
                yield PromptInput(placeholder="Ask anything.", id="prompt")
        yield KeybindBar(id="keybind-bar")

    async def on_mount(self) -> None:
        self._apply_theme_with_fallback(self.state.theme)
        scroll = self._safe_query_one("#transcript", VerticalScroll)
        if scroll is not None:
            scroll.remove_children()
        self._start_new_chat(initial=True)
        self.set_focus(self.query_one("#prompt", TextArea))
        await self.update_status()
        self._refresh_sessions()
        asyncio.create_task(self._load_default_model())
        asyncio.create_task(self.tools.refresh_custom_tools())

    def apply_theme(self, name: str) -> None:
        selected_name = name if name in THEMES else "cyberpunk"
        for source_key in self._loaded_theme_sources:
            self.stylesheet.source.pop((source_key, ""), None)

        base_file = files("aichat.assets").joinpath("base.tcss")
        theme_file = files("aichat.assets").joinpath(f"themes/{THEMES[selected_name]}")

        with as_file(base_file) as base_path, as_file(theme_file) as theme_path:
            self.stylesheet.read_all([base_path, theme_path])
            self._loaded_theme_sources = {str(base_path.resolve()), str(theme_path.resolve())}

        self.state.theme = selected_name
        self.refresh_css(animate=False)

    def _apply_theme_with_fallback(self, requested_theme: str) -> None:
        try:
            self.apply_theme(requested_theme)
        except Exception as exc:
            self.apply_theme("cyberpunk")
            self.notify(f"Theme '{requested_theme}' failed; reverted to cyberpunk ({exc})", severity="warning")

    async def update_status(self) -> None:
        up = await self.client.health()
        model_line = self._safe_query_one("#model-line", Static)
        status_line = self._safe_query_one("#status", Static)
        if model_line is None or status_line is None:
            return
        model_line.update(
            f"model={self.state.model} | base={self.state.base_url} | server={'UP' if up else 'DOWN'}"
        )
        status_line.update(
            "stream="
            + ("ON" if self.state.streaming else "OFF")
            + f" | approval={self.state.approval.value}"
            + f" | concise={'ON' if self.state.concise_mode else 'OFF'}"
            + f" | shell={'ON' if self.state.shell_enabled else 'OFF'}"
            + f" | persona={self.state.personality_id}"
            + f" | cmd={self.state.cwd}"
        )

    async def action_send(self) -> None:
        prompt = self.query_one("#prompt", TextArea)
        text = prompt.text
        if not text.strip():
            prompt.text = ""
            return
        text = text.rstrip()
        prompt.text = ""
        if not text:
            return
        await self.handle_submit(text)

    async def handle_submit(self, text: str) -> None:
        if text.startswith("/"):
            asyncio.create_task(self.handle_command(text))
            return
        if text.startswith("!"):
            asyncio.create_task(self._handle_shell_command(text[1:], allow_toggle=False))
            return
        self.tools.reset_turn()
        user = Message("user", text)
        self.messages.append(user)
        self.transcript_store.append(user)
        self._write_transcript("You", text)
        self.active_task = asyncio.create_task(self.run_llm_turn())
        await self.active_task

    async def run_llm_turn(self) -> None:
        self.state.busy = True
        content = ""
        tools = self.tools.tool_definitions(self.state.shell_enabled)
        _live_msg: ChatMessage | None = None
        try:
            if self.state.streaming:
                _live_msg = ChatMessage("Assistant", "_…_", classes="chat-msg chat-assistant")
                scroll = self._safe_query_one("#transcript", VerticalScroll)
                if scroll is not None:
                    scroll.mount(_live_msg)
                    self.call_after_refresh(scroll.scroll_end, animate=False)

                def _on_chunk(text: str) -> None:
                    if _live_msg is not None:
                        _live_msg.update_content(text)

                content, tool_calls = await self._stream_with_tools(tools, on_chunk=_on_chunk)
                if _live_msg is not None:
                    _live_msg.remove()
                    _live_msg = None
            else:
                response = await self.client.chat_once_with_tools(
                    self.state.model,
                    self._llm_messages(),
                    tools=tools,
                )
                content = response.get("content", "")
                tool_calls = response.get("tool_calls", [])
            if tool_calls:
                await self._handle_tool_calls(tool_calls)
            else:
                self._finalize_assistant_response(content)
        except asyncio.CancelledError:
            self._write_transcript("Assistant", "Request cancelled.")
            raise
        except LLMClientError as exc:
            self._write_transcript("Assistant", f"LLM error: {exc}")
        finally:
            if _live_msg is not None:
                try:
                    _live_msg.remove()
                except Exception:
                    pass
            self.state.busy = False
            self.active_task = None
            await self.update_status()
            self._refresh_sessions()

    async def handle_command(self, text: str) -> None:
        self.tools.reset_turn()
        try:
            if text.startswith("/concise"):
                await self._toggle_concise(text)
                return
            if text.startswith("/verbose"):
                await self._set_concise(False)
                return
            if text.startswith("/persona") or text.startswith("/personality"):
                await self._handle_personality_command(text)
                return
            if text.startswith("/new"):
                await self.action_new_chat()
                return
            if text.startswith("/clear"):
                await self.action_clear_transcript()
                return
            if text.startswith("/copy"):
                await self.action_copy_last()
                return
            if text.startswith("/export"):
                await self.action_export_chat()
                return
            if text.startswith("/help"):
                await self.action_help()
                return
            if text.startswith("/shell"):
                await self._handle_shell_command(text[6:].strip(), allow_toggle=True)
                return
            if text.startswith("/vibecode"):
                name = text[len("/vibecode"):].strip()
                if not name:
                    self._write_transcript("Assistant", "Usage: /vibecode <project>")
                    return
                await self._create_project(name)
                return
            if text.startswith("/rss "):
                args = text[5:].strip()
                if not args:
                    self._write_transcript("Assistant", "RSS command requires a topic.")
                    return
                if args.startswith("store "):
                    topic = args[6:].strip()
                    if not topic:
                        self._write_transcript("Assistant", "Usage: /rss store <topic>")
                        return
                    await self._rss_store_topic(topic)
                    return
                if args.startswith("ingest "):
                    remainder = args[7:].strip()
                    if not remainder or " " not in remainder:
                        self._write_transcript("Assistant", "Usage: /rss ingest <topic> <feed_url>")
                        return
                    topic, feed_url = remainder.split(maxsplit=1)
                    call = ToolCall(
                        index=0,
                        name="researchbox_push",
                        args={"feed_url": feed_url, "topic": topic},
                        call_id="",
                        label="researchbox_push",
                    )
                    results = await self._run_tool_batch([call])
                    self._log_tool_results(results)
                    if any(not res.ok for res in results):
                        self._notify_tool_failures(results)
                    else:
                        self._write_transcript("Assistant", f"Stored feed for '{topic}'. See Tools panel.")
                    return
                if args == "ingest":
                    await self._rss_ingest_modal()
                    return
                topic = args
                call = ToolCall(index=0, name="rss_latest", args={"topic": topic}, call_id="", label="rss_latest")
                results = await self._run_tool_batch([call])
                self._log_tool_results(results)
                if any(not res.ok for res in results):
                    self._notify_tool_failures(results)
                else:
                    self._write_transcript("Assistant", "RSS fetched. See Tools panel for details.")
                return
            if text.startswith("/researchbox "):
                topic = text[13:].strip()
                if not topic:
                    self._write_transcript("Assistant", "Researchbox command requires a topic.")
                    return
                call = ToolCall(index=0, name="researchbox_search", args={"topic": topic}, call_id="", label="researchbox_search")
                results = await self._run_tool_batch([call])
                self._log_tool_results(results)
                if any(not res.ok for res in results):
                    self._notify_tool_failures(results)
                else:
                    self._write_transcript("Assistant", "Researchbox search complete. See Tools panel for details.")
                return
            if text.startswith("/endpoint"):
                new_url = text[len("/endpoint"):].strip()
                if not new_url:
                    self._write_transcript("Assistant", f"Current endpoint: {self.state.base_url}\nUsage: /endpoint <url>  (e.g. /endpoint http://localhost:1234)")
                    return
                if not new_url.startswith(("http://", "https://")):
                    self._write_transcript("Assistant", "Endpoint must start with http:// or https://")
                    return
                self.state.base_url = new_url
                self.client = LLMClient(new_url)
                self._persist_config()
                await self.update_status()
                self._write_transcript("Assistant", f"Endpoint set to {new_url}")
                return
            if text.startswith("/fetch "):
                url = text[7:].strip()
                if not url:
                    self._write_transcript("Assistant", "Usage: /fetch <url>")
                    return
                call = ToolCall(index=0, name="web_fetch", args={"url": url}, call_id="", label="web_fetch")
                results = await self._run_tool_batch([call])
                self._log_tool_results(results)
                if any(not res.ok for res in results):
                    self._notify_tool_failures(results)
                else:
                    self._write_transcript("Assistant", "Fetched. See Tools panel for content.")
                return
            if text.startswith("/memory "):
                args_str = text[8:].strip()
                parts = args_str.split(maxsplit=1)
                sub = parts[0] if parts else ""
                if sub == "recall":
                    key = parts[1].strip() if len(parts) > 1 else ""
                    call = ToolCall(index=0, name="memory_recall", args={"key": key}, call_id="", label="memory_recall")
                    results = await self._run_tool_batch([call])
                    self._log_tool_results(results)
                    if any(not res.ok for res in results):
                        self._notify_tool_failures(results)
                    else:
                        self._write_transcript("Assistant", "Memory recalled. See Tools panel.")
                    return
                if sub == "store":
                    remainder = parts[1].strip() if len(parts) > 1 else ""
                    if " " not in remainder:
                        self._write_transcript("Assistant", "Usage: /memory store <key> <value>")
                        return
                    key, value = remainder.split(maxsplit=1)
                    call = ToolCall(index=0, name="memory_store", args={"key": key, "value": value}, call_id="", label="memory_store")
                    results = await self._run_tool_batch([call])
                    self._log_tool_results(results)
                    if any(not res.ok for res in results):
                        self._notify_tool_failures(results)
                    else:
                        self._write_transcript("Assistant", f"Stored memory key '{key}'.")
                    return
                self._write_transcript("Assistant", "Usage: /memory recall [key] | /memory store <key> <value>")
                return
            if text.startswith("/tool"):
                await self._handle_tool_command(text[5:].strip())
                return
            self._write_transcript("Assistant", "Unknown command.")
        except ToolDeniedError as exc:
            self._tool_log(f"[denied] {exc}")
        except Exception as exc:
            self._tool_log(f"[tool error] {exc}")


    async def _handle_tool_command(self, sub: str) -> None:
        parts = sub.split(maxsplit=1)
        verb = parts[0].lower() if parts else ""
        if verb == "list":
            call = ToolCall(index=0, name="list_custom_tools", args={}, call_id="", label="list_custom_tools")
            results = await self._run_tool_batch([call])
            self._log_tool_results(results)
            if any(not res.ok for res in results):
                self._notify_tool_failures(results)
            else:
                self._write_transcript("Assistant", "Custom tools listed. See Tools panel.")
            return
        if verb == "delete":
            tool_name = parts[1].strip() if len(parts) > 1 else ""
            if not tool_name:
                self._write_transcript("Assistant", "Usage: /tool delete <name>")
                return
            call = ToolCall(
                index=0,
                name="delete_custom_tool",
                args={"tool_name": tool_name},
                call_id="",
                label="delete_custom_tool",
            )
            results = await self._run_tool_batch([call])
            self._log_tool_results(results)
            if any(not res.ok for res in results):
                self._notify_tool_failures(results)
            else:
                self._write_transcript("Assistant", f"Deleted tool '{tool_name}'.")
            return
        self._write_transcript("Assistant", "Usage: /tool list | /tool delete <name>")

    async def _show_modal(self, screen: object) -> object:
        """Show a modal screen and await dismissal without requiring a Textual worker."""
        loop = asyncio.get_running_loop()
        result_future: asyncio.Future[object] = loop.create_future()

        def _on_dismiss(result: object) -> None:
            if not result_future.done():
                result_future.set_result(result)

        self.push_screen(screen, callback=_on_dismiss)
        return await result_future

    def _push_modal(self, screen: object, on_dismiss: Callable[[object], object] | None = None) -> None:
        """Push a modal screen and handle results via callback to avoid blocking the UI loop."""

        def _handle(result: object) -> None:
            if on_dismiss is None:
                return
            try:
                outcome = on_dismiss(result)
                if asyncio.iscoroutine(outcome):
                    asyncio.create_task(outcome)
            except Exception as exc:
                self.notify(f"Modal handler failed: {exc}", severity="error")

        self.push_screen(screen, callback=_handle)

    def _merge_tool_call_deltas(self, state: dict[int, dict[str, object]], deltas: list[dict[str, object]]) -> None:
        for call in deltas:
            index = int(call.get("index", 0))
            entry = state.setdefault(
                index,
                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
            )
            if "id" in call:
                entry["id"] = call["id"]
            if "type" in call:
                entry["type"] = call["type"]
            if "function" in call and isinstance(call["function"], dict):
                func = entry.setdefault("function", {"name": "", "arguments": ""})
                if "name" in call["function"]:
                    func["name"] = call["function"]["name"]
                if "arguments" in call["function"]:
                    func["arguments"] = str(func.get("arguments", "")) + str(call["function"]["arguments"])

    async def _stream_with_tools(
        self,
        tools: list[dict[str, object]],
        on_chunk: Callable[[str], None] | None = None,
    ) -> tuple[str, list[dict[str, object]]]:
        content = ""
        tool_call_state: dict[int, dict[str, object]] = {}
        async for event in self.client.chat_stream_events(
            self.state.model,
            self._llm_messages(),
            tools=tools,
        ):
            if event.get("type") == "content":
                chunk = str(event.get("value", ""))
                if not chunk:
                    continue
                content += chunk
                if on_chunk:
                    on_chunk(content)
            elif event.get("type") == "tool_calls":
                deltas = event.get("value", [])
                if isinstance(deltas, list):
                    self._merge_tool_call_deltas(tool_call_state, deltas)
        tool_calls = [tool_call_state[idx] for idx in sorted(tool_call_state)]
        return content, tool_calls

    async def _handle_tool_calls(self, tool_calls: list[dict[str, object]]) -> None:
        if not tool_calls:
            return
        assistant = Message("assistant", "", full_content="", metadata={"tool_calls": tool_calls})
        self.messages.append(assistant)
        self.transcript_store.append(assistant)
        calls: list[ToolCall] = []
        immediate_results: list[ToolResult] = []
        for index, call in enumerate(tool_calls[: self.state.max_tool_calls_per_turn]):
            name = ""
            args_text = ""
            call_id = ""
            if isinstance(call.get("function"), dict):
                name = str(call["function"].get("name", ""))
                args_text = str(call["function"].get("arguments", ""))
            call_id = str(call.get("id", "")) if call.get("id") else ""
            args, error = parse_tool_args(name, args_text)
            if error:
                immediate_results.append(
                    ToolResult(
                        call=ToolCall(index=index, name=name, args={}, call_id=call_id, label=name),
                        ok=False,
                        output="",
                        attempts=1,
                        duration=0.0,
                        error=error,
                    )
                )
                continue
            calls.append(ToolCall(index=index, name=name, args=args, call_id=call_id, label=name))

        results: list[ToolResult] = []
        if calls:
            results = await self._run_tool_batch(calls)
        results = results + immediate_results
        results.sort(key=lambda r: r.call.index)
        self._log_tool_results(results)
        self._append_shell_output(results)
        self._append_tool_messages(results)
        self._notify_tool_failures(results)
        await self._run_followup_response()

    async def _execute_tool_call(self, name: str, args: dict[str, object]) -> str:
        if name == "rss_latest":
            topic = str(args.get("topic", "")).strip()
            if not topic:
                return "rss_latest: missing 'topic'"
            payload = await self.tools.run_rss(topic, self.state.approval, self._confirm_tool)
            return json.dumps(payload, ensure_ascii=False)
        if name == "researchbox_search":
            topic = str(args.get("topic", "")).strip()
            if not topic:
                return "researchbox_search: missing 'topic'"
            payload = await self.tools.run_researchbox(topic, self.state.approval, self._confirm_tool)
            return json.dumps(payload, ensure_ascii=False)
        if name == "researchbox_push":
            feed_url = str(args.get("feed_url", "")).strip()
            topic = str(args.get("topic", "")).strip()
            if not feed_url or not topic:
                return "researchbox_push: missing 'feed_url' or 'topic'"
            payload = await self.tools.run_researchbox_push(
                feed_url,
                topic,
                self.state.approval,
                self._confirm_tool,
            )
            return json.dumps(payload, ensure_ascii=False)
        if name == "shell_exec":
            if not self.state.shell_enabled:
                return "shell_exec: shell access is disabled"
            command = str(args.get("command", "")).strip()
            if not command:
                return "shell_exec: missing 'command'"
            output, new_cwd = await self.tools.run_shell(
                command, self.state.approval, self._confirm_tool, cwd=self.state.cwd
            )
            if new_cwd:
                self.state.cwd = new_cwd
                await self.update_status()
            return output or "(no output)"
        if name == "web_fetch":
            url = str(args.get("url", "")).strip()
            if not url:
                return "web_fetch: missing 'url'"
            max_chars = int(args.get("max_chars", 4000))
            max_chars = max(500, min(max_chars, 16000))
            payload = await self.tools.run_web_fetch(url, max_chars, self.state.approval, self._confirm_tool)
            text = payload.get("text", "")
            truncated = payload.get("truncated", False)
            suffix = "\n...[truncated]" if truncated else ""
            return f"{text}{suffix}" if text else "(no content)"
        if name == "memory_store":
            key = str(args.get("key", "")).strip()
            value = str(args.get("value", "")).strip()
            if not key or not value:
                return "memory_store: missing 'key' or 'value'"
            payload = await self.tools.run_memory_store(key, value, self.state.approval, self._confirm_tool)
            return json.dumps(payload, ensure_ascii=False)
        if name == "memory_recall":
            key = str(args.get("key", "")).strip()
            payload = await self.tools.run_memory_recall(key, self.state.approval, self._confirm_tool)
            return json.dumps(payload, ensure_ascii=False)
        if name == "create_tool":
            tool_name = str(args.get("tool_name", "")).strip()
            description = str(args.get("description", "")).strip()
            parameters = args.get("parameters_schema", {})
            if not isinstance(parameters, dict):
                parameters = {}
            code = str(args.get("code", "")).strip()
            if not tool_name or not code:
                return "create_tool: missing 'tool_name' or 'code'"
            payload = await self.tools.run_create_tool(
                tool_name, description, parameters, code, self.state.approval, self._confirm_tool
            )
            return json.dumps(payload, ensure_ascii=False)
        if name == "list_custom_tools":
            tools_list = await self.tools.run_list_custom_tools(self.state.approval, self._confirm_tool)
            return json.dumps({"tools": tools_list}, ensure_ascii=False)
        if name == "delete_custom_tool":
            tool_name = str(args.get("tool_name", "")).strip()
            if not tool_name:
                return "delete_custom_tool: missing 'tool_name'"
            payload = await self.tools.run_delete_custom_tool(
                tool_name, self.state.approval, self._confirm_tool
            )
            return json.dumps(payload, ensure_ascii=False)
        if name == "browser":
            action = str(args.get("action", "")).strip()
            if not action:
                return "browser: missing 'action'"
            payload = await self.tools.run_browser(
                action,
                self.state.approval,
                self._confirm_tool,
                url=str(args["url"]).strip() if args.get("url") else None,
                selector=str(args["selector"]).strip() if args.get("selector") else None,
                value=str(args["value"]) if args.get("value") is not None else None,
                code=str(args["code"]).strip() if args.get("code") else None,
            )
            return json.dumps(payload, ensure_ascii=False)
        if self.tools.is_custom_tool(name):
            payload = await self.tools.run_custom_tool(
                name, dict(args), self.state.approval, self._confirm_tool
            )
            return payload.get("result", "(no output)")
        return f"unknown tool '{name}'"

    async def _run_followup_response(self) -> None:
        content = ""
        if self.state.streaming:
            async for chunk in self.client.chat_stream(self.state.model, self._llm_messages()):
                content += chunk
        else:
            content = await self.client.chat_once(self.state.model, self._llm_messages())
        self._finalize_assistant_response(content)

    def _llm_messages(self) -> list[dict[str, object]]:
        return [{"role": "system", "content": self.system_prompt}, *[m.as_chat_dict() for m in self.messages]]

    def _build_system_prompt(self) -> str:
        base = "You are a helpful assistant."
        persona = self._current_personality_prompt()
        base += (
            f" {persona} When creating new projects, use {self._project_root}/<project>."
            " Use exact names and paths requested by the user; do not invent or alter names."
            " If a request is ambiguous, choose the most likely interpretation, state the key"
            " assumption briefly, answer fully, and ask one short clarifying question."
            " Be natural, conversational, and helpful."
            " You can create new persistent tools at any time using the create_tool function."
            " Created tools run in an isolated Docker container and are available immediately"
            " in this session and all future sessions."
            " Use list_custom_tools to see what tools you have already created,"
            " and delete_custom_tool to remove ones you no longer need."
        )
        if self.state.concise_mode:
            return (
                base
                + " Respond with the final answer only. Be concise. No planning. No meta commentary."
                + " No <think>. No JSON/XML. Use short bullets when helpful."
                + " Max ~8-12 lines unless the user asks for more."
            )
        return (
            base
            + " Respond with the final answer only. Be clear, detailed, and more verbose when helpful."
            + " No <think>. Use short bullets when helpful. Include extra context and practical guidance."
            + " Ask a brief follow-up question when it makes sense."
        )

    def _write_transcript(self, speaker: str, text: str) -> None:
        scroll = self._safe_query_one("#transcript", VerticalScroll)
        if scroll is None:
            return
        role_class = "chat-user" if speaker == "You" else "chat-assistant"
        msg = ChatMessage(speaker, text or " ", classes=f"chat-msg {role_class}")
        scroll.mount(msg)
        self.call_after_refresh(scroll.scroll_end, animate=False)

    def _finalize_assistant_response(self, content: str) -> None:
        sanitized = sanitize_response(content)
        if sanitized.structured_hidden:
            display = format_structured(sanitized.text)
        else:
            display = sanitized.text.strip()
            if not display:
                display = "No response."
        self._write_transcript("Assistant", display)
        assistant = Message("assistant", display, full_content=content)
        self.messages.append(assistant)
        self.transcript_store.append(assistant)

    def _tool_log(self, message: str) -> None:
        log = self._safe_query_one("#toolpane", Log)
        if log is None:
            return
        log.write_line(message)

    def _log_tool_results(self, results: list[ToolResult]) -> None:
        for result in results:
            if result.ok:
                output = result.output or "(no output)"
            else:
                output = result.error or "Tool failed"
            snippet = output.strip()
            if result.call.name == "shell_exec":
                snippet = self._redact_secrets(snippet)
            if len(snippet) > 4000:
                snippet = snippet[:4000] + " ...[truncated]"
            self._tool_log(f"result [{result.call.name}] {snippet}")

    def _append_shell_output(self, results: list[ToolResult]) -> None:
        for result in results:
            if result.call.name != "shell_exec" or not result.ok:
                continue
            output = self._redact_secrets(result.output or "").strip()
            if not output:
                continue
            if len(output) > 4000:
                output = output[:4000] + " ...[truncated]"
            self._write_transcript("Shell", output)

    def _append_tool_messages(self, results: list[ToolResult]) -> None:
        for result in results:
            payload = result.output if result.ok else (result.error or "Tool failed")
            tool_msg = Message(
                "tool",
                payload[:4000],
                full_content=payload,
                metadata={"tool_call_id": result.call.call_id},
            )
            self.messages.append(tool_msg)
            self.transcript_store.append(tool_msg)

    def _notify_tool_failures(self, results: list[ToolResult]) -> None:
        failures = [res for res in results if not res.ok]
        if not failures:
            return
        summary = "; ".join(
            f"{res.call.name}: {(res.error or 'failed').splitlines()[0][:120]}" for res in failures[:3]
        )
        self._write_transcript("Assistant", f"Tool error: {summary}. See Tools panel.")

    async def _run_tool_batch(self, calls: list[ToolCall]) -> list[ToolResult]:
        if not calls:
            return []
        self._pending_tools = len(calls)
        self._refresh_sessions()
        scheduler = ToolScheduler(
            self._execute_tool_from_call,
            log=self._tool_log,
            concurrency=self.state.tool_concurrency,
        )
        try:
            return await scheduler.run_batch(calls)
        finally:
            self._pending_tools = 0
            self._refresh_sessions()

    async def _execute_tool_from_call(self, call: ToolCall) -> str:
        return await self._execute_tool_call(call.name, call.args)

    def _refresh_sessions(self) -> None:
        log = self._safe_query_one("#sessionpane", Log)
        if log is None:
            return
        log.clear()
        log.write_line(f"LLM busy: {'yes' if self.state.busy else 'no'}")
        log.write_line(f"Tool queue: {self._pending_tools}")
        sessions = self.tools.active_sessions()
        log.write_line("Shell sessions: " + (", ".join(sessions) if sessions else "none"))
        for note in self._session_notes[-5:]:
            log.write_line(note)

    def _log_session(self, message: str) -> None:
        self._session_notes.append(message)
        if len(self._session_notes) > 20:
            self._session_notes = self._session_notes[-20:]
        self._refresh_sessions()

    def _safe_query_one(self, selector: str, expect_type):
        try:
            return self.query_one(selector, expect_type)
        except NoMatches:
            return None

    def _start_new_chat(self, initial: bool = False) -> None:
        archived = self.transcript_store.archive_to(Path("/tmp/context"))
        if archived:
            self._log_session(f"Archived chat to {archived}")
        self.transcript_store.clear()
        self.messages = []
        scroll = self._safe_query_one("#transcript", VerticalScroll)
        if scroll is not None:
            scroll.remove_children()
        if not initial:
            self._write_transcript("Assistant", "New chat started.")

    def _clear_transcript(self) -> None:
        self.transcript_store.clear()
        self.messages = []
        scroll = self._safe_query_one("#transcript", VerticalScroll)
        if scroll is not None:
            scroll.remove_children()

    async def _rss_store_topic(self, topic: str) -> None:
        search_call = ToolCall(
            index=0,
            name="researchbox_search",
            args={"topic": topic},
            call_id="",
            label="researchbox_search",
        )
        search_results = await self._run_tool_batch([search_call])
        self._log_tool_results(search_results)
        if any(not res.ok for res in search_results):
            self._notify_tool_failures(search_results)
            return
        try:
            search_payload = json.loads(search_results[0].output)
        except (IndexError, json.JSONDecodeError):
            self._write_transcript("Assistant", "Search returned invalid data. See Tools panel.")
            return
        feeds = search_payload.get("feeds") if isinstance(search_payload, dict) else None
        if not isinstance(feeds, list) or not feeds:
            self._write_transcript(
                "Assistant",
                "No feeds found. Try https://rssfinder.app/ or https://rss.app/rss-feed to locate a feed URL.",
            )
            return
        push_calls: list[ToolCall] = []
        for index, feed_url in enumerate(feeds):
            if not isinstance(feed_url, str) or not feed_url.strip():
                continue
            push_calls.append(
                ToolCall(
                    index=index,
                    name="researchbox_push",
                    args={"feed_url": feed_url, "topic": topic},
                    call_id="",
                    label="researchbox_push",
                )
            )
        if not push_calls:
            self._write_transcript(
                "Assistant",
                "No valid feeds returned. Try https://rssfinder.app/ or https://rss.app/rss-feed.",
            )
            return
        push_results = await self._run_tool_batch(push_calls)
        self._log_tool_results(push_results)
        if any(not res.ok for res in push_results):
            self._notify_tool_failures(push_results)
            self._write_transcript(
                "Assistant",
                "If feeds fail to ingest, try https://rssfinder.app/ or https://rss.app/rss-feed and use /rss ingest.",
            )
            return
        self._write_transcript(
            "Assistant",
            f"Stored {len(push_calls)} feed(s) for '{topic}'. See Tools panel.",
        )

    async def _rss_ingest_modal(self) -> None:
        values = await self._show_modal(RssIngestModal())
        if not isinstance(values, dict) or not values:
            return
        topic = str(values.get("topic", "")).strip()
        feed_url = str(values.get("feed_url", "")).strip()
        if not topic or not feed_url:
            self._write_transcript("Assistant", "Usage: /rss ingest <topic> <feed_url>")
            return
        call = ToolCall(
            index=0,
            name="researchbox_push",
            args={"feed_url": feed_url, "topic": topic},
            call_id="",
            label="researchbox_push",
        )
        results = await self._run_tool_batch([call])
        self._log_tool_results(results)
        if any(not res.ok for res in results):
            self._notify_tool_failures(results)
            return
        self._write_transcript("Assistant", f"Stored feed for '{topic}'. See Tools panel.")

    async def _handle_personality_command(self, text: str) -> None:
        parts = text.split(maxsplit=2)
        if len(parts) == 1:
            await self.action_personality()
            return
        sub = parts[1].strip().lower()
        if sub == "list":
            names = ", ".join(p["name"] for p in self.personalities)
            self._write_transcript("Assistant", f"Personalities: {names}")
            return
        if sub == "add":
            await self._personality_add_modal()
            return
        await self._set_personality_by_query(" ".join(parts[1:]))

    async def _personality_add_modal(self) -> None:
        values = await self._show_modal(PersonalityAddModal())
        if not isinstance(values, dict) or not values:
            return
        name = str(values.get("name", "")).strip()
        prompt = str(values.get("prompt", "")).strip()
        if not name or not prompt:
            self._write_transcript("Assistant", "Personality name and prompt are required.")
            return
        new_id = self._unique_personality_id(name)
        self.personalities.append({"id": new_id, "name": name, "prompt": prompt})
        self.state.personality_id = new_id
        self.system_prompt = self._build_system_prompt()
        self._persist_config()
        self._write_transcript("Assistant", f"Personality added: {name}")

    def _unique_personality_id(self, name: str) -> str:
        base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "custom"
        existing = {p.get("id") for p in self.personalities if isinstance(p, dict)}
        candidate = base
        counter = 2
        while candidate in existing:
            candidate = f"{base}-{counter}"
            counter += 1
        return candidate

    async def _set_personality_by_query(self, query: str) -> None:
        key = query.strip().lower()
        if not key:
            return
        for persona in self.personalities:
            if persona.get("id", "").lower() == key or persona.get("name", "").lower() == key:
                self.state.personality_id = persona["id"]
                self.system_prompt = self._build_system_prompt()
                self._persist_config()
                self._write_transcript("Assistant", f"Personality set to {persona['name']}.")
                return
        self._write_transcript("Assistant", "Personality not found. Use /persona list.")

    async def _create_project(self, name: str) -> None:
        cleaned = name.strip()
        if not cleaned or cleaned in {".", ".."}:
            self._write_transcript("Assistant", "Invalid project name.")
            return
        if re.search(r"[\\\\/]", cleaned):
            self._write_transcript("Assistant", "Project name must not contain slashes.")
            return
        if not re.fullmatch(r"[A-Za-z0-9._ -]+", cleaned) or cleaned.strip(" ._-") == "":
            self._write_transcript(
                "Assistant",
                "Project name must use letters, numbers, spaces, dot, underscore, or dash.",
            )
            return
        root = self._project_root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        project_path = (root / cleaned).resolve()
        if not str(project_path).startswith(str(root)):
            self._write_transcript("Assistant", "Project path escapes the workspace.")
            return
        project_path.mkdir(parents=True, exist_ok=True)
        self.state.cwd = str(project_path)
        await self.update_status()
        self._log_session(f"Project set to {project_path}")
        self._write_transcript("Assistant", f"Project ready at {project_path}.")

    def _current_personality_prompt(self) -> str:
        for persona in self.personalities:
            if persona.get("id") == self.state.personality_id:
                return str(persona.get("prompt", "")).strip()
        return ""

    async def action_personality(self) -> None:
        options: list[tuple[str, str]] = []
        for persona in self.personalities:
            options.append((str(persona.get("name", "")), str(persona.get("id", ""))))
        options.append(("Add Custom...", "__add__"))

        async def _apply(selected: object) -> None:
            if selected == "__add__":
                await self._personality_add_modal()
                return
            if isinstance(selected, str) and selected:
                await self._set_personality_by_query(selected)

        self._push_modal(ChoiceModal("Choose Personality", options), _apply)

    def _persist_config(self) -> None:
        save_config(
            {
                "base_url": self.state.base_url,
                "model": self.state.model,
                "theme": self.state.theme,
                "approval": self.state.approval.value,
                "shell_enabled": self.state.shell_enabled,
                "concise_mode": self.state.concise_mode,
                "active_personality": self.state.personality_id,
                "personalities": merge_personalities(self.personalities),
                "project_root": str(self._project_root),
            }
        )

    async def _confirm_tool(self, tool_name: str) -> bool:
        if self.state.approval == ApprovalMode.AUTO:
            return True
        if self.state.approval == ApprovalMode.DENY:
            return False
        choice = await self._show_modal(ChoiceModal(f"Allow tool '{tool_name}'?", ["yes", "no"]))
        return choice == "yes"

    async def _toggle_concise(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) == 1:
            self._write_transcript(
                "Assistant",
                f"Concise mode is {'ON' if self.state.concise_mode else 'OFF'}.",
            )
            return
        value = parts[1].strip().lower()
        if value in {"on", "true", "yes"}:
            await self._set_concise(True)
            return
        if value in {"off", "false", "no"}:
            await self._set_concise(False)
            return
        self._write_transcript("Assistant", "Usage: /concise on|off")

    async def _set_concise(self, enabled: bool) -> None:
        self.state.concise_mode = enabled
        self.system_prompt = self._build_system_prompt()
        self._persist_config()
        await self.update_status()
        self._write_transcript("Assistant", f"Concise mode {'enabled' if enabled else 'disabled'}.")

    async def _handle_shell_command(self, text: str, *, allow_toggle: bool = True) -> None:
        self.tools.reset_turn()
        arg = text.strip()
        if allow_toggle and arg.lower() in {"on", "off"}:
            enabled = arg.lower() == "on"
            self.state.shell_enabled = enabled
            self._persist_config()
            await self.update_status()
            self._write_transcript("Assistant", f"Shell {'enabled' if enabled else 'disabled'}.")
            return
        if not arg:
            self._write_transcript(
                "Assistant",
                f"Shell is {'ON' if self.state.shell_enabled else 'OFF'}. Use /shell on|off.",
            )
            return
        if not self.state.shell_enabled:
            self._write_transcript("Assistant", "Shell is OFF. Use /shell on to enable.")
            return
        tool_log = self.query_one("#toolpane", Log)
        tool_log.write_line(f"shell> {arg}")
        start = time.monotonic()
        exit_code = 0
        output = ""

        def _on_output(chunk: str) -> None:
            redacted = self._redact_secrets(chunk)
            for line in redacted.splitlines():
                tool_log.write_line(line)

        try:
            exit_code, output, new_cwd = await self.tools.run_shell_stream(
                arg,
                self.state.approval,
                self._confirm_tool,
                cwd=self.state.cwd,
                on_output=_on_output,
            )
            if new_cwd:
                self.state.cwd = new_cwd
                await self.update_status()
        except ToolDeniedError as exc:
            tool_log.write_line(f"[denied] {exc}")
            self._write_transcript("Assistant", f"Shell denied: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            tool_log.write_line(f"[shell error] {exc}")
            self._write_transcript("Assistant", f"Shell error: {exc}")
            return
        elapsed = time.monotonic() - start
        self._log_session(f"Shell command executed (exit {exit_code}, {elapsed:.2f}s)")
        summary = f"Ran command: {arg} (exit {exit_code})."
        output_text = output.strip()
        if output_text:
            max_chars = 4000
            if len(output_text) > max_chars:
                output_text = output_text[:max_chars] + " ...[truncated]"
            self._write_transcript("Assistant", summary)
            self._write_transcript("Shell", output_text)
        else:
            self._write_transcript("Assistant", summary + " (no output)")

    def _redact_secrets(self, text: str) -> str:
        redacted = text
        for key, value in os.environ.items():
            if not value:
                continue
            upper = key.upper()
            if any(token in upper for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                if len(value) >= 4:
                    redacted = redacted.replace(value, "****")
        return redacted

    async def _load_default_model(self) -> None:
        try:
            models = await self.client.list_models()
        except LLMClientError as exc:
            self._log_session(f"Model load failed: {exc}")
            return
        if not models:
            return
        if self.state.model not in models:
            self.state.model = models[0]
            self._persist_config()
            await self.update_status()
            self._log_session(f"Default model set to {self.state.model}")

    async def action_toggle_shell(self) -> None:
        self.state.shell_enabled = not self.state.shell_enabled
        self._persist_config()
        await self.update_status()
        self._write_transcript(
            "Assistant",
            f"Shell {'enabled' if self.state.shell_enabled else 'disabled'}.",
        )

    async def action_scroll_up(self) -> None:
        scroll = self._safe_query_one("#transcript", VerticalScroll)
        if scroll is not None:
            scroll.scroll_page_up()

    async def action_scroll_down(self) -> None:
        scroll = self._safe_query_one("#transcript", VerticalScroll)
        if scroll is not None:
            scroll.scroll_page_down()

    def on_mouse_scroll_up(self, event) -> None:
        scroll = self._safe_query_one("#transcript", VerticalScroll)
        if scroll is not None:
            scroll.scroll_up(animate=False)

    def on_mouse_scroll_down(self, event) -> None:
        scroll = self._safe_query_one("#transcript", VerticalScroll)
        if scroll is not None:
            scroll.scroll_down(animate=False)

    async def action_cancel(self) -> None:
        if self.active_task and not self.active_task.done():
            self.active_task.cancel()

    async def action_cycle_approval(self) -> None:
        self.state.approval = self.state.approval.cycle()
        await self.update_status()

    async def action_theme_picker(self) -> None:
        def _apply(selected: object) -> asyncio.Future | None:
            if isinstance(selected, str) and selected in THEMES:
                self._apply_theme_with_fallback(selected)
                return self.update_status()
            return None

        self._push_modal(ChoiceModal("Choose Theme", list(THEMES)), _apply)

    async def action_toggle_streaming(self) -> None:
        self.state.streaming = not self.state.streaming
        await self.update_status()


    async def action_help(self) -> None:
        self.notify(render_keybinds())

    async def action_pick_model(self) -> None:
        try:
            models = await self.client.list_models()
        except LLMClientError as exc:
            self.notify(str(exc), severity="error")
            return
        options = model_options(models or [self.state.model])
        if not options:
            options = [(self.state.model, self.state.model)]
        async def _apply(selected: object) -> None:
            if isinstance(selected, str) and selected:
                try:
                    await self.client.ensure_model(selected)
                except LLMClientError as exc:
                    self.notify(str(exc), severity="error")
                    return
                self.state.model = selected
                self._persist_config()
                await self.update_status()

        self._push_modal(ChoiceModal("Pick Model", options), _apply)

    async def action_search(self) -> None:
        def _apply(query: object) -> None:
            if not isinstance(query, str) or not query:
                return
            hits = self.transcript_store.search(query)
            log = self.query_one("#toolpane", Log)
            log.clear()
            for hit in hits[:20]:
                log.write_line(f"[{hit['timestamp']}] {hit['role']}: {hit['content'][:120]}")

        self._push_modal(SearchModal(), _apply)

    async def action_sessions(self) -> None:
        self._refresh_sessions()

    async def action_new_chat(self) -> None:
        self._start_new_chat()
        await self.update_status()

    async def action_clear_transcript(self) -> None:
        self._clear_transcript()
        await self.update_status()

    async def action_settings(self) -> None:
        models = await self.client.list_models() if await self.client.health() else [self.state.model]
        if self.state.model not in models:
            models = [self.state.model, *models]
        async def _apply(values: object) -> None:
            if not isinstance(values, dict) or not values:
                return
            selected_model = str(values["model"])
            new_url = str(values.get("base_url", self.state.base_url)).strip() or self.state.base_url
            # Build the client against the *new* URL so health/ensure_model checks use it.
            new_client = LLMClient(new_url)
            if await new_client.health():
                try:
                    await new_client.ensure_model(selected_model)
                except LLMClientError as exc:
                    self.notify(str(exc), severity="warning")
                    # Don't abort — user may have intentionally switched endpoints.
            self.state.base_url = new_url
            self.state.model = selected_model
            self.state.approval = ApprovalMode(str(values["approval"]))
            self.state.shell_enabled = bool(values.get("shell_enabled", self.state.shell_enabled))
            self.state.concise_mode = bool(values.get("concise_mode", self.state.concise_mode))
            self.system_prompt = self._build_system_prompt()
            self.client = new_client
            self._apply_theme_with_fallback(str(values["theme"]))
            self._persist_config()
            await self.update_status()

        self._push_modal(
            SettingsModal(
                current={
                    "base_url": self.state.base_url,
                    "model": self.state.model,
                    "theme": self.state.theme,
                    "approval": self.state.approval.value,
                    "shell_enabled": self.state.shell_enabled,
                    "concise_mode": self.state.concise_mode,
                },
                models=models,
                themes=list(THEMES),
            ),
            _apply,
        )


    async def action_copy_last(self) -> None:
        for message in reversed(self.messages):
            if message.role == "assistant":
                self.copy_to_clipboard(message.full_content or message.content)
                self.notify("Copied last assistant message")
                return
        self.notify("No assistant message to copy", severity="warning")

    async def action_export_chat(self) -> None:
        out = Path.home() / ".local" / "share" / "aichat" / f"export-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
        self.transcript_store.export_markdown(out)
        self.notify(f"exported to {out}")


def main() -> None:
    AIChatApp().run()


if __name__ == "__main__":
    main()
