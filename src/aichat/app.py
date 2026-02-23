from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
import time
from collections.abc import Callable
from datetime import datetime
from importlib.resources import as_file, files
from pathlib import Path

import yaml
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Log, Static, TextArea


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
from .config import LM_STUDIO_BASE_URL, load_config, save_config
from .model_labels import model_options
from .personalities import DEFAULT_PERSONALITY_ID, merge_personalities
from .sanitizer import sanitize_response
from .state import AppState, ApprovalMode, Message
from .themes import THEMES
from .tool_scheduler import ToolCall, ToolResult, ToolScheduler
from .tools.manager import ToolDeniedError, ToolManager
from .transcript import TranscriptStore
from .ui.keybind_bar import KeybindBar
from .ui.keybinds import binding_list, render_keybinds
from .ui.modals import ChoiceModal, PersonalityAddModal, RssIngestModal, SearchModal, SettingsModal


class AIChatApp(App):
    BINDINGS = binding_list() + [
        Binding("pageup", "scroll_up", "ScrollUp", priority=True),
        Binding("pagedown", "scroll_down", "ScrollDown", priority=True),
        Binding("ctrl+;", "personality", "Persona", show=False, priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._project_root = Path("~/git")
        cfg = load_config()
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
                    yield Log(id="transcript", auto_scroll=True)
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
        transcript = self.query_one("#transcript", Log)
        transcript.clear()
        self._start_new_chat(initial=True)
        self.set_focus(self.query_one("#prompt", TextArea))
        await self.update_status()
        self._refresh_sessions()
        asyncio.create_task(self._load_default_model())

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
        self.query_one("#model-line", Static).update(
            f"model={self.state.model} | base={self.state.base_url} | server={'UP' if up else 'DOWN'}"
        )
        self.query_one("#status", Static).update(
            "stream="
            + ("ON" if self.state.streaming else "OFF")
            + f" | approval={self.state.approval.value}"
            + f" | concise={'ON' if self.state.concise_mode else 'OFF'}"
            + f" | shell={'ON' if self.state.shell_enabled else 'OFF'}"
            + f" | persona={self.state.personality_id}"
            + f" | cwd={self.state.cwd}"
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
            asyncio.create_task(self._handle_shell_command(text[1:]))
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
        try:
            if self.state.streaming:
                content, tool_calls = await self._stream_with_tools(tools)
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
                await self._handle_shell_command(text[6:].strip())
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
            self._write_transcript("Assistant", "Unknown command.")
        except ToolDeniedError as exc:
            self._tool_log(f"[denied] {exc}")
        except Exception as exc:
            self._tool_log(f"[tool error] {exc}")


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

    async def _stream_with_tools(self, tools: list[dict[str, object]]) -> tuple[str, list[dict[str, object]]]:
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
            args, error = self._parse_tool_args(name, args_text)
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
            output = await self.tools.run_shell(command, self.state.approval, self._confirm_tool, cwd=self.state.cwd)
            return output or "(no output)"
        return f"unknown tool '{name}'"

    def _parse_tool_args(self, name: str, args_text: str) -> tuple[dict[str, object], str | None]:
        if not args_text:
            return {}, None
        cleaned = args_text.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.splitlines()[1:-1]).strip()
        parsed: object
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                parsed = yaml.safe_load(cleaned)
            except Exception as exc:  # noqa: BLE001
                if name == "shell_exec" and cleaned:
                    return {"command": cleaned}, None
                return {}, f"invalid tool arguments: {exc}"
        if isinstance(parsed, dict):
            return parsed, None
        if name == "shell_exec" and isinstance(parsed, str) and parsed.strip():
            return {"command": parsed.strip()}, None
        if name == "shell_exec" and cleaned:
            return {"command": cleaned}, None
        return {}, "invalid tool arguments: expected object payload"

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
        base += f" {persona} When creating new projects, use ~/git/<project>."
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
        log = self.query_one("#transcript", Log)
        raw_lines = (text or "").splitlines() or [""]
        prefix = f"{speaker}:"
        pad = " " * (len(prefix) + 1)
        width = max(log.size.width - len(prefix) - 1, 20)
        first = True
        for raw in raw_lines:
            wrapped = textwrap.wrap(raw, width=width) or [""]
            for segment in wrapped:
                if first:
                    log.write_line(f"{prefix} {segment}".rstrip())
                    first = False
                else:
                    log.write_line(f"{pad}{segment}".rstrip())

    def _finalize_assistant_response(self, content: str) -> None:
        sanitized = sanitize_response(content)
        if sanitized.structured_hidden:
            display = "Output contained structured data. See Tools panel for details."
        else:
            display = sanitized.text.strip()
            if not display:
                display = "No response."
        self._write_transcript("Assistant", display)
        assistant = Message("assistant", display, full_content=content)
        self.messages.append(assistant)
        self.transcript_store.append(assistant)

    def _tool_log(self, message: str) -> None:
        self.query_one("#toolpane", Log).write_line(message)

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
        log = self.query_one("#sessionpane", Log)
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

    def _start_new_chat(self, initial: bool = False) -> None:
        archived = self.transcript_store.archive_to(Path("/tmp/context"))
        if archived:
            self._log_session(f"Archived chat to {archived}")
        self.transcript_store.clear()
        self.messages = []
        self.query_one("#transcript", Log).clear()
        if not initial:
            self._write_transcript("Assistant", "New chat started.")

    def _clear_transcript(self) -> None:
        self.transcript_store.clear()
        self.messages = []
        self.query_one("#transcript", Log).clear()

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
        if not re.fullmatch(r"[A-Za-z0-9._-]+", cleaned):
            self._write_transcript(
                "Assistant",
                "Project name must use letters, numbers, dot, underscore, or dash.",
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

    async def _handle_shell_command(self, text: str) -> None:
        self.tools.reset_turn()
        arg = text.strip()
        if arg.lower() in {"on", "off"}:
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
            exit_code, output = await self.tools.run_shell_stream(
                arg,
                self.state.approval,
                self._confirm_tool,
                cwd=self.state.cwd,
                on_output=_on_output,
            )
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
        log = self.query_one("#transcript", Log)
        log.auto_scroll = False
        log.scroll_page_up()

    async def action_scroll_down(self) -> None:
        log = self.query_one("#transcript", Log)
        log.scroll_page_down()
        if log.is_vertical_scroll_end:
            log.auto_scroll = True

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
            if await self.client.health():
                try:
                    await self.client.ensure_model(selected_model)
                except LLMClientError as exc:
                    self.notify(str(exc), severity="error")
                    return
            self.state.base_url = LM_STUDIO_BASE_URL
            self.state.model = selected_model
            self.state.approval = ApprovalMode(str(values["approval"]))
            self.state.shell_enabled = bool(values.get("shell_enabled", self.state.shell_enabled))
            self.state.concise_mode = bool(values.get("concise_mode", self.state.concise_mode))
            self.system_prompt = self._build_system_prompt()
            self.client = LLMClient(self.state.base_url)
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
