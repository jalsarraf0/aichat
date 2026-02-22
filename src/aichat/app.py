from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import datetime
from importlib.resources import as_file, files
from pathlib import Path

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Log, Static, TextArea

from .client import LLMClient, LLMClientError
from .config import LM_STUDIO_BASE_URL, load_config, save_config
from .state import AppState, ApprovalMode, Message
from .themes import THEMES
from .tools.manager import ToolDeniedError, ToolManager
from .transcript import TranscriptStore
from .ui.modals import ChoiceModal, SearchModal, SettingsModal


class AIChatApp(App):
    BINDINGS = [
        Binding("f1", "help", "Help", priority=True),
        Binding("f2", "pick_model", "Model", priority=True),
        Binding("f3", "search", "Search", priority=True),
        Binding("f4", "cycle_approval", "Approval", priority=True),
        Binding("f5", "theme_picker", "Theme", priority=True),
        Binding("f6", "toggle_streaming", "Streaming", priority=True),
        Binding("f7", "sessions", "Sessions", priority=True),
        Binding("f8", "settings", "Settings", priority=True),
        Binding("f9", "copy_last", "Copy", priority=True),
        Binding("f10", "export_chat", "Export", priority=True),
        Binding("f11", "cancel", "Cancel", priority=True),
        Binding("f12", "quit", "Quit", priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        cfg = load_config()
        self.state = AppState(
            model=cfg["model"],
            base_url=cfg["base_url"],
            theme=cfg["theme"],
            approval=ApprovalMode(cfg["approval"]),
            allow_host_shell=cfg.get("allow_host_shell", True),
            cwd=str(Path.cwd()),
        )
        self.client = LLMClient(self.state.base_url)
        self.tools = ToolManager(max_tool_calls_per_turn=self.state.max_tool_calls_per_turn)
        self.transcript_store = TranscriptStore()
        self.messages = self.transcript_store.load_messages()
        self.active_task: asyncio.Task[None] | None = None
        self._stream_line = ""
        self._loaded_theme_sources: set[str] = set()
        self.system_prompt = (
            "You are a senior ultra gigabrain Linux engineer with deep Docker, Python, Rust, "
            "and multi-language expertise. Provide concise, correct, production-grade guidance."
        )

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
                yield TextArea(placeholder="Ask anything. Commands: /shell /rss /researchbox", id="prompt")
        yield Footer()

    async def on_mount(self) -> None:
        self._apply_theme_with_fallback(self.state.theme)
        transcript = self.query_one("#transcript", Log)
        for message in self.messages[-100:]:
            transcript.write_line(self._format_transcript_line(message))
        self.set_focus(self.query_one("#prompt", TextArea))
        await self.update_status()

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
            + f" | shell={'ON' if self.state.allow_host_shell else 'OFF'}"
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
        self.tools.reset_turn()
        user = Message("user", text)
        self.messages.append(user)
        self.transcript_store.append(user)
        self.query_one("#transcript", Log).write_line(self._format_transcript_line(user))
        self.active_task = asyncio.create_task(self.run_llm_turn())
        await self.active_task

    async def run_llm_turn(self) -> None:
        self.state.busy = True
        transcript = self.query_one("#transcript", Log)
        content = ""
        tools = self.tools.tool_definitions(self.state.allow_host_shell)
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
                if not self.state.streaming:
                    transcript.write_line(f"assistant> {content}")
                assistant = Message("assistant", content[:4000], full_content=content)
                self.messages.append(assistant)
                self.transcript_store.append(assistant)
        except asyncio.CancelledError:
            transcript.write_line("[stream cancelled]")
            raise
        except LLMClientError as exc:
            transcript.write_line(f"[llm error] {exc}")
        finally:
            self.state.busy = False
            self.active_task = None
            await self.update_status()

    async def handle_command(self, text: str) -> None:
        self.tools.reset_turn()
        log = self.query_one("#toolpane", Log)
        try:
            if text.startswith("/shell "):
                output = await self.tools.run_shell(text[7:], self.state.approval, self._confirm_tool, cwd=self.state.cwd)
                log.write_line(output or "(no output)")
            elif text.startswith("/rss "):
                payload = await self.tools.run_rss(text[5:], self.state.approval, self._confirm_tool)
                log.write_line(str(payload))
            elif text.startswith("/researchbox "):
                payload = await self.tools.run_researchbox(text[13:], self.state.approval, self._confirm_tool)
                log.write_line(str(payload))
            else:
                self.notify("Unknown command")
        except ToolDeniedError as exc:
            log.write_line(f"[denied] {exc}")
        except Exception as exc:
            log.write_line(f"[tool error] {exc}")


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
        transcript = self.query_one("#transcript", Log)
        content = ""
        self._stream_line = f"{self._model_tag()} "
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
                self._stream_line += chunk
                if "\n" in self._stream_line:
                    for line in self._stream_line.splitlines()[:-1]:
                        transcript.write_line(line)
                    self._stream_line = self._stream_line.splitlines()[-1]
            elif event.get("type") == "tool_calls":
                deltas = event.get("value", [])
                if isinstance(deltas, list):
                    self._merge_tool_call_deltas(tool_call_state, deltas)
        if self._stream_line:
            transcript.write_line(self._stream_line.rstrip())
            self._stream_line = ""
        tool_calls = [tool_call_state[idx] for idx in sorted(tool_call_state)]
        return content, tool_calls

    async def _handle_tool_calls(self, tool_calls: list[dict[str, object]]) -> None:
        if not tool_calls:
            return
        log = self.query_one("#toolpane", Log)
        assistant = Message("assistant", "", full_content="", metadata={"tool_calls": tool_calls})
        self.messages.append(assistant)
        self.transcript_store.append(assistant)
        for call in tool_calls[: self.state.max_tool_calls_per_turn]:
            name = ""
            args_text = ""
            call_id = ""
            if isinstance(call.get("function"), dict):
                name = str(call["function"].get("name", ""))
                args_text = str(call["function"].get("arguments", ""))
            call_id = str(call.get("id", "")) if call.get("id") else ""
            result_text = await self._execute_tool_call(name, args_text)
            log.write_line(f"[tool:{name}] {result_text[:4000]}")
            tool_msg = Message("tool", result_text[:4000], full_content=result_text, metadata={"tool_call_id": call_id})
            self.messages.append(tool_msg)
            self.transcript_store.append(tool_msg)
        await self._run_followup_response()

    async def _execute_tool_call(self, name: str, args_text: str) -> str:
        log = self.query_one("#toolpane", Log)
        try:
            args = json.loads(args_text) if args_text else {}
        except json.JSONDecodeError as exc:
            error = f"invalid tool arguments: {exc}"
            log.write_line(f"[tool error] {error}")
            return error
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
        if name == "shell_exec":
            command = str(args.get("command", "")).strip()
            if not command:
                return "shell_exec: missing 'command'"
            output = await self.tools.run_shell(command, self.state.approval, self._confirm_tool, cwd=self.state.cwd)
            return output or "(no output)"
        return f"unknown tool '{name}'"

    async def _run_followup_response(self) -> None:
        transcript = self.query_one("#transcript", Log)
        content = ""
        self._stream_line = f"{self._model_tag()} "
        if self.state.streaming:
            async for chunk in self.client.chat_stream(self.state.model, self._llm_messages()):
                content += chunk
                self._stream_line += chunk
                if "\n" in self._stream_line:
                    for line in self._stream_line.splitlines()[:-1]:
                        transcript.write_line(line)
                    self._stream_line = self._stream_line.splitlines()[-1]
            if self._stream_line:
                transcript.write_line(self._stream_line.rstrip())
                self._stream_line = ""
        else:
            content = await self.client.chat_once(self.state.model, self._llm_messages())
            transcript.write_line(f"{self._model_tag()} {content}")
        assistant = Message("assistant", content[:4000], full_content=content)
        self.messages.append(assistant)
        self.transcript_store.append(assistant)

    def _model_tag(self) -> str:
        name = self.state.model.strip() if self.state.model else "Model"
        return f"<{name}>"

    def _format_transcript_line(self, message: Message) -> str:
        if message.role == "user":
            tag = "<User>"
        elif message.role == "assistant":
            tag = self._model_tag()
            if not message.content and "tool_calls" in message.metadata:
                return f"{tag} [tool call]"
        elif message.role == "tool":
            tag = "<Tool>"
        else:
            tag = f"<{message.role}>"
        return f"{tag} {message.content}".rstrip()

    def _llm_messages(self) -> list[dict[str, object]]:
        return [{"role": "system", "content": self.system_prompt}, *[m.as_chat_dict() for m in self.messages]]
    async def _confirm_tool(self, tool_name: str) -> bool:
        if self.state.approval == ApprovalMode.AUTO:
            return True
        if self.state.approval == ApprovalMode.DENY:
            return False
        choice = await self._show_modal(ChoiceModal(f"Allow tool '{tool_name}'?", ["yes", "no"]))
        return choice == "yes"

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
        self.notify(
            "F1 Help | F2 Model | F3 Search | F4 Approval | F5 Theme | F6 Streaming | F7 Sessions | F8 Settings | F9 Copy | F10 Export | F11 Cancel | F12 Quit"
        )

    async def action_pick_model(self) -> None:
        try:
            models = await self.client.list_models()
        except LLMClientError as exc:
            self.notify(str(exc), severity="error")
            return
        def _apply(selected: object) -> asyncio.Future | None:
            if isinstance(selected, str) and selected:
                self.state.model = selected
                return self.update_status()
            return None

        self._push_modal(ChoiceModal("Pick Model", models), _apply)

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
        sessions = self.tools.active_sessions()
        log = self.query_one("#sessionpane", Log)
        log.write_line("active sessions: " + (", ".join(sessions) if sessions else "none"))

    async def action_settings(self) -> None:
        models = await self.client.list_models() if await self.client.health() else [self.state.model]
        def _apply(values: object) -> asyncio.Future | None:
            if not isinstance(values, dict) or not values:
                return None
            self.state.base_url = LM_STUDIO_BASE_URL
            self.state.model = str(values["model"])
            self.state.approval = ApprovalMode(str(values["approval"]))
            self.state.allow_host_shell = bool(values.get("allow_host_shell", self.state.allow_host_shell))
            self.client = LLMClient(self.state.base_url)
            self._apply_theme_with_fallback(str(values["theme"]))
            save_config(
                {
                    "base_url": self.state.base_url,
                    "model": self.state.model,
                    "theme": self.state.theme,
                    "approval": self.state.approval.value,
                    "allow_host_shell": self.state.allow_host_shell,
                }
            )
            return self.update_status()

        self._push_modal(
            SettingsModal(
                current={
                    "base_url": self.state.base_url,
                    "model": self.state.model,
                    "theme": self.state.theme,
                    "approval": self.state.approval.value,
                    "allow_host_shell": self.state.allow_host_shell,
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

    async def on_key(self, event: events.Key) -> None:
        if (
            event.key == "enter"
            and not event.shift
            and self.focused
            and self.focused.id == "prompt"
        ):
            await self.action_send()
            event.stop()


def main() -> None:
    AIChatApp().run()


if __name__ == "__main__":
    main()
