from __future__ import annotations

import asyncio
from datetime import datetime
from importlib.resources import as_file, files
from pathlib import Path

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Log, Static

from .client import LLMClient, LLMClientError
from .config import LM_STUDIO_BASE_URL, load_config, save_config
from .state import AppState, ApprovalMode, Message
from .themes import THEMES
from .tools.manager import ToolDeniedError, ToolManager
from .transcript import TranscriptStore
from .ui.modals import ChoiceModal, SearchModal, SettingsModal


class AIChatApp(App):
    BINDINGS = [
        Binding("enter", "send", "Send", priority=True),
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
            cwd=str(Path.cwd()),
        )
        self.client = LLMClient(self.state.base_url)
        self.tools = ToolManager(max_tool_calls_per_turn=self.state.max_tool_calls_per_turn)
        self.transcript_store = TranscriptStore()
        self.messages = self.transcript_store.load_messages()
        self.active_task: asyncio.Task[None] | None = None
        self._stream_line = ""
        self._loaded_theme_sources: set[str] = set()

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="topbar"):
            yield Static("", id="model-line")
            yield Static("", id="status")
        with Horizontal(id="main"):
            yield Log(id="transcript", auto_scroll=True)
            yield Log(id="toolpane", auto_scroll=True)
        yield Input(placeholder="Ask anything. Commands: /shell /rss /researchbox", id="prompt")
        yield Footer()

    async def on_mount(self) -> None:
        self._apply_theme_with_fallback(self.state.theme)
        transcript = self.query_one("#transcript", Log)
        for message in self.messages[-100:]:
            transcript.write_line(f"{message.role}> {message.content}")
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
            f"stream={'ON' if self.state.streaming else 'OFF'} | approval={self.state.approval.value} | cwd={self.state.cwd}"
        )

    async def action_send(self) -> None:
        prompt = self.query_one("#prompt", Input)
        text = prompt.value.strip()
        prompt.value = ""
        if not text:
            return
        await self.handle_submit(text)

    async def handle_submit(self, text: str) -> None:
        if text.startswith("/"):
            await self.handle_command(text)
            return
        self.tools.reset_turn()
        user = Message("user", text)
        self.messages.append(user)
        self.transcript_store.append(user)
        self.query_one("#transcript", Log).write_line(f"user> {text}")
        self.active_task = asyncio.create_task(self.run_llm_turn())
        await self.active_task

    async def run_llm_turn(self) -> None:
        self.state.busy = True
        transcript = self.query_one("#transcript", Log)
        content = ""
        try:
            if self.state.streaming:
                transcript.write_line("assistant> ")
                async for chunk in self.client.chat_stream(self.state.model, [m.as_chat_dict() for m in self.messages]):
                    content += chunk
                    self._stream_line += chunk
                    if "\n" in self._stream_line:
                        for line in self._stream_line.splitlines()[:-1]:
                            transcript.write_line(line)
                        self._stream_line = self._stream_line.splitlines()[-1]
                if self._stream_line:
                    transcript.write_line(self._stream_line)
                    self._stream_line = ""
            else:
                content = await self.client.chat_once(self.state.model, [m.as_chat_dict() for m in self.messages])
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
        selected = await self._show_modal(ChoiceModal("Choose Theme", list(THEMES)))
        if selected in THEMES:
            self._apply_theme_with_fallback(selected)
            await self.update_status()

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
        selected = await self._show_modal(ChoiceModal("Pick Model", models))
        if selected:
            self.state.model = selected
            await self.update_status()

    async def action_search(self) -> None:
        query = await self._show_modal(SearchModal())
        if not query:
            return
        hits = self.transcript_store.search(query)
        log = self.query_one("#toolpane", Log)
        log.clear()
        for hit in hits[:20]:
            log.write_line(f"[{hit['timestamp']}] {hit['role']}: {hit['content'][:120]}")

    async def action_sessions(self) -> None:
        sessions = self.tools.active_sessions()
        log = self.query_one("#toolpane", Log)
        log.write_line("active sessions: " + (", ".join(sessions) if sessions else "none"))

    async def action_settings(self) -> None:
        models = await self.client.list_models() if await self.client.health() else [self.state.model]
        values = await self._show_modal(
            SettingsModal(
                current={
                    "base_url": self.state.base_url,
                    "model": self.state.model,
                    "theme": self.state.theme,
                    "approval": self.state.approval.value,
                },
                models=models,
                themes=list(THEMES),
            )
        )
        if not values:
            return
        self.state.base_url = LM_STUDIO_BASE_URL
        self.state.model = values["model"]
        self.state.approval = ApprovalMode(values["approval"])
        self.client = LLMClient(self.state.base_url)
        self._apply_theme_with_fallback(values["theme"])
        save_config(
            {
                "base_url": self.state.base_url,
                "model": self.state.model,
                "theme": self.state.theme,
                "approval": self.state.approval.value,
                "allow_host_shell": False,
            }
        )
        await self.update_status()


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
        if event.key == "enter" and self.focused and self.focused.id == "prompt":
            await self.action_send()
            event.stop()


def main() -> None:
    AIChatApp().run()


if __name__ == "__main__":
    main()
