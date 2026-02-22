from __future__ import annotations

import asyncio
from pathlib import Path

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Log, Static

from .client import LLMClient
from .config import load_config, save_config
from .state import AppState, ApprovalMode, Message
from .themes import THEMES
from .tools.manager import ToolManager
from .transcript import TranscriptStore
from .ui.modals import ThemePicker


class AIChatApp(App):
    CSS = """
    #status { height: 1; }
    #main { height: 1fr; }
    #toolpane { width: 40%; border: round #666; }
    #transcript { width: 60%; border: round #666; }
    """

    BINDINGS = [
        Binding("enter", "send", "Send", priority=True),
        Binding("shift+enter", "newline", "Newline", priority=True),
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+o", "command_palette", "Palette", priority=True),
        Binding("ctrl+i", "theme_picker", "Theme", priority=True),
        Binding("ctrl+a", "cycle_approval", "Approval", priority=True),
        Binding("f1", "help", "Help", priority=True),
        Binding("f2", "pick_model", "Model", priority=True),
        Binding("f3", "search", "Search", priority=True),
        Binding("f4", "toggle_auto", "Auto", priority=True),
        Binding("f5", "refresh_models", "Refresh", priority=True),
        Binding("f6", "toggle_streaming", "Streaming", priority=True),
        Binding("f7", "sessions", "Sessions", priority=True),
        Binding("f8", "settings", "Settings", priority=True),
        Binding("f9", "copy_last", "Copy", priority=True),
        Binding("f10", "export_chat", "Export", priority=True),
        Binding("f11", "focus_mode", "Focus", priority=True),
        Binding("f12", "debug", "Debug", priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        cfg = load_config()
        self.state = AppState(
            model=cfg["model"],
            base_url=cfg["base_url"],
            theme=cfg["theme"],
            approval=ApprovalMode(cfg["approval"]),
        )
        self.client = LLMClient(self.state.base_url)
        self.tools = ToolManager()
        self.transcript_store = TranscriptStore()
        self.messages: list[Message] = []
        self.active_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="status")
        with Horizontal(id="main"):
            yield Log(id="transcript", auto_scroll=True)
            yield Log(id="toolpane", auto_scroll=True)
        yield Input(placeholder="Type message, /research, /operator, /continue", id="prompt")
        yield Footer()

    async def on_mount(self) -> None:
        self.apply_theme(self.state.theme)
        await self.update_status()

    async def update_status(self) -> None:
        up = await self.client.health()
        self.query_one("#status", Static).update(
            f"MODEL:{self.state.model} | BASE:{self.state.base_url} ({'UP' if up else 'DOWN'}) | "
            f"APPROVAL:{self.state.approval.value} | MODE:{self.state.mode} | THEME:{self.state.theme}"
        )

    def apply_theme(self, name: str) -> None:
        self.state.theme = name
        self.stylesheet.read_all([THEMES.get(name, THEMES["cyberpunk"])])

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.run_worker(self.handle_submit(event.value), exclusive=True)
        event.input.value = ""

    async def handle_submit(self, text: str) -> None:
        if not text.strip():
            return
        if text.startswith("/"):
            await self.handle_command(text)
            return
        user = Message("user", text)
        self.messages.append(user)
        self.transcript_store.append(user)
        self.query_one("#transcript", Log).write_line(f"You: {text}")
        await self.run_llm_turn()

    async def run_llm_turn(self) -> None:
        self.state.busy = True
        content = ""
        self.active_task = asyncio.current_task()
        try:
            if self.state.streaming:
                async for chunk in self.client.chat_stream(
                    self.state.model,
                    [{"role": m.role, "content": m.content} for m in self.messages],
                ):
                    content += chunk + "\n"
                    self.query_one("#transcript", Log).write_line(chunk)
            else:
                content = await self.client.chat_once(
                    self.state.model,
                    [{"role": m.role, "content": m.content} for m in self.messages],
                )
            assistant = Message("assistant", content[:2000], full_content=content)
            self.messages.append(assistant)
            self.transcript_store.append(assistant)
        except asyncio.CancelledError:
            self.query_one("#transcript", Log).write_line("Canceled")
        except Exception as exc:
            self.query_one("#transcript", Log).write_line(f"Error: {exc}")
        finally:
            self.state.busy = False
            self.active_task = None
            await self.update_status()

    async def handle_command(self, text: str) -> None:
        if text in ("/research",):
            self.state.mode = "research"
        elif text in ("/operator", "/code"):
            self.state.mode = "operator"
        elif text == "/continue":
            await self.run_llm_turn()
        self.query_one("#transcript", Log).write_line(f"Mode -> {self.state.mode}")
        await self.update_status()

    async def action_send(self) -> None:
        pass

    async def action_newline(self) -> None:
        prompt = self.query_one("#prompt", Input)
        prompt.insert_text("\n")

    async def action_cancel(self) -> None:
        if self.active_task:
            self.active_task.cancel()
        for sid in list(self.tools.shell.sessions.keys()):
            await self.tools.shell.sh_interrupt(sid)
        self.query_one("#transcript", Log).write_line("Canceled")

    async def action_theme_picker(self) -> None:
        def done(theme: str) -> None:
            if theme:
                self.apply_theme(theme)
                cfg = load_config()
                cfg["theme"] = theme
                save_config(cfg)
                self.run_worker(self.update_status())

        self.push_screen(ThemePicker(list(THEMES.keys())), done)

    async def action_cycle_approval(self) -> None:
        self.state.approval = self.state.approval.cycle()
        cfg = load_config()
        cfg["approval"] = self.state.approval.value
        save_config(cfg)
        await self.update_status()

    async def action_toggle_streaming(self) -> None:
        self.state.streaming = not self.state.streaming
        await self.update_status()

    async def action_export_chat(self) -> None:
        out = Path.cwd() / "chat-export.md"
        self.transcript_store.export_markdown(out)
        self.notify(f"Exported: {out}")

    async def action_copy_last(self) -> None:
        for m in reversed(self.messages):
            if m.role == "assistant":
                self.app.copy_to_clipboard(m.full_content or m.content)
                self.notify("Copied")
                return

    async def action_help(self) -> None:
        self.notify("F1..F12 and Ctrl+O/Ctrl+I/Ctrl+A available")

    async def action_pick_model(self) -> None:
        self.notify("Model picker TODO")

    async def action_search(self) -> None:
        self.notify("Transcript search TODO")

    async def action_toggle_auto(self) -> None:
        self.state.approval = ApprovalMode.AUTO if self.state.approval != ApprovalMode.AUTO else ApprovalMode.ASK
        await self.update_status()

    async def action_refresh_models(self) -> None:
        await self.update_status()

    async def action_sessions(self) -> None:
        self.notify("Sessions UI TODO")

    async def action_settings(self) -> None:
        self.notify("Settings modal TODO")

    async def action_focus_mode(self) -> None:
        self.notify("Focus mode toggled")

    async def action_debug(self) -> None:
        self.notify(f"busy={self.state.busy} active_task={self.active_task is not None}")

    async def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            await self.action_cancel()
            event.stop()


def run() -> None:
    AIChatApp().run()


if __name__ == "__main__":
    run()
