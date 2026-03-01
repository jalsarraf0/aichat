from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid as _uuid
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
from .ui.modals import ChoiceModal, PersonalityAddModal, SearchModal, SettingsModal

# ---------------------------------------------------------------------------
# Contextual compaction constants
# ---------------------------------------------------------------------------
_COMPACT_MIN_MSGS: int = 8        # min visible messages before auto-compact triggers
_COMPACT_KEEP_RATIO: float = 0.5  # compact oldest 50%, keep newest 50%


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
        self._context_length: int = int(cfg.get("context_length", 35063))
        self._max_response_tokens: int = int(cfg.get("max_response_tokens", 4096))
        self._compact_threshold_pct: int = int(cfg.get("compact_threshold_pct", 95))
        self._compact_min_msgs: int = int(cfg.get("compact_min_msgs", 8))
        self._compact_keep_ratio: float = float(cfg.get("compact_keep_ratio", 0.5))
        self._compact_tool_turns: bool = bool(cfg.get("compact_tool_turns", True))
        self.state.compaction_enabled = bool(cfg.get("compaction_enabled", True))
        self.client = LLMClient(self.state.base_url)
        self.tools = ToolManager(max_tool_calls_per_turn=self.state.max_tool_calls_per_turn)
        self.transcript_store = TranscriptStore()
        self.messages = self.transcript_store.load_messages()
        self.active_task: asyncio.Task[None] | None = None
        self._loaded_theme_sources: set[str] = set()
        self.system_prompt = self._build_system_prompt()
        self._session_notes: list[str] = []
        self._pending_tools: int = 0
        self._last_status_ts: float = 0.0
        self._last_refresh_ts: float = 0.0
        self._rag_context_query: str = ""
        self._rag_context_cache: str = ""
        self._compact_summary: str = ""    # LLM-generated summary of compacted turns
        self._compact_from_idx: int = 0    # messages[:idx] are covered by _compact_summary
        self._compact_pending: bool = False
        self._compact_model: str = str(cfg.get("compact_model", ""))
        self._tool_result_max_chars: int = int(cfg.get("tool_result_max_chars", 2000))
        self._rag_recency_days: float = float(cfg.get("rag_recency_days", 30.0))
        self._compact_events: list[dict] = []   # history of compaction events this session
        self._ctx_history: list[int] = []       # rolling CTX% readings (last 20)
        self._thinking_enabled: bool = bool(cfg.get("thinking_enabled", False))
        self._thinking_paths: int = int(cfg.get("thinking_paths", 3))
        self._thinking_model: str = str(cfg.get("thinking_model", ""))
        self._thinking_temperature: float = float(cfg.get("thinking_temperature", 0.8))
        self._thinking_count: int = 0   # total think_and_answer calls this session
        self.tools.think_tool.model = self._thinking_model

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
        asyncio.create_task(self._maybe_resume_last_session())

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

    _STATUS_CACHE_TTL = 4.0  # seconds between network health-check polls

    def _effective_threshold_pct(self) -> int:
        """Scale compact threshold: 80% for ≤8k ctx, up to _compact_threshold_pct for ≥128k.

        Small-context models need to compact sooner; large-context models can wait longer.
        Uses log-linear interpolation between 8k and 128k.
        """
        import math
        ctx = self._context_length or 35063
        lo_ctx, hi_ctx = 8192, 131072
        lo_thr, hi_thr = 80, self._compact_threshold_pct
        if ctx <= lo_ctx:
            return lo_thr
        if ctx >= hi_ctx:
            return hi_thr
        t = (math.log2(ctx) - math.log2(lo_ctx)) / (math.log2(hi_ctx) - math.log2(lo_ctx))
        return int(lo_thr + t * (hi_thr - lo_thr))

    def _ctx_sparkline(self) -> str:
        """Return ASCII sparkline string of recent CTX% history (last 20 readings)."""
        bars = "▁▂▃▄▅▆▇█"
        if not self._ctx_history:
            return "(no data yet)"
        mx = max(self._ctx_history) or 1
        spark = "".join(bars[min(7, int(v * 8 // (mx + 1)))] for v in self._ctx_history)
        return f"`{spark}` ({len(self._ctx_history)} readings, peak {mx}%)"

    def _context_pct(self) -> int:
        """Return context usage % from _compact_from_idx onward (0-100).

        Uses only uncompacted history + compact summary so the number reflects
        what is actually sent to the LLM.
        """
        ctx = self._context_length or 35063
        reserve = self._max_response_tokens or 0
        budget = max(1, ctx - reserve)
        summary_tokens = len(self._compact_summary) // 4 if self._compact_summary else 0
        history_tokens = sum(
            len(m.content) // 4 for m in self.messages[self._compact_from_idx:]
        )
        pct = min(100, int((summary_tokens + history_tokens) * 100 // budget))
        if not self._ctx_history or self._ctx_history[-1] != pct:
            self._ctx_history = (self._ctx_history + [pct])[-20:]
        return pct

    async def update_status(self) -> None:
        now = time.monotonic()
        if now - self._last_status_ts < self._STATUS_CACHE_TTL:
            return
        self._last_status_ts = now
        up = await self.client.health()
        model_line = self._safe_query_one("#model-line", Static)
        status_line = self._safe_query_one("#status", Static)
        if model_line is None or status_line is None:
            return
        ctx_pct = self._context_pct()
        ctx_str = f" | CTX:{ctx_pct}%" if ctx_pct > 20 else ""
        rag_str = " | RAG:ON" if self.state.rag_context_enabled else " | RAG:OFF"
        cmp_str = f" | CMP:{self._compact_from_idx}" if self._compact_from_idx > 0 else ""
        thk_str = " | THK:ON" if self._thinking_enabled else ""
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
            + ctx_str
            + rag_str
            + cmp_str
            + thk_str
        )

    async def action_send(self) -> None:
        if self.state.busy or (self.active_task and not self.active_task.done()):
            return  # prevent double-submit race condition
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
        self._rag_context_query = ""
        self._rag_context_cache = ""
        user = Message("user", text)
        self.messages.append(user)
        self.transcript_store.append(user)
        self._write_transcript("You", text)
        asyncio.create_task(self._auto_save_turn(user, len(self.messages) - 1))
        if self._thinking_enabled:
            _think_out = await self._apply_thinking(text)
            if _think_out:
                self._write_transcript("Assistant", f"**[Parallel Think]**\n\n{_think_out}")
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

                try:
                    async with asyncio.timeout(300.0):
                        content, tool_calls = await self._stream_with_tools(tools, on_chunk=_on_chunk)
                except asyncio.TimeoutError:
                    if _live_msg is not None:
                        try:
                            _live_msg.remove()
                        except Exception:
                            pass
                        _live_msg = None
                    if content:
                        self._tool_log("[stream] 5-minute watchdog fired — using partial response")
                        self._finalize_assistant_response(content)
                    else:
                        self._write_transcript("Assistant", "Request timed out — no response received.")
                    return
                if _live_msg is not None:
                    _live_msg.remove()
                    _live_msg = None
            else:
                response = await self.client.chat_once_with_tools(
                    self.state.model,
                    await self._llm_messages(),
                    tools=tools,
                    max_tokens=self._max_response_tokens or None,
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
            self._last_status_ts = 0.0  # force status bar refresh after turn
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
            if text.startswith("/rss ") or text == "/rss":
                # /rss <topic> now searches the PostgreSQL database for stored articles
                topic = text[4:].strip() if text.startswith("/rss ") else ""
                if not topic:
                    self._write_transcript("Assistant", "Usage: /rss <topic>  — search stored articles by topic.\nTo store new feeds use: /rss store <topic>")
                    return
                if topic.startswith("store "):
                    feed_topic = topic[6:].strip()
                    if not feed_topic:
                        self._write_transcript("Assistant", "Usage: /rss store <topic>")
                        return
                    await self._rss_store_topic(feed_topic)
                    return
                call = ToolCall(index=0, name="db_search", args={"topic": topic}, call_id="", label="db_search")
                results = await self._run_tool_batch([call])
                self._log_tool_results(results)
                if any(not res.ok for res in results):
                    self._notify_tool_failures(results)
                else:
                    self._write_transcript("Assistant", "Database search complete. See Tools panel for details.")
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
            if text.startswith("/db ") or text == "/db":
                args_str = text[4:].strip() if text.startswith("/db ") else ""
                if args_str.startswith("search "):
                    q = args_str[7:].strip()
                    if not q:
                        self._write_transcript("Assistant", "Usage: /db search <query>")
                        return
                    call = ToolCall(index=0, name="db_search", args={"q": q}, call_id="", label="db_search")
                    results = await self._run_tool_batch([call])
                    self._log_tool_results(results)
                    if any(not res.ok for res in results):
                        self._notify_tool_failures(results)
                    else:
                        self._write_transcript("Assistant", "Database search complete. See Tools panel for details.")
                    return
                self._write_transcript("Assistant", "Usage: /db search <query>")
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
            if text.startswith("/search ") or text == "/search":
                query = text[8:].strip() if text.startswith("/search ") else ""
                if not query:
                    self._write_transcript("Assistant", "Usage: /search <query>")
                    return
                call = ToolCall(index=0, name="web_search", args={"query": query}, call_id="", label="web_search")
                results = await self._run_tool_batch([call])
                self._log_tool_results(results)
                if any(not res.ok for res in results):
                    self._notify_tool_failures(results)
                else:
                    self._write_transcript("Assistant", "Search complete. See Tools panel for results.")
                return
            if text.startswith("/fetch "):
                url = text[7:].strip()
                if not url:
                    self._write_transcript("Assistant", "Usage: /fetch <url>")
                    return
                # web_fetch is routed through the human_browser container (a12fdfeaaf78)
                call = ToolCall(index=0, name="web_fetch", args={"url": url}, call_id="", label="web_fetch")
                results = await self._run_tool_batch([call])
                self._log_tool_results(results)
                if any(not res.ok for res in results):
                    self._notify_tool_failures(results)
                else:
                    self._write_transcript("Assistant", "Fetched via browser. See Tools panel for content.")
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
            if text.startswith("/screenshots"):
                await self._handle_screenshots_command()
                return
            if text.startswith("/history"):
                await self._handle_history_command(text[8:].strip())
                return
            if text.startswith("/sessions"):
                await self._handle_sessions_command()
                return
            if text.startswith("/context"):
                await self._handle_context_toggle(text[8:].strip())
                return
            if text.startswith("/resume"):
                await self._handle_resume_command(text[7:].strip())
                return
            if text.startswith("/stats"):
                await self._handle_stats_command()
                return
            if text.startswith("/compact"):
                await self._handle_compact_command(text[8:].strip())
                return
            if text.startswith("/fork"):
                await self._handle_fork_command()
                return
            if text.startswith("/think ") or text == "/think":
                query = text[7:].strip() if text.startswith("/think ") else ""
                if not query:
                    self._write_transcript("Assistant", "Usage: /think <question>")
                    return
                self._write_transcript("Assistant", "_Thinking in parallel\u2026_")
                out = await self._apply_thinking(query)
                self._write_transcript(
                    "Assistant", out or "Thinking failed \u2014 no answer generated."
                )
                return
            if text.startswith("/thinking"):
                await self._handle_thinking_command(text[9:].strip())
                return
            if text.startswith("/ctx"):
                arg = text[4:].strip()
                if arg in ("graph", ""):
                    self._write_transcript(
                        "Assistant",
                        f"**CTX history:** {self._ctx_sparkline()}\n\nCurrent: **{self._context_pct()}%**",
                    )
                else:
                    self._write_transcript("Assistant", "Usage: /ctx  or  /ctx graph")
                return
            self._write_transcript("Assistant", "Unknown command.")
        except ToolDeniedError as exc:
            self._tool_log(f"[denied] {exc}")
        except Exception as exc:
            self._tool_log(f"[tool error] {exc}")


    # ------------------------------------------------------------------
    # Screenshot helpers
    # ------------------------------------------------------------------

    def _format_screenshot_result(self, payload: dict) -> str:
        """Return a user-friendly Markdown message for a screenshot result."""
        error = payload.get("error", "")
        host_path = payload.get("host_path", "")
        if error and not host_path:
            return f"Screenshot failed: {error}"
        lines = ["**Screenshot saved.**", ""]
        if host_path:
            lines += [f"**File:** `{host_path}`", ""]
        title = payload.get("title", "")
        url = payload.get("url", "")
        if title or url:
            lines += [f"**Page:** {title}{' — ' + url if url else ''}", ""]
        if host_path:
            lines += [
                "**Open with:**",
                f"```",
                f"xdg-open {host_path}",
                f"```",
                "",
                "_Metadata stored in the image database. Use `/screenshots` to list all saved screenshots._",
            ]
        return "\n".join(lines)

    def _format_image_list(self, payload: dict) -> str:
        """Format a list of images from the database for display."""
        images = payload.get("images", [])
        if not images:
            return "No screenshots stored yet. Use `browser` with `action=screenshot` to take one."
        lines = [f"**{len(images)} screenshot(s) stored:**", ""]
        for img in images:
            host_path = img.get("host_path") or img.get("url", "")
            alt = img.get("alt_text", "")
            stored_at = img.get("stored_at", "")[:19].replace("T", " ") if img.get("stored_at") else ""
            lines.append(f"- `{host_path}`" + (f"  _{alt}_" if alt else "") + (f"  ({stored_at})" if stored_at else ""))
        lines += ["", "Open with `xdg-open <path>` or `eog <path>`."]
        return "\n".join(lines)

    def _format_fetch_image_result(self, payload: dict) -> str:
        """Format a fetch_image result for the TUI."""
        error = payload.get("error", "")
        if error:
            return f"fetch_image failed: {error}"
        host_path = payload.get("host_path", "")
        url = payload.get("url", "")
        content_type = payload.get("content_type", "")
        size = payload.get("size", 0)
        lines = ["**Image saved.**", ""]
        if host_path:
            lines += [f"**File:** `{host_path}`", ""]
        if url:
            lines += [f"**Source:** {url}", ""]
        if content_type or size:
            meta = []
            if content_type:
                meta.append(content_type)
            if size:
                meta.append(f"{size:,} bytes")
            lines += [f"**Info:** {' — '.join(meta)}", ""]
        if host_path:
            lines += [
                "**Open with:**",
                "```",
                f"xdg-open {host_path}",
                "```",
                "",
                "_Saved to the image database. Use `/screenshots` to list all saved images._",
            ]
        return "\n".join(lines)

    def _format_screenshot_search_result(self, payload: dict) -> str:
        """Format the result of screenshot_search for the TUI."""
        query = payload.get("query", "")
        error = payload.get("error", "")
        screenshots = payload.get("screenshots", [])
        if error and not screenshots:
            return f"Screenshot search failed: {error}"
        lines = [f"**Visual search: '{query}'**", ""]
        if not screenshots:
            lines.append("No results found.")
            return "\n".join(lines)
        for i, shot in enumerate(screenshots, 1):
            url = shot.get("url", "")
            title = shot.get("title", "")
            host_path = shot.get("host_path", "")
            err = shot.get("error", "")
            if err and not host_path:
                lines.append(f"{i}. {url or '(unknown)'} — failed: {err}")
            else:
                lines.append(f"**{i}. {title or url}**")
                if url:
                    lines.append(f"   URL: {url}")
                if host_path:
                    lines += [
                        f"   File: `{host_path}`",
                        f"   Open: `xdg-open {host_path}`",
                    ]
            lines.append("")
        lines.append("_All screenshots saved. Use `/screenshots` to list them._")
        return "\n".join(lines)

    async def _handle_screenshots_command(self) -> None:
        """List all saved screenshots from the database."""
        call = ToolCall(
            index=0,
            name="db_list_images",
            args={"limit": 20},
            call_id="",
            label="db_list_images",
        )
        results = await self._run_tool_batch([call])
        self._log_tool_results(results)
        if any(not res.ok for res in results):
            self._notify_tool_failures(results)
        else:
            text = results[0].output if results else "No output."
            self._write_transcript("Assistant", text)

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
            await self._llm_messages(),
            tools=tools,
            max_tokens=self._max_response_tokens or None,
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
        if name == "web_search":
            query = str(args.get("query", "")).strip()
            if not query:
                return "web_search: missing 'query'"
            max_chars = int(args.get("max_chars", 4000))
            payload = await self.tools.run_web_search(query, max_chars, self.state.approval, self._confirm_tool)
            tier = payload.get("tier", 0)
            tier_name = payload.get("tier_name", "unknown")
            content = payload.get("content", "")
            error = payload.get("error", "")
            if error and not content:
                return f"web_search failed: {error}"
            header = f"[Search via {tier_name} (tier {tier})]\n\n"
            return f"{header}{content}" if content else "(no results)"
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
        if name == "db_store_article":
            url = str(args.get("url", "")).strip()
            if not url:
                return "db_store_article: missing 'url'"
            payload = await self.tools.run_db_store_article(
                url,
                str(args.get("title", "")),
                str(args.get("content", "")),
                str(args.get("topic", "")),
                self.state.approval,
                self._confirm_tool,
            )
            return json.dumps(payload, ensure_ascii=False)
        if name == "db_search":
            payload = await self.tools.run_db_search(
                str(args.get("topic", "")),
                str(args.get("q", "")),
                self.state.approval,
                self._confirm_tool,
            )
            return json.dumps(payload, ensure_ascii=False)
        if name == "db_cache_store":
            url = str(args.get("url", "")).strip()
            content = str(args.get("content", "")).strip()
            if not url or not content:
                return "db_cache_store: missing 'url' or 'content'"
            payload = await self.tools.run_db_cache_store(
                url, content, str(args.get("title", "")),
                self.state.approval, self._confirm_tool,
            )
            return json.dumps(payload, ensure_ascii=False)
        if name == "db_cache_get":
            url = str(args.get("url", "")).strip()
            if not url:
                return "db_cache_get: missing 'url'"
            payload = await self.tools.run_db_cache_get(url, self.state.approval, self._confirm_tool)
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
                find_text=str(args["find_text"]).strip() if args.get("find_text") else None,
            )
            if action == "screenshot":
                return self._format_screenshot_result(payload)
            return json.dumps(payload, ensure_ascii=False)
        if name == "fetch_image":
            url = str(args.get("url", "")).strip()
            if not url:
                return "fetch_image: 'url' is required"
            payload = await self.tools.run_fetch_image(url, self.state.approval, self._confirm_tool)
            return self._format_fetch_image_result(payload)
        if name == "screenshot_search":
            query = str(args.get("query", "")).strip()
            if not query:
                return "screenshot_search: 'query' is required"
            max_results = int(args.get("max_results", 3))
            payload = await self.tools.run_screenshot_search(
                query, max_results, self.state.approval, self._confirm_tool
            )
            return self._format_screenshot_search_result(payload)
        if name == "db_store_image":
            url = str(args.get("url", "")).strip()
            if not url:
                return "db_store_image: missing 'url'"
            payload = await self.tools.run_db_store_image(
                url,
                str(args.get("host_path", "")),
                str(args.get("alt_text", "")),
                self.state.approval,
                self._confirm_tool,
            )
            return json.dumps(payload, ensure_ascii=False)
        if name == "db_list_images":
            limit = int(args.get("limit", 20))
            payload = await self.tools.run_db_list_images(limit, self.state.approval, self._confirm_tool)
            return self._format_image_list(payload)
        if self.tools.is_custom_tool(name):
            payload = await self.tools.run_custom_tool(
                name, dict(args), self.state.approval, self._confirm_tool
            )
            return payload.get("result", "(no output)")
        return f"unknown tool '{name}'"

    async def _run_followup_response(self) -> None:
        content = ""
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

                try:
                    async with asyncio.timeout(300.0):
                        async for chunk in self.client.chat_stream(
                            self.state.model, await self._llm_messages(),
                            max_tokens=self._max_response_tokens or None,
                        ):
                            content += chunk
                            _on_chunk(content)
                except asyncio.TimeoutError:
                    if content:
                        self._tool_log("[followup] 5-minute watchdog fired — using partial response")
                    else:
                        self._write_transcript("Assistant", "Follow-up timed out — no response received.")
                        return
                if _live_msg is not None:
                    try:
                        _live_msg.remove()
                    except Exception:
                        pass
                    _live_msg = None
            else:
                content = await self.client.chat_once(self.state.model, await self._llm_messages(),
                                                       max_tokens=self._max_response_tokens or None)
            self._finalize_assistant_response(content)
        except asyncio.CancelledError:
            raise
        except LLMClientError as exc:
            self._write_transcript("Assistant", f"Follow-up error: {exc}")
        finally:
            if _live_msg is not None:
                try:
                    _live_msg.remove()
                except Exception:
                    pass

    async def _llm_messages(self) -> list[dict[str, object]]:
        def _est(text: str) -> int:
            return max(1, len(text) // 4) + 4  # chars/4 + per-message overhead

        # Build system content (compact summary appended when available)
        system_content = self.system_prompt
        if self._compact_summary:
            system_content += (
                "\n\n[Earlier conversation summary]\n"
                + self._compact_summary
                + "\n[/summary]"
            )

        # History from uncompacted portion only
        history = [m.as_chat_dict() for m in self.messages[self._compact_from_idx:]]

        ctx = self._context_length
        reserve = self._max_response_tokens
        if ctx and reserve and ctx > reserve:
            budget = ctx - reserve - _est(system_content)
            total_tokens = sum(_est(str(msg.get("content", "") or "")) for msg in history)

            # Auto-compaction when approaching context threshold (adaptive per ctx size)
            if (
                total_tokens > budget * self._effective_threshold_pct() / 100
                and self.state.compaction_enabled
                and not self._compact_pending
                and len(self.messages) - self._compact_from_idx >= self._compact_min_msgs
            ):
                await self._maybe_compact()
                # Rebuild after potential compaction
                system_content = self.system_prompt
                if self._compact_summary:
                    system_content += (
                        "\n\n[Earlier conversation summary]\n"
                        + self._compact_summary
                        + "\n[/summary]"
                    )
                history = [m.as_chat_dict() for m in self.messages[self._compact_from_idx:]]
                budget = ctx - reserve - _est(system_content)

            # Standard trimming fallback (handles disabled compaction or insufficient compaction)
            trimmed: list[dict[str, object]] = []
            used = 0
            for msg in reversed(history):
                content = msg.get("content", "")
                if isinstance(content, str):
                    tokens = _est(content)
                elif isinstance(content, list):
                    tokens = sum(
                        _est(str(b.get("text", b) if isinstance(b, dict) else b))
                        for b in content
                    ) + 4
                else:
                    tokens = 50
                if used + tokens > budget and trimmed:
                    break  # oldest messages dropped to stay within context budget
                trimmed.insert(0, msg)
                used += tokens
        else:
            trimmed = history

        system_msg: dict[str, object] = {"role": "system", "content": system_content}

        # RAG: inject semantically relevant past turns into the system prompt
        if self.state.rag_context_enabled and self.state.session_id:
            last_user = next((m for m in reversed(self.messages) if m.role == "user"), None)
            if last_user:
                rag = await self._fetch_rag_context(last_user.content)
                if rag:
                    system_msg = {"role": "system", "content": system_content + rag}

        return [system_msg, *trimmed]

    async def _fetch_rag_context(self, query: str) -> str:
        """Embed *query*, search past turns by cosine similarity, return formatted snippet block."""
        if query == self._rag_context_query:
            return self._rag_context_cache
        try:
            vecs = await self.tools.lm.embed([query[:2000]])
            if not vecs:
                self._rag_context_query = query
                self._rag_context_cache = ""
                return ""
            results = await self.tools.conv.search_turns(
                vecs[0], limit=4, exclude_session=self.state.session_id
            )
            if not results:
                self._rag_context_query = query
                self._rag_context_cache = ""
                return ""
            lines = ["\n\n---\n**Past context (from earlier sessions):**"]
            any_relevant = False
            import math as _math
            from datetime import datetime as _dt, timezone as _tz
            for r in results:
                sim = r.get("similarity", 0)
                if sim < 0.25:
                    continue
                # Date-weighted: score = sim * exp(-age_days / rag_recency_days)
                ts_str = r.get("timestamp", "")
                age_days = 0.0
                if ts_str:
                    try:
                        ts_dt = _dt.fromisoformat(ts_str.replace("Z", "+00:00"))
                        age_days = (_dt.now(_tz.utc) - ts_dt).total_seconds() / 86400
                    except Exception:
                        pass
                weighted = sim * _math.exp(-age_days / max(1.0, self._rag_recency_days))
                if weighted < 0.2:
                    continue
                any_relevant = True
                ts_label = ts_str[:10]
                role = r.get("role", "")
                snippet = r.get("content", "")[:200].replace("\n", " ")
                lines.append(f"> [{ts_label}] {role} (w={weighted:.2f}): {snippet}")
            if not any_relevant:
                self._rag_context_query = query
                self._rag_context_cache = ""
                return ""
            lines.append("---")
            rag = "\n".join(lines)
            self._rag_context_query = query
            self._rag_context_cache = rag
            return rag
        except Exception:
            self._rag_context_query = query
            self._rag_context_cache = ""
            return ""

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
        from .sanitizer import extract_thinking as _extract_thinking
        _thinking_txt, content = _extract_thinking(content)
        if _thinking_txt:
            self._write_transcript(
                "Assistant",
                f"> **[thinking]** {_thinking_txt[:300]}{'...' if len(_thinking_txt) > 300 else ''}",
            )
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
        asyncio.create_task(self._auto_save_turn(assistant, len(self.messages) - 1))
        # Proactive background compaction: warm up summary before hitting the threshold
        if (
            self._context_pct() >= max(60, self._effective_threshold_pct() - 20)
            and self.state.compaction_enabled
            and not self._compact_pending
            and len(self.messages) - self._compact_from_idx >= self._compact_min_msgs
        ):
            asyncio.create_task(self._maybe_compact())
        # Generate a session title after ≥3 user turns — only once per session
        user_count = sum(1 for m in self.messages if m.role == "user")
        if user_count >= 3 and not self.state.session_title_set:
            self.state.session_title_set = True
            asyncio.create_task(self._auto_generate_title())

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
                payload[:self._tool_result_max_chars],
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
        cached = self.tools.check_cache(call.name, call.args)
        if cached is not None:
            self._tool_log(f"[cache hit] {call.name}")
            return cached
        result = await self._execute_tool_call(call.name, call.args)
        self.tools.store_cache(call.name, call.args, result)
        return result

    def _refresh_sessions(self) -> None:
        now = time.monotonic()
        if now - self._last_refresh_ts < 0.5:
            return
        self._last_refresh_ts = now
        log = self._safe_query_one("#sessionpane", Log)
        if log is None:
            return
        log.clear()
        log.write_line(f"LLM busy: {'yes' if self.state.busy else 'no'}")
        if self._pending_tools > 0:
            log.write_line(f"⟳ {self._pending_tools} tools running in parallel")
        else:
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
        self.state.session_id = str(_uuid.uuid4())
        # Session row is created lazily on first _auto_save_turn (avoids orphan empty sessions)
        self._last_refresh_ts = 0.0
        self._last_status_ts = 0.0
        self.state.session_title_set = False
        self._rag_context_query = ""
        self._rag_context_cache = ""
        self._compact_summary = ""
        self._compact_from_idx = 0
        self._compact_pending = False
        if not initial:
            self._write_transcript("Assistant", "New chat started.")

    def _clear_transcript(self) -> None:
        self.transcript_store.clear()
        self.messages = []
        scroll = self._safe_query_one("#transcript", VerticalScroll)
        if scroll is not None:
            scroll.remove_children()

    # ------------------------------------------------------------------
    # Conversation persistence helpers
    # ------------------------------------------------------------------

    async def _auto_save_turn(self, msg: "Message", turn_index: int) -> None:
        """Fire-and-forget: persist a conversation turn (with optional embedding)."""
        content = msg.full_content or msg.content
        if not content.strip() or not self.state.session_id:
            return
        # Idempotent create_session ensures the FK constraint is satisfied before the turn insert.
        # Session row is only created when there is actual content to store (no orphan empty sessions).
        await self.tools.conv.create_session(self.state.session_id, model=self.state.model)
        embedding: list[float] | None = None
        if self.state.rag_context_enabled and msg.role in ("user", "assistant"):
            try:
                vecs = await self.tools.lm.embed([content[:2000]])
                embedding = vecs[0] if vecs else None
            except Exception:
                pass
        await self.tools.conv.store_turn(
            self.state.session_id,
            msg.role,
            content,
            turn_index=turn_index,
            embedding=embedding,
        )

    async def _auto_generate_title(self) -> None:
        """Generate a 5-word session title from the last 3 user messages (fire-and-forget)."""
        if not self.state.session_id:
            return
        try:
            user_msgs = [m.content for m in self.messages if m.role == "user"][-3:]
            prompt = (
                "In 5 words or fewer, give a short title for this conversation. "
                "Reply with only the title, no punctuation:\n\n"
                + "\n".join(user_msgs)
            )
            title = await self.tools.lm.chat(
                [{"role": "user", "content": prompt}], max_tokens=20
            )
            if title and title.strip():
                await self.tools.conv.update_title(self.state.session_id, title.strip()[:80])
        except Exception:
            pass

    async def _handle_history_command(self, query: str) -> None:
        """Search past conversation turns semantically: /history <query>"""
        if not query:
            self._write_transcript("Assistant", "Usage: /history <query>  — search past conversations semantically.")
            return
        try:
            vecs = await self.tools.lm.embed([query])
            if vecs:
                results = await self.tools.conv.search_turns(vecs[0], limit=8)
            else:
                results = []
        except Exception:
            vecs = []
            results = []
        # Full-text fallback when embedding unavailable or returned no results
        if not results:
            try:
                results = await self.tools.conv.search_turns_text(query, limit=8)
                mode_label = "full-text"
            except Exception as exc:
                self._write_transcript("Assistant", f"History search error: {exc}")
                return
        else:
            mode_label = "semantic"
        if not results:
            self._write_transcript("Assistant", "No matching past conversation turns found.")
            return
        lines = [f"**Past turns matching** `{query}` ({mode_label}):", ""]
        for r in results:
            ts = r.get("timestamp", "")[:10]
            role = r.get("role", "")
            snippet = r.get("content", "")[:200].replace("\n", " ")
            sim = r.get("similarity")
            sim_str = f" (sim={round(sim, 3)})" if sim is not None else ""
            lines.append(f"- [{ts}] **{role}**{sim_str}: {snippet}")
        self._write_transcript("Assistant", "\n".join(lines))

    async def _handle_sessions_command(self) -> None:
        """List recent conversation sessions: /sessions"""
        try:
            sessions = await self.tools.conv.list_sessions(20)
            if not sessions:
                self._write_transcript("Assistant", "No conversation sessions found in the database.")
                return
            lines = ["**Recent conversation sessions:**", ""]
            for s in sessions:
                ts = s.get("updated_at", "")[:10]
                title = s.get("title") or "(untitled)"
                turns = s.get("turn_count", 0)
                model = s.get("model", "")
                sid = s.get("session_id", "")[:8]
                lines.append(f"- [{ts}] **{title}** — {turns} turns, model={model}, id={sid}…")
            self._write_transcript("Assistant", "\n".join(lines))
        except Exception as exc:
            self._write_transcript("Assistant", f"Sessions list error: {exc}")

    async def _handle_context_toggle(self, arg: str) -> None:
        """/context [on|off] — toggle or set RAG context injection."""
        if arg.lower() in ("on", "1", "true", "yes"):
            self.state.rag_context_enabled = True
        elif arg.lower() in ("off", "0", "false", "no"):
            self.state.rag_context_enabled = False
        else:
            self.state.rag_context_enabled = not self.state.rag_context_enabled
        state_str = "ON" if self.state.rag_context_enabled else "OFF"
        self._write_transcript("Assistant", f"RAG context injection: **{state_str}**")
        self._last_status_ts = 0.0  # force status bar refresh
        await self.update_status()

    async def _handle_resume_command(self, partial_id: str) -> None:
        """/resume <partial_session_id> — load a past session by id prefix."""
        if not partial_id:
            self._write_transcript("Assistant", "Usage: /resume <session-id-prefix>")
            return
        try:
            sessions = await self.tools.conv.list_sessions(50)
            match = next((s for s in sessions if s["session_id"].startswith(partial_id)), None)
            if not match:
                self._write_transcript("Assistant", f"No session found matching `{partial_id}`.")
                return
            data = await self.tools.conv.get_session(match["session_id"], limit=200)
            turns = data.get("turns", [])
            if not turns:
                self._write_transcript("Assistant", "Session has no turns.")
                return
            self._start_new_chat(initial=True)  # suppress "New chat started" — resume has its own message
            for t in turns:
                msg = Message(t["role"], t["content"])
                self.messages.append(msg)
                self._write_transcript(t["role"].capitalize(), t["content"])
            self.state.session_id = match["session_id"]  # continue in same session
            # Restore compaction overlay from persisted DB state
            self._compact_summary = data.get("compact_summary", "") or ""
            self._compact_from_idx = int(data.get("compact_from_idx", 0) or 0)
            # Integrity check: idx must not exceed actual turn count
            if self._compact_from_idx > len(turns):
                self._tool_log(
                    f"[compact] integrity: idx {self._compact_from_idx} > {len(turns)} turns → reset"
                )
                self._compact_summary = ""
                self._compact_from_idx = 0
            self._compact_pending = False
            self._last_status_ts = 0.0
            await self.update_status()
            cmp_note = (
                f" (compact overlay: {self._compact_from_idx} turns summarized)"
                if self._compact_from_idx > 0
                else ""
            )
            self._write_transcript(
                "Assistant",
                f"Resumed session `{match['session_id'][:8]}…` — {len(turns)} turns loaded{cmp_note}.",
            )
        except Exception as exc:
            self._write_transcript("Assistant", f"Resume error: {exc}")

    async def _handle_stats_command(self) -> None:
        """/stats — show conversation DB statistics."""
        try:
            sessions = await self.tools.conv.list_sessions(1000)
            total_turns = sum(s.get("turn_count", 0) for s in sessions)
            newest = sessions[0].get("updated_at", "")[:10] if sessions else "—"
            oldest = sessions[-1].get("created_at", "")[:10] if sessions else "—"
            current_sid = (
                f"- Current session: `{self.state.session_id[:8]}…`"
                if self.state.session_id
                else "- No active session"
            )
            self._write_transcript(
                "Assistant",
                "\n".join(
                    [
                        "**Conversation DB stats:**",
                        f"- Sessions: {len(sessions)}",
                        f"- Total turns: {total_turns}",
                        f"- Newest: {newest}  Oldest: {oldest}",
                        f"- RAG: {'ON' if self.state.rag_context_enabled else 'OFF'}",
                        current_sid,
                    ]
                ),
            )
        except Exception as exc:
            self._write_transcript("Assistant", f"Stats error: {exc}")

    async def _apply_thinking(self, question: str) -> str | None:
        """Fan-out parallel thinking and return formatted output, or None on failure."""
        if not question.strip():
            return None
        try:
            result = await self.tools.run_think(
                question, self._thinking_paths, self._thinking_temperature,
                self.state.approval, self._confirm_tool,
            )
        except Exception:
            return None
        if not result.answer:
            return None
        self._thinking_count += 1
        meta = f"\n\n*({result.paths_tried} paths, score {result.best_score:.2f}, {result.duration_ms}ms)*"
        if result.reasoning and result.reasoning != result.answer:
            snippet = result.reasoning[:300] + ("..." if len(result.reasoning) > 300 else "")
            return f"> **[thinking]** {snippet}\n\n{result.answer}{meta}"
        return f"{result.answer}{meta}"

    async def _handle_thinking_command(self, arg: str) -> None:
        """/thinking [on|off|status|paths N|model NAME]"""
        parts = arg.split()
        sub = parts[0].lower() if parts else ""
        if sub == "on":
            self._thinking_enabled = True
            self._write_transcript(
                "Assistant",
                "Thinking mode **ON** \u2014 every query will use parallel chain-of-thought.",
            )
        elif sub == "off":
            self._thinking_enabled = False
            self._write_transcript("Assistant", "Thinking mode **OFF**.")
        elif sub == "paths" and len(parts) >= 2 and parts[1].isdigit():
            self._thinking_paths = max(1, min(10, int(parts[1])))
            self._write_transcript(
                "Assistant", f"Thinking paths set to **{self._thinking_paths}**."
            )
        elif sub == "model" and len(parts) >= 2:
            self._thinking_model = " ".join(parts[1:])
            self.tools.think_tool.model = self._thinking_model
            self._write_transcript(
                "Assistant", f"Thinking model set to `{self._thinking_model}`."
            )
        else:
            status = "ON" if self._thinking_enabled else "OFF"
            self._write_transcript(
                "Assistant",
                f"**Thinking:** {status} | paths={self._thinking_paths} | "
                f"temp={self._thinking_temperature} | "
                f"model={'(main)' if not self._thinking_model else self._thinking_model} | "
                f"calls this session={self._thinking_count}\n\n"
                "Commands: `/thinking on` \u00b7 `/thinking off` \u00b7 `/thinking paths N` \u00b7 "
                "`/thinking model NAME` \u00b7 `/thinking status`",
            )
        await self.update_status()

    async def _handle_fork_command(self) -> None:
        """/fork — start a new chat, pre-seeding compact summary from the current session."""
        summary = self._compact_summary
        turn_count = len(self.messages)
        await self.action_new_chat()
        if summary:
            self._compact_summary = f"[Forked from {turn_count}-turn session]\n\n{summary}"
            self._compact_from_idx = 0
            self._write_transcript(
                "Assistant",
                f"Forked — {turn_count}-turn session context pre-loaded. "
                "The compaction summary from the previous session is active.",
            )
        else:
            self._write_transcript("Assistant", "Forked — new session started (no prior summary).")

    async def _maybe_resume_last_session(self) -> None:
        """On startup, hint (via notify toast) if a recent session exists (< 2 hours old)."""
        try:
            sessions = await self.tools.conv.list_sessions(1)
            if not sessions:
                return
            recent = sessions[0]
            if recent.get("turn_count", 0) < 2:
                return  # skip sessions with no real content (avoids test churn)
            from datetime import datetime, timezone, timedelta
            ts_str = recent.get("updated_at", "")
            if not ts_str:
                return
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) - ts > timedelta(hours=2):
                return
            title = recent.get("title") or "(untitled)"
            sid = recent["session_id"][:8]
            turns = recent.get("turn_count", 0)
            self.notify(
                f"Recent: {title} ({turns} turns) — type /resume {sid} to continue",
                timeout=10,
            )
        except Exception:
            pass

    async def _run_compact(self, to_compact: "list[Message]", n_advance: int) -> None:
        """Summarize *to_compact* messages and advance _compact_from_idx by *n_advance*.

        Fail-open: on any exception (LM Studio offline, etc.) state is unchanged.
        Meta-compaction: when a prior summary exists, merge it with new turns into a
        single fresh summary (keeps summary size bounded across multiple compactions).
        """
        valid_roles: set[str] = {"user", "assistant"}
        if self._compact_tool_turns:
            valid_roles.add("tool")

        def _fmt(m: "Message") -> str:
            body = (m.full_content or m.content)[:300 if m.role == "tool" else 400]
            return f"{m.role.upper()}: {body}"

        conv_text = "\n".join(_fmt(m) for m in to_compact if m.role in valid_roles)
        if not conv_text.strip():
            return

        # Persona-aware compaction prompt
        persona_name = next(
            (p.get("name", "") for p in self.personalities
             if p.get("id") == self.state.personality_id), ""
        )
        persona_note = (
            f" You are compacting a conversation with persona '{persona_name}'."
            if persona_name else ""
        )

        self._compact_pending = True
        try:
            if self._compact_summary:
                # Meta-compaction: merge old summary + new turns into a single fresh summary
                sys_txt = (
                    "You are a conversation compactor. "
                    "Merge the PREVIOUS SUMMARY with the NEW TURNS into a single "
                    "concise summary. Preserve all key facts, decisions, context, "
                    "code snippets, and user intent. Output under 400 words. "
                    f"Use bullet points.{persona_note}"
                )
                user_txt = (
                    f"PREVIOUS SUMMARY:\n{self._compact_summary}\n\n"
                    f"NEW TURNS:\n{conv_text}"
                )
            else:
                sys_txt = (
                    "You are a conversation compactor. "
                    "Summarize the following conversation turns concisely, "
                    "preserving key facts, decisions, context, code snippets, "
                    "and user intent. Be brief (under 300 words). "
                    f"Use bullet points.{persona_note}"
                )
                user_txt = conv_text

            # Use dedicated compact_model if configured, else fall back to main LM
            from .tools.lm_studio import LMStudioTool as _LMT
            _lm = _LMT(self.tools.lm.base_url, self._compact_model) if self._compact_model else self.tools.lm
            summary = await _lm.chat(
                [
                    {"role": "system", "content": sys_txt},
                    {"role": "user", "content": user_txt},
                ],
                max_tokens=400,
                temperature=0.2,
            )
            if summary and summary.strip():
                self._compact_summary = summary.strip()  # always replace (meta-compact)
                self._compact_from_idx += n_advance
                self._tool_log(
                    f"[compact] {n_advance} turns → {len(summary.split())} word summary"
                )
                # Track compaction event
                from datetime import timezone as _tz
                self._compact_events.append({
                    "time": datetime.now(_tz.utc).isoformat(timespec="seconds"),
                    "n_turns": n_advance,
                    "words": len(summary.split()),
                })
                # Persist compact state to DB (fire-and-forget, fail-open)
                if self.state.session_id:
                    asyncio.create_task(
                        self.tools.conv.update_compact_state(
                            self.state.session_id,
                            self._compact_summary,
                            self._compact_from_idx,
                        )
                    )
        except Exception:
            pass  # fail-open — normal trimming takes over
        finally:
            self._compact_pending = False

    async def _maybe_compact(self) -> None:
        """Auto-triggered compaction: compact oldest 50% of visible messages."""
        if self._compact_pending or not self.state.compaction_enabled:
            return
        visible = self.messages[self._compact_from_idx:]
        n = len(visible)
        if n < self._compact_min_msgs:
            return
        n_compact = max(4, int(n * self._compact_keep_ratio))

        # Differential compaction: prefer complete user→assistant pairs
        pairs: list[Message] = []
        i = 0
        window = visible[:n_compact]
        while i < len(window) - 1:
            if window[i].role == "user" and window[i + 1].role == "assistant":
                pairs.extend([window[i], window[i + 1]])
                i += 2
            else:
                i += 1
        to_compact = pairs if len(pairs) >= 4 else [m for m in window if m.role in ("user", "assistant")]
        if len(to_compact) < 4:
            return
        await self._run_compact(to_compact, n_compact)

    async def _handle_compact_command(self, arg: str) -> None:
        """/compact [on|off|now|status|dry|history] — control contextual compaction."""
        if arg.lower() == "dry":
            visible = self.messages[self._compact_from_idx:]
            n = len(visible)
            if n < 4:
                self._write_transcript("Assistant", "Too few messages to compact (need ≥4).")
                return
            n_compact = max(2, n // 2)
            chars = sum(len(m.content) for m in visible[:n_compact])
            tokens = chars // 4
            budget = max(1, (self._context_length or 35063) - (self._max_response_tokens or 0))
            pct_freed = min(100, int(tokens * 100 // budget))
            self._write_transcript(
                "Assistant",
                f"**Dry-run compact:** would compact {n_compact}/{n} messages, "
                f"freeing ~{tokens:,} tokens (~{pct_freed}% of context budget). "
                f"Run `/compact now` to execute.",
            )
            return
        if arg.lower() == "history":
            if not self._compact_events:
                self._write_transcript("Assistant", "No compaction events this session.")
                return
            lines = ["**Compaction history (this session):**"]
            for i, ev in enumerate(self._compact_events, 1):
                lines.append(f"{i}. `{ev['time']}` — {ev['n_turns']} turns → {ev['words']} words")
            self._write_transcript("Assistant", "\n".join(lines))
            return
        if arg.lower() == "status":
            if self._compact_summary:
                self._write_transcript(
                    "Assistant",
                    f"**Compaction active** — {self._compact_from_idx} turns summarized.\n\n"
                    f"**Current summary:**\n{self._compact_summary}",
                )
            else:
                self._write_transcript(
                    "Assistant",
                    "No active compaction summary. Context is uncompacted.",
                )
            return
        if arg.lower() in ("off", "0", "false", "no"):
            self.state.compaction_enabled = False
            self._write_transcript("Assistant", "Contextual compaction: **OFF**")
            self._last_status_ts = 0.0
            await self.update_status()
            from .config import load_config, save_config as _save_cfg
            _cfg = dict(load_config())
            _cfg["compaction_enabled"] = False
            _save_cfg(_cfg)
            return
        if arg.lower() in ("on", "1", "true", "yes"):
            self.state.compaction_enabled = True
            self._write_transcript("Assistant", "Contextual compaction: **ON**")
            self._last_status_ts = 0.0
            await self.update_status()
            from .config import load_config, save_config as _save_cfg
            _cfg = dict(load_config())
            _cfg["compaction_enabled"] = True
            _save_cfg(_cfg)
            return
        # /compact or /compact now — force immediate compaction
        visible = self.messages[self._compact_from_idx:]
        n = len(visible)
        if n < 4:
            self._write_transcript("Assistant", "Not enough messages to compact (need ≥4).")
            return
        if self._compact_pending:
            self._write_transcript("Assistant", "Compaction already in progress.")
            return
        n_compact = max(2, n // 2)
        to_compact = [m for m in visible[:n_compact] if m.role in ("user", "assistant")]
        if not to_compact:
            self._write_transcript("Assistant", "No user/assistant messages to compact.")
            return
        before = self._compact_from_idx
        await self._run_compact(to_compact, n_compact)
        if self._compact_from_idx > before:
            compacted = self._compact_from_idx - before
            self._last_status_ts = 0.0
            await self.update_status()
            self._write_transcript(
                "Assistant",
                f"Compacted **{compacted}** turns into summary. "
                "Context freed; summary injected into system prompt.",
            )
        else:
            self._write_transcript(
                "Assistant",
                "Compaction failed (LM Studio unavailable or too few messages).",
            )

    async def _rss_store_topic(self, topic: str) -> None:
        """Discover feeds for a topic and store their articles in PostgreSQL."""
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
                "If feeds fail to store, try https://rssfinder.app/ or https://rss.app/rss-feed.",
            )
            return
        self._write_transcript(
            "Assistant",
            f"Stored {len(push_calls)} feed(s) for '{topic}' in PostgreSQL. See Tools panel.",
        )

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
        # Always try to update context length from model metadata (fail-open)
        try:
            info = await self.client.model_info()
            if self.state.model in info and info[self.state.model] > 0:
                self._context_length = info[self.state.model]
        except Exception:
            pass
        if self.state.model not in models:
            self.state.model = models[0]
            self._persist_config()
            await self.update_status()
            self._log_session(f"Default model set to {self.state.model}")
            # Update context length for the newly selected model
            try:
                info = await self.client.model_info()
                ctx = info.get(self.state.model, 0)
                if ctx > 0:
                    self._context_length = ctx
                    self._log_session(f"Context length: {ctx:,} tokens")
                    self._last_status_ts = 0.0
                    await self.update_status()
            except Exception:
                pass

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
        help_text = (
            "**Keyboard shortcuts:**\n"
            + render_keybinds()
            + "\n\n**Slash commands:**\n"
            "- `/new` · `/clear` · `/export` · `/copy`\n"
            "- `/history <query>` — semantic (or full-text) search past conversations\n"
            "- `/sessions` — list recent sessions\n"
            "- `/resume <id>` — reload a past session by id prefix\n"
            "- `/stats` — conversation DB statistics\n"
            "- `/context [on|off]` — toggle RAG context injection\n"
            "- `/search <query>` · `/fetch <url>`\n"
            "- `/db search <q>` · `/rss <topic>`\n"
            "- `/memory recall [key]` · `/memory store <k> <v>`\n"
            "- `/shell [cmd]` · `/endpoint <url>`\n"
            "- `/persona [id]` · `/concise` · `/verbose`\n"
            "- `/tool list|delete <name>`\n"
            "- `/compact [on|off|now|status]` — contextual compaction (summarize old turns)\n"
            "- `/help` — this message\n"
        )
        self._write_transcript("Assistant", help_text)

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
