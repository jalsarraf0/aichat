from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from textual.binding import Binding


@dataclass(frozen=True)
class KeybindSpec:
    key: str
    action: str
    label: str


KEYBINDS: list[KeybindSpec] = [
    KeybindSpec("f1", "help", "Help"),
    KeybindSpec("f2", "pick_model", "Model"),
    KeybindSpec("f3", "search", "Search"),
    KeybindSpec("f4", "cycle_approval", "Approval"),
    KeybindSpec("f5", "theme_picker", "Theme"),
    KeybindSpec("f6", "toggle_streaming", "Streaming"),
    KeybindSpec("f7", "sessions", "Sessions"),
    KeybindSpec("f8", "settings", "Settings"),
    KeybindSpec("f9", "new_chat", "New Chat"),
    KeybindSpec("f10", "clear_transcript", "Clear"),
    KeybindSpec("f11", "cancel", "Cancel"),
    KeybindSpec("f12", "quit", "Quit"),
    KeybindSpec("ctrl+h", "help", "Help"),
]


def binding_list() -> list["Binding"]:
    from textual.binding import Binding

    return [Binding(spec.key, spec.action, spec.label, priority=True) for spec in KEYBINDS]


def display_key(key: str) -> str:
    key_lower = key.lower()
    if key_lower.startswith("f") and key_lower[1:].isdigit():
        return key_upper(key_lower)
    if key_lower == "ctrl+h":
        return "^H"
    return key


def key_upper(key: str) -> str:
    return key.upper()


def render_keybinds() -> str:
    parts: list[str] = []
    for spec in KEYBINDS:
        label = spec.label
        key_label = display_key(spec.key)
        parts.append(f"{key_label} {label}")
    return "  ".join(parts)
