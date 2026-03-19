from __future__ import annotations

from textual.widgets import Static

from .keybinds import render_keybinds


class KeybindBar(Static):
    """Bottom keybind bar — renders F1-F12 + Ctrl combos."""

    DEFAULT_CSS = """
    KeybindBar {
        height: 1;
        padding: 0 1;
        text-style: bold;
    }
    """

    def on_mount(self) -> None:
        self.update(render_keybinds())
