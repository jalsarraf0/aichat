from __future__ import annotations

from textual.widgets import Static

from .keybinds import render_keybinds


class KeybindBar(Static):
    def on_mount(self) -> None:
        self.update(render_keybinds())
