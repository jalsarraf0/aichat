from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Select


class ChoiceModal(ModalScreen[str]):
    def __init__(self, title: str, choices: list[str]) -> None:
        super().__init__()
        self._title = title
        self._choices = choices

    def compose(self) -> ComposeResult:
        with Vertical(id="choice-modal"):
            yield Label(self._title)
            yield ListView(*[ListItem(Label(item)) for item in self._choices], id="choice-list")
            yield Button("Close", id="close")

    def on_mount(self) -> None:
        list_view = self.query_one("#choice-list", ListView)
        if self._choices:
            list_view.index = 0
        self.set_focus(list_view)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        index = event.index
        if 0 <= index < len(self._choices):
            self.dismiss(self._choices[index])
            return
        self.dismiss("")

    def on_button_pressed(self, _: Button.Pressed) -> None:
        self.dismiss("")

    def on_key(self, event: events.Key) -> None:
        list_view = self.query_one("#choice-list", ListView)
        if event.key == "up":
            list_view.action_cursor_up()
            event.stop()
            return
        if event.key == "down":
            list_view.action_cursor_down()
            event.stop()
            return
        if event.key == "enter":
            list_view.action_select_cursor()
            event.stop()
            return
        if event.key == "escape":
            self.dismiss("")
            event.stop()


class SearchModal(ModalScreen[str]):
    def compose(self) -> ComposeResult:
        with Vertical(id="search-modal"):
            yield Label("Search transcript")
            yield Input(placeholder="Type search text", id="search-query")
            yield Button("Search", id="submit-search")

    def on_mount(self) -> None:
        self.set_focus(self.query_one("#search-query", Input))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit-search":
            query = self.query_one("#search-query", Input).value.strip()
            self.dismiss(query)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search-query":
            self.dismiss(event.value.strip())

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.dismiss("")
            event.stop()


class SettingsModal(ModalScreen[dict]):
    def __init__(self, current: dict[str, str], models: list[str], themes: list[str]) -> None:
        super().__init__()
        self.current = current
        self.models = models
        self.themes = themes

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-modal"):
            yield Label("Settings")
            yield Label(f"Base URL: {self.current['base_url']} (fixed)")
            yield Select.from_values(self.models or [self.current["model"]], value=self.current["model"], id="model")
            yield Select.from_values(self.themes, value=self.current["theme"], id="theme")
            yield Select.from_values(["DENY", "ASK", "AUTO"], value=self.current["approval"], id="approval")
            yield Button("Save", id="save")
            yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.set_focus(self.query_one("#model", Select))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss({})
            return
        if event.button.id == "save":
            self.dismiss(
                {
                    "base_url": self.current["base_url"],
                    "model": str(self.query_one("#model", Select).value),
                    "theme": str(self.query_one("#theme", Select).value),
                    "approval": str(self.query_one("#approval", Select).value),
                }
            )

    def on_key(self, event: events.Key) -> None:
        if event.key == "ctrl+s":
            self.dismiss(
                {
                    "base_url": self.current["base_url"],
                    "model": str(self.query_one("#model", Select).value),
                    "theme": str(self.query_one("#theme", Select).value),
                    "approval": str(self.query_one("#approval", Select).value),
                }
            )
            event.stop()
            return
        if event.key == "escape":
            self.dismiss({})
            event.stop()
