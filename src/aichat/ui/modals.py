from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from dataclasses import dataclass

from textual.widgets import Button, Input, Label, ListItem, ListView, Select, TextArea

from ..model_labels import model_options


@dataclass(frozen=True)
class Choice:
    label: str
    value: str


class ChoiceItem(ListItem):
    def __init__(self, label: str, value: str) -> None:
        super().__init__(Label(label))
        self.value = value


class ChoiceModal(ModalScreen[str]):
    def __init__(self, title: str, choices: list[str | Choice]) -> None:
        super().__init__()
        self._title = title
        normalized: list[Choice] = []
        for choice in choices:
            if isinstance(choice, Choice):
                normalized.append(choice)
            elif isinstance(choice, (tuple, list)) and len(choice) == 2:
                label = str(choice[0])
                value = str(choice[1])
                normalized.append(Choice(label=label, value=value))
            else:
                normalized.append(Choice(label=str(choice), value=str(choice)))
        self._choices = normalized

    def compose(self) -> ComposeResult:
        with Vertical(id="choice-modal"):
            yield Label(self._title)
            yield ListView(
                *[ChoiceItem(item.label, item.value) for item in self._choices],
                id="choice-list",
            )
            yield Button("Close", id="close")

    def on_mount(self) -> None:
        self.set_focus(self.query_one("#choice-list", ListView))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        choice = ""
        item = getattr(event, "item", None)
        if isinstance(item, ChoiceItem):
            choice = item.value
        else:
            index = event.index
            if 0 <= index < len(self._choices):
                choice = self._choices[index].value
        self.dismiss(choice)

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


class RssIngestModal(ModalScreen[dict]):
    def compose(self) -> ComposeResult:
        with Vertical(id="rss-ingest-modal"):
            yield Label("RSS ingest")
            yield Input(placeholder="Topic (e.g., linux)", id="rss-topic")
            yield Input(placeholder="Feed URL", id="rss-feed-url")
            yield Button("Ingest", id="ingest")
            yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.set_focus(self.query_one("#rss-topic", Input))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss({})
            return
        if event.button.id == "ingest":
            topic = self.query_one("#rss-topic", Input).value.strip()
            feed_url = self.query_one("#rss-feed-url", Input).value.strip()
            self.dismiss({"topic": topic, "feed_url": feed_url})

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "rss-feed-url":
            topic = self.query_one("#rss-topic", Input).value.strip()
            feed_url = self.query_one("#rss-feed-url", Input).value.strip()
            self.dismiss({"topic": topic, "feed_url": feed_url})

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.dismiss({})
            event.stop()


class PersonalityAddModal(ModalScreen[dict]):
    def compose(self) -> ComposeResult:
        with Vertical(id="persona-add-modal"):
            yield Label("Add Personality")
            yield Input(placeholder="Name", id="persona-name")
            yield TextArea(id="persona-prompt")
            yield Button("Save", id="save")
            yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.set_focus(self.query_one("#persona-name", Input))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss({})
            return
        if event.button.id == "save":
            name = self.query_one("#persona-name", Input).value.strip()
            prompt = self.query_one("#persona-prompt", TextArea).text.strip()
            self.dismiss({"name": name, "prompt": prompt})

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.dismiss({})
            event.stop()


class SettingsModal(ModalScreen[dict]):
    def __init__(self, current: dict[str, str], models: list[str], themes: list[str]) -> None:
        super().__init__()
        self.current = current
        self.models = models
        self.themes = themes

    def compose(self) -> ComposeResult:
        model_list = self.models or [self.current["model"]]
        if self.current["model"] not in model_list:
            model_list = [self.current["model"], *model_list]
        options = model_options(model_list)
        if not options:
            options = [(self.current["model"], self.current["model"])]
        with Vertical(id="settings-modal"):
            yield Label("Settings")
            yield Label("Base URL (LM Studio endpoint)")
            yield Input(value=self.current["base_url"], id="base_url")
            yield Select(options, value=self.current["model"], id="model")
            yield Select.from_values(self.themes, value=self.current["theme"], id="theme")
            yield Select.from_values(["DENY", "ASK", "AUTO"], value=self.current["approval"], id="approval")
            yield Label("Host shell tool access")
            yield Select.from_values(
                ["true", "false"],
                value="true" if self.current.get("shell_enabled") else "false",
                id="shell_enabled",
            )
            yield Label("Concise mode")
            yield Select.from_values(
                ["true", "false"],
                value="true" if self.current.get("concise_mode") else "false",
                id="concise_mode",
            )
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
                    "base_url": self.query_one("#base_url", Input).value.strip() or self.current["base_url"],
                    "model": str(self.query_one("#model", Select).value),
                    "theme": str(self.query_one("#theme", Select).value),
                    "approval": str(self.query_one("#approval", Select).value),
                    "shell_enabled": str(self.query_one("#shell_enabled", Select).value) == "true",
                    "concise_mode": str(self.query_one("#concise_mode", Select).value) == "true",
                }
            )

    def on_key(self, event: events.Key) -> None:
        if event.key == "ctrl+s":
            self.dismiss(
                {
                    "base_url": self.query_one("#base_url", Input).value.strip() or self.current["base_url"],
                    "model": str(self.query_one("#model", Select).value),
                    "theme": str(self.query_one("#theme", Select).value),
                    "approval": str(self.query_one("#approval", Select).value),
                    "shell_enabled": str(self.query_one("#shell_enabled", Select).value) == "true",
                    "concise_mode": str(self.query_one("#concise_mode", Select).value) == "true",
                }
            )
            event.stop()
            return
        if event.key == "escape":
            self.dismiss({})
            event.stop()
