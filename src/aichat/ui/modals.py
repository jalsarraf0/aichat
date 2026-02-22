from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, ListItem, ListView


class ThemePicker(ModalScreen[str]):
    def __init__(self, themes: list[str]) -> None:
        super().__init__()
        self._themes = themes

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Choose Theme")
            lv = ListView(*[ListItem(Label(t)) for t in self._themes], id="themes")
            yield lv
            yield Button("Close", id="close")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(str(event.item.children[0].renderable))

    def on_button_pressed(self) -> None:
        self.dismiss("")
