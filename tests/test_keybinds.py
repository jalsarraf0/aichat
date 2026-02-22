import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aichat.ui.keybinds import KEYBINDS, display_key


class TestKeybindOrdering(unittest.TestCase):
    def test_ordering(self) -> None:
        keys = [display_key(spec.key) for spec in KEYBINDS]
        self.assertEqual(keys[:12], [f"F{i}" for i in range(1, 13)])
        self.assertEqual(keys[-2:], ["^S", "^G"])
