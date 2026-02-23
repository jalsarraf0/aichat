import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aichat.shell_paths import detect_project_name, rewrite_cd_commands


class TestShellPaths(unittest.TestCase):
    def test_detect_project_name_simple(self) -> None:
        text = "Please create a directory called caesar"
        self.assertEqual(detect_project_name(text), "caesar")

    def test_detect_project_name_quoted(self) -> None:
        text = "make project named \"My Project\""
        self.assertEqual(detect_project_name(text), "My Project")

    def test_rewrite_cd_under_root(self) -> None:
        root = Path("~/git")
        locked = root / "caesar"
        command = "cd ~/git/caesar_cipher3\nls"
        rewritten = rewrite_cd_commands(command, root, locked)
        self.assertIn(str(locked), rewritten)
        self.assertNotIn("caesar_cipher3", rewritten)

    def test_rewrite_cd_outside_root(self) -> None:
        root = Path("~/git")
        locked = root / "caesar"
        command = "cd /tmp\nls"
        rewritten = rewrite_cd_commands(command, root, locked)
        self.assertEqual(command, rewritten)
