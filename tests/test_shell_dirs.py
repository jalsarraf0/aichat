import sys
import unittest
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aichat.tools.manager import ensure_project_dirs


class TestShellDirs(unittest.TestCase):
    def test_creates_cwd_under_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            target = root / "proj"
            ensure_project_dirs("", str(target), root=root)
            self.assertTrue(target.exists())

    def test_creates_cd_target_under_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            target = root / "proj2"
            command = f"cd {target}\npython -V"
            ensure_project_dirs(command, None, root=root)
            self.assertTrue(target.exists())

    def test_does_not_create_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            outside = Path(tmp) / "outside"
            command = f"cd {outside}\nls"
            ensure_project_dirs(command, None, root=root)
            self.assertFalse(outside.exists())
