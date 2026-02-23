import sys
import unittest
from pathlib import Path
import tempfile
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aichat.tools.manager import ensure_project_dirs


class TestShellDirs(unittest.TestCase):
    def test_creates_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "proj"
            ensure_project_dirs("", str(target))
            self.assertTrue(target.exists())

    def test_creates_cd_target_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "proj2"
            command = f"cd {target}\npython -V"
            ensure_project_dirs(command, None)
            self.assertTrue(target.exists())

    def test_creates_cd_target_relative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            command = "cd relproj\nls"
            ensure_project_dirs(command, str(base))
            self.assertTrue((base / "relproj").exists())

    def test_creates_cd_after_preamble(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            command = "set -e\ncd relproj\npwd"
            ensure_project_dirs(command, str(base))
            self.assertTrue((base / "relproj").exists())

    def test_creates_chained_cd_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            command = f"cd {base} && cd child\npwd"
            ensure_project_dirs(command, None)
            self.assertTrue((base / "child").exists())

    def test_creates_relative_cd_from_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            base.mkdir(parents=True)
            original = Path.cwd()
            try:
                os.chdir(base)
                command = "cd nested && pwd"
                ensure_project_dirs(command, None)
                self.assertTrue((base / "nested").exists())
            finally:
                os.chdir(original)
