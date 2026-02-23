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

    def test_cd_home_and_expandvars(self) -> None:
        home = Path.home()
        target = home / "aichat_home_test"
        if target.exists():
            target.rmdir()
        ensure_project_dirs("cd ~\ncd aichat_home_test", None)
        self.assertTrue(target.exists())

    def test_cd_home_tilde_path(self) -> None:
        target = Path.home() / "aichat_home_tilde"
        if target.exists():
            target.rmdir()
        ensure_project_dirs("cd ~/aichat_home_tilde", None)
        self.assertTrue(target.exists())

    def test_cd_with_double_dash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "dash"
            ensure_project_dirs(f"cd -- {target}", None)
            self.assertTrue(target.exists())

    def test_cd_dash_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "-"
            if target.exists():
                target.rmdir()
            ensure_project_dirs("cd -\ncd other", tmp)
            self.assertFalse(target.exists())
            self.assertTrue((Path(tmp) / "other").exists())

    def test_cd_with_spaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "dir with spaces"
            command = f'cd "{target}"\npwd'
            ensure_project_dirs(command, None)
            self.assertTrue(target.exists())

    def test_pushd_creates_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "pushd_dir"
            ensure_project_dirs(f"pushd {target}", None)
            self.assertTrue(target.exists())
