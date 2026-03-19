import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aichat.github_repo import select_working_ssh_key


class DummyResult:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def make_runner(success_key: str):
    def runner(cmd, capture_output=True, text=True, env=None):
        key_path = cmd[cmd.index("-i") + 1]
        if Path(key_path).name == success_key:
            return DummyResult(stdout="Hi! You've successfully authenticated.", returncode=1)
        return DummyResult(stderr="Permission denied", returncode=255)

    return runner


class TestSSHKeySelection(unittest.TestCase):
    def test_prefers_ed25519(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ssh_dir = Path(tmp)
            (ssh_dir / "id_ed25519").write_text("dummy")
            (ssh_dir / "id_rsa").write_text("dummy")
            key = select_working_ssh_key(ssh_dir, runner=make_runner("id_ed25519"))
            self.assertIsNotNone(key)
            self.assertEqual(key.name, "id_ed25519")

    def test_falls_back_to_rsa(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ssh_dir = Path(tmp)
            (ssh_dir / "id_ed25519").write_text("dummy")
            (ssh_dir / "id_rsa").write_text("dummy")
            key = select_working_ssh_key(ssh_dir, runner=make_runner("id_rsa"))
            self.assertIsNotNone(key)
            self.assertEqual(key.name, "id_rsa")
