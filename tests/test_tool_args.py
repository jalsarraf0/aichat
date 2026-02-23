import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aichat.tool_args import parse_tool_args


class TestToolArgs(unittest.TestCase):
    def test_shell_args_jsonish_multiline(self) -> None:
        args_text = """{"command": "cat > /tmp/test.py <<'PY'
print(\"hi\")
PY
python /tmp/test.py
"}"""
        args, error = parse_tool_args("shell_exec", args_text)
        self.assertIsNone(error)
        command = args.get("command", "")
        self.assertIn("cat > /tmp/test.py", command)
        self.assertNotIn("{\"command\"", command)
        self.assertIn("\nprint", command)

    def test_shell_args_yaml_block(self) -> None:
        args_text = """command: |
  echo one
  echo two
"""
        args, error = parse_tool_args("shell_exec", args_text)
        self.assertIsNone(error)
        self.assertEqual(args.get("command"), "echo one\necho two")

    def test_shell_args_json(self) -> None:
        args_text = "{\"command\": \"echo ok\"}"
        args, error = parse_tool_args("shell_exec", args_text)
        self.assertIsNone(error)
        self.assertEqual(args.get("command"), "echo ok")
