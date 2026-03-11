import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aichat.tool_args import parse_tool_args, _parse_xml_args


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


class TestXmlArgParsing(unittest.TestCase):
    """Tests for the XML arg_key/arg_value format some models emit."""

    def test_basic_xml_args(self) -> None:
        text = "<arg_key>topic</arg_key><arg_value>Iran News</arg_value>"
        result = _parse_xml_args(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["topic"], "Iran News")

    def test_url_encoded_value(self) -> None:
        text = "<arg_key>topic</arg_key><arg_value>Iran%20News</arg_value>"
        result = _parse_xml_args(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["topic"], "Iran News")

    def test_multiple_pairs(self) -> None:
        text = (
            "<arg_key>topic</arg_key><arg_value>Iran%20News</arg_value>"
            "<arg_key>limit</arg_key><arg_value>5</arg_value>"
        )
        result = _parse_xml_args(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["topic"], "Iran News")
        self.assertEqual(result["limit"], 5)

    def test_numeric_coercion(self) -> None:
        text = "<arg_key>max_chars</arg_key><arg_value>4000</arg_value>"
        result = _parse_xml_args(text)
        self.assertEqual(result["max_chars"], 4000)
        self.assertIsInstance(result["max_chars"], int)

    def test_no_pairs_returns_none(self) -> None:
        self.assertIsNone(_parse_xml_args('{"topic": "hello"}'))
        self.assertIsNone(_parse_xml_args(""))
        self.assertIsNone(_parse_xml_args("topic: hello"))

    def test_case_insensitive_tags(self) -> None:
        text = "<ARG_KEY>url</ARG_KEY><ARG_VALUE>https://example.com</ARG_VALUE>"
        result = _parse_xml_args(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["url"], "https://example.com")

    def test_whitespace_between_tags(self) -> None:
        text = "<arg_key>topic</arg_key>  \n  <arg_value>AI News</arg_value>"
        result = _parse_xml_args(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["topic"], "AI News")

    def test_parse_tool_args_dispatches_xml(self) -> None:
        """parse_tool_args should auto-detect XML format."""
        text = "<arg_key>topic</arg_key><arg_value>technology</arg_value>"
        args, error = parse_tool_args("rss_latest", text)
        self.assertIsNone(error)
        self.assertEqual(args["topic"], "technology")

    def test_xml_embedded_in_tool_output(self) -> None:
        """Realistic example: tool name prefix then XML args."""
        text = "rss_latest <arg_key>topic</arg_key><arg_value>sports</arg_value>"
        result = _parse_xml_args(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["topic"], "sports")
