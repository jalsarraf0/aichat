import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aichat.sanitizer import sanitize_response


class TestSanitizer(unittest.TestCase):
    def test_strip_think_blocks(self) -> None:
        text = "Hello<think>secret\nline</think>World"
        result = sanitize_response(text)
        self.assertNotIn("secret", result.text)
        self.assertIn("Hello", result.text)

    def test_strip_model_tags(self) -> None:
        result = sanitize_response("<model-x> hello")
        self.assertTrue(result.text.startswith("hello"))

    def test_detect_structured(self) -> None:
        result = sanitize_response('{"a":1,"b":2}')
        self.assertTrue(result.structured_hidden)

    # --- XML arg-tag leakage (the reported bug) ---

    def test_strip_residual_arg_close_tags(self) -> None:
        """Closing arg tags mid-line must not reach the bubble."""
        raw = "rss_latest topic</arg_key> Iran%20News</arg_value> limit</arg_key> 5</arg_value>"
        result = sanitize_response(raw)
        self.assertNotIn("</arg_key>", result.text)
        self.assertNotIn("</arg_value>", result.text)

    def test_residual_arg_tags_mark_structured(self) -> None:
        """Content with leftover closing arg tags is flagged as structured_hidden."""
        raw = "researchbox_show_news topic</arg_key> Iran News</arg_value>"
        result = sanitize_response(raw)
        self.assertTrue(result.structured_hidden)

    def test_full_inline_tool_blob_suppressed(self) -> None:
        """A complete <tool_call>...</tool_call> blob is removed and flagged."""
        raw = (
            "<tool_call>rss_latest"
            "<arg_key>topic</arg_key><arg_value>AI</arg_value>"
            "</tool_call>"
        )
        result = sanitize_response(raw)
        self.assertNotIn("arg_key", result.text)
        self.assertNotIn("arg_value", result.text)

    def test_opening_arg_tags_stripped(self) -> None:
        raw = "<arg_key>topic</arg_key><arg_value>Sports</arg_value> some trailing text"
        result = sanitize_response(raw)
        self.assertNotIn("<arg_key>", result.text)
        self.assertNotIn("<arg_value>", result.text)
        self.assertNotIn("</arg_key>", result.text)
        self.assertNotIn("</arg_value>", result.text)

    def test_function_calls_blob_suppressed(self) -> None:
        raw = "<function_calls><invoke><tool>web_fetch</tool></invoke></function_calls>"
        result = sanitize_response(raw)
        self.assertNotIn("<invoke>", result.text)

    def test_clean_text_unchanged(self) -> None:
        """Plain prose should pass through with no modification."""
        raw = "The capital of France is Paris."
        result = sanitize_response(raw)
        self.assertEqual(result.text, raw)
        self.assertFalse(result.structured_hidden)

    def test_glm4_box_token_unwrapped(self) -> None:
        raw = "<|begin_of_box|>Hello world<|end_of_box|>"
        result = sanitize_response(raw)
        self.assertEqual(result.text, "Hello world")
        self.assertFalse(result.structured_hidden)

    def test_pipe_tokens_stripped(self) -> None:
        raw = "<|im_start|>assistant\nHello<|im_end|>"
        result = sanitize_response(raw)
        self.assertNotIn("<|", result.text)
        self.assertIn("Hello", result.text)
