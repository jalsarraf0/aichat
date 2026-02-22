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
