import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aichat.tool_scheduler import ToolCall, ToolScheduler
from aichat.tools.errors import ToolRequestError


class TestToolScheduler(unittest.IsolatedAsyncioTestCase):
    async def test_retry_on_429(self) -> None:
        attempts = 0
        sleeps: list[float] = []

        async def runner(call: ToolCall) -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ToolRequestError("rate limit", status_code=429, retryable=True)
            return "ok"

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        scheduler = ToolScheduler(runner, sleep=fake_sleep, jitter=lambda: 0, max_attempts=4)
        call = ToolCall(index=0, name="rss_latest", args={"topic": "x"}, call_id="", label="rss_latest")
        results = await scheduler.run_batch([call])
        self.assertEqual(attempts, 3)
        self.assertEqual(len(sleeps), 2)
        self.assertTrue(results[0].ok)

    async def test_caps_attempts(self) -> None:
        sleeps: list[float] = []

        async def runner(call: ToolCall) -> str:
            raise ToolRequestError("rate limit", status_code=429, retryable=True)

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        scheduler = ToolScheduler(runner, sleep=fake_sleep, jitter=lambda: 0, max_attempts=3)
        call = ToolCall(index=0, name="rss_latest", args={"topic": "x"}, call_id="", label="rss_latest")
        results = await scheduler.run_batch([call])
        self.assertFalse(results[0].ok)
        self.assertEqual(results[0].attempts, 3)
