from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from .tools.errors import ToolRequestError


@dataclass(frozen=True)
class ToolCall:
    index: int
    name: str
    args: dict[str, object]
    call_id: str
    label: str


@dataclass(frozen=True)
class ToolResult:
    call: ToolCall
    ok: bool
    output: str
    attempts: int
    duration: float
    error: str | None = None


class ToolScheduler:
    def __init__(
        self,
        runner: Callable[[ToolCall], Awaitable[str]],
        *,
        log: Callable[[str], None] | None = None,
        concurrency: int = 1,
        max_attempts: int = 4,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        jitter: Callable[[], float] | None = None,
    ) -> None:
        self.runner = runner
        self.log = log
        self.concurrency = max(1, min(concurrency, 2))
        self.max_attempts = max_attempts
        self.sleep = sleep or asyncio.sleep
        self.jitter = jitter or random.random

    async def run_batch(self, calls: list[ToolCall]) -> list[ToolResult]:
        if not calls:
            return []
        queue: asyncio.Queue[ToolCall] = asyncio.Queue()
        results: dict[int, ToolResult] = {}
        for call in calls:
            queue.put_nowait(call)
            self._log(f"queued [{call.name}] {self._format_args(call.args)}")

        workers = [asyncio.create_task(self._worker(queue, results)) for _ in range(self.concurrency)]
        await queue.join()
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        return [results[call.index] for call in calls if call.index in results]

    async def _worker(self, queue: asyncio.Queue[ToolCall], results: dict[int, ToolResult]) -> None:
        while True:
            call = await queue.get()
            try:
                result = await self._run_with_retry(call)
                results[call.index] = result
            finally:
                queue.task_done()

    async def _run_with_retry(self, call: ToolCall) -> ToolResult:
        start = time.monotonic()
        attempt = 1
        while True:
            self._log(f"running [{call.name}] attempt {attempt}")
            try:
                output = await self.runner(call)
                duration = time.monotonic() - start
                self._log(f"success [{call.name}] {duration:.2f}s")
                return ToolResult(call=call, ok=True, output=output, attempts=attempt, duration=duration)
            except Exception as exc:  # noqa: BLE001
                retryable = isinstance(exc, ToolRequestError) and exc.retryable
                status = getattr(exc, "status_code", None)
                reason = f"{status}" if status else type(exc).__name__
                if retryable and attempt < self.max_attempts:
                    delay = self._backoff_delay(attempt)
                    self._log(f"retry [{call.name}] attempt {attempt + 1} in {delay:.2f}s ({reason})")
                    await self.sleep(delay)
                    attempt += 1
                    continue
                duration = time.monotonic() - start
                error_text = str(exc)
                self._log(f"fail [{call.name}] {duration:.2f}s ({reason})")
                return ToolResult(
                    call=call,
                    ok=False,
                    output="",
                    attempts=attempt,
                    duration=duration,
                    error=error_text,
                )

    def _backoff_delay(self, attempt: int) -> float:
        base = 0.5 * (2 ** (attempt - 1))
        capped = min(base, 8.0)
        jitter = self.jitter() * 0.25 * capped
        return capped + jitter

    def _format_args(self, args: dict[str, object]) -> str:
        try:
            return json.dumps(args, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(args)

    def _log(self, message: str) -> None:
        if self.log:
            self.log(message)
