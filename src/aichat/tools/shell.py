from __future__ import annotations

import asyncio
import os
import signal
from dataclasses import dataclass


@dataclass
class ShellSession:
    process: asyncio.subprocess.Process


class ShellTool:
    def __init__(self) -> None:
        self.sessions: dict[str, ShellSession] = {}
        self._next_id = 1

    async def sh_start(self, cwd: str | None = None) -> str:
        proc = await asyncio.create_subprocess_exec(
            "bash",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            start_new_session=True,
        )
        sid = str(self._next_id)
        self._next_id += 1
        self.sessions[sid] = ShellSession(proc)
        return sid

    async def sh_send(self, session_id: str, text: str) -> None:
        s = self.sessions[session_id]
        assert s.process.stdin is not None
        s.process.stdin.write(text.encode())
        await s.process.stdin.drain()

    async def sh_read(self, session_id: str, max_bytes: int = 4096, timeout_ms: int = 200) -> str:
        s = self.sessions[session_id]
        assert s.process.stdout is not None
        try:
            data = await asyncio.wait_for(s.process.stdout.read(max_bytes), timeout=timeout_ms / 1000)
        except asyncio.TimeoutError:
            return ""
        return data.decode(errors="replace")

    async def sh_interrupt(self, session_id: str) -> None:
        s = self.sessions[session_id]
        os.killpg(os.getpgid(s.process.pid), signal.SIGINT)

    async def sh_close(self, session_id: str) -> None:
        s = self.sessions.pop(session_id)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            if s.process.returncode is not None:
                return
            os.killpg(os.getpgid(s.process.pid), sig)
            await asyncio.sleep(0.15)
