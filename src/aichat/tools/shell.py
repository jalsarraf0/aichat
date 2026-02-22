from __future__ import annotations

import asyncio
import os
import signal
from dataclasses import dataclass


class ShellToolError(RuntimeError):
    pass


@dataclass
class ShellSession:
    process: asyncio.subprocess.Process
    cwd: str


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
        self.sessions[sid] = ShellSession(process=proc, cwd=cwd or os.getcwd())
        return sid

    def _session(self, session_id: str) -> ShellSession:
        if session_id not in self.sessions:
            raise ShellToolError(f"Unknown shell session: {session_id}")
        return self.sessions[session_id]

    async def sh_send(self, session_id: str, text: str) -> None:
        session = self._session(session_id)
        if session.process.stdin is None:
            raise ShellToolError("Shell session stdin unavailable")
        session.process.stdin.write(text.encode())
        await session.process.stdin.drain()

    async def sh_read(self, session_id: str, max_bytes: int = 8192, timeout_ms: int = 250) -> str:
        session = self._session(session_id)
        if session.process.stdout is None:
            raise ShellToolError("Shell session stdout unavailable")
        try:
            data = await asyncio.wait_for(session.process.stdout.read(max_bytes), timeout=timeout_ms / 1000)
        except asyncio.TimeoutError:
            return ""
        return data.decode(errors="replace")

    async def sh_interrupt(self, session_id: str) -> None:
        session = self._session(session_id)
        if session.process.returncode is None:
            os.killpg(os.getpgid(session.process.pid), signal.SIGINT)

    async def sh_close(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is None:
            return
        process = session.process
        if process.returncode is not None:
            return
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            if process.returncode is not None:
                return
            os.killpg(os.getpgid(process.pid), sig)
            await asyncio.sleep(0.2)

    async def close_all(self) -> None:
        for session_id in list(self.sessions):
            await self.sh_close(session_id)
