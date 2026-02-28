"""CodeInterpreterTool — sandboxed Python subprocess executor.

Runs arbitrary Python code in a subprocess with:
  - configurable wall-clock timeout (default 30 s)
  - separate stdout/stderr capture
  - optional pip-install of extra packages before execution
  - tempfile cleanup in a finally block

Security note: this runs code directly; use only in trusted contexts (the AI
assistant calling its own generated code, not arbitrary user input).
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time


class CodeInterpreterError(RuntimeError):
    pass


class CodeInterpreterTool:
    """Sandboxed Python execution via subprocess.

    Parameters
    ----------
    timeout:
        Maximum wall-clock seconds allowed for code execution (default 30).
    python:
        Python interpreter path (defaults to the same interpreter running this
        process, so installed packages are available).
    """

    def __init__(
        self,
        timeout: int = 30,
        python: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.python  = python or sys.executable

    async def run(
        self,
        code: str,
        packages: list[str] | None = None,
        timeout: int | None = None,
    ) -> dict:
        """Execute *code* in a subprocess and return a result dict.

        Parameters
        ----------
        code:
            Python source code to execute.
        packages:
            Optional list of pip package names to install before running.
            Each install is attempted with a 5 s timeout and best-effort
            (failure is logged in the result but does not abort execution).
        timeout:
            Per-call override for the wall-clock limit (seconds).

        Returns
        -------
        dict with keys:
            stdout (str), stderr (str), exit_code (int), duration_ms (int),
            install_log (str)  — pip install output, if any.
        """
        max_time   = timeout if timeout is not None else self.timeout
        install_log: list[str] = []
        tmp_path:   str | None = None
        t0 = time.monotonic()

        try:
            # ── Optional package installation ──────────────────────────────
            if packages:
                for pkg in packages:
                    pkg = pkg.strip()
                    if not pkg:
                        continue
                    try:
                        proc_pip = await asyncio.create_subprocess_exec(
                            self.python, "-m", "pip", "install", "--quiet", pkg,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.STDOUT,
                        )
                        out_pip, _ = await asyncio.wait_for(
                            proc_pip.communicate(), timeout=5.0
                        )
                        install_log.append(
                            f"pip install {pkg}: exit {proc_pip.returncode}\n"
                            + (out_pip.decode(errors="replace") if out_pip else "")
                        )
                    except asyncio.TimeoutError:
                        install_log.append(f"pip install {pkg}: timed out (5s)")
                    except Exception as exc:
                        install_log.append(f"pip install {pkg}: {exc}")

            # ── Write code to tempfile ─────────────────────────────────────
            with tempfile.NamedTemporaryFile(
                suffix=".py", mode="w", encoding="utf-8", delete=False
            ) as f:
                f.write(code)
                tmp_path = f.name

            # ── Execute ───────────────────────────────────────────────────
            elapsed_install = time.monotonic() - t0
            remaining       = max(1, max_time - int(elapsed_install))

            proc = await asyncio.create_subprocess_exec(
                self.python, tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ},
            )

            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=float(remaining)
                )
                exit_code = proc.returncode
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return {
                    "stdout":      "",
                    "stderr":      f"Execution timed out after {remaining}s.",
                    "exit_code":   -1,
                    "duration_ms": int((time.monotonic() - t0) * 1000),
                    "install_log": "\n".join(install_log),
                    "error":       "timeout",
                }

            duration_ms = int((time.monotonic() - t0) * 1000)
            return {
                "stdout":      stdout_b.decode(errors="replace"),
                "stderr":      stderr_b.decode(errors="replace"),
                "exit_code":   exit_code,
                "duration_ms": duration_ms,
                "install_log": "\n".join(install_log),
                "error":       None,
            }

        except Exception as exc:
            return {
                "stdout":      "",
                "stderr":      str(exc),
                "exit_code":   -1,
                "duration_ms": int((time.monotonic() - t0) * 1000),
                "install_log": "\n".join(install_log),
                "error":       str(exc),
            }

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
