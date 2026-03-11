"""aichat-jupyter: persistent Python kernel service.

Each session_id gets its own isolated Jupyter kernel.
Kernels survive between HTTP requests — variables, imports, DataFrames,
and trained models all persist within a session.

Endpoints:
  POST /exec          — execute code, return stdout/stderr/outputs/plots
  GET  /sessions      — list active kernels
  DELETE /sessions/{id} — shut down a kernel
  GET  /health        — liveness probe
"""
from __future__ import annotations

import asyncio
import base64
import logging
import queue
import threading
import time
from typing import Any

import jupyter_client
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("aichat-jupyter")

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_API_KEY = os.environ.get("JUPYTER_API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_key(key: str | None = Security(_api_key_header)) -> None:
    if not _API_KEY:
        return  # key auth disabled when env var is unset (internal-only mode)
    if key != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


app = FastAPI(title="aichat-jupyter")

# ---------------------------------------------------------------------------
# Kernel registry
# ---------------------------------------------------------------------------

# session_id → (KernelManager, KernelClient)
_kernels: dict[str, tuple[jupyter_client.KernelManager, jupyter_client.KernelClient]] = {}
_lock = threading.Lock()

_SETUP_CODE = """\
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
"""


def _start_kernel(session_id: str) -> tuple:
    km = jupyter_client.KernelManager(kernel_name="python3")
    km.start_kernel()
    kc = km.client()
    kc.start_channels()
    try:
        kc.wait_for_ready(timeout=30)
    except RuntimeError:
        pass  # kernel may still be usable

    # Run setup code (matplotlib Agg backend, etc.) — drain outputs
    kc.execute(_SETUP_CODE)
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            msg = kc.get_iopub_msg(timeout=0.5)
            if (msg.get("msg_type") == "status" and
                    msg.get("content", {}).get("execution_state") == "idle"):
                break
        except queue.Empty:
            break

    _kernels[session_id] = (km, kc)
    log.info("started kernel session=%s", session_id)
    return km, kc


def _get_or_create_kernel(session_id: str, reset: bool = False) -> tuple:
    with _lock:
        if reset and session_id in _kernels:
            _shutdown_kernel(session_id)
        if session_id not in _kernels:
            return _start_kernel(session_id)
        return _kernels[session_id]


def _shutdown_kernel(session_id: str) -> bool:
    """Shut down kernel for session_id. Must be called with _lock held."""
    if session_id not in _kernels:
        return False
    km, kc = _kernels.pop(session_id)
    try:
        kc.stop_channels()
    except Exception:
        pass
    try:
        km.shutdown_kernel(now=True)
    except Exception:
        pass
    log.info("shut down kernel session=%s", session_id)
    return True


# ---------------------------------------------------------------------------
# Synchronous execution (runs in thread executor)
# ---------------------------------------------------------------------------

def _execute_sync(session_id: str, code: str, timeout: int, reset: bool) -> dict[str, Any]:
    import re as _re_jup
    km, kc = _get_or_create_kernel(session_id, reset=reset)

    msg_id = kc.execute(code)

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    text_outputs: list[str] = []
    images: list[str] = []       # base64-encoded PNGs
    error_text: str | None = None

    deadline = time.time() + timeout
    idle_seen = False

    while time.time() < deadline and not idle_seen:
        try:
            remaining = max(0.1, deadline - time.time())
            msg = kc.get_iopub_msg(timeout=min(1.0, remaining))
        except queue.Empty:
            continue

        # Only process messages belonging to our execute request
        parent_msg_id = msg.get("parent_header", {}).get("msg_id", "")
        if parent_msg_id != msg_id:
            continue

        msg_type = msg.get("msg_type", "")
        content  = msg.get("content", {})

        if msg_type == "stream":
            if content.get("name") == "stdout":
                stdout_parts.append(content.get("text", ""))
            elif content.get("name") == "stderr":
                stderr_parts.append(content.get("text", ""))

        elif msg_type in ("execute_result", "display_data"):
            data = content.get("data", {})
            if "image/png" in data:
                images.append(data["image/png"])   # already base64
            if "text/plain" in data:
                text_outputs.append(data["text/plain"])

        elif msg_type == "error":
            tb = content.get("traceback", [])
            clean = _re_jup.sub(r"\x1b\[[0-9;]*m", "", "\n".join(tb))
            error_text = clean or content.get("evalue", "unknown error")

        elif msg_type == "status":
            if content.get("execution_state") == "idle":
                idle_seen = True

    if not idle_seen:
        error_text = (error_text or "") + f"\n(execution may have timed out after {timeout}s)"

    return {
        "stdout":  "".join(stdout_parts),
        "stderr":  "".join(stderr_parts),
        "outputs": text_outputs,
        "images":  images,
        "error":   error_text,
        "session_id": session_id,
    }


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.post("/exec")
async def execute(payload: dict, _: None = Depends(_require_key)) -> dict:
    session_id = str(payload.get("session_id", "default")).strip() or "default"
    code       = str(payload.get("code", "")).strip()
    timeout    = max(1, min(300, int(payload.get("timeout", 60))))
    reset      = bool(payload.get("reset", False))

    if not code:
        return {"error": "'code' is required", "stdout": "", "stderr": "",
                "outputs": [], "images": [], "session_id": session_id}

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, _execute_sync, session_id, code, timeout, reset
    )
    return result


@app.get("/sessions")
def list_sessions(_: None = Depends(_require_key)) -> dict:
    return {"sessions": list(_kernels.keys()), "count": len(_kernels)}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, _: None = Depends(_require_key)) -> dict:
    with _lock:
        if _shutdown_kernel(session_id):
            return {"deleted": session_id}
    raise HTTPException(status_code=404, detail=f"session '{session_id}' not found")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "sessions": len(_kernels), "service": "aichat-jupyter"}
