"""
pytest conftest — auto-discover aichat service URLs.

Priority order for each service URL:
  1. Explicit env var (CI / manual override)
  2. Docker container IP auto-detected via docker inspect
  3. Localhost fallback (for port-published setups)

This means `pytest tests/test_full_regression.py -m regression` works from the
host without any env setup, as long as Docker is running.
"""
from __future__ import annotations

import os
import subprocess


# Container-name → (env-var, internal-port)
_SERVICES = {
    "aichat-aichat-data-1":    ("DATA_URL",    8091),
    "aichat-aichat-vision-1":  ("VISION_URL",  8099),
    "aichat-aichat-docs-1":    ("DOCS_URL",    8101),
    "aichat-aichat-sandbox-1": ("SANDBOX_URL", 8095),
    "aichat-aichat-mcp-1":     ("MCP_URL",     8096),
    "aichat-aichat-jupyter-1": ("JUPYTER_URL", 8098),
}


def _docker_ip(container: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["docker", "inspect", "--format",
             "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
             container],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        ip = out.decode().strip()
        return ip if ip else None
    except Exception:
        return None


def pytest_configure(config):  # noqa: ARG001 - config param required by pytest hookspec
    """Inject service URLs into os.environ before any test module is imported."""
    for container, (env_var, port) in _SERVICES.items():
        if os.environ.get(env_var):
            continue  # already set — respect explicit override
        ip = _docker_ip(container)
        if ip:
            os.environ[env_var] = f"http://{ip}:{port}"
        # else: leave unset → test module falls back to localhost default
