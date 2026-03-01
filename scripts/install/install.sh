#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

log() { printf '[aichat-install] %s\n' "$*"; }
warn() { printf '[aichat-install][warn] %s\n' "$*" >&2; }
fail() { printf '[aichat-install][error] %s\n' "$*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  fail "Do not run with sudo. Run: bash install.sh"
fi

for script in install.sh uninstall.sh scripts/bin/install scripts/bin/uninstall scripts/install/install.sh scripts/uninstall/uninstall.sh; do
  sed -i 's/\r$//' "$REPO_ROOT/$script"
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  fail "python3 is required. Please install Python 3."
fi

"$PYTHON_BIN" - <<'PY' || fail "Python 3.12+ is required."
import sys
raise SystemExit(0 if sys.version_info >= (3, 12) else 1)
PY

INSTALL_HOME="${HOME}/.local/share/aichat"
VENV_DIR="$INSTALL_HOME/venv"
BIN_DIR="${HOME}/.local/bin"
LAUNCHER="$BIN_DIR/aichat"

mkdir -p "$INSTALL_HOME" "$BIN_DIR"

log "Creating/updating virtual environment at $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -e "$REPO_ROOT"

cat > "$LAUNCHER" <<'WRAP'
#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
exec "$HOME/.local/share/aichat/venv/bin/python" -m aichat "$@"
WRAP
chmod 0755 "$LAUNCHER"

if [[ ":${PATH}:" != *":${HOME}/.local/bin:"* ]]; then
  warn "~/.local/bin is not currently in PATH. Add this line to your shell profile:"
  warn "export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

log "Creating aichat config directories."
mkdir -p "${HOME}/.config/aichat/tools"
mkdir -p "${HOME}/git"

if command -v docker >/dev/null 2>&1; then
  INTEL_VIDEO_GID="$(getent group video 2>/dev/null | cut -d: -f3 || true)"
  INTEL_RENDER_GID="$(getent group render 2>/dev/null | cut -d: -f3 || true)"
  INTEL_DRI_DEVICE="/dev/dri"
  [[ -n "$INTEL_VIDEO_GID" ]] || INTEL_VIDEO_GID="39"
  [[ -n "$INTEL_RENDER_GID" ]] || INTEL_RENDER_GID="105"
  INTEL_GPU_LINE=""
  if command -v lspci >/dev/null 2>&1; then
    INTEL_GPU_LINE="$(lspci -nn 2>/dev/null | grep -Ei 'VGA|3D|Display' | grep -Ei 'Intel|Arc|DG2' | head -1 || true)"
  fi
  if [[ -n "$INTEL_GPU_LINE" ]]; then
    log "Detected Intel GPU: $INTEL_GPU_LINE"
  else
    warn "No Intel GPU line found via lspci. Continuing with generic defaults."
  fi
  if [[ ! -d "$INTEL_DRI_DEVICE" ]]; then
    warn "$INTEL_DRI_DEVICE is missing; GPU acceleration inside containers will be unavailable."
  else
    log "Using DRI path: $INTEL_DRI_DEVICE (video GID=$INTEL_VIDEO_GID, render GID=$INTEL_RENDER_GID)"
  fi
  if ! groups | tr ' ' '\n' | grep -qx docker; then
    warn "Docker is installed, but your user may not be in the 'docker' group."
  fi
  if docker compose version >/dev/null 2>&1; then
    log "Starting docker-backed tools (docker compose up -d --build)."
    if ! INTEL_VIDEO_GID="$INTEL_VIDEO_GID" \
         INTEL_RENDER_GID="$INTEL_RENDER_GID" \
         INTEL_DRI_DEVICE="$INTEL_DRI_DEVICE" \
         docker compose up -d --build; then
      warn "Docker compose failed to start containers. Check docker permissions and that the daemon is running."
    fi
    # Connect human_browser to the aichat compose network so the MCP container
    # can reach it by hostname (human_browser:7081) for screenshot capture.
    AICHAT_NET="$(docker compose ps -q 2>/dev/null | head -1 | xargs -r docker inspect --format '{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{end}}' 2>/dev/null | head -1 || echo '')"
    if [[ -z "$AICHAT_NET" ]]; then
      AICHAT_NET="aichat_default"
    fi
    if docker inspect human_browser >/dev/null 2>&1; then
      if ! docker inspect human_browser --format '{{json .NetworkSettings.Networks}}' | grep -q "\"${AICHAT_NET}\""; then
        log "Connecting human_browser to ${AICHAT_NET} for MCP screenshot access."
        docker network connect "$AICHAT_NET" human_browser 2>/dev/null || warn "Could not connect human_browser to ${AICHAT_NET} (may already be connected)."
      else
        log "human_browser already connected to ${AICHAT_NET}."
      fi
      # Ensure browser_server.py is running inside human_browser (provides screenshot API on :7081).
      # It may not be running after a container restart since the entrypoint only starts VNC/noVNC.
      if ! docker exec human_browser python3 -c "
import urllib.request
try:
    urllib.request.urlopen('http://localhost:7081/health', timeout=3)
    raise SystemExit(0)
except OSError:
    raise SystemExit(1)
" 2>/dev/null; then
        log "Starting browser_server.py inside human_browser (screenshot API)."
        docker exec -d human_browser bash -c \
          "DISPLAY=:99 HOME=/home/ai python3 /workspace/browser_server.py >> /workspace/browser_server.log 2>&1" \
          || warn "Could not start browser_server.py in human_browser."
        # Allow time for Playwright/Chromium startup
        sleep 5
      else
        log "browser_server.py already running in human_browser."
      fi
    else
      warn "human_browser container not found — screenshot tool will be unavailable until it is running."
    fi

    # Best-effort GPU diagnostics for the MCP container. This confirms whether
    # /dev/dri is visible and whether OpenCV reports OpenCL availability.
    MCP_CID="$(docker compose ps -q aichat-mcp 2>/dev/null || true)"
    if [[ -n "$MCP_CID" ]]; then
      if ! docker exec "$MCP_CID" python3 - <<'PY' 2>/dev/null; then
import os
try:
    import cv2
    have = bool(cv2.ocl.haveOpenCL())
    use_before = bool(cv2.ocl.useOpenCL())
    cv2.ocl.setUseOpenCL(True)
    use_after = bool(cv2.ocl.useOpenCL())
    print(f"[aichat-install] MCP cv2={cv2.__version__} opencl_have={have} opencl_use_before={use_before} opencl_use_after={use_after}")
except Exception as exc:
    print(f"[aichat-install][warn] MCP OpenCV/OpenCL check failed: {exc}")
print(f"[aichat-install] MCP /dev/dri present={os.path.isdir('/dev/dri')}")
PY
        warn "MCP GPU diagnostics could not be completed."
      fi
    fi
  else
    warn "Docker is installed, but the 'docker compose' plugin is unavailable or permission was denied."
  fi
else
  warn "Docker is not installed; skipping docker-backed tools."
fi

log "Install complete."
log "Run: $HOME/.local/bin/aichat"
log "WhatsApp bot: visit http://localhost:8097 to scan the QR code and pair a number."
