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

if command -v docker >/dev/null 2>&1; then
  if ! groups | tr ' ' '\n' | grep -qx docker; then
    warn "Docker is installed, but your user may not be in the 'docker' group."
  fi
  if docker compose version >/dev/null 2>&1; then
    log "Starting docker-backed tools (docker compose up -d --build)."
    if ! docker compose up -d --build; then
      warn "Docker compose failed to start containers. Check docker permissions and that the daemon is running."
    fi
  else
    warn "Docker is installed, but the 'docker compose' plugin is unavailable or permission was denied."
  fi
else
  warn "Docker is not installed; skipping docker-backed tools."
fi

log "Install complete."
log "Run: $HOME/.local/bin/aichat"
