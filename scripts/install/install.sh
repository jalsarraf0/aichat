#!/usr/bin/env bash
# Title: AIChat System Installer
# Purpose: Install AIChat with minimal user effort, including Python/bootstrap dependencies and Docker services.
# NIST 800-53 Controls: CM-6, SI-2, AC-6, AU-2, RA-5
# FIPS Dependencies: No (installer workflow only; no cryptographic implementation)
# Authorisation Boundary: Subsystem
# Written by: Jamal Al-Sarraf
# Last-Reviewed: 2026-02-22
set -Eeuo pipefail

log() { printf '[aichat-install] %s\n' "$*"; }
warn() { printf '[aichat-install][warn] %s\n' "$*" >&2; }
fail() { printf '[aichat-install][error] %s\n' "$*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
[[ -f pyproject.toml ]] || fail "pyproject.toml missing; run inside AIChat repository."

ensure_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1
}

install_pkg() {
  local pkg="$1"
  if ensure_cmd dnf; then sudo dnf install -y "$pkg"; return 0; fi
  if ensure_cmd apt-get; then sudo apt-get update && sudo apt-get install -y "$pkg"; return 0; fi
  if ensure_cmd zypper; then sudo zypper --non-interactive install "$pkg"; return 0; fi
  if ensure_cmd pacman; then sudo pacman -Sy --noconfirm "$pkg"; return 0; fi
  return 1
}

pick_python() {
  for bin in python3.12 python3.11 python3.10 python3; do
    if ensure_cmd "$bin"; then
      "$bin" - <<'PY' >/dev/null 2>&1 || continue
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
      echo "$bin"
      return 0
    fi
  done
  return 1
}

if ! PY_BIN="$(pick_python)"; then
  warn "Python 3.10+ not found. Attempting package installation."
  install_pkg python3 || fail "Unable to install python3 automatically. Please install Python 3.10+ manually."
  PY_BIN="$(pick_python || true)"
fi
[[ -n "${PY_BIN:-}" ]] || fail "No usable Python found after attempted install."

if ! ensure_cmd docker; then
  warn "Docker not found. Attempting package installation."
  install_pkg docker || warn "Could not install docker automatically."
fi
if ! ensure_cmd docker; then
  fail "Docker is required for rss/researchbox services. Install Docker then rerun."
fi

if ! docker compose version >/dev/null 2>&1; then
  fail "docker compose plugin is required. Install Docker Compose plugin and rerun."
fi

log "Using Python interpreter: $PY_BIN"
"$PY_BIN" -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .

mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/aichat" <<WRAP
#!/usr/bin/env bash
exec "$REPO_ROOT/.venv/bin/aichat" "\$@"
WRAP
chmod 0755 "$HOME/.local/bin/aichat"

if [[ ":${PATH}:" != *":$HOME/.local/bin:"* ]]; then
  warn "~/.local/bin is not currently in PATH. Add: export PATH=\"$HOME/.local/bin:\$PATH\""
fi

log "Starting Docker services..."
docker compose up -d --build

log "Install complete."
log "Run now with: $HOME/.local/bin/aichat"
log "Or activate venv: source .venv/bin/activate && aichat"
