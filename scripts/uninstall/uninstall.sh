#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

log() { printf '[aichat-uninstall] %s\n' "$*"; }
warn() { printf '[aichat-uninstall][warn] %s\n' "$*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

for script in install.sh uninstall.sh scripts/bin/install scripts/bin/uninstall scripts/install/install.sh scripts/uninstall/uninstall.sh; do
  sed -i 's/\r$//' "$REPO_ROOT/$script"
done

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  log "Stopping and removing plugin containers..."
  if ! docker compose down --volumes --remove-orphans; then
    warn "docker compose down failed; continuing uninstall"
  fi
else
  warn "Docker compose unavailable; skipping container teardown"
fi

rm -f "$HOME/.local/bin/aichat"
rm -rf "$HOME/.local/share/aichat/venv"

log "Uninstall complete."
