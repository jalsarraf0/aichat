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

if command -v docker >/dev/null 2>&1; then
  if docker compose version >/dev/null 2>&1; then
    # Preserve the PostgreSQL data volume (aichatdb) so stored articles, images,
    # and web-cache survive reinstalls.  Use --volumes only for the memory store.
    log "Stopping docker-backed tools (preserving aichatdb PostgreSQL volume)."
    if ! docker compose down --remove-orphans; then
      warn "Docker compose failed to stop containers. Check docker permissions and that the daemon is running."
    fi
  else
    warn "Docker is installed, but the 'docker compose' plugin is unavailable or permission was denied."
  fi
else
  warn "Docker is not installed; skipping docker-backed tools cleanup."
fi

rm -f "$HOME/.local/bin/aichat"
rm -rf "$HOME/.local/share/aichat/venv"

log "Uninstall complete."
