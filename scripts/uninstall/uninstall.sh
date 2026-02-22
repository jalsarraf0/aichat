#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

log() { printf '[aichat-uninstall] %s\n' "$*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

for script in install.sh uninstall.sh scripts/bin/install scripts/bin/uninstall scripts/install/install.sh scripts/uninstall/uninstall.sh; do
  sed -i 's/\r$//' "$REPO_ROOT/$script"
done

rm -f "$HOME/.local/bin/aichat"
rm -rf "$HOME/.local/share/aichat/venv"

log "Uninstall complete."
