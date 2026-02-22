#!/usr/bin/env bash
# Title: AIChat System Uninstaller
# Purpose: Remove AIChat virtualenv, launchers, runtime state, and Docker resources created by project installation.
# NIST 800-53 Controls: CM-6, SI-2, AC-6, AU-2
# FIPS Dependencies: No (uninstall workflow only; no cryptographic implementation)
# Authorisation Boundary: Subsystem
# Written by: Jamal Al-Sarraf
# Last-Reviewed: 2026-02-22
set -Eeuo pipefail

log() { printf '[aichat-uninstall] %s\n' "$*"; }
warn() { printf '[aichat-uninstall][warn] %s\n' "$*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  log "Stopping/removing Docker services, volumes, and local images..."
  docker compose down --volumes --rmi local --remove-orphans || warn "docker compose down encountered an issue"
else
  warn "Docker compose unavailable; skipping container cleanup"
fi

if [[ -d .venv ]]; then
  rm -rf .venv
  log "Removed .venv"
fi

rm -f "$HOME/.local/bin/aichat"
rm -f "$HOME/.local/bin/aichat-cli"
rm -f "$HOME/.local/bin/aichat-dev"

rm -rf "$HOME/.config/aichat" "$HOME/.local/share/aichat" "$HOME/.cache/aichat"

if command -v python3 >/dev/null 2>&1; then
  python3 -m pip uninstall -y aichat >/dev/null 2>&1 || true
fi

log "Uninstall complete."
log "Removed project venv, launchers, config/data/cache, and local docker resources."
