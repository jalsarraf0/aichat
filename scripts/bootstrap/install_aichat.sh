#!/usr/bin/env bash
# Title: AIChat Bootstrap Installer
# Purpose: Install AIChat reliably on Linux by locating Python, creating a venv, and installing editable package deps.
# NIST 800-53 Controls: CM-6, SI-2, AC-6, AU-2
# FIPS Dependencies: No (installer only; no cryptography primitives implemented)
# Authorisation Boundary: Subsystem
# Written by: Jamal Al-Sarraf
# Last-Reviewed: 2026-02-22
set -Eeuo pipefail

log() { printf '[aichat-install] %s\n' "$*"; }
fail() { printf '[aichat-install][error] %s\n' "$*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
[[ -f pyproject.toml ]] || fail "Run this from inside the AIChat repository (pyproject.toml not found)."

pick_python() {
  for bin in python3.12 python3.11 python3.10 python3; do
    if command -v "$bin" >/dev/null 2>&1; then
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

PY_BIN="${PYTHON_BIN:-}"
if [[ -z "$PY_BIN" ]]; then
  PY_BIN="$(pick_python || true)"
fi
[[ -n "$PY_BIN" ]] || fail "No usable Python found. Install python3.12+ (or 3.10+) and retry."

log "Using Python interpreter: $PY_BIN"
"$PY_BIN" -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .

log "Install completed."
log "Activate with: source .venv/bin/activate"
log "Run with: aichat"
