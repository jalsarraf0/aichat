# Bootstrap Installer (`install_aichat.sh`)

## What this script does

`install_aichat.sh` provides a safe, repeatable install path for AIChat:

- Verifies the command is run from the repository root (checks `pyproject.toml`).
- Finds a usable Python interpreter (`python3.12`, `python3.11`, `python3.10`, then `python3`).
- Creates `.venv`, upgrades packaging tooling, and installs AIChat in editable mode.
- Prints exact activation/run commands on success.

## Prerequisites

- Linux shell (bash)
- Python 3.10+ available in `PATH`
- Internet access for pip dependency resolution

## Usage

From repository root:

```bash
./scripts/bootstrap/install_aichat.sh
```

Or from any directory:

```bash
bash /path/to/aichat/scripts/bootstrap/install_aichat.sh
```

Optional override:

```bash
PYTHON_BIN=python3.12 ./scripts/bootstrap/install_aichat.sh
```

## NIST 800-53 mapping

- **CM-6 (Configuration Settings):** standardized install flow enforces consistent local setup.
- **SI-2 (Flaw Remediation):** updates pip/setuptools/wheel before install.
- **AC-6 (Least Privilege):** installs into local virtualenv rather than system interpreter.
- **AU-2 (Event Logging):** script emits deterministic operational logs.

## Embedded lint/security checks

Run repository checks from root:

```bash
make security-checks
```

This target executes:

- `shellcheck`, `shfmt` for bash
- `bandit`, `safety`, `semgrep` for python
- `trivy fs` for repository scanning

## Expected output and exit codes

- `0`: install completed
- non-zero: failure (missing python, bad working directory, pip failure, etc.)

## Maintenance notes

- CI workflow is `Security Checks` in `.github/workflows/security-checks.yml`.
- Keep interpreter preference list aligned with project support policy.
