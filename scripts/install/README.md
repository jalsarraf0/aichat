# AIChat Install Script

## What it does

`scripts/install/install.sh` performs a full user-friendly install:

- Requires Python 3.12+ and uses it to create/update `$HOME/.local/share/aichat/venv`.
- Installs AIChat in editable mode into that venv.
- Creates `$HOME/.local/bin/aichat` launcher so command works without activating a venv.
- Creates `~/.config/aichat/tools` and `~/git` directories if missing.
- Starts Docker services with `docker compose up -d --build` (best effort).

## Usage

```bash
./install.sh
# or
bash scripts/install/install.sh
```

## NIST control mapping

- CM-6: consistent install configuration.
- SI-2: dependency update/install workflow.
- AC-6: local venv and user-space launcher.
- AU-2: structured installer logs.
- RA-5: integrates with repo security checks.

## Security checks

```bash
make security-checks
```

## Exit codes

- `0` success
- non-zero on missing requirements or install errors

## Maintenance notes

- Installer tries `dnf`, `apt-get`, `zypper`, `pacman` in that order.
- Docker must be present with compose plugin.
