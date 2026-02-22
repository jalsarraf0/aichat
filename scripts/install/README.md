# AIChat Install Script

## What it does

`scripts/install/install.sh` performs a full user-friendly install:

- Detects/installs Python (best effort via distro package manager).
- Creates project `.venv` and installs AIChat.
- Installs `aichat` launcher in `/usr/local/bin` when allowed, fallback to `$HOME/.local/bin`.
- Starts Docker services with `docker compose up -d --build`.

## Usage

```bash
./install.sh
# or
bash scripts/install/install.sh
```

## NIST control mapping

- CM-6: consistent install configuration.
- SI-2: dependency update/install workflow.
- AC-6: local venv with least-privilege launcher placement and fallback.
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
