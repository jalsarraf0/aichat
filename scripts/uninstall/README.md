# AIChat Uninstall Script

## What it does

`scripts/uninstall/uninstall.sh` removes local AIChat traces:

- Stops/removes the AIChat Docker stack (`docker compose down --remove-orphans`).
- Preserves named volumes (including `aichatdb`) so data survives reinstall.
- Removes launcher script from `$HOME/.local/bin/aichat`.
- Removes installed venv at `$HOME/.local/share/aichat/venv`.

## Usage

```bash
./uninstall.sh
# or
bash scripts/uninstall/uninstall.sh
```

## NIST control mapping

- CM-6: controlled teardown path.
- SI-2: clean removal of installed components.
- AC-6: removes user-space privileges/artifacts.
- AU-2: uninstaller logs for operator visibility.

## Security checks

```bash
make security-checks
```

## Exit codes

- `0` success
- non-zero on unexpected script failure

## Maintenance notes

- Docker cleanup is best effort when engine/compose is unavailable.
