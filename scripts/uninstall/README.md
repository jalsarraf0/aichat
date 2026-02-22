# AIChat Uninstall Script

## What it does

`scripts/uninstall/uninstall.sh` removes local AIChat traces:

- Stops/removes Docker stack, volumes, and local images.
- Deletes project `.venv`.
- Removes launcher from `/usr/local/bin` and user-level launcher paths.
- Removes config/data/cache directories under `$HOME`.

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
