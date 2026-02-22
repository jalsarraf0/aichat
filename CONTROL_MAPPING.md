# CONTROL_MAPPING

## NIST 800-53 Rev-5 Crosswalk

- **CM-6**
  - `scripts/bootstrap/install_aichat.sh` (standardized installation workflow and interpreter selection)
  - `Makefile` (`security-checks` target enforces consistent checks)

- **SI-2**
  - `scripts/bootstrap/install_aichat.sh` (`pip/setuptools/wheel` upgrade + deterministic dependency installation)
  - `.github/workflows/security-checks.yml` (automated static/security checks on every push/PR)

- **AC-6**
  - `scripts/bootstrap/install_aichat.sh` (virtualenv-based least-privilege install instead of global/system site-packages)

- **AU-2**
  - `scripts/bootstrap/install_aichat.sh` (structured logging through `log`/`fail` functions)

- **RA-5**
  - `Makefile` (`bandit`, `safety`, `semgrep`, `trivy`) 
  - `.github/workflows/security-checks.yml` (automated execution in CI)


- **CM-6**
  - `scripts/install/install.sh` (standardized install flow including launcher + docker startup)
  - `scripts/uninstall/uninstall.sh` (standardized teardown/removal workflow)

- **SI-2**
  - `scripts/install/install.sh` (dependency/toolchain remediation with venv bootstrap)
  - `scripts/uninstall/uninstall.sh` (clean removal of stale installations and runtime artifacts)

- **AC-6**
  - `scripts/install/install.sh` (launcher installation at `/usr/local/bin` with least-privilege fallback to `$HOME/.local/bin`)
  - `scripts/uninstall/uninstall.sh` (removes system and user-scoped launchers and local state)

- **AU-2**
  - `scripts/install/install.sh` (`log/warn/fail` operational logging)
  - `scripts/uninstall/uninstall.sh` (`log/warn` teardown logging)
