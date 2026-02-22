from __future__ import annotations

from pathlib import Path
import yaml

CONFIG_PATH = Path.home() / ".config" / "aichat" / "config.yml"

DEFAULTS = {
    "base_url": "http://localhost:1234",
    "model": "local-model",
    "theme": "cyberpunk",
    "approval": "ASK",
    "allow_host_shell": False,
}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return DEFAULTS.copy()
    data = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    cfg = DEFAULTS.copy()
    cfg.update(data)
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=True))
