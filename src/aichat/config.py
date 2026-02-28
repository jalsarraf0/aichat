from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .state import ApprovalMode
from .personalities import DEFAULT_PERSONALITY_ID, default_personalities, merge_personalities, normalize_personalities

# Default LM Studio endpoint – override via LM_STUDIO_URL env var or config file.
_DEFAULT_BASE_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234")

CONFIG_PATH = Path.home() / ".config" / "aichat" / "config.yml"


@dataclass(frozen=True)
class AppConfig:
    base_url: str = _DEFAULT_BASE_URL
    model: str = "local-model"
    theme: str = "cyberpunk"
    approval: str = ApprovalMode.AUTO.value
    concise_mode: bool = False
    shell_enabled: bool = True
    active_personality: str = DEFAULT_PERSONALITY_ID
    personalities: list[dict[str, str]] = field(default_factory=default_personalities)
    project_root: str = str(Path.home() / "git")
    # Context window management — set to your model's context length.
    # History is trimmed to fit within context_length - max_response_tokens tokens.
    context_length: int = 35063        # mistralai/magistral-small-2509 default
    max_response_tokens: int = 4096    # tokens reserved for the assistant's response
    # Contextual compaction settings
    compact_threshold_pct: int = 95    # trigger auto-compact when CTX >= this %
    compact_min_msgs: int = 8          # min visible messages before auto-compact fires
    compact_keep_ratio: float = 0.5    # compact oldest N fraction of visible messages
    compact_tool_turns: bool = True    # include tool-result messages in compaction input
    compaction_enabled: bool = True    # persisted default for new sessions
    compact_model: str = ""            # dedicated fast model for compaction (empty = use main model)
    tool_result_max_chars: int = 2000  # max chars per tool result stored in history
    rag_recency_days: float = 30.0     # recency half-life for date-weighted RAG scoring
    thinking_enabled: bool = False          # apply parallel thinking to every query
    thinking_paths: int = 3                 # parallel reasoning chains (1–10)
    thinking_model: str = ""                # dedicated model for thinking (empty = main)
    thinking_temperature: float = 0.8      # temperature for reasoning chains
    config_version: int = 6


def _validate(cfg: dict[str, Any]) -> dict[str, Any]:
    defaults = AppConfig().__dict__.copy()
    merged = {**defaults, **cfg}
    raw_version = cfg.get("config_version", 1)
    if isinstance(raw_version, (int, str)) and str(raw_version).isdigit():
        cfg_version = int(raw_version)
    else:
        cfg_version = 1
    # Validate base_url is a non-empty string; keep user value or fall back to default.
    if not isinstance(merged.get("base_url"), str) or not merged["base_url"].strip():
        merged["base_url"] = defaults["base_url"]
    if not isinstance(merged["model"], str) or not merged["model"].strip():
        merged["model"] = defaults["model"]
    if not isinstance(merged["theme"], str) or not merged["theme"].strip():
        merged["theme"] = defaults["theme"]
    if merged["approval"] not in {m.value for m in ApprovalMode}:
        merged["approval"] = defaults["approval"]
    if cfg_version < 2:
        merged["concise_mode"] = defaults["concise_mode"]
    else:
        merged["concise_mode"] = bool(merged.get("concise_mode", defaults["concise_mode"]))
    if cfg_version < 5:
        merged["shell_enabled"] = defaults["shell_enabled"]
    else:
        merged["shell_enabled"] = bool(
            merged.get("shell_enabled", merged.get("allow_host_shell", defaults["shell_enabled"]))
        )
    if cfg_version < 4:
        merged["personalities"] = defaults["personalities"]
        merged["active_personality"] = defaults["active_personality"]
    else:
        merged["personalities"] = merge_personalities(merged.get("personalities"))
        merged["active_personality"] = str(merged.get("active_personality") or defaults["active_personality"])
    active = merged["active_personality"]
    ids = {p.get("id") for p in merged["personalities"] if isinstance(p, dict)}
    if active not in ids:
        merged["active_personality"] = defaults["active_personality"]
    if not isinstance(merged.get("project_root"), str) or not merged["project_root"].strip():
        merged["project_root"] = defaults["project_root"]
    # Context window settings (added in config_version 6)
    raw_ctx = merged.get("context_length", defaults["context_length"])
    merged["context_length"] = int(raw_ctx) if isinstance(raw_ctx, (int, float)) and int(raw_ctx) > 0 else defaults["context_length"]
    raw_mrt = merged.get("max_response_tokens", defaults["max_response_tokens"])
    merged["max_response_tokens"] = int(raw_mrt) if isinstance(raw_mrt, (int, float)) and int(raw_mrt) > 0 else defaults["max_response_tokens"]
    # Compaction settings (added in config_version 7)
    raw_ctp = merged.get("compact_threshold_pct", defaults["compact_threshold_pct"])
    merged["compact_threshold_pct"] = int(raw_ctp) if isinstance(raw_ctp, (int, float)) and 1 <= int(raw_ctp) <= 100 else defaults["compact_threshold_pct"]
    raw_cmm = merged.get("compact_min_msgs", defaults["compact_min_msgs"])
    merged["compact_min_msgs"] = int(raw_cmm) if isinstance(raw_cmm, (int, float)) and int(raw_cmm) >= 2 else defaults["compact_min_msgs"]
    raw_ckr = merged.get("compact_keep_ratio", defaults["compact_keep_ratio"])
    merged["compact_keep_ratio"] = float(raw_ckr) if isinstance(raw_ckr, (int, float)) and 0.0 < float(raw_ckr) < 1.0 else defaults["compact_keep_ratio"]
    merged["compact_tool_turns"] = bool(merged.get("compact_tool_turns", defaults["compact_tool_turns"]))
    merged["compaction_enabled"] = bool(merged.get("compaction_enabled", defaults["compaction_enabled"]))
    merged["compact_model"] = str(merged.get("compact_model", defaults["compact_model"]))
    raw_trmc = merged.get("tool_result_max_chars", defaults["tool_result_max_chars"])
    merged["tool_result_max_chars"] = int(raw_trmc) if isinstance(raw_trmc, (int, float)) and int(raw_trmc) >= 100 else defaults["tool_result_max_chars"]
    raw_rrd = merged.get("rag_recency_days", defaults["rag_recency_days"])
    merged["rag_recency_days"] = float(raw_rrd) if isinstance(raw_rrd, (int, float)) and float(raw_rrd) > 0 else defaults["rag_recency_days"]
    merged["thinking_enabled"] = bool(merged.get("thinking_enabled", defaults["thinking_enabled"]))
    raw_tp = merged.get("thinking_paths", defaults["thinking_paths"])
    merged["thinking_paths"] = int(raw_tp) if isinstance(raw_tp, (int, float)) and 1 <= int(raw_tp) <= 10 else defaults["thinking_paths"]
    merged["thinking_model"] = str(merged.get("thinking_model", defaults["thinking_model"]))
    raw_tt = merged.get("thinking_temperature", defaults["thinking_temperature"])
    merged["thinking_temperature"] = float(raw_tt) if isinstance(raw_tt, (int, float)) and 0.0 < float(raw_tt) <= 2.0 else defaults["thinking_temperature"]
    merged["config_version"] = defaults["config_version"]
    return merged


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        cfg = _validate({})
        save_config(cfg, path)
        return cfg

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = _validate(raw if isinstance(raw, dict) else {})
    if cfg != raw:
        save_config(cfg, path)
    return cfg


def save_config(cfg: dict[str, Any], path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    validated = _validate(cfg)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(yaml.safe_dump(validated, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)
