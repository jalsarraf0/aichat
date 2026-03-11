"""Unit tests for vision/mcp-server/app/config.py."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Load mcp-server config module explicitly to avoid collision with
# vision-router's app.config module.
# ---------------------------------------------------------------------------

_repo_root = Path(__file__).parent.parent.parent.parent  # .../aichat
_mcp_root = _repo_root / "vision" / "mcp-server"


def _load_mcp_config():
    """Load app.config from vision/mcp-server with a fresh module cache."""
    # Purge all app.* modules from cache to ensure fresh import
    for key in list(sys.modules.keys()):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]

    # Ensure mcp-server is at the front of sys.path
    mcp_root_str = str(_mcp_root)
    router_root_str = str(_repo_root / "vision" / "services" / "vision-router")

    # Remove router root if present to avoid collision
    path_backup = sys.path[:]
    sys.path = [p for p in sys.path if p != router_root_str]
    if mcp_root_str not in sys.path:
        sys.path.insert(0, mcp_root_str)

    try:
        from app.config import Settings  # type: ignore[import]
        return Settings
    finally:
        sys.path[:] = path_backup


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDefaultConfig:
    def test_compreface_default_url(self):
        Settings = _load_mcp_config()
        s = Settings()
        assert s.compreface.url == "http://192.168.50.2:8080"

    def test_compreface_default_similarity_threshold(self):
        Settings = _load_mcp_config()
        s = Settings()
        assert s.compreface.similarity_threshold == 0.85

    def test_compreface_default_det_prob_threshold(self):
        Settings = _load_mcp_config()
        s = Settings()
        assert s.compreface.det_prob_threshold == 0.8

    def test_compreface_default_limit(self):
        Settings = _load_mcp_config()
        s = Settings()
        assert s.compreface.limit == 5

    def test_server_default_port(self):
        Settings = _load_mcp_config()
        s = Settings()
        assert s.server.port == 8097

    def test_server_default_host(self):
        Settings = _load_mcp_config()
        s = Settings()
        assert s.server.host == "0.0.0.0"

    def test_server_default_workers(self):
        Settings = _load_mcp_config()
        s = Settings()
        assert s.server.workers == 2

    def test_router_default_url(self):
        Settings = _load_mcp_config()
        s = Settings()
        assert s.router.url == "http://192.168.50.2:8090"

    def test_feature_flags_defaults(self):
        Settings = _load_mcp_config()
        s = Settings()
        assert s.enable_compreface is True
        assert s.enable_triton is True
        assert s.enable_url_input is True
        assert s.enable_embeddings is True

    def test_api_key_defaults_empty(self):
        Settings = _load_mcp_config()
        s = Settings()
        assert s.compreface.api_key == ""
        assert s.compreface.recognition_api_key == ""
        assert s.compreface.detection_api_key == ""
        assert s.compreface.verification_api_key == ""


class TestEnvVarOverrides:
    def test_server_port_override(self):
        with patch.dict("os.environ", {"VISION_SERVER__PORT": "9000"}):
            Settings = _load_mcp_config()
            s = Settings()
            assert s.server.port == 9000

    def test_compreface_url_override(self):
        with patch.dict("os.environ", {"VISION_COMPREFACE__URL": "http://10.0.0.5:8080"}):
            Settings = _load_mcp_config()
            s = Settings()
            assert s.compreface.url == "http://10.0.0.5:8080"

    def test_router_url_override(self):
        with patch.dict("os.environ", {"VISION_ROUTER__URL": "http://10.0.0.5:8090"}):
            Settings = _load_mcp_config()
            s = Settings()
            assert s.router.url == "http://10.0.0.5:8090"

    def test_similarity_threshold_override(self):
        with patch.dict("os.environ", {"VISION_COMPREFACE__SIMILARITY_THRESHOLD": "0.9"}):
            Settings = _load_mcp_config()
            s = Settings()
            assert s.compreface.similarity_threshold == pytest.approx(0.9)

    def test_feature_flag_disable(self):
        with patch.dict("os.environ", {"VISION_ENABLE_COMPREFACE": "false"}):
            Settings = _load_mcp_config()
            s = Settings()
            assert s.enable_compreface is False


class TestApiKeyFanOut:
    def test_master_key_fans_to_all_services(self):
        with patch.dict("os.environ", {"VISION_COMPREFACE__API_KEY": "master-key"}):
            Settings = _load_mcp_config()
            s = Settings()
            assert s.compreface.recognition_api_key == "master-key"
            assert s.compreface.detection_api_key == "master-key"
            assert s.compreface.verification_api_key == "master-key"

    def test_per_service_key_not_overridden_when_set(self):
        with patch.dict(
            "os.environ",
            {
                "VISION_COMPREFACE__API_KEY": "master-key",
                "VISION_COMPREFACE__RECOGNITION_API_KEY": "custom-key",
            },
        ):
            Settings = _load_mcp_config()
            s = Settings()
            assert s.compreface.recognition_api_key == "custom-key"
            assert s.compreface.detection_api_key == "master-key"
            assert s.compreface.verification_api_key == "master-key"

    def test_all_per_service_keys_override_master(self):
        with patch.dict(
            "os.environ",
            {
                "VISION_COMPREFACE__API_KEY": "master-key",
                "VISION_COMPREFACE__RECOGNITION_API_KEY": "recog-key",
                "VISION_COMPREFACE__DETECTION_API_KEY": "detect-key",
                "VISION_COMPREFACE__VERIFICATION_API_KEY": "verify-key",
            },
        ):
            Settings = _load_mcp_config()
            s = Settings()
            assert s.compreface.recognition_api_key == "recog-key"
            assert s.compreface.detection_api_key == "detect-key"
            assert s.compreface.verification_api_key == "verify-key"

    def test_no_master_key_no_fan_out(self):
        import os
        # Build a clean env with no COMPREFACE keys
        env_clean = {
            k: v for k, v in os.environ.items()
            if not k.startswith("VISION_COMPREFACE")
        }
        with patch.dict("os.environ", env_clean, clear=True):
            Settings = _load_mcp_config()
            s = Settings()
            assert s.compreface.recognition_api_key == ""
            assert s.compreface.detection_api_key == ""
            assert s.compreface.verification_api_key == ""
