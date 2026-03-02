"""
GPU acceleration tests for human_browser + aichat-toolkit.

Structure
---------
TestBrowserGpuConfig      — unit tests for BrowserGpuConfig (host-side class)
TestBrowserServerVersion  — introspection tests: version bump + _SERVER_SRC content
TestToolkitGpuRuntime     — unit tests for ToolkitGpuRuntime
TestBrowserGpuSmoke       — @pytest.mark.smoke: live browser server health endpoint
TestToolkitGpuSmoke       — @pytest.mark.smoke: live toolkit health endpoint
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Load modules under test via importlib (avoids polluting sys.modules)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent


def _load_browser_mod():
    spec = importlib.util.spec_from_file_location(
        "browser_module",
        _REPO_ROOT / "src" / "aichat" / "tools" / "browser.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _load_toolkit_mod():
    from pathlib import Path
    from unittest.mock import patch as _patch
    spec = importlib.util.spec_from_file_location(
        "toolkit_app",
        _REPO_ROOT / "docker" / "toolkit" / "app.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # The module calls TOOLS_DIR.mkdir() at import time; /data doesn't exist
    # outside the container — patch mkdir to be a no-op during load.
    with _patch.object(Path, "mkdir", return_value=None):
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


try:
    _browser_mod       = _load_browser_mod()
    BrowserGpuConfig   = _browser_mod.BrowserGpuConfig
    _SERVER_SRC        = _browser_mod._SERVER_SRC
    _REQ_VER           = _browser_mod._REQUIRED_SERVER_VERSION
    _BROWSER_LOAD_OK   = True
except Exception as _be:
    _BROWSER_LOAD_OK   = False
    _BROWSER_LOAD_ERR  = str(_be)

try:
    _toolkit_mod        = _load_toolkit_mod()
    ToolkitGpuRuntime   = _toolkit_mod.ToolkitGpuRuntime
    _TOOLKIT_LOAD_OK    = True
except Exception as _te:
    _TOOLKIT_LOAD_OK    = False
    _TOOLKIT_LOAD_ERR   = str(_te)

skip_browser_load = pytest.mark.skipif(
    not _BROWSER_LOAD_OK,
    reason=f"browser.py load failed: {_BROWSER_LOAD_ERR if not _BROWSER_LOAD_OK else ''}",
)
skip_toolkit_load = pytest.mark.skipif(
    not _TOOLKIT_LOAD_OK,
    reason=f"toolkit app.py load failed: {_TOOLKIT_LOAD_ERR if not _TOOLKIT_LOAD_OK else ''}",
)

# Live-service checks
_BROWSER_URL  = "http://localhost:7081"
_TOOLKIT_URL  = "http://localhost:8095"
_BROWSER_UP   = False
_TOOLKIT_UP   = False

try:
    _BROWSER_UP = httpx.get(f"{_BROWSER_URL}/health", timeout=2).status_code == 200
except Exception:
    pass
try:
    _TOOLKIT_UP = httpx.get(f"{_TOOLKIT_URL}/health", timeout=2).status_code == 200
except Exception:
    pass

skip_browser_svc = pytest.mark.skipif(not _BROWSER_UP, reason="browser server not reachable")
skip_toolkit_svc = pytest.mark.skipif(not _TOOLKIT_UP, reason="toolkit service not reachable")


# ===========================================================================
# 1. BrowserGpuConfig — unit tests
# ===========================================================================

@skip_browser_load
class TestBrowserGpuConfig:
    """Unit tests for BrowserGpuConfig (host-side class in browser.py)."""

    def setup_method(self):
        # Reset any monkeypatching of class attributes between tests
        BrowserGpuConfig._has_dri.cache_clear() if hasattr(BrowserGpuConfig._has_dri, "cache_clear") else None

    def test_no_dri_gives_disable_gpu(self, monkeypatch):
        """When /dev/dri is not accessible, --disable-gpu must be in args."""
        monkeypatch.setattr(BrowserGpuConfig, "_has_dri", classmethod(lambda cls: False))
        monkeypatch.delenv("INTEL_GPU", raising=False)
        args = BrowserGpuConfig.launch_args()
        assert "--disable-gpu" in args

    def test_dri_removes_disable_gpu(self, monkeypatch):
        """When /dev/dri/renderD* is present, --disable-gpu must NOT be in args."""
        monkeypatch.setattr(BrowserGpuConfig, "_has_dri", classmethod(lambda cls: True))
        monkeypatch.delenv("INTEL_GPU", raising=False)
        args = BrowserGpuConfig.launch_args()
        assert "--disable-gpu" not in args

    def test_dri_adds_egl(self, monkeypatch):
        """--use-gl=egl must be present when GPU is available."""
        monkeypatch.setattr(BrowserGpuConfig, "_has_dri", classmethod(lambda cls: True))
        monkeypatch.delenv("INTEL_GPU", raising=False)
        args = BrowserGpuConfig.launch_args()
        assert "--use-gl=egl" in args

    def test_dri_adds_vaapi_feature(self, monkeypatch):
        """VaapiVideoDecoder must appear in an --enable-features arg."""
        monkeypatch.setattr(BrowserGpuConfig, "_has_dri", classmethod(lambda cls: True))
        monkeypatch.delenv("INTEL_GPU", raising=False)
        args = BrowserGpuConfig.launch_args()
        features_args = [a for a in args if a.startswith("--enable-features=")]
        assert features_args, "--enable-features arg missing"
        assert any("VaapiVideoDecoder" in a for a in features_args)

    def test_intel_gpu_env_enables_gpu(self, monkeypatch):
        """INTEL_GPU=1 env var must trigger GPU args even without /dev/dri."""
        monkeypatch.setattr(BrowserGpuConfig, "_has_dri", classmethod(lambda cls: False))
        monkeypatch.setenv("INTEL_GPU", "1")
        args = BrowserGpuConfig.launch_args()
        assert "--use-gl=egl" in args
        assert "--disable-gpu" not in args

    def test_base_args_always_present_no_gpu(self, monkeypatch):
        """--no-sandbox and --lang=en-US must be present in non-GPU mode."""
        monkeypatch.setattr(BrowserGpuConfig, "_has_dri", classmethod(lambda cls: False))
        monkeypatch.delenv("INTEL_GPU", raising=False)
        args = BrowserGpuConfig.launch_args()
        assert "--no-sandbox" in args
        assert "--lang=en-US" in args

    def test_base_args_always_present_gpu(self, monkeypatch):
        """--no-sandbox and --lang=en-US must be present in GPU mode too."""
        monkeypatch.setattr(BrowserGpuConfig, "_has_dri", classmethod(lambda cls: True))
        monkeypatch.delenv("INTEL_GPU", raising=False)
        args = BrowserGpuConfig.launch_args()
        assert "--no-sandbox" in args
        assert "--lang=en-US" in args

    def test_info_has_required_keys(self, monkeypatch):
        monkeypatch.setattr(BrowserGpuConfig, "_has_dri", classmethod(lambda cls: False))
        monkeypatch.delenv("INTEL_GPU", raising=False)
        info = BrowserGpuConfig.info()
        assert "gpu_available" in info
        assert "dri_accessible" in info
        assert "intel_gpu_env" in info

    def test_gpu_available_false_by_default(self, monkeypatch):
        monkeypatch.setattr(BrowserGpuConfig, "_has_dri", classmethod(lambda cls: False))
        monkeypatch.delenv("INTEL_GPU", raising=False)
        assert BrowserGpuConfig.gpu_available() is False

    def test_gpu_available_true_with_dri(self, monkeypatch):
        monkeypatch.setattr(BrowserGpuConfig, "_has_dri", classmethod(lambda cls: True))
        monkeypatch.delenv("INTEL_GPU", raising=False)
        assert BrowserGpuConfig.gpu_available() is True

    def test_no_duplicate_args(self, monkeypatch):
        """launch_args() must not contain duplicate flags."""
        for dri in (True, False):
            monkeypatch.setattr(BrowserGpuConfig, "_has_dri", classmethod(lambda cls, d=dri: d))
            monkeypatch.delenv("INTEL_GPU", raising=False)
            args = BrowserGpuConfig.launch_args()
            assert len(args) == len(set(args)), f"Duplicate args when dri={dri}: {args}"

    def test_has_dri_false_when_dir_missing(self, monkeypatch):
        """_has_dri() returns False when /dev/dri doesn't exist."""
        import os as _os
        monkeypatch.setattr(_os, "listdir", lambda p: (_ for _ in ()).throw(FileNotFoundError()))
        assert BrowserGpuConfig._has_dri() is False

    def test_has_dri_true_when_render_node_present(self, monkeypatch):
        """_has_dri() returns True when renderD* is in /dev/dri listing."""
        import os as _os
        original_listdir = _os.listdir

        def _mock_listdir(path):
            if "/dev/dri" in str(path):
                return ["renderD128", "card0"]
            return original_listdir(path)

        monkeypatch.setattr(_os, "listdir", _mock_listdir)
        assert BrowserGpuConfig._has_dri() is True


# ===========================================================================
# 2. BrowserServerVersion — introspection of _SERVER_SRC + host version
# ===========================================================================

@skip_browser_load
class TestBrowserServerVersion:
    """Verify the browser server version was bumped and GPU class is embedded."""

    def test_required_version_is_19(self):
        assert _REQ_VER == "19", f"Expected '19', got {_REQ_VER!r}"

    def test_server_src_contains_gpu_class(self):
        assert "BrowserGpuConfig" in _SERVER_SRC, (
            "BrowserGpuConfig class not found inside _SERVER_SRC"
        )

    def test_server_src_uses_launch_args_method(self):
        assert "BrowserGpuConfig.launch_args()" in _SERVER_SRC, (
            "_LAUNCH_ARGS = BrowserGpuConfig.launch_args() not found in _SERVER_SRC"
        )

    def test_server_src_version_is_19(self):
        assert '_VERSION = "19"' in _SERVER_SRC, (
            "_VERSION inside _SERVER_SRC was not bumped to '19'"
        )

    def test_server_src_health_includes_gpu(self):
        assert "BrowserGpuConfig.info()" in _SERVER_SRC, (
            "/health endpoint in _SERVER_SRC does not call BrowserGpuConfig.info()"
        )


# ===========================================================================
# 3. ToolkitGpuRuntime — unit tests
# ===========================================================================

@skip_toolkit_load
class TestToolkitGpuRuntime:
    """Unit tests for ToolkitGpuRuntime in docker/toolkit/app.py."""

    def test_available_false_by_default(self, monkeypatch):
        monkeypatch.setattr(ToolkitGpuRuntime, "_has_dri", classmethod(lambda cls: False))
        monkeypatch.delenv("INTEL_GPU", raising=False)
        assert ToolkitGpuRuntime.available() is False

    def test_available_true_with_env(self, monkeypatch):
        monkeypatch.setattr(ToolkitGpuRuntime, "_has_dri", classmethod(lambda cls: False))
        monkeypatch.setenv("INTEL_GPU", "1")
        assert ToolkitGpuRuntime.available() is True

    def test_available_true_with_dri(self, monkeypatch):
        monkeypatch.setattr(ToolkitGpuRuntime, "_has_dri", classmethod(lambda cls: True))
        monkeypatch.delenv("INTEL_GPU", raising=False)
        assert ToolkitGpuRuntime.available() is True

    def test_packages_returns_list(self):
        result = ToolkitGpuRuntime.packages()
        assert isinstance(result, list)

    def test_packages_contains_numpy_if_importable(self):
        """numpy is in the host venv (and now in toolkit requirements) — should appear."""
        try:
            import numpy  # noqa: F401
            numpy_available = True
        except ImportError:
            numpy_available = False
        if numpy_available:
            assert "numpy" in ToolkitGpuRuntime.packages()

    def test_info_has_required_keys(self, monkeypatch):
        monkeypatch.setattr(ToolkitGpuRuntime, "_has_dri", classmethod(lambda cls: False))
        monkeypatch.delenv("INTEL_GPU", raising=False)
        info = ToolkitGpuRuntime.info()
        assert "available" in info
        assert "dri_accessible" in info
        assert "packages" in info

    def test_info_packages_is_list(self, monkeypatch):
        monkeypatch.setattr(ToolkitGpuRuntime, "_has_dri", classmethod(lambda cls: False))
        monkeypatch.delenv("INTEL_GPU", raising=False)
        assert isinstance(ToolkitGpuRuntime.info()["packages"], list)

    def test_has_dri_false_when_missing(self, monkeypatch):
        import os as _os
        monkeypatch.setattr(_os, "listdir", lambda p: (_ for _ in ()).throw(FileNotFoundError()))
        assert ToolkitGpuRuntime._has_dri() is False

    def test_has_dri_true_when_render_node(self, monkeypatch):
        import os as _os
        original = _os.listdir

        def _mock(path):
            if "/dev/dri" in str(path):
                return ["renderD128", "card0"]
            return original(path)

        monkeypatch.setattr(_os, "listdir", _mock)
        assert ToolkitGpuRuntime._has_dri() is True


# ===========================================================================
# 4. Smoke tests — live services
# ===========================================================================

@skip_browser_svc
@skip_browser_load
@pytest.mark.smoke
class TestBrowserGpuSmoke:
    """Smoke tests for browser server GPU health endpoint."""

    def test_health_has_gpu_key(self):
        r = httpx.get(f"{_BROWSER_URL}/health", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "gpu" in data, f"/health missing 'gpu' key: {data}"

    def test_health_gpu_available_is_bool(self):
        r = httpx.get(f"{_BROWSER_URL}/health", timeout=5)
        gpu = r.json().get("gpu", {})
        assert isinstance(gpu.get("gpu_available"), bool)

    def test_health_version_is_18(self):
        r = httpx.get(f"{_BROWSER_URL}/health", timeout=5)
        assert r.json().get("version") == "18", (
            f"Expected version '18', got {r.json().get('version')!r} "
            "(may need to trigger a screenshot to force server redeploy)"
        )


@skip_toolkit_svc
@skip_toolkit_load
@pytest.mark.smoke
class TestToolkitGpuSmoke:
    """Smoke tests for toolkit GPU health endpoint."""

    def test_health_has_gpu_key(self):
        r = httpx.get(f"{_TOOLKIT_URL}/health", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "gpu" in data, f"/health missing 'gpu' key: {data}"

    def test_health_gpu_available_is_bool(self):
        r = httpx.get(f"{_TOOLKIT_URL}/health", timeout=5)
        gpu = r.json().get("gpu", {})
        assert isinstance(gpu.get("available"), bool)

    def test_health_gpu_packages_is_list(self):
        r = httpx.get(f"{_TOOLKIT_URL}/health", timeout=5)
        gpu = r.json().get("gpu", {})
        assert isinstance(gpu.get("packages"), list)
