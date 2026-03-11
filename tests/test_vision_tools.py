"""Unit tests for vision MCP tool integration in the TUI."""
from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, "src")


# ─── VisionMCPTool unit tests ───────────────────────────────────────────────

class TestVisionMCPToolCall:
    """Tests for VisionMCPTool.call() — the JSON-RPC dispatcher."""

    def _mock_response(self, result_text: str, status_code: int = 200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = {
            "jsonrpc": "2.0", "id": 1,
            "result": {"content": [{"type": "text", "text": result_text}]}
        }
        resp.raise_for_status = MagicMock()
        return resp

    @pytest.mark.asyncio
    async def test_call_success_parses_json(self):
        from aichat.tools.vision import VisionMCPTool
        tool = VisionMCPTool("http://localhost:8097")
        mock_data = {"subjects": ["alice", "bob"], "count": 2}
        resp = self._mock_response(json.dumps(mock_data))
        with patch("httpx.AsyncClient") as mock_client_cls:
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=ctx)
            ctx.__aexit__ = AsyncMock(return_value=False)
            ctx.post = AsyncMock(return_value=resp)
            mock_client_cls.return_value = ctx
            result = await tool.call("list_face_subjects", {})
        assert result == mock_data

    @pytest.mark.asyncio
    async def test_call_returns_error_on_jsonrpc_error(self):
        from aichat.tools.vision import VisionMCPTool
        tool = VisionMCPTool("http://localhost:8097")
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "jsonrpc": "2.0", "id": 1,
            "error": {"code": -32601, "message": "Method not found"}
        }
        with patch("httpx.AsyncClient") as mock_client_cls:
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=ctx)
            ctx.__aexit__ = AsyncMock(return_value=False)
            ctx.post = AsyncMock(return_value=resp)
            mock_client_cls.return_value = ctx
            result = await tool.call("bad_tool", {})
        assert "error" in result
        assert "Method not found" in result["error"]

    @pytest.mark.asyncio
    async def test_call_returns_error_on_connection_failure(self):
        import httpx
        from aichat.tools.vision import VisionMCPTool
        tool = VisionMCPTool("http://localhost:8097")
        with patch("httpx.AsyncClient") as mock_client_cls:
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=ctx)
            ctx.__aexit__ = AsyncMock(return_value=False)
            ctx.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client_cls.return_value = ctx
            result = await tool.call("list_face_subjects", {})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_call_non_json_text_returned_as_result(self):
        from aichat.tools.vision import VisionMCPTool
        tool = VisionMCPTool("http://localhost:8097")
        resp = self._mock_response("plain text result")
        with patch("httpx.AsyncClient") as mock_client_cls:
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=ctx)
            ctx.__aexit__ = AsyncMock(return_value=False)
            ctx.post = AsyncMock(return_value=resp)
            mock_client_cls.return_value = ctx
            result = await tool.call("some_tool", {})
        assert result == {"result": "plain text result"}

    def test_image_source_url(self):
        from aichat.tools.vision import VisionMCPTool
        tool = VisionMCPTool()
        assert tool._image_source(image_url="http://example.com/img.jpg") == {"url": "http://example.com/img.jpg"}

    def test_image_source_base64(self):
        from aichat.tools.vision import VisionMCPTool
        tool = VisionMCPTool()
        assert tool._image_source(image_base64="abc123") == {"base64": "abc123"}

    def test_image_source_file(self):
        from aichat.tools.vision import VisionMCPTool
        tool = VisionMCPTool()
        assert tool._image_source(image_file="/tmp/test.jpg") == {"file_path": "/tmp/test.jpg"}

    def test_image_source_empty(self):
        from aichat.tools.vision import VisionMCPTool
        tool = VisionMCPTool()
        assert tool._image_source() == {}


class TestVisionMCPToolHealth:
    @pytest.mark.asyncio
    async def test_health_true_when_tools_present(self):
        from aichat.tools.vision import VisionMCPTool
        tool = VisionMCPTool("http://localhost:8097")
        resp = MagicMock()
        resp.json.return_value = {"status": "ok", "tools": 11}
        with patch("httpx.AsyncClient") as mock_client_cls:
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=ctx)
            ctx.__aexit__ = AsyncMock(return_value=False)
            ctx.get = AsyncMock(return_value=resp)
            mock_client_cls.return_value = ctx
            result = await tool.health()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_false_on_exception(self):
        import httpx
        from aichat.tools.vision import VisionMCPTool
        tool = VisionMCPTool("http://localhost:8097")
        with patch("httpx.AsyncClient") as mock_client_cls:
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=ctx)
            ctx.__aexit__ = AsyncMock(return_value=False)
            ctx.get = AsyncMock(side_effect=httpx.ConnectError("down"))
            mock_client_cls.return_value = ctx
            result = await tool.health()
        assert result is False


# ─── ToolManager integration tests (mocked VisionMCPTool) ───────────────────

class TestToolManagerVisionMethods:
    """Tests for the 11 run_* methods in ToolManager."""

    def _make_manager(self, vision_mock: Any) -> Any:
        from aichat.tools.manager import ToolManager
        from aichat.state import ApprovalMode
        manager = ToolManager()
        manager.vision = vision_mock
        return manager, ApprovalMode.AUTO

    @pytest.mark.asyncio
    async def test_run_list_face_subjects_returns_result(self):
        from aichat.tools.vision import VisionMCPTool
        mock_vision = MagicMock(spec=VisionMCPTool)
        mock_vision.list_face_subjects = AsyncMock(return_value={"subjects": ["alice"], "count": 1})
        manager, mode = self._make_manager(mock_vision)
        result = await manager.run_list_face_subjects(mode, None)
        assert result == {"subjects": ["alice"], "count": 1}
        mock_vision.list_face_subjects.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_recognize_face_calls_with_url(self):
        from aichat.tools.vision import VisionMCPTool
        mock_vision = MagicMock(spec=VisionMCPTool)
        mock_vision.recognize_face = AsyncMock(return_value={"faces": [], "count": 0})
        manager, mode = self._make_manager(mock_vision)
        result = await manager.run_recognize_face(
            "http://example.com/img.jpg", None, None, 5, 0.7, mode, None
        )
        mock_vision.recognize_face.assert_called_once_with(
            image_url="http://example.com/img.jpg",
            image_base64=None,
            image_file=None,
            limit=5,
            min_confidence=0.7,
        )
        assert result == {"faces": [], "count": 0}

    @pytest.mark.asyncio
    async def test_run_detect_objects_calls_correctly(self):
        from aichat.tools.vision import VisionMCPTool
        mock_vision = MagicMock(spec=VisionMCPTool)
        mock_vision.detect_objects = AsyncMock(return_value={"objects": [{"label": "person", "confidence": 0.9}], "count": 1})
        manager, mode = self._make_manager(mock_vision)
        result = await manager.run_detect_objects(
            "http://example.com/img.jpg", None, None, 0.25, mode, None
        )
        assert result["count"] == 1
        assert result["objects"][0]["label"] == "person"

    @pytest.mark.asyncio
    async def test_run_analyze_image_calls_correctly(self):
        from aichat.tools.vision import VisionMCPTool
        mock_vision = MagicMock(spec=VisionMCPTool)
        mock_vision.analyze_image = AsyncMock(return_value={
            "objects": [], "labels": [{"label": "cat", "confidence": 0.95}], "inference_ms": 42.0
        })
        manager, mode = self._make_manager(mock_vision)
        result = await manager.run_analyze_image(
            "http://example.com/img.jpg", None, None,
            True, True, False, 0.25, 5, mode, None
        )
        assert result["labels"][0]["label"] == "cat"


# ─── App _execute_tool_call tests (mocked ToolManager vision methods) ─────────

def _make_app_with_mocked_vision(**vision_return_values: Any):
    """Build a minimal AIChatApp with mocked tool manager vision methods."""
    from aichat.app import AIChatApp
    from aichat.state import ApprovalMode
    app = AIChatApp.__new__(AIChatApp)
    # Minimal state
    app.state = MagicMock()
    app.state.approval = ApprovalMode.AUTO
    app.state.shell_enabled = False
    app._confirm_tool = None
    # Mock tool manager
    app.tools = MagicMock()
    for method_name, return_val in vision_return_values.items():
        setattr(app.tools, method_name, AsyncMock(return_value=return_val))
    app.tools.is_custom_tool = MagicMock(return_value=False)
    return app


class TestExecuteToolCallVision:
    """Tests for _execute_tool_call vision dispatching."""

    @pytest.mark.asyncio
    async def test_list_face_subjects_empty(self):
        app = _make_app_with_mocked_vision(run_list_face_subjects={"subjects": [], "count": 0})
        result = await app._execute_tool_call("list_face_subjects", {})
        assert "No face subjects enrolled" in result

    @pytest.mark.asyncio
    async def test_list_face_subjects_with_entries(self):
        app = _make_app_with_mocked_vision(
            run_list_face_subjects={"subjects": ["alice", "bob"], "count": 2}
        )
        result = await app._execute_tool_call("list_face_subjects", {})
        assert "alice" in result
        assert "bob" in result
        assert "2" in result

    @pytest.mark.asyncio
    async def test_recognize_face_missing_image(self):
        app = _make_app_with_mocked_vision()
        result = await app._execute_tool_call("recognize_face", {})
        assert "provide image_url" in result

    @pytest.mark.asyncio
    async def test_recognize_face_with_url_no_matches(self):
        app = _make_app_with_mocked_vision(
            run_recognize_face={"faces": [{"matches": [], "box": {}}], "count": 1}
        )
        result = await app._execute_tool_call(
            "recognize_face", {"image_url": "http://example.com/img.jpg"}
        )
        assert "unknown" in result.lower()

    @pytest.mark.asyncio
    async def test_recognize_face_with_match(self):
        app = _make_app_with_mocked_vision(
            run_recognize_face={"faces": [{"matches": [{"subject": "alice", "similarity": 0.92}], "box": {}}], "count": 1}
        )
        result = await app._execute_tool_call(
            "recognize_face", {"image_url": "http://example.com/img.jpg"}
        )
        assert "alice" in result
        assert "92%" in result

    @pytest.mark.asyncio
    async def test_verify_face_match(self):
        app = _make_app_with_mocked_vision(
            run_verify_face={"verified": True, "similarity": 0.95}
        )
        result = await app._execute_tool_call("verify_face", {
            "image_a_url": "http://example.com/a.jpg",
            "image_b_url": "http://example.com/b.jpg",
        })
        assert "MATCH" in result
        assert "95%" in result

    @pytest.mark.asyncio
    async def test_verify_face_no_match(self):
        app = _make_app_with_mocked_vision(
            run_verify_face={"verified": False, "similarity": 0.32}
        )
        result = await app._execute_tool_call("verify_face", {
            "image_a_url": "http://example.com/a.jpg",
            "image_b_url": "http://example.com/b.jpg",
        })
        assert "NO MATCH" in result

    @pytest.mark.asyncio
    async def test_verify_face_missing_images(self):
        app = _make_app_with_mocked_vision()
        result = await app._execute_tool_call("verify_face", {"image_a_url": "http://example.com/a.jpg"})
        assert "provide image_a_url" in result

    @pytest.mark.asyncio
    async def test_detect_objects_returns_summary(self):
        app = _make_app_with_mocked_vision(
            run_detect_objects={
                "objects": [
                    {"label": "person", "confidence": 0.9},
                    {"label": "car", "confidence": 0.8},
                    {"label": "person", "confidence": 0.75},
                ],
                "count": 3,
            }
        )
        result = await app._execute_tool_call(
            "detect_objects", {"image_url": "http://example.com/scene.jpg"}
        )
        assert "2" in result and "person" in result
        assert "car" in result

    @pytest.mark.asyncio
    async def test_detect_objects_no_detections(self):
        app = _make_app_with_mocked_vision(run_detect_objects={"objects": [], "count": 0})
        result = await app._execute_tool_call(
            "detect_objects", {"image_url": "http://example.com/img.jpg"}
        )
        assert "No objects detected" in result

    @pytest.mark.asyncio
    async def test_classify_image_returns_ranked_list(self):
        app = _make_app_with_mocked_vision(
            run_classify_image={
                "labels": [
                    {"label": "golden retriever", "confidence": 0.82},
                    {"label": "Labrador retriever", "confidence": 0.11},
                ]
            }
        )
        result = await app._execute_tool_call(
            "classify_image", {"image_url": "http://example.com/dog.jpg"}
        )
        assert "golden retriever" in result
        assert "82%" in result

    @pytest.mark.asyncio
    async def test_detect_clothing_returns_items(self):
        app = _make_app_with_mocked_vision(
            run_detect_clothing={
                "items": [
                    {"category": "t-shirt", "confidence": 0.75},
                    {"category": "jeans", "confidence": 0.68},
                ],
                "count": 2,
            }
        )
        result = await app._execute_tool_call(
            "detect_clothing", {"image_url": "http://example.com/person.jpg"}
        )
        assert "t-shirt" in result
        assert "jeans" in result

    @pytest.mark.asyncio
    async def test_embed_image_returns_snippet(self):
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5] + [0.0] * 507
        app = _make_app_with_mocked_vision(
            run_embed_image={"embedding": embedding, "dimension": 512, "model": "clip_vit_b32"}
        )
        result = await app._execute_tool_call(
            "embed_image", {"image_url": "http://example.com/img.jpg"}
        )
        assert "512" in result
        assert "clip_vit_b32" in result

    @pytest.mark.asyncio
    async def test_analyze_image_combined(self):
        app = _make_app_with_mocked_vision(
            run_analyze_image={
                "objects": [{"label": "cat", "confidence": 0.91}],
                "labels": [{"label": "tabby cat", "confidence": 0.85}],
                "inference_ms": 120.0,
            }
        )
        result = await app._execute_tool_call(
            "analyze_image", {"image_url": "http://example.com/cat.jpg"}
        )
        assert "cat" in result
        assert "tabby cat" in result

    @pytest.mark.asyncio
    async def test_vision_error_propagated(self):
        app = _make_app_with_mocked_vision(
            run_detect_objects={"error": "vision-mcp unreachable: Connection refused"}
        )
        result = await app._execute_tool_call(
            "detect_objects", {"image_url": "http://example.com/img.jpg"}
        )
        assert "failed" in result
        assert "Connection refused" in result

    @pytest.mark.asyncio
    async def test_enroll_face_missing_subject(self):
        app = _make_app_with_mocked_vision()
        result = await app._execute_tool_call("enroll_face", {"image_url": "http://example.com/img.jpg"})
        assert "subject_name" in result

    @pytest.mark.asyncio
    async def test_delete_face_subject_found(self):
        app = _make_app_with_mocked_vision(
            run_delete_face_subject={"subject": "alice", "deleted": True}
        )
        result = await app._execute_tool_call("delete_face_subject", {"subject_name": "alice"})
        assert "deleted" in result


# ─── Tool definitions schema validation ──────────────────────────────────────

class TestVisionToolDefinitions:
    """Validate that vision tool definitions have correct schema structure."""

    def _get_vision_defs(self) -> list[dict]:
        from aichat.tools.manager import ToolManager
        manager = ToolManager()
        defs = manager.tool_definitions(shell_enabled=False)
        vision_names = {
            "recognize_face", "verify_face", "detect_faces", "enroll_face",
            "list_face_subjects", "delete_face_subject",
            "detect_objects", "classify_image", "detect_clothing",
            "embed_image", "analyze_image",
        }
        return [d for d in defs if d.get("function", {}).get("name") in vision_names]

    def test_all_11_vision_tools_present(self):
        defs = self._get_vision_defs()
        names = {d["function"]["name"] for d in defs}
        assert "recognize_face" in names
        assert "verify_face" in names
        assert "detect_faces" in names
        assert "enroll_face" in names
        assert "list_face_subjects" in names
        assert "delete_face_subject" in names
        assert "detect_objects" in names
        assert "classify_image" in names
        assert "detect_clothing" in names
        assert "embed_image" in names
        assert "analyze_image" in names
        assert len(defs) == 11

    def test_all_defs_have_required_openai_fields(self):
        for defn in self._get_vision_defs():
            assert defn["type"] == "function"
            fn = defn["function"]
            assert "name" in fn
            assert "description" in fn
            assert len(fn["description"]) > 20  # substantive description
            assert "parameters" in fn
            assert fn["parameters"]["type"] == "object"

    def test_enroll_face_has_required_subject_name(self):
        defs = self._get_vision_defs()
        enroll = next(d for d in defs if d["function"]["name"] == "enroll_face")
        assert "subject_name" in enroll["function"]["parameters"].get("required", [])

    def test_delete_face_subject_has_required_subject_name(self):
        defs = self._get_vision_defs()
        delete = next(d for d in defs if d["function"]["name"] == "delete_face_subject")
        assert "subject_name" in delete["function"]["parameters"].get("required", [])
