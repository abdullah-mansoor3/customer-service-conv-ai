"""
Unit tests for all backend endpoints.

Endpoints covered:
  GET  /api/health
  POST /api/sessions
  GET  /api/sessions
  GET  /api/sessions/{session_id}
  DELETE /api/sessions/{session_id}
  WS   /ws/chat

Run with:
    cd backend
    pytest tests/test_endpoints.py -v
"""

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── App import ─────────────────────────────────────────────────────────────
from app.main import app
from app.store import DEFAULT_STATE, sessions

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clear_sessions():
    """Wipe the in-memory session store before every test."""
    sessions.clear()
    yield
    sessions.clear()


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


# ══════════════════════════════════════════════════════════════════════════
# Health endpoint
# ══════════════════════════════════════════════════════════════════════════


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_body_has_status_ok(self, client):
        data = client.get("/api/health").json()
        assert data["status"] == "ok"

    def test_health_body_has_llama_server_key(self, client):
        data = client.get("/api/health").json()
        assert "llama_server" in data

    def test_health_llama_server_is_string(self, client):
        data = client.get("/api/health").json()
        assert isinstance(data["llama_server"], str)


# ══════════════════════════════════════════════════════════════════════════
# Sessions — POST /api/sessions  (create)
# ══════════════════════════════════════════════════════════════════════════


class TestCreateSession:
    def test_create_returns_201(self, client):
        resp = client.post("/api/sessions")
        assert resp.status_code == 201

    def test_create_response_has_session_id(self, client):
        data = client.post("/api/sessions").json()
        assert "session_id" in data
        assert data["session_id"]  # non-empty

    def test_create_response_has_created_at(self, client):
        data = client.post("/api/sessions").json()
        assert "created_at" in data

    def test_create_session_stored_in_memory(self, client):
        data = client.post("/api/sessions").json()
        assert data["session_id"] in sessions

    def test_create_two_sessions_have_different_ids(self, client):
        id1 = client.post("/api/sessions").json()["session_id"]
        id2 = client.post("/api/sessions").json()["session_id"]
        assert id1 != id2

    def test_new_session_has_empty_messages(self, client):
        sid = client.post("/api/sessions").json()["session_id"]
        assert sessions[sid]["messages"] == []

    def test_new_session_state_matches_default(self, client):
        sid = client.post("/api/sessions").json()["session_id"]
        assert sessions[sid]["state"] == DEFAULT_STATE


# ══════════════════════════════════════════════════════════════════════════
# Sessions — GET /api/sessions  (list)
# ══════════════════════════════════════════════════════════════════════════


class TestListSessions:
    def test_list_empty_when_no_sessions(self, client):
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_returns_all_created_sessions(self, client):
        client.post("/api/sessions")
        client.post("/api/sessions")
        data = client.get("/api/sessions").json()
        assert len(data) == 2

    def test_list_item_has_required_fields(self, client):
        client.post("/api/sessions")
        item = client.get("/api/sessions").json()[0]
        assert "session_id" in item
        assert "created_at" in item
        assert "message_count" in item

    def test_list_message_count_starts_at_zero(self, client):
        client.post("/api/sessions")
        item = client.get("/api/sessions").json()[0]
        assert item["message_count"] == 0

    def test_list_message_count_reflects_manual_injection(self, client):
        sid = client.post("/api/sessions").json()["session_id"]
        sessions[sid]["messages"].append({"role": "user", "content": "hi"})
        item = client.get("/api/sessions").json()[0]
        assert item["message_count"] == 1


# ══════════════════════════════════════════════════════════════════════════
# Sessions — GET /api/sessions/{session_id}  (detail)
# ══════════════════════════════════════════════════════════════════════════


class TestGetSession:
    def test_get_existing_session_returns_200(self, client):
        sid = client.post("/api/sessions").json()["session_id"]
        resp = client.get(f"/api/sessions/{sid}")
        assert resp.status_code == 200

    def test_get_non_existent_session_returns_404(self, client):
        resp = client.get("/api/sessions/does-not-exist")
        assert resp.status_code == 404

    def test_get_session_response_has_session_id(self, client):
        sid = client.post("/api/sessions").json()["session_id"]
        data = client.get(f"/api/sessions/{sid}").json()
        assert data["session_id"] == sid

    def test_get_session_response_has_messages(self, client):
        sid = client.post("/api/sessions").json()["session_id"]
        data = client.get(f"/api/sessions/{sid}").json()
        assert "messages" in data
        assert isinstance(data["messages"], list)

    def test_get_session_messages_reflects_history(self, client):
        sid = client.post("/api/sessions").json()["session_id"]
        sessions[sid]["messages"].append({"role": "user", "content": "hello"})
        data = client.get(f"/api/sessions/{sid}").json()
        assert len(data["messages"]) == 1
        assert data["messages"][0]["content"] == "hello"

    def test_get_session_404_detail_message(self, client):
        data = client.get("/api/sessions/fake-id").json()
        assert "not found" in data["detail"].lower()


# ══════════════════════════════════════════════════════════════════════════
# Sessions — DELETE /api/sessions/{session_id}
# ══════════════════════════════════════════════════════════════════════════


class TestDeleteSession:
    def test_delete_existing_session_returns_200(self, client):
        sid = client.post("/api/sessions").json()["session_id"]
        resp = client.delete(f"/api/sessions/{sid}")
        assert resp.status_code == 200

    def test_delete_removes_session_from_store(self, client):
        sid = client.post("/api/sessions").json()["session_id"]
        client.delete(f"/api/sessions/{sid}")
        assert sid not in sessions

    def test_delete_non_existent_returns_404(self, client):
        resp = client.delete("/api/sessions/ghost-id")
        assert resp.status_code == 404

    def test_delete_response_has_message_key(self, client):
        sid = client.post("/api/sessions").json()["session_id"]
        data = client.delete(f"/api/sessions/{sid}").json()
        assert "message" in data

    def test_delete_then_get_returns_404(self, client):
        sid = client.post("/api/sessions").json()["session_id"]
        client.delete(f"/api/sessions/{sid}")
        assert client.get(f"/api/sessions/{sid}").status_code == 404


# ══════════════════════════════════════════════════════════════════════════
# WebSocket /ws/chat  — helpers & mocks
# ══════════════════════════════════════════════════════════════════════════


def _make_sse_lines(content: str) -> list[str]:
    """
    Build a minimal list of SSE lines that mimic llama-server streaming
    a single assistant turn with a STATE block followed by a reply.
    """
    payload = {
        "choices": [{"delta": {"content": content}, "finish_reason": None}]
    }
    return [
        f"data: {json.dumps(payload)}",
        "data: [DONE]",
    ]


class _FakeStreamResponse:
    """
    Minimal async context-manager that mimics httpx streaming response.
    """

    def __init__(self, lines: list[str], status_code: int = 200):
        self._lines = lines
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass


class _FakeAsyncClient:
    """Replaces httpx.AsyncClient so no real network calls are made."""

    def __init__(self, lines: list[str]):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    def stream(self, method, url, **kwargs):
        return _FakeStreamResponse(self._lines)


# ══════════════════════════════════════════════════════════════════════════
# WebSocket /ws/chat — tests
# ══════════════════════════════════════════════════════════════════════════


class TestWebSocketChat:
    # ── helper ─────────────────────────────────────────────────────────────

    def _chat(self, client, payload: dict, sse_lines: list[str]) -> list[dict]:
        """
        Open the WebSocket, send one message, collect all received frames
        until the server sends done=True or an error frame, then return them.
        """
        with patch("app.routers.chat.httpx.AsyncClient") as MockClient:
            MockClient.return_value = _FakeAsyncClient(sse_lines)
            with client.websocket_connect("/ws/chat") as ws:
                ws.send_text(json.dumps(payload))
                frames = []
                while True:
                    raw = ws.receive_text()
                    frame = json.loads(raw)
                    frames.append(frame)
                    if frame.get("type") == "token" and frame.get("done"):
                        break
                    if frame.get("type") == "error":
                        break
                return frames

    # ── basic flow ─────────────────────────────────────────────────────────

    def test_ws_sends_session_id_frame(self, client):
        lines = _make_sse_lines("Hello!")
        frames = self._chat(client, {"message": "hi"}, lines)
        types = [f["type"] for f in frames]
        assert "session_id" in types

    def test_ws_sends_done_frame(self, client):
        lines = _make_sse_lines("Hello!")
        frames = self._chat(client, {"message": "hi"}, lines)
        done_frames = [f for f in frames if f.get("done") is True]
        assert len(done_frames) == 1

    def test_ws_respects_provided_session_id(self, client):
        sid = client.post("/api/sessions").json()["session_id"]
        lines = _make_sse_lines("Sure, I can help.")
        frames = self._chat(client, {"message": "hello", "session_id": sid}, lines)
        id_frame = next(f for f in frames if f.get("type") == "session_id")
        assert id_frame["session_id"] == sid

    def test_ws_creates_new_session_when_none_given(self, client):
        lines = _make_sse_lines("Hi there!")
        frames = self._chat(client, {"message": "hello"}, lines)
        id_frame = next(f for f in frames if f.get("type") == "session_id")
        assert id_frame["session_id"] in sessions

    def test_ws_appends_user_message_to_history(self, client):
        sid = client.post("/api/sessions").json()["session_id"]
        lines = _make_sse_lines("Got it.")
        self._chat(client, {"message": "my router is broken", "session_id": sid}, lines)
        user_msgs = [m for m in sessions[sid]["messages"] if m["role"] == "user"]
        assert any("router" in m["content"] for m in user_msgs)

    def test_ws_appends_assistant_message_to_history(self, client):
        sid = client.post("/api/sessions").json()["session_id"]
        lines = _make_sse_lines("Let me help you with that.")
        self._chat(client, {"message": "hi", "session_id": sid}, lines)
        asst_msgs = [m for m in sessions[sid]["messages"] if m["role"] == "assistant"]
        assert len(asst_msgs) == 1

    # ── state extraction ────────────────────────────────────────────────────

    def test_ws_extracts_state_from_response(self, client):
        sid = client.post("/api/sessions").json()["session_id"]
        content = (
            '<STATE>{"router_model": "TP-Link", "lights_status": null, '
            '"error_message": null, "connection_type": "wifi", '
            '"has_restarted": null}</STATE>\n'
            "Thanks for that info!"
        )
        lines = _make_sse_lines(content)
        self._chat(client, {"message": "TP-Link wifi", "session_id": sid}, lines)
        assert sessions[sid]["state"]["router_model"] == "TP-Link"
        assert sessions[sid]["state"]["connection_type"] == "wifi"

    def test_ws_state_null_values_do_not_overwrite_existing(self, client):
        sid = client.post("/api/sessions").json()["session_id"]
        sessions[sid]["state"]["router_model"] = "Netgear"
        content = (
            '<STATE>{"router_model": null, "lights_status": null, '
            '"error_message": null, "connection_type": null, '
            '"has_restarted": null}</STATE>\nOkay!'
        )
        lines = _make_sse_lines(content)
        self._chat(client, {"message": "follow-up", "session_id": sid}, lines)
        # null from LLM must NOT overwrite the already-known value
        assert sessions[sid]["state"]["router_model"] == "Netgear"

    def test_ws_state_block_not_forwarded_to_client(self, client):
        sid = client.post("/api/sessions").json()["session_id"]
        content = (
            '<STATE>{"router_model": "ASUS", "lights_status": null, '
            '"error_message": null, "connection_type": null, '
            '"has_restarted": null}</STATE>\nI can help!'
        )
        lines = _make_sse_lines(content)
        frames = self._chat(client, {"message": "ASUS router", "session_id": sid}, lines)
        token_text = "".join(
            f.get("token", "")
            for f in frames
            if f.get("type") == "token"
        )
        assert "<STATE>" not in token_text
        assert "</STATE>" not in token_text

    # ── error handling ──────────────────────────────────────────────────────

    def test_ws_error_on_invalid_json(self, client):
        with patch("app.routers.chat.httpx.AsyncClient"):
            with client.websocket_connect("/ws/chat") as ws:
                ws.send_text("not valid json at all")
                frame = json.loads(ws.receive_text())
                assert frame["type"] == "error"

    def test_ws_error_on_empty_message(self, client):
        with patch("app.routers.chat.httpx.AsyncClient"):
            with client.websocket_connect("/ws/chat") as ws:
                ws.send_text(json.dumps({"message": "   "}))
                frame = json.loads(ws.receive_text())
                assert frame["type"] == "error"

    def test_ws_error_on_missing_message_field(self, client):
        with patch("app.routers.chat.httpx.AsyncClient"):
            with client.websocket_connect("/ws/chat") as ws:
                ws.send_text(json.dumps({"session_id": "abc"}))
                frame = json.loads(ws.receive_text())
                assert frame["type"] == "error"

    def test_ws_error_on_llm_server_unreachable(self, client):
        import httpx

        with patch("app.routers.chat.httpx.AsyncClient") as MockClient:
            # Make the stream() call raise a RequestError
            mock_instance = MagicMock()
            mock_cm = MagicMock()
            mock_cm.__aenter__ = AsyncMock(side_effect=httpx.RequestError("unreachable"))
            mock_cm.__aexit__ = AsyncMock(return_value=False)
            mock_instance.stream.return_value = mock_cm
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            with client.websocket_connect("/ws/chat") as ws:
                ws.send_text(json.dumps({"message": "hello"}))
                # Skip the session_id frame if it arrives before the error
                frames = []
                for _ in range(5):
                    raw = ws.receive_text()
                    frame = json.loads(raw)
                    frames.append(frame)
                    if frame.get("type") in ("error", "token"):
                        break
                error_frames = [f for f in frames if f.get("type") == "error"]
                assert len(error_frames) == 1
