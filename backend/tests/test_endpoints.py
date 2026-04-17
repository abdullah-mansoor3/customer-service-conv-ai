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
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# ── App import ─────────────────────────────────────────────────────────────
from app.main import app
from app.orchestration import conversation_orchestrator
from app.orchestration.langgraph_engine import (
    AgentTurnResult,
    extract_visible_response_text,
    parse_state_block,
)
from app.store import DEFAULT_STATE, sessions

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clear_sessions():
    """Wipe the in-memory session store before every test."""
    conversation_orchestrator.reset_memory()
    sessions.clear()
    yield
    conversation_orchestrator.reset_memory()
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


def _fake_orchestrator_stream(content: str):
    """Build an async generator that mimics LangGraph token + final events."""
    parsed_state, assistant_text = parse_state_block(content)

    async def _stream(*_args, **_kwargs):
        if assistant_text:
            yield {"type": "token", "token": assistant_text}
        yield {
            "type": "final",
            "result": AgentTurnResult(
                route="dialogue",
                raw_response=content,
                assistant_text=assistant_text,
                state_update=parsed_state,
            ),
        }

    return _stream


# ══════════════════════════════════════════════════════════════════════════
# WebSocket /ws/chat — tests
# ══════════════════════════════════════════════════════════════════════════


class TestWebSocketChat:
    # ── helper ─────────────────────────────────────────────────────────────

    def _chat(self, client, payload: dict, content: str) -> list[dict]:
        """
        Open the WebSocket, send one message, collect all received frames
        until the server sends done=True or an error frame, then return them.
        """
        with patch(
            "app.routers.chat.conversation_orchestrator.stream_turn_events",
            new=_fake_orchestrator_stream(content),
        ):
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
        frames = self._chat(client, {"message": "hi"}, "Hello!")
        types = [f["type"] for f in frames]
        assert "session_id" in types

    def test_ws_sends_done_frame(self, client):
        frames = self._chat(client, {"message": "hi"}, "Hello!")
        done_frames = [f for f in frames if f.get("done") is True]
        assert len(done_frames) == 1

    def test_ws_sends_done_on_visible_done_event(self, client):
        async def _stream(*_args, **_kwargs):
            yield {"type": "token", "token": "Hello"}
            yield {"type": "visible_done"}
            yield {
                "type": "final",
                "result": AgentTurnResult(
                    route="dialogue",
                    raw_response="Hello",
                    assistant_text="Hello",
                    state_update={},
                ),
            }

        with patch("app.routers.chat.conversation_orchestrator.stream_turn_events", new=_stream):
            with client.websocket_connect("/ws/chat") as ws:
                ws.send_text(json.dumps({"message": "hi"}))
                frames = []
                while True:
                    frame = json.loads(ws.receive_text())
                    frames.append(frame)
                    if frame.get("type") == "token" and frame.get("done"):
                        break

        done_frames = [f for f in frames if f.get("type") == "token" and f.get("done") is True]
        assert len(done_frames) == 1

    def test_ws_respects_provided_session_id(self, client):
        sid = client.post("/api/sessions").json()["session_id"]
        frames = self._chat(client, {"message": "hello", "session_id": sid}, "Sure, I can help.")
        id_frame = next(f for f in frames if f.get("type") == "session_id")
        assert id_frame["session_id"] == sid

    def test_ws_creates_new_session_when_none_given(self, client):
        frames = self._chat(client, {"message": "hello"}, "Hi there!")
        id_frame = next(f for f in frames if f.get("type") == "session_id")
        assert id_frame["session_id"] in sessions

    def test_ws_appends_user_message_to_history(self, client):
        sid = client.post("/api/sessions").json()["session_id"]
        self._chat(client, {"message": "my router is broken", "session_id": sid}, "Got it.")
        user_msgs = [m for m in sessions[sid]["messages"] if m["role"] == "user"]
        assert any("router" in m["content"] for m in user_msgs)

    def test_ws_appends_assistant_message_to_history(self, client):
        sid = client.post("/api/sessions").json()["session_id"]
        self._chat(client, {"message": "hi", "session_id": sid}, "Let me help you with that.")
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
        self._chat(client, {"message": "TP-Link wifi", "session_id": sid}, content)
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
        self._chat(client, {"message": "follow-up", "session_id": sid}, content)
        # null from LLM must NOT overwrite the already-known value
        assert sessions[sid]["state"]["router_model"] == "Netgear"

    def test_ws_state_block_not_forwarded_to_client(self, client):
        sid = client.post("/api/sessions").json()["session_id"]
        content = (
            '<STATE>{"router_model": "ASUS", "lights_status": null, '
            '"error_message": null, "connection_type": null, '
            '"has_restarted": null}</STATE>\nI can help!'
        )
        frames = self._chat(client, {"message": "ASUS router", "session_id": sid}, content)
        token_text = "".join(
            f.get("token", "")
            for f in frames
            if f.get("type") == "token"
        )
        assert "<STATE>" not in token_text
        assert "</STATE>" not in token_text

    # ── error handling ──────────────────────────────────────────────────────

    def test_ws_error_on_invalid_json(self, client):
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text("not valid json at all")
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "error"

    def test_ws_error_on_empty_message(self, client):
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text(json.dumps({"message": "   "}))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "error"

    def test_ws_error_on_missing_message_field(self, client):
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text(json.dumps({"session_id": "abc"}))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "error"

    def test_ws_error_on_llm_server_unreachable(self, client):
        async def _broken_stream(*_args, **_kwargs):
            raise RuntimeError("unreachable")
            if False:  # pragma: no cover
                yield {}

        with patch("app.routers.chat.conversation_orchestrator.stream_turn_events", new=_broken_stream):
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


class TestStateVisibility:
    def test_parse_state_block_from_trailing_state(self):
        raw = (
            'I can help.\n'
            '<STATE>{"router_model": "ASUS", "lights_status": null, '
            '"error_message": null, "connection_type": "wifi", '
            '"has_restarted": null}</STATE>'
        )
        parsed, clean = parse_state_block(raw)
        assert parsed["router_model"] == "ASUS"
        assert parsed["connection_type"] == "wifi"
        assert clean == "I can help."

    def test_visible_text_hides_partial_trailing_state_prefix(self):
        partial = "I can help with that issue.\n<ST"
        assert extract_visible_response_text(partial, final=False) == "I can help with that issue.\n"

    def test_visible_text_removes_trailing_state_block(self):
        raw = 'I can help.\n<STATE>{"router_model": "ASUS"}</STATE>'
        assert extract_visible_response_text(raw, final=False) == "I can help."

    def test_visible_text_passthrough_when_no_state_prefix(self):
        raw = "Hello there"
        assert extract_visible_response_text(raw, final=False) == "Hello there"
