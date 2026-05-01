"""
Integration tests for the ISP Customer Service AI backend.

Tests WebSocket chat endpoint using FastAPI TestClient:
- Connection and session creation
- Full chat turn with token streaming
- STATE extraction from streamed responses
- Tool hint forwarding
- Session persistence across turns
"""

import json
import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
BACKEND_ROOT = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_health_endpoint():
    """Health endpoint should return 200."""
    response = client.get("/health")
    assert response.status_code == 200


@pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION_TESTS"),
    reason="Integration tests require running llama.cpp backend (set RUN_INTEGRATION_TESTS=1)",
)
class TestWebSocketChat:
    """WebSocket integration tests — require running llama.cpp server."""

    def test_websocket_session_creation(self):
        """Connecting and sending a message should return a session_id."""
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "Hello", "session_id": "integration_test_1"})
            response = ws.receive_json()
            assert response["type"] == "session_id"
            assert response["session_id"] == "integration_test_1"

    def test_websocket_full_turn(self):
        """A full chat turn should produce tokens and a done signal."""
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "Hi there", "session_id": "integration_test_2"})
            # Consume session_id
            ws.receive_json()
            # Collect tokens
            tokens = []
            done = False
            while not done:
                msg = ws.receive_json()
                if msg["type"] == "token":
                    if msg.get("done"):
                        done = True
                    elif msg.get("token"):
                        tokens.append(msg["token"])
                elif msg["type"] == "status":
                    continue
            assert len(tokens) > 0
            assert done

    def test_websocket_error_on_empty_message(self):
        """Sending an empty message should return an error."""
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "", "session_id": "integration_test_3"})
            response = ws.receive_json()
            assert response["type"] == "error"

    def test_websocket_invalid_json(self):
        """Sending invalid JSON should return an error."""
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text("not json")
            response = ws.receive_json()
            assert response["type"] == "error"

    def test_tool_hints_accepted(self):
        """Tool hints should be accepted without error."""
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({
                "message": "What is PTCL helpline?",
                "session_id": "integration_test_hints",
                "tool_hints": ["rag", "websearch"],
            })
            response = ws.receive_json()
            assert response["type"] == "session_id"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])