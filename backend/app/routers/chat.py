"""WebSocket streaming chat endpoint backed by LangChain + LangGraph orchestration."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.orchestration import conversation_orchestrator
from app.orchestration.langgraph_engine import AgentTurnResult, merge_non_null_state
from app.store import DEFAULT_STATE, sessions

router = APIRouter()
logger = logging.getLogger(__name__)


def _ensure_session(session_id: str) -> None:
    """Create session container on first use."""
    if session_id in sessions:
        return

    sessions[session_id] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "messages": [],
        "state": {**DEFAULT_STATE},
    }


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
                user_message = str(data.get("message", "")).strip()
                session_id = data.get("session_id")
            except (json.JSONDecodeError, AttributeError):
                await websocket.send_json({"type": "error", "error": "Payload must be valid JSON."})
                continue

            if not user_message:
                await websocket.send_json({"type": "error", "error": "Field 'message' is missing or empty."})
                continue

            session_id = str(session_id or uuid.uuid4())
            _ensure_session(session_id)

            session = sessions[session_id]

            await conversation_orchestrator.ensure_thread_state(
                session_id=session_id,
                known_state=session["state"],
                messages=session["messages"],
            )

            await websocket.send_json({"type": "session_id", "session_id": session_id})

            final_result: AgentTurnResult | None = None
            done_sent = False

            try:
                async for event in conversation_orchestrator.stream_turn_events(
                    session_id=session_id,
                    user_message=user_message,
                    emit_state_block=True,
                ):
                    if event.get("type") == "token":
                        token = str(event.get("token", ""))
                        if token:
                            await websocket.send_json({"type": "token", "token": token, "done": False})
                    elif event.get("type") == "visible_done":
                        if not done_sent:
                            await websocket.send_json({"type": "token", "token": "", "done": True})
                            done_sent = True
                    elif event.get("type") == "final":
                        final_result = event.get("result")
            except Exception as exc:  # noqa: BLE001
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": f"LLM orchestration failed: {exc}",
                    }
                )
                continue

            if not isinstance(final_result, AgentTurnResult):
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": "Conversation graph did not return a final result.",
                    }
                )
                continue

            if final_result.messages:
                session["messages"] = final_result.messages
            else:
                session["messages"].append({"role": "user", "content": user_message})
                if final_result.assistant_text:
                    session["messages"].append(
                        {
                            "role": "assistant",
                            "content": final_result.assistant_text,
                        }
                    )

            if final_result.known_state:
                session["state"] = final_result.known_state
                logger.info("state_ready session_id=%s state=%s", session_id, final_result.known_state)
            else:
                session["state"] = merge_non_null_state(session["state"], final_result.state_update)
                if final_result.state_update:
                    logger.info("state_ready session_id=%s state=%s", session_id, session["state"])

            if not done_sent:
                await websocket.send_json({"type": "token", "token": "", "done": True})
                done_sent = True

    except WebSocketDisconnect:
        pass
