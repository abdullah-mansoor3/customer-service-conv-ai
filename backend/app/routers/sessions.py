import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.orchestration import conversation_orchestrator
from app.store import DEFAULT_STATE, SessionCreatedResponse, SessionDetail, SessionSummary, sessions

router = APIRouter(prefix="/api/sessions")


@router.post("", response_model=SessionCreatedResponse, status_code=201)
async def create_session():
    """Create a new empty chat session and return its ID."""
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    sessions[session_id] = {
        "created_at": now,
        "messages": [],
        "state": {**DEFAULT_STATE},
    }
    await conversation_orchestrator.ensure_thread_state(
        session_id=session_id,
        known_state=sessions[session_id]["state"],
        messages=sessions[session_id]["messages"],
    )
    return SessionCreatedResponse(session_id=session_id, created_at=now)


@router.get("", response_model=list[SessionSummary])
async def list_sessions():
    """Return a summary of every active session."""
    return [
        SessionSummary(
            session_id=sid,
            created_at=data["created_at"],
            message_count=len(data["messages"]),
        )
        for sid, data in sessions.items()
    ]


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str):
    """Return full message history for a session (used to restore the chat UI)."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    graph_messages, graph_state = await conversation_orchestrator.get_session_snapshot(session_id)

    data = sessions[session_id]
    if graph_messages:
        data["messages"] = graph_messages
    data["state"] = graph_state

    return SessionDetail(
        session_id=session_id,
        created_at=data["created_at"],
        messages=data["messages"],
    )


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its history (maps to the frontend 'New Chat' button)."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    await conversation_orchestrator.delete_session_thread(session_id)
    del sessions[session_id]
    return {"message": "Session deleted"}
