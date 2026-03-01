"""
In-memory session store shared across all routers.

Structure:
    sessions[session_id] = {
        "created_at": "2026-01-01T10:00:00",
        "messages":   [{"role": "user"|"assistant", "content": "..."}, ...]
    }

NOTE: data is lost on server restart — acceptable for this assignment.
If you need persistence, swap the dict for Redis.
"""

from pydantic import BaseModel

# ── Shared state ──────────────────────────────────────────────────────────

sessions: dict = {}

# ── Pydantic models ───────────────────────────────────────────────────────


class SessionCreatedResponse(BaseModel):
    session_id: str
    created_at: str


class SessionSummary(BaseModel):
    session_id: str
    created_at: str
    message_count: int


class SessionDetail(BaseModel):
    session_id: str
    created_at: str
    messages: list  # list of {"role": ..., "content": ...}
