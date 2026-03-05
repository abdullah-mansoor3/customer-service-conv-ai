"""
In-memory session store shared across all routers.

Structure:
    sessions[session_id] = {
        "created_at": "2026-01-01T10:00:00",
        "messages":   [{"role": "user"|"assistant", "content": "..."}, ...],
        "state":      {"router_model": None, "lights_status": None, ...}
    }

The "state" dict is a signal-to-noise reduction technique: the LLM is
instructed to emit a <STATE>{…}</STATE> block at the start of every reply.
The backend intercepts that block (never shown to the user), parses
the JSON, and merges it back into session["state"].  On the next turn
the accumulated state is injected into the system prompt so the model
always has a clean summary of known facts — regardless of how long the
conversation history grows.

NOTE: data is lost on server restart — acceptable for this assignment.
If you need persistence, swap the dict for Redis.
"""

from pydantic import BaseModel

# ── Shared state ──────────────────────────────────────────────────────────

sessions: dict = {}

# Default state keys tracked per session.  Add more as needed.
DEFAULT_STATE: dict = {
    "router_model": None,
    "lights_status": None,
    "error_message": None,
    "connection_type": None,
    "has_restarted": None,
}

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
