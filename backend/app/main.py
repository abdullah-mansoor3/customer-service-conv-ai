"""
Customer Service Conversational AI — Backend entry point.

Registers all routers and middleware. Logic lives in:
    app/routers/health.py    — GET /api/health
    app/routers/sessions.py  — POST/GET/DELETE /api/sessions/…
    app/routers/chat.py      — WS /ws/chat
    app/store.py             — in-memory session store + Pydantic models
    app/config.py            — env vars (LLAMA_SERVER_URL, SYSTEM_PROMPT)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import chat, health, sessions

app = FastAPI(title="Customer Service Conversational AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(sessions.router)
app.include_router(chat.router)
