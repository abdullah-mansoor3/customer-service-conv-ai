"""
Customer Service Conversational AI — Backend entry point.

Registers all routers and middleware. Logic lives in:
    app/routers/health.py    — GET /api/health
    app/routers/sessions.py  — POST/GET/DELETE /api/sessions/…
    app/routers/chat.py      — WS /ws/chat
    app/routers/voice.py     — POST /api/asr, POST /api/tts, WS /ws/voice-chat
    app/store.py             — in-memory session store + Pydantic models
    app/config.py            — env vars (LLAMA_SERVER_URL, SYSTEM_PROMPT)
"""

import logging
import os
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import chat, health, sessions, voice


def _configure_logging() -> None:
    """Configure app logger level so orchestration diagnostics are visible."""
    level_name = os.getenv("BACKEND_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    else:
        root_logger.setLevel(level)

    logging.getLogger("app.orchestration.langgraph_engine").setLevel(level)


_configure_logging()


def _preload_rag_models() -> None:
    """Preload RAG embedding models in background thread."""

    def _load():
        try:
            from rag.inference import preload_models

            preload_models()
            logging.getLogger(__name__).info("RAG models preloaded successfully")
        except Exception as exc:
            logging.getLogger(__name__).warning(f"RAG model preload failed: {exc}")

    thread = threading.Thread(target=_load, name="rag-model-preload", daemon=True)
    thread.start()


_preload_rag_models()

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
app.include_router(voice.router)
