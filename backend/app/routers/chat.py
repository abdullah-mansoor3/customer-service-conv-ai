"""
WebSocket streaming chat endpoint.

Flow per user turn
------------------
1. Receive JSON from client:     {"message": "Hello"}
2. Append user message to session history.
3. POST to llama-server with full history + system prompt (stream=True).
4. Forward each decoded token to the client:
       {"type": "token", "token": "Hi",  "done": false}
       {"type": "token", "token": " there", "done": false}
       ...
       {"type": "token", "token": "",    "done": true}   ← end-of-stream marker
5. Append the complete assistant reply to session history.

On any error the client receives:
       {"type": "error", "error": "<description>"}
The socket stays open so the user can try again.
"""

import json
from datetime import datetime

import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import DEFAULT_SYSTEM_PROMPT, LLAMA_SERVER_URL
from app.store import sessions

router = APIRouter()


@router.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    await websocket.accept()

    # Auto-create session if it doesn't exist
    if session_id not in sessions:
        sessions[session_id] = {
            "created_at": datetime.utcnow().isoformat(),
            "messages": [],
        }

    try:
        while True:
            # ── Receive client message ─────────────────────────────────
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
                user_message = data.get("message", "").strip()
            except (json.JSONDecodeError, AttributeError):
                await websocket.send_json(
                    {"type": "error", "error": "Payload must be valid JSON."}
                )
                continue

            if not user_message:
                await websocket.send_json(
                    {"type": "error", "error": "Field 'message' is missing or empty."}
                )
                continue

            # ── Build messages array for llama-server ──────────────────
            sessions[session_id]["messages"].append(
                {"role": "user", "content": user_message}
            )

            messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
            messages.extend(sessions[session_id]["messages"])

            # ── Stream from llama-server ───────────────────────────────
            full_response = ""
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST",
                        f"{LLAMA_SERVER_URL}/v1/chat/completions",
                        json={
                            "messages": messages,
                            "stream": True,
                            "temperature": 0.7,
                            "max_tokens": -1,
                        },
                    ) as response:
                        response.raise_for_status()

                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            chunk = line[len("data:"):].strip()
                            if chunk == "[DONE]":
                                break
                            try:
                                delta = json.loads(chunk)
                                token = delta["choices"][0]["delta"].get("content", "")
                                if token:
                                    full_response += token
                                    await websocket.send_json(
                                        {"type": "token", "token": token, "done": False}
                                    )
                            except (json.JSONDecodeError, KeyError):
                                continue

            except httpx.HTTPStatusError as exc:
                sessions[session_id]["messages"].pop()  # roll back user message
                await websocket.send_json(
                    {"type": "error", "error": f"LLM server returned {exc.response.status_code}."}
                )
                continue

            except httpx.RequestError:
                sessions[session_id]["messages"].pop()  # roll back user message
                await websocket.send_json(
                    {"type": "error", "error": "Could not connect to the LLM server. Is llama-server running?"}
                )
                continue

            # ── End-of-stream signal ───────────────────────────────────
            await websocket.send_json({"type": "token", "token": "", "done": True})

            # ── Persist assistant reply ────────────────────────────────
            sessions[session_id]["messages"].append(
                {"role": "assistant", "content": full_response}
            )

    except WebSocketDisconnect:
        pass  # client closed the connection — nothing to clean up
