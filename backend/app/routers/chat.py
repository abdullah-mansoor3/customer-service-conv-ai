"""
WebSocket streaming chat endpoint.

Flow per user turn
------------------
1. Receive JSON from client:     {"message": "Hello", "session_id": "optional-uuid"}
2. Resolve or create session; append user message to history.
3. Inject the accumulated state dict into the system prompt so the LLM
   always has a clean summary of known facts (signal-to-noise technique).
4. POST to llama-server with history + enriched system prompt (stream=True).
5. Intercept the <STATE>{…}</STATE> block the LLM emits — parse it,
   merge into the session state, and **never** forward it to the client.
6. Forward only the human-readable tokens to the client:
       {"type": "token", "token": "Hi",  "done": false}
       ...
       {"type": "token", "token": "",    "done": true}   ← end-of-stream
7. Persist only the clean assistant reply (without state block) in history.

On any error the client receives:
       {"type": "error", "error": "<description>"}
The socket stays open so the user can try again.
"""

import json
import re
import uuid
from datetime import datetime

import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import DEFAULT_SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, LLAMA_SERVER_URL, MAX_HISTORY
from app.store import DEFAULT_STATE, sessions

router = APIRouter()


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # ── Receive client message ─────────────────────────────────
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
                user_message = data.get("message", "").strip()
                session_id = data.get("session_id")
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

            # ── Resolve / create session ───────────────────────────────
            session_id = session_id or str(uuid.uuid4())
            if session_id not in sessions:
                sessions[session_id] = {
                    "created_at": datetime.utcnow().isoformat(),
                    "messages": [],
                    "state": {**DEFAULT_STATE},
                }

            sessions[session_id]["messages"].append(
                {"role": "user", "content": user_message}
            )

            # Keep only the last MAX_HISTORY messages for context window
            history = sessions[session_id]["messages"][-MAX_HISTORY:]

            # ── Build system prompt with injected state ────────────────
            state_str = json.dumps(sessions[session_id]["state"])
            system_prompt = (
                f"{DEFAULT_SYSTEM_PROMPT}\n\n"
                "Track the user's issue state as JSON. "
                "Always reply in EXACTLY this format — the STATE JSON "
                "block on its own line, then a newline, then your "
                "short reply:\n"
                "<STATE>{\"router_model\": null, \"lights_status\": null, "
                "\"error_message\": null, \"connection_type\": null, "
                "\"has_restarted\": null}</STATE>\n"
                "Your short reply here.\n\n"
                f"CURRENT KNOWN STATE:\n{state_str}"
            )

            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(FEW_SHOT_EXAMPLES)
            messages.extend(history)

            payload = {
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": -1,
                "stream": True,
            }

            # ── Stream from llama-server ───────────────────────────────
            full_response = ""
            is_state_block = True
            state_buffer = ""
            user_response = ""

            async def _send_token(text: str) -> None:
                """Helper: send a visible token to the client."""
                nonlocal user_response
                if text:
                    user_response += text
                    await websocket.send_json(
                        {"type": "token", "token": text, "done": False}
                    )

            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST",
                        f"{LLAMA_SERVER_URL}/v1/chat/completions",
                        json=payload,
                    ) as response:
                        response.raise_for_status()

                        # Tell the client which session they are in
                        await websocket.send_json(
                            {"type": "session_id", "session_id": session_id}
                        )

                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            chunk = line[len("data:"):].strip()
                            if chunk == "[DONE]":
                                break

                            try:
                                delta = json.loads(chunk)
                                token = (
                                    delta["choices"][0]["delta"].get("content", "")
                                )
                                if not token:
                                    continue

                                full_response += token

                                # ── Intercept <STATE>{…}</STATE> block ─
                                # The model is asked to start every reply
                                # with <STATE>{"key":…}</STATE> followed
                                # by the human-readable text.  We buffer
                                # tokens until we can decide whether a
                                # valid state block is present.
                                if is_state_block:
                                    state_buffer += token

                                    if "</STATE>" in state_buffer:
                                        # End tag arrived — check for JSON
                                        is_state_block = False
                                        before_end, after_end = state_buffer.split(
                                            "</STATE>", 1
                                        )
                                        # Try to pull a JSON dict out of
                                        # <STATE>…</STATE>
                                        inner = before_end
                                        if "<STATE>" in inner:
                                            inner = inner.split("<STATE>", 1)[1]
                                        brace = inner.find("{")
                                        has_json = False
                                        if brace != -1:
                                            try:
                                                json.loads(inner[brace:])
                                                has_json = True
                                            except json.JSONDecodeError:
                                                pass

                                        if has_json:
                                            # Valid state block — only
                                            # forward what comes after it
                                            await _send_token(
                                                after_end.lstrip("\n")
                                            )
                                        else:
                                            # Model wrapped its whole reply
                                            # in <STATE> without JSON —
                                            # strip the tags and flush
                                            clean = state_buffer.replace(
                                                "<STATE>", ""
                                            ).replace("</STATE>", "")
                                            await _send_token(
                                                clean.lstrip("\n")
                                            )

                                    elif len(state_buffer) > 500:
                                        # Buffer too large — the model
                                        # probably wrapped its entire
                                        # response in <STATE>. Flush.
                                        is_state_block = False
                                        clean = state_buffer.replace(
                                            "<STATE>", ""
                                        )
                                        await _send_token(clean.lstrip("\n"))

                                    elif (
                                        len(state_buffer) > 15
                                        and "<STATE>" not in state_buffer
                                        and "<" not in state_buffer[-1:]
                                    ):
                                        # No <STATE> tag at all — just
                                        # flush the buffer as normal text
                                        is_state_block = False
                                        await _send_token(
                                            state_buffer.lstrip("\n")
                                        )
                                else:
                                    await _send_token(token)

                            except (json.JSONDecodeError, KeyError):
                                continue

                    # If stream ended while we were still buffering,
                    # flush whatever we have
                    if is_state_block and state_buffer:
                        clean = (
                            state_buffer.replace("<STATE>", "")
                            .replace("</STATE>", "")
                        )
                        await _send_token(clean.lstrip("\n"))

                    # ── End-of-stream signal ───────────────────────────
                    await websocket.send_json(
                        {"type": "token", "token": "", "done": True}
                    )

                    # ── Parse & merge state dict ──────────────────────
                    match = re.search(
                        r"<STATE>\s*(\{.*?\})\s*</STATE>",
                        full_response,
                        re.DOTALL,
                    )
                    if match:
                        try:
                            new_state = json.loads(match.group(1))
                            if isinstance(new_state, dict):
                                for key, value in new_state.items():
                                    if (
                                        value is not None
                                        and str(value).strip().lower()
                                        not in ("", "null", "none")
                                        and key in sessions[session_id]["state"]
                                    ):
                                        sessions[session_id]["state"][key] = value
                                print(
                                    f"✅ Extracted State Update: "
                                    f"{sessions[session_id]['state']}"
                                )
                        except json.JSONDecodeError:
                            print("❌ Warning: State block JSON was invalid.")
                    else:
                        print(
                            "⚠️  No valid <STATE>{…}</STATE> block found "
                            "in LLM response."
                        )

                    # ── Persist only the clean assistant reply ─────────
                    sessions[session_id]["messages"].append(
                        {"role": "assistant", "content": user_response.strip()}
                    )

            except httpx.HTTPStatusError as exc:
                sessions[session_id]["messages"].pop()  # roll back user msg
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": f"LLM server returned {exc.response.status_code}.",
                    }
                )
                continue

            except httpx.RequestError:
                sessions[session_id]["messages"].pop()  # roll back user msg
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": "Could not connect to the LLM server. "
                        "Is llama-server running?",
                    }
                )
                continue

    except WebSocketDisconnect:
        pass  # client closed the connection — nothing to clean up
