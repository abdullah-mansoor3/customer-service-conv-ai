from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
import certifi
import uuid
import json

app = FastAPI(title="Llama.cpp Backend Proxy")

# Allow all origins for seamless frontend integration during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")

# In-memory session store
sessions = {}
MAX_HISTORY = 10

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Wait for message from client
            # Format expected: {"message": "Hello", "session_id": "optional-uuid"}
            raw_data = await websocket.receive_text()
            
            try:
                data = json.loads(raw_data)
                user_message = data.get("message", "").strip()
                session_id = data.get("session_id")
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "error": "Invalid JSON format"})
                continue
                
            if not user_message:
                await websocket.send_json({"type": "error", "error": "Message cannot be empty"})
                continue
                
            # Generate or retrieve session
            session_id = session_id or str(uuid.uuid4())
            if session_id not in sessions:
                sessions[session_id] = []
                
            sessions[session_id].append({"role": "user", "content": user_message})
            
            # Keep only the last MAX_HISTORY messages
            history = sessions[session_id][-MAX_HISTORY:]

            # Construct messages array
            system_prompt = "You are a helpful customer service assistant. Be concise and friendly."
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history)
            
            payload = {
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": -1,
                "stream": True
            }

            full_response = ""
            try:
                # Use a larger timeout since we are streaming tokens as they generate
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST", 
                        f"{LLAMA_SERVER_URL}/v1/chat/completions",
                        json=payload
                    ) as response:
                        response.raise_for_status()
                        
                        # First item back is the generated session ID in case the client needs it
                        await websocket.send_json({"type": "session_id", "session_id": session_id})
                        
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
                                    # Forward token to client
                                    await websocket.send_json({
                                        "type": "token",
                                        "token": token,
                                        "done": False
                                    })
                            except (json.JSONDecodeError, KeyError):
                                continue
                                
                    # Send end of stream marker
                    await websocket.send_json({"type": "token", "token": "", "done": True})
                    
                    # Store assistant response in history
                    sessions[session_id].append({"role": "assistant", "content": full_response})
                    
            except httpx.HTTPStatusError as e:
                sessions[session_id].pop() # Revert user message
                await websocket.send_json({"type": "error", "error": f"LLM Server Error: {e.response.status_code}"})
            except httpx.RequestError:
                sessions[session_id].pop() # Revert user message
                await websocket.send_json({"type": "error", "error": "Could not connect to the local Llama Server. Is it running?"})
                
    except WebSocketDisconnect:
        # Client disconnected
        pass

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "llama_server": LLAMA_SERVER_URL}
