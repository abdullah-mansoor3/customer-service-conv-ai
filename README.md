# Customer Service Conversational AI

A locally-running conversational AI system.  
The backend proxies requests to a **llama.cpp** inference server (`llama-server`) and streams tokens to the frontend over WebSocket.

```
Frontend  ←──WebSocket──►  FastAPI Backend  ←──HTTP──►  llama-server (llama.cpp)
```

---

## Directory Structure

```
customer-service-conv-ai/
│
├── backend/
│   ├── app/
│   │   ├── main.py               # ✅ App factory — creates FastAPI app, mounts routers
│   │   ├── config.py             # ✅ Env vars: LLAMA_SERVER_URL, SYSTEM_PROMPT
│   │   ├── store.py              # ✅ In-memory sessions dict + Pydantic models
│   │   └── routers/
│   │       ├── health.py         # ✅ GET /api/health
│   │       ├── sessions.py       # ✅ POST/GET/DELETE /api/sessions/…
│   │       └── chat.py           # ✅ WS /ws/chat/{session_id}  (streaming)
│   ├── requirements.txt          # ✅ fastapi, uvicorn, httpx, pydantic
│   └── Dockerfile                # ✅ Simple single-stage Python image
│
├── frontend/                     # Web UI — Frontend Team
│   └── Dockerfile                # placeholder nginx; replace with real build
│
├── docker-compose.yml            # ✅ Orchestrates backend + frontend
├── .gitignore
└── README.md
```

---

## Backend — What's Implemented

Logic is split across these files:

| File | Responsibility |
|------|---------------|
| `main.py` | Creates the FastAPI app, registers routers and CORS middleware |
| `config.py` | Reads `LLAMA_SERVER_URL` and `SYSTEM_PROMPT` from env |
| `store.py` | Shared in-memory `sessions` dict and Pydantic response models |
| `routers/health.py` | `GET /api/health` |
| `routers/sessions.py` | Session CRUD endpoints |
| `routers/chat.py` | WebSocket streaming chat |

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Liveness check — returns llama-server URL |
| `POST` | `/api/sessions` | Create a new chat session → returns `session_id` |
| `GET` | `/api/sessions` | List all active sessions |
| `GET` | `/api/sessions/{session_id}` | Session info + full message history |
| `DELETE` | `/api/sessions/{session_id}` | Delete session and its history |

### WebSocket — `WS /ws/chat/{session_id}`

Real-time streaming chat.

**Client → Server** (each turn):
```json
{"message": "What are your opening hours?"}
```

**Server → Client** (token stream):
```json
{"type": "token", "token": "Our",    "done": false}
{"type": "token", "token": " hours", "done": false}
{"type": "token", "token": "",       "done": true}
```

**Server → Client** (on error):
```json
{"type": "error", "error": "Could not connect to the LLM server."}
```

**Flow per turn:**
1. Receive `{"message": "..."}` from the client.
2. Append the user message to in-memory session history.
3. POST to `llama-server /v1/chat/completions` with `stream: true` and the full session history (system prompt + all prior turns).
4. Forward each token chunk to the client as it arrives.
5. Send `{"done": true}` end-of-stream marker.
6. Append the full assistant reply to session history.

The session is **auto-created** if the `session_id` in the URL does not exist, so the frontend can open the socket without calling `POST /api/sessions` first.

### Session Store

Sessions are stored in a plain Python `dict` (in-process memory).  
Data is lost on server restart — acceptable for this assignment.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_SERVER_URL` | `http://127.0.0.1:8080` | URL of the running llama-server |
| `SYSTEM_PROMPT` | `"You are a helpful customer service assistant..."` | System prompt prepended to every conversation |

---

## What Remains

### Frontend Team → `frontend/`
Replace `frontend/Dockerfile` with the real build for your chosen framework (plain HTML/JS or React).  
See the comments inside that file for Dockerfile examples.

Required features:
- `POST /api/sessions` on page load to get a `session_id`.
- Open `WS /ws/chat/{session_id}`.
- Send `{"message": "..."}` on form submit.
- Render streaming tokens as they arrive (append on `done: false`, close bubble on `done: true`).
- "New Chat" button → `DELETE /api/sessions/{session_id}` → create new session.
- Page reload → `GET /api/sessions/{session_id}/history` to restore chat.

### LLM / Conversation Team → llama-server
The backend expects a **llama-server** (llama.cpp) process reachable at `LLAMA_SERVER_URL`.  
It must expose the OpenAI-compatible `/v1/chat/completions` endpoint with SSE streaming support.

To run llama-server locally:
```bash
llama-server -m /path/to/model.gguf --host 0.0.0.0 --port 8080 --ctx-size 2048
```

To integrate it into Docker Compose, uncomment the `llama-server` service block in `docker-compose.yml` and change `LLAMA_SERVER_URL` to `http://llama-server:8080`.

The system prompt is set via the `SYSTEM_PROMPT` environment variable — update it in `docker-compose.yml` to match your business domain.

---

## Quick Start

### Prerequisites
- Docker ≥ 24 + Docker Compose v2
- A running `llama-server` (see above)

### 1. Clone

```bash
git clone <repo-url>
cd customer-service-conv-ai
```

### 2. Start

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| Backend API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Frontend | http://localhost:3000 |

### 3. Local dev (no Docker)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
LLAMA_SERVER_URL=http://127.0.0.1:8080 uvicorn app.main:app --reload
```
