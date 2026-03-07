# Customer Service Conversational AI

A locally-running conversational AI system for ISP technical support.  
The backend streams tokens from a **llama.cpp** inference server to the browser over WebSocket.

```
Browser
  │
  ├─ HTTP GET → :3000 ─────► Nginx (frontend)  serves React build
  │                                │  proxies /ws/* → backend:8000
  │
  └─ WebSocket ws://:8000 ────────► FastAPI Backend  session store + prompt logic
                                          │
                                          └──► llama-server:8080  LLM inference (llama.cpp)
```

---

## Directory Structure

```
customer-service-conv-ai/
│
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app factory — registers routers + CORS
│   │   ├── config.py             # Env vars: LLAMA_SERVER_URL, SYSTEM_PROMPT, MAX_HISTORY
│   │   ├── store.py              # In-memory sessions dict + Pydantic models
│   │   └── routers/
│   │       ├── health.py         # GET /api/health
│   │       ├── sessions.py       # POST/GET/DELETE /api/sessions/…
│   │       └── chat.py           # WS /ws/chat  (streaming, state-tracking)
│   ├── requirements.txt          # fastapi, uvicorn, httpx, pydantic
│   ├── run_tests.py
│   └── Dockerfile                # python:3.11-slim single-stage image
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx               # React chat UI — WebSocket client + streaming render
│   │   └── App.css
│   ├── nginx.conf                # Nginx config — serves static files + proxies /ws/
│   ├── package.json              # React 19 + Vite 7
│   └── Dockerfile                # Multi-stage: Node 20 build → Nginx serve
│
├── models/                       # ← put your .gguf file here (git-ignored)
│
├── docker-compose.yml            # Orchestrates all three services
├── .env.example                  # Template — copy to .env before running
├── serve_model.sh                # Run llama-server on the host (outside Docker)
├── check_llama.sh                # Install / verify llama.cpp on the host
└── README.md
```

---

## Services

| Service | Image / Build | Port | Role |
|---------|--------------|------|------|
| `llama-server` | `ghcr.io/ggerganov/llama.cpp:server` | 8080 | LLM inference; OpenAI-compatible `/v1/chat/completions` with SSE streaming |
| `backend` | `./backend/Dockerfile` | 8000 | FastAPI — REST API + WebSocket; manages sessions, streams tokens |
| `frontend` | `./frontend/Dockerfile` | 3000 | React UI built with Vite, served by Nginx |

---

## Backend API Reference

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Liveness check — returns configured `llama_server` URL |
| `POST` | `/api/sessions` | Create a new chat session → `{"session_id": "...", "created_at": "..."}` |
| `GET` | `/api/sessions` | List all active sessions |
| `GET` | `/api/sessions/{session_id}` | Full message history for a session |
| `DELETE` | `/api/sessions/{session_id}` | Delete session and its history |

Interactive docs: **http://localhost:8000/docs**

### WebSocket — `WS /ws/chat`

**Client → Server** (each turn):
```json
{"message": "My internet keeps disconnecting.", "session_id": "optional-uuid"}
```

**Server → Client** (token stream):
```json
{"type": "token", "token": "I",      "done": false}
{"type": "token", "token": " see",   "done": false}
{"type": "token", "token": "",       "done": true}
```

**Server → Client** (on error):
```json
{"type": "error", "error": "Could not connect to the LLM server."}
```

If `session_id` is omitted or unknown, the backend auto-creates a new session.

### State Tracking

The LLM is prompted to emit a `<STATE>{…}</STATE>` JSON block at the start of every reply.  
The backend intercepts it (never shown to the user), merges it into the session state dict, and reinjects the accumulated state into the system prompt on the next turn — keeping the model grounded without relying on long context alone.

Default tracked fields: `router_model`, `lights_status`, `error_message`, `connection_type`, `has_restarted`.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_SERVER_URL` | `http://llama-server:8080` | URL of the running llama-server |
| `SYSTEM_PROMPT` | `"You are a helpful customer service assistant..."` | System prompt for every conversation |
| `MAX_HISTORY` | `10` | Conversation turns kept in the LLM context window |

---

## Quick Start (Docker Compose — recommended)

### Prerequisites

- **Docker ≥ 24** with the Compose v2 plugin (`docker compose`)
- A **GGUF model file** — download from Hugging Face, e.g.:
  - [`mistral-7b-instruct-v0.2.Q4_K_M.gguf`](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) (~4 GB, good quality/speed on CPU)
  - [`phi-2.Q4_K_M.gguf`](https://huggingface.co/TheBloke/phi-2-GGUF) (~1.5 GB, fast on CPU)

### 1. Clone

```bash
git clone <repo-url>
cd customer-service-conv-ai
```

### 2. Add your model

```bash
mkdir -p models
cp ~/Downloads/your-model.gguf models/
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set `MODEL_PATH` to the path **inside the container** (always `/models/<filename>`):

```ini
MODEL_PATH=/models/your-model.gguf
```

Optionally adjust `CTX_SIZE`, `LLAMA_THREADS`, or `SYSTEM_PROMPT`.

### 4. Build and start all services

```bash
docker compose up --build
```

> The llama-server takes **1–2 minutes** to load the model on the first run (especially on CPU).  
> Watch progress: `docker compose logs -f llama-server`

| Service | URL |
|---------|-----|
| Frontend / Chat UI | http://localhost:3000 |
| Backend REST API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| llama-server (raw) | http://localhost:8080 |

### 5. Stop

```bash
docker compose down
```

---

## Useful Docker Commands

```bash
# Rebuild everything from scratch
docker compose up --build

# Start only the backend (useful for API development)
docker compose up --build backend

# Tail logs for a specific service
docker compose logs -f llama-server
docker compose logs -f backend
docker compose logs -f frontend

# Open a shell inside the backend container
docker compose exec backend sh

# Stop and remove containers (keeps images)
docker compose down

# Stop and remove containers + built images
docker compose down --rmi local
```

---

## Local Development (without Docker)

### llama-server on the host

```bash
# 1. Check / install llama.cpp
bash check_llama.sh

# 2. Start the server (reads MODEL_PATH from .env)
bash serve_model.sh

# Or run directly:
llama-server -m /path/to/model.gguf --host 0.0.0.0 --port 8080 --ctx-size 2048
```

### Backend on the host

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

LLAMA_SERVER_URL=http://127.0.0.1:8080 uvicorn app.main:app --reload --port 8000
```

### Frontend on the host

```bash
cd frontend
npm install
npm run dev     # Vite dev server → http://localhost:5173
```

> When running outside Docker the React app connects directly to `ws://localhost:8000/ws/chat`,
> so the backend must be running on port 8000 on the same machine.

---

## GPU Support (optional)

To use an NVIDIA GPU, update the `llama-server` service in `docker-compose.yml`:

```yaml
llama-server:
  image: ghcr.io/ggerganov/llama.cpp:server-cuda
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

Also add `--n-gpu-layers 99` to the `command:` block to offload all layers to the GPU.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `docker compose` not found | Old Docker | Upgrade to Docker ≥ 24 or use `docker-compose` |
| llama-server exits immediately | Wrong `MODEL_PATH` | Path must be `/models/<filename>` matching a file in `./models/` |
| llama-server still loading after 2 min | Large model + slow CPU | Use a smaller quantised model (Q4_K_M recommended) |
| Backend errors `Connection refused` | Model not loaded yet | Wait — `docker compose logs -f llama-server` |
| WebSocket fails in browser | Backend not running | Check `docker compose ps` and `docker compose logs backend` |
| Frontend build fails (`npm ci`) | Missing `package-lock.json` | Run `npm install` inside `frontend/` to generate it, then rebuild |
