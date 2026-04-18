# Customer Service Conversational AI with Voice

A fully local, streaming conversational AI system for ISP (Internet Service Provider) technical support with **voice input and output capabilities**.
No cloud APIs, no data ever leaves your machine. Models run on CPU via **llama.cpp**, **Faster Whisper**, and **Piper TTS**.

---

## Table of Contents

1. [What the Project Does](#1-what-the-project-does)
2. [Voice Features](#2-voice-features)
3. [Architecture Overview](#3-architecture-overview)
4. [How Communication Works](#4-how-communication-works)
5. [The LLM Layer](#5-the-llm-layer)
6. [Voice Processing (ASR & TTS)](#6-voice-processing-asr--tts)
7. [Conversation & Session Management](#7-conversation--session-management)
8. [Signal-to-Noise / State Tracking Logic](#8-signal-to-noise--state-tracking-logic)
9. [Few-Shot Prompting](#9-few-shot-prompting)
10. [Directory Structure](#10-directory-structure)
11. [Backend API Reference](#11-backend-api-reference)
12. [Configuration](#12-configuration)
13. [Running with Docker (recommended)](#13-running-with-docker-recommended)
14. [Running without Docker](#14-running-without-docker)
15. [Voice Model Download](#15-voice-model-download)
16. [Tests](#16-tests)
17. [Model Download — Qwen3.5-0.8B](#17-model-download--qwen350-8b)
18. [Standalone RAG Module](#18-standalone-rag-module)
19. [LangGraph RAG Tool Calling Integration](#19-langgraph-rag-tool-calling-integration)

---

## 1. What the Project Does

The system is an **ISP tech-support chatbot** that:

- Holds a real-time, streaming conversation with a user through a browser UI
- Supports both **text and voice** input/output for natural interaction
- Progressively extracts structured facts from the free-form conversation (router model, connection type, error message, etc.)
- Uses those facts to stay on-topic and ask the right follow-up questions — even for small models that struggle with long context
- Runs entirely locally: no subscription, no API key, no cloud
- Handles up to 4 concurrent users with sub-second latency

The default use-case is diagnosing internet connectivity problems, but the tracked state fields and system prompt are fully configurable via environment variables.

---

## 2. Voice Features

- **Voice Input**: Speak naturally to the chatbot using your microphone
- **Voice Output**: AI responses are spoken back to you automatically
- **Real-time Processing**: Sub-second latency for voice interactions
- **Seamless Switching**: Toggle between text and voice modes instantly
- **Local Processing**: All voice processing happens on your machine (ASR via Faster Whisper, TTS via Piper)

---

## 3. Architecture Overview

Four services communicate over a private Docker network:

```
Browser (React)
   │
   │  HTTP GET /  → serves the React app
   ├──────────────────────────────► Nginx : 3000
   │                                  │  /ws/* proxied to backend
   │                                  │
   │  WebSocket ws://localhost:8000/ws/chat (text)
   │  WebSocket ws://localhost:8000/ws/voice-chat (voice)
   └──────────────────────────────► FastAPI Backend : 8000
                                       │
                                       │  HTTP POST /v1/chat/completions
                                       │  (OpenAI-compatible, SSE stream)
                                       ├────────────────────────────────► llama-server : 8080
                                       │                                     └── loads model.gguf
                                       │
                                       ├─ Voice Processing ──────────────┐
                                       │  Faster Whisper (ASR)           │
                                       │  Piper TTS (TTS)               │
                                       └─────────────────────────────────┘
```

| Service | Technology | Port | Responsibility |
|---|---|---|---|
| `llama-server` | llama.cpp (official Docker image) | 8080 | LLM inference — OpenAI-compatible REST, streams tokens as Server-Sent Events |
| `backend` | Python / FastAPI | 8000 | Session store, prompt construction, state extraction, token relay over WebSocket + voice processing |
| `frontend` | React 19 + Vite, served by Nginx | 3000 | Chat UI with voice controls, WebSocket client, streaming token render, audio recording/playback |

---

## 3. How Communication Works

### Browser ↔ Backend (WebSocket)

The frontend opens a single, persistent WebSocket connection to `ws://localhost:8000/ws/chat`.

Every time the user sends a message the browser sends one JSON frame:

```json
{"message": "My TP-Link router keeps disconnecting.", "session_id": "optional-uuid"}
```

The backend responds with a stream of token frames followed by a final done frame:

```json
{"type": "session_id", "session_id": "abc-123"}      ← sent once per turn, carries the session UUID
{"type": "token",      "token": "I",    "done": false}
{"type": "token",      "token": " see", "done": false}
{"type": "token",      "token": "",     "done": true}  ← signals end of turn
```

On any error (bad JSON, missing fields, LLM server down) the backend sends:

```json
{"type": "error", "error": "Could not connect to the LLM server."}
```

The connection stays open after an error so the user can retry without a page reload.

Nginx proxies the WebSocket path from port 3000 to port 8000, keeping the browser pointed at a single origin.

### Backend ↔ llama-server (HTTP SSE Streaming)

For each user turn the backend makes an HTTP POST to `llama-server`'s OpenAI-compatible endpoint:

```
POST http://llama-server:8080/v1/chat/completions
Content-Type: application/json

{
  "messages": [...],
  "temperature": 0.3,
  "max_tokens": -1,
  "stream": true
}
```

With `stream: true` llama-server responds with a sequence of Server-Sent Event lines:

```
data: {"choices":[{"delta":{"content":"I "},"finish_reason":null}]}
data: {"choices":[{"delta":{"content":"see "},"finish_reason":null}]}
data: [DONE]
```

The backend reads these with `httpx`'s async streaming client, extracts the `content` delta from each line, and relays each token over the WebSocket to the browser as it arrives — achieving true word-by-word streaming in the UI.

---

## 5. The LLM Layer

### llama.cpp

[llama.cpp](https://github.com/ggerganov/llama.cpp) is a C++ inference engine that runs transformer models on CPU (and optionally GPU) using quantized GGUF weight files. The official Docker image `ghcr.io/ggerganov/llama.cpp:server` exposes an OpenAI-compatible HTTP server.

Key advantages for this project:
- Runs on any hardware — no GPU required
- GGUF quantization (Q4, Q8, etc.) drastically shrinks model size while preserving most quality
- Streams tokens as they are generated, so the UI feels responsive

### Recommended Model — Qwen3.5-0.8B

The project is configured to use **Qwen/Qwen3.5-0.8B** (quantized GGUF). At only ~500 MB it loads instantly and runs on CPU fast enough for real conversation.

The `serve_model.sh` script automatically detects the model filename and applies the correct chat template (`chatml` for Qwen models) so the model formats its replies as expected.

### Temperature & Context

| Parameter | Value | Reason |
|---|---|---|
| `temperature` | 0.3 | Low randomness — support agents should be factual and consistent |
| `max_tokens` | -1 | No artificial truncation; the model decides when it's done |
| `ctx_size` | 2048 | Safe default for 0.8B models on CPU |

---

## 6. Voice Processing (ASR & TTS)

### Automatic Speech Recognition (ASR)

**Faster Whisper** converts user speech to text in real-time:
- **Model**: OpenAI Whisper `tiny` (39MB) for fast CPU inference
- **Latency**: ~0.5-1 second for short utterances
- **Quality**: Good English recognition with VAD (Voice Activity Detection)
- **Integration**: Python library with async support for concurrent users

### Text-to-Speech (TTS)

**Piper** converts AI responses to natural speech:
- **Model**: Neural TTS with 20+ voices (English: `en_US-lessac-medium` ~20MB)
- **Latency**: ~100-500ms generation + playback
- **Quality**: High-quality neural voices, no cloud dependency
- **Streaming**: Supports real-time audio streaming

### Voice Chat Flow

1. User clicks 🎤 and speaks → browser records audio
2. Audio sent via WebSocket to backend
3. **ASR**: Faster Whisper transcribes speech to text
4. **LLM**: Processes text through conversation state
5. **TTS**: Piper generates speech from AI response
6. Audio streamed back to browser for playback

### Performance Optimization

- **Concurrent Users**: Up to 4 simultaneous voice sessions
- **CPU Usage**: Models quantized for efficiency
- **Latency**: Sub-second end-to-end for voice interactions
- **Memory**: ~200MB additional RAM for voice models

## 5. Conversation & Session Management

### Sessions

Every browser tab or conversation is tracked as a **session**. A session is a plain Python dict:

```python
sessions[session_id] = {
    "created_at": "2026-01-01T10:00:00",
    "messages":   [{"role": "user"|"assistant", "content": "..."}, ...],
    "state":      {"router_model": None, "lights_status": None, ...}
}
```

Sessions live in memory (`app/store.py`). They are lost on server restart, which is acceptable for this assignment. To add persistence, replace the in-memory dict with Redis.

### History Window

Only the **last N messages** (default `MAX_HISTORY=10`) are sent to the LLM per turn. This caps the prompt size regardless of conversation length, preventing context-window overflow on small models. The full history is always stored in memory for session restore.

### REST API for session management

The frontend (and tests) can create, list, inspect, and delete sessions over HTTP — independent of the WebSocket. This cleanly separates session lifecycle from the chat stream:

```
POST   /api/sessions          → create session → {session_id, created_at}
GET    /api/sessions          → list all sessions
GET    /api/sessions/{id}     → full message history
DELETE /api/sessions/{id}     → end / clear session
```

---

## 6. Signal-to-Noise / State Tracking Logic

This is the core technique that makes small models (≤1B parameters) work reliably for structured support tasks.

### The Problem

Small LLMs have limited working memory. With a growing chat history their attention spreads thin and they start forgetting facts the user mentioned three messages ago ("what router did you say you had?").

### The Solution — Extracted State + System Prompt Injection

The LLM is instructed to **emit a structured JSON block at the start of every reply**:

```
<STATE>{"router_model": "TP-Link", "lights_status": null, "error_message": "disconnects", "connection_type": "wifi", "has_restarted": null}</STATE>
I see — a TP-Link on wifi that keeps disconnecting. Have you tried restarting it yet?
```

The backend intercepts this block **before forwarding any tokens to the browser**:

1. The `<STATE>{…}</STATE>` block is parsed with regex + `json.loads`
2. Non-null values are merged into `session["state"]` (null values never overwrite existing facts)
3. The block is stripped — the user never sees it
4. On the **next turn**, the accumulated state is injected into the system prompt:

```
CURRENT KNOWN STATE:
{"router_model": "TP-Link", "lights_status": null, "error_message": "disconnects", "connection_type": "wifi", "has_restarted": null}
```

### Why This Works

Instead of asking the model to recall facts from a growing, noisy conversation history, we give it a **compact, always-current summary** alongside the last few messages. The model only needs to:

1. Update the state with any new facts in the latest user message
2. Use the state to formulate the next useful question

This is a form of **explicit working memory** — the information is always present at the front of the context rather than buried in history.

### State Fields

| Field | Tracks |
|---|---|
| `router_model` | Brand/model of the user's router |
| `lights_status` | What the router indicator lights look like |
| `error_message` | The on-screen error the user sees |
| `connection_type` | `"wifi"` or `"wired"` |
| `has_restarted` | Whether the user has already restarted the router |

Add more fields by editing `DEFAULT_STATE` in `app/store.py` and updating the system prompt in `app/config.py`.

---

## 7. Few-Shot Prompting

Small models often struggle to follow a multi-part format instruction (produce structured JSON *and* a conversational reply) from a system prompt alone.

To fix this, two synthetic conversation turns are **prepended to every request** after the system prompt, immediately before the real history:

```
User:      "Hi, I need help with my internet."
Assistant: "<STATE>{...all null...}</STATE>\nHello! What router model are you using?"

User:      "I have a TP-Link and it keeps disconnecting on wifi."
Assistant: "<STATE>{...router_model: TP-Link, connection_type: wifi...}</STATE>\nHave you tried restarting the router yet?"
```

These are fake turns the model never actually generated — they demonstrate the exact output format by example. The model sees the pattern twice before generating its first real reply, which dramatically increases format compliance without any fine-tuning.

These turns are defined in `FEW_SHOT_EXAMPLES` inside `app/config.py` and are never stored in the session history or shown to the user.

---

## 8. Directory Structure

```
customer-service-conv-ai/
│
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI app factory — registers routers + CORS
│   │   ├── config.py         # Env vars: LLAMA_SERVER_URL, SYSTEM_PROMPT,
│   │   │                     #   MAX_HISTORY, FEW_SHOT_EXAMPLES
│   │   ├── store.py          # In-memory sessions dict + Pydantic response models
│   │   └── routers/
│   │       ├── health.py     # GET /api/health
│   │       ├── sessions.py   # POST/GET/DELETE /api/sessions/…
│   │       └── chat.py       # WS /ws/chat — streaming, state extraction
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_endpoints.py # Pytest unit tests for all endpoints
│   ├── requirements.txt
│   ├── run_tests.py          # Manual WebSocket integration smoke-tests
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # React chat UI — WebSocket client, token streaming
│   │   └── App.css
│   ├── nginx.conf            # Serves static files + proxies /ws/ to backend
│   ├── package.json
│   ├── vite.config.js
│   └── Dockerfile            # Multi-stage: Node 20 build → Nginx serve
│
├── models/                   # ← place your .gguf file here (git-ignored)
│
├── docker-compose.yml        # Orchestrates all three services
├── .env.example              # Template — copy to .env
├── serve_model.sh            # Run llama-server directly on the host (no Docker)
├── check_llama.sh            # Verify / install llama.cpp on the host
└── README.md
```

---

## 9. Backend API Reference

### REST Endpoints

| Method | Path | Status | Description |
|---|---|---|---|
| `GET` | `/api/health` | 200 | Liveness check — returns `{"status":"ok","llama_server":"..."}` |
| `POST` | `/api/sessions` | 201 | Create session → `{"session_id":"...","created_at":"..."}` |
| `GET` | `/api/sessions` | 200 | List all active sessions with message counts |
| `GET` | `/api/sessions/{id}` | 200 / 404 | Full message history for a session |
| `DELETE` | `/api/sessions/{id}` | 200 / 404 | Delete session and its history |

Interactive Swagger docs: **http://localhost:8000/docs**

### WebSocket — `WS /ws/chat`

**Client → Server** (per turn):
```json
{"message": "My router keeps disconnecting.", "session_id": "optional-uuid"}
```
Omit `session_id` the first time — the server creates and returns one.

**Server → Client** (token stream):
```json
{"type": "session_id", "session_id": "abc-123"}
{"type": "token", "token": "Let",  "done": false}
{"type": "token", "token": " me",  "done": false}
{"type": "token", "token": "",     "done": true}
```

**Server → Client** (on error):
```json
{"type": "error", "error": "Field 'message' is missing or empty."}
```

---

## 10. Configuration

All settings are read from environment variables (or a `.env` file in the project root).

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/models/model.gguf` | Path to the GGUF file **inside the container** |
| `LLAMA_SERVER_URL` | `http://127.0.0.1:8080` | URL the backend uses to reach llama-server |
| `SYSTEM_PROMPT` | `"You are a helpful customer service assistant..."` | Base system prompt for every session |
| `MAX_HISTORY` | `10` | Max conversation turns sent to the LLM per request |
| `CTX_SIZE` | `2048` | llama-server context window size (tokens) |
| `LLAMA_THREADS` | auto-detected | CPU threads for token generation |
| `LLAMA_BATCH_THREADS` | same as `LLAMA_THREADS` | CPU threads for prompt evaluation |

Copy `.env.example` to `.env` and edit before running.

---

## 11. Running with Docker (recommended)

### Prerequisites

- **Docker ≥ 24** with Compose v2 (`docker compose version`)
- A GGUF model file in `./models/` (see [Section 14](#14-model-download--qwen350-8b))

### Step-by-step

```bash
# 1. Clone the repo
git clone <repo-url>
cd customer-service-conv-ai

# 2. Place your model
mkdir -p models
cp ~/Downloads/Qwen3.5-0.8B-Q4_K_M.gguf models/

# 3. Configure environment
cp .env.example .env
# Edit .env — set MODEL_PATH to the in-container path:
#   MODEL_PATH=/models/Qwen3.5-0.8B-Q4_K_M.gguf

# 4. Build and start everything
docker compose up --build
```

> Backend image now installs both `backend/requirements.txt` and `rag/requirements.txt`, and includes `rag/` + `rag_data/` so LangGraph tool-calling can execute RAG inside Docker.

The first run downloads the llama-server Docker image (~600 MB) and builds the backend and frontend images. Subsequent starts are fast.

| URL | Service |
|---|---|
| http://localhost:3000 | Chat UI |
| http://localhost:8000/docs | Backend Swagger |
| http://localhost:8000/api/health | Health check |
| http://localhost:8080/health | llama-server raw health |

```bash
# Watch the model load (first boot takes ~30 sec for 0.8B)
docker compose logs -f llama-server

# Tail backend logs (shows state extraction output)
docker compose logs -f backend

# Start only the backend (useful while developing)
docker compose up --build backend

# Stop all services
docker compose down

# Rebuild from scratch and remove old images
docker compose down --rmi local && docker compose up --build
```

---

## 12. Running without Docker

Use this if you want to run on bare metal, develop the backend, or avoid Docker overhead.

### Prerequisites

- Python 3.11+
- Node.js 20+ (only for the frontend dev server)
- `llama-server` installed on `$PATH` (run `./check_llama.sh` to verify or install)

### Step-by-step

#### 1. Start llama-server (the model)

```bash
# Copy and edit .env
cp .env.example .env
# Set MODEL_PATH to the absolute path on your machine, e.g.:
#   MODEL_PATH=/home/you/models/Qwen3.5-0.8B-Q4_K_M.gguf

./serve_model.sh
```

`serve_model.sh` reads `.env`, detects your CPU core count, picks the right chat template for Qwen (`chatml`), and starts `llama-server` on port **8080**.

#### 2. Start the backend

```bash
cd backend

# Install dependencies (first time only)
pip install -r requirements.txt

# Run with hot-reload
LLAMA_SERVER_URL=http://127.0.0.1:8080 uvicorn app.main:app --reload --port 8000
```

Check it is up: http://localhost:8000/api/health

#### 3. Start the frontend (optional)

```bash
cd frontend

# Install dependencies (first time only)
npm install

# Start Vite dev server
npm run dev
```

Open http://localhost:5173 (Vite default).  
> The Vite dev server connects the WebSocket directly to `ws://localhost:8000/ws/chat`.  
> No Nginx proxy is needed in development.

#### All three processes together (split-terminal approach)

```
Terminal 1  →  ./serve_model.sh
Terminal 2  →  cd backend  && uvicorn app.main:app --reload --port 8000
Terminal 3  →  cd frontend && npm run dev
```

---

## 13. Voice Model Download

Voice models are downloaded automatically on first use, but you can pre-download them to avoid delays.

### Download Voice Models

```bash
# Download ASR (Whisper) and TTS (Piper) models
./download_voice_models.sh
```

This creates:
- `models/voice/whisper/` — Faster Whisper tiny model (~39MB)
- `models/voice/tts/` — Piper TTS voice model (~20MB)

### Model Details

| Component | Model | Size | Purpose |
|---|---|---|---|
| **ASR** | Faster Whisper `tiny` | ~39MB | Speech-to-text (English) |
| **TTS** | Piper `en_US-lessac-medium` | ~20MB | Text-to-speech (English female voice) |

Models are optimized for CPU inference and provide good quality with low latency.

---

## 14. Tests

### Unit Tests (pytest) — no running server needed

The unit tests mock the LLM server with `unittest.mock` so they run instantly without Docker.

```bash
cd backend
pip install -r requirements.txt   # includes pytest, pytest-asyncio

pytest tests/test_endpoints.py -v
```

**Coverage:**

| Test Class | Endpoint | What is tested |
|---|---|---|
| `TestHealth` | `GET /api/health` | 200 status, `status: ok`, `llama_server` field |
| `TestCreateSession` | `POST /api/sessions` | 201 status, unique IDs, session stored, default state |
| `TestListSessions` | `GET /api/sessions` | Empty list, correct count, field shapes, message count |
| `TestGetSession` | `GET /api/sessions/{id}` | 200, 404, messages list, history content |
| `TestDeleteSession` | `DELETE /api/sessions/{id}` | 200, 404, removal from store, subsequent GET is 404 |
| `TestWebSocketChat` | `WS /ws/chat` | session_id frame, done frame, session creation, history persistence, **state extraction from `<STATE>` block**, null values do not overwrite existing facts, state block stripped from token stream, error on bad JSON, error on empty message, error on missing field, error when LLM server unreachable |

### Integration / Smoke Tests (live server required)

`run_tests.py` sends real WebSocket messages to a running backend and prints the AI responses. Requires the full stack to be up.

```bash
# With Docker
docker compose up -d
python backend/run_tests.py

# Without Docker (all three processes running)
python backend/run_tests.py
```

Test scenarios:
1. **Remembrance** — mentions router model early, asks about it later
2. **Pivot-back** — off-topic question; does the agent return to diagnosis?
3. **End-of-state** — all 5 fields given in one message; check backend logs for state extraction
4. **Identity integrity** — attempts to break character / expose system prompt
5. **Multi-variable extraction** — verifies regex parsing in backend logs

---

## 15. Model Download — Qwen3.5-0.8B

### Option A — Hugging Face website (no tools needed)

1. Open **https://huggingface.co/bartowski/Qwen3.5-0.8B-GGUF/tree/main** in your browser
2. Click the file you want:

   | File | Size | Recommendation |
   |---|---|---|
   | `Qwen3.5-0.8B-Q4_K_M.gguf` | ~500 MB | Best balance of speed and quality ✅ |
   | `Qwen3.5-0.8B-Q8_0.gguf` | ~900 MB | Highest quality, needs more RAM |

3. Click the **download icon (↓)** on the right side of the file row
4. Move the file into `models/`:

```bash
mv ~/Downloads/Qwen3.5-0.8B-Q4_K_M.gguf models/
```

### Option B — Hugging Face CLI

```bash
pip install huggingface_hub

huggingface-cli download \
  bartowski/Qwen3.5-0.8B-GGUF \
  Qwen3.5-0.8B-Q4_K_M.gguf \
  --local-dir ./models
```

### Set MODEL_PATH in `.env`

```ini
MODEL_PATH=/models/Qwen3.5-0.8B-Q4_K_M.gguf
```

> Use the exact filename you downloaded. The path is always `/models/<filename>` inside the container — the `./models/` folder on your host is mounted at `/models` in Docker.

---

## 18. Standalone RAG Module

This repo includes a separate RAG pipeline in [rag/README.md](rag/README.md), and it is also wired into the backend orchestration as a LangChain tool (`retrieve_isp_knowledge`) for factual ISP support queries.

Quick start:

```bash
# install rag-specific dependencies
/opt/miniconda3/bin/python -m pip install -r rag/requirements.txt

# build vector index from ./rag_data
/opt/miniconda3/bin/python -m rag.ingest --reset

# test retrieval quality (no llm call)
/opt/miniconda3/bin/python -m rag.inference "My Nayatel connection keeps dropping" --no-llm --json
```

For full pipeline logic (chunking, metadata filtering, reranking, context compression, cache, and all CLI flags), use [rag/README.md](rag/README.md).

## 19. LangGraph RAG Tool Calling Integration

The backend orchestrator now supports conditional tool-calling with LangGraph/LangChain:

- The graph flow remains `START -> route -> dialogue -> parse_state -> END`.
- In `dialogue`, the model is bound to a function tool named `retrieve_isp_knowledge`.
- Tooling is only attempted for RAG-like queries (e.g., troubleshooting/device/setup/error intent), not greetings/chit-chat.
- If the model issues a tool call, backend executes retrieval via `rag.inference.RAGEngine`, appends a `ToolMessage`, and asks the model for the final user-facing response.
- `<STATE>...</STATE>` extraction and merge logic is unchanged.

Key files:

- [backend/app/orchestration/langgraph_engine.py](backend/app/orchestration/langgraph_engine.py)
- [backend/app/orchestration/tools/rag_tool.py](backend/app/orchestration/tools/rag_tool.py)

Notes:

- RAG engine initialization is lazy (first tool call), so normal conversation startup remains fast.
- If RAG dependencies are missing outside Docker, the tool returns a graceful "RAG retrieval unavailable" message instead of crashing the turn.
