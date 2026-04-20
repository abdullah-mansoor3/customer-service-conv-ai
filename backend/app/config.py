import os

# ── Load .env from project root ──────────────────────────────────────────
# When running via `uvicorn app.main:app` from backend/, the .env lives
# one level up (the repository root).  We also check the cwd as a fallback.
for _candidate in [
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
    os.path.join(os.getcwd(), "..", ".env"),
]:
    if os.path.isfile(_candidate):
        with open(_candidate, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _key, _val = _line.split("=", 1)
                    os.environ.setdefault(_key.strip(), _val.strip(' "\''))
        break

LLAMA_SERVER_URL: str = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
LLAMA_MODEL_NAME: str = os.getenv("LLAMA_MODEL_NAME", "local-model")
LLAMA_API_KEY: str = os.getenv("LLAMA_API_KEY", "not-needed")
LLAMA_TEMPERATURE: float = float(os.getenv("LLAMA_TEMPERATURE", "0.35"))
LLAMA_TOP_P: float = float(os.getenv("LLAMA_TOP_P", "0.9"))
LLAMA_TIMEOUT_SEC: float = float(os.getenv("LLAMA_TIMEOUT_SEC", "90"))
LLAMA_TOOL_PLANNER_TIMEOUT_SEC: float = float(os.getenv("LLAMA_TOOL_PLANNER_TIMEOUT_SEC", "8"))

DEFAULT_SYSTEM_PROMPT: str = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful customer service assistant. Be concise and friendly.",
)

MAX_HISTORY: int = int(os.getenv("MAX_HISTORY", "10"))

# ── Few-shot seed messages ────────────────────────────────────────────────
# Previously we injected fake user/assistant turns, but small models treat
# them as real conversation history and hallucinate the example data.
# The format demonstration is now embedded directly in the system prompt
# (see chat.py) so the model learns the pattern without pollution.

FEW_SHOT_EXAMPLES: list[dict] = []
