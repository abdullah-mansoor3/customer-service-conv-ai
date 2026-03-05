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

DEFAULT_SYSTEM_PROMPT: str = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful customer service assistant. Be concise and friendly.",
)

MAX_HISTORY: int = int(os.getenv("MAX_HISTORY", "10"))

# ── Few-shot seed messages ────────────────────────────────────────────────
# Small models (≤4 B params) struggle to follow complex formatting rules
# from a system prompt alone.  We prepend two fake conversation turns so
# the model can learn the <STATE>{json}</STATE>\nreply pattern by example.
# These are never shown to the user and are always the first messages
# after the system prompt in every request to llama-server.

FEW_SHOT_EXAMPLES: list[dict] = [
    {
        "role": "user",
        "content": "Hi, I need help with my internet.",
    },
    {
        "role": "assistant",
        "content": (
            '<STATE>{"router_model": null, "lights_status": null, '
            '"error_message": null, "connection_type": null, '
            '"has_restarted": null}</STATE>\n'
            "Hello! Sorry to hear about your internet issues. "
            "What router model are you using?"
        ),
    },
    {
        "role": "user",
        "content": "I have a TP-Link and it keeps disconnecting on wifi.",
    },
    {
        "role": "assistant",
        "content": (
            '<STATE>{"router_model": "TP-Link", "lights_status": null, '
            '"error_message": "keeps disconnecting", '
            '"connection_type": "wifi", "has_restarted": null}</STATE>\n'
            "That sounds frustrating. Have you tried restarting "
            "the router yet?"
        ),
    },
]
