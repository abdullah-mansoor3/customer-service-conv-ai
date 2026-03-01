import os

LLAMA_SERVER_URL: str = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")

# TODO (Conversation Team): replace with your domain-specific system prompt.
DEFAULT_SYSTEM_PROMPT: str = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful customer service assistant. Be concise and friendly.",
)
