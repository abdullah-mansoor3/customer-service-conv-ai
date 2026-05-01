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
LLAMA_TOOL_PLANNER_TIMEOUT_SEC: float = float(os.getenv("LLAMA_TOOL_PLANNER_TIMEOUT_SEC", "30"))
LLAMA_TOOL_PLANNER_MAX_STEPS: int = int(os.getenv("LLAMA_TOOL_PLANNER_MAX_STEPS", "4"))
LLAMA_TOOL_PLANNER_MAX_TOKENS: int = int(os.getenv("LLAMA_TOOL_PLANNER_MAX_TOKENS", "160"))

DEFAULT_SYSTEM_PROMPT: str = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful customer service assistant. Be concise and friendly.",
)

PLANNER_PROMPT: str = os.getenv(
    "PLANNER_PROMPT",
    (
        "You are the planning model for an ISP support assistant. Read the latest user request and recent "
        "context, then decide whether calling tools would improve factual accuracy, personalization, or workflow "
        "quality. You may choose zero, one, or multiple tool calls over multiple rounds. If tools are not needed, "
        "explicitly continue with no tool calls. "
        "CRITICAL RULES:\n"
        "1. CLEAN QUERY EXTRACTION: Before calling ANY tool, ALWAYS extract a clean query by removing ALL greetings "
        "(hey, hi, hello, how are you, what\'s up, dude, by the way, amazing ui, who made you, are you AI) and "
        "filler phrases (can you, could you, please find, find me, look up). Keep ONLY the core question. "
        "Example: 'hey dude, find me the latest coverage area for ptcl, amazing ui by the way' becomes "
        "'PTCL coverage area'. Example: 'hey can you tell me about nayatel prices' becomes 'Nayatel prices'.\n"
        "2. WEB SEARCH + PAGE CONTENT FLOW: When user asks for current/latest/info that requires web search:\n"
        "   - First call search_web with the CLEAN query\n"
        "   - THEN you MUST call get_page_content for EACH URL returned by search_web\n"
        "   - After all page content is fetched, STOP calling more tools - the LLM will answer using that context\n"
        "3. URL-BASED QUERY: If user provides a URL and asks to get content/answer from it, call get_page_content directly\n"
        "   with the URL and the user's actual question (extracted as clean query).\n"
        "4. MULTI-STEP VALIDATION: After each tool call, check if the output answers the user's question. "
        "If it does, stop calling more tools. If not, continue to the next appropriate tool.\n"
        "5. After get_page_content calls are done (or if no web search needed), STOP - do not call more tools."
    ),
)

MAX_HISTORY: int = int(os.getenv("MAX_HISTORY", "10"))

# ── Few-shot seed messages ────────────────────────────────────────────────
# Previously we injected fake user/assistant turns, but small models treat
# them as real conversation history and hallucinate the example data.
# The format demonstration is now embedded directly in the system prompt
# (see chat.py) so the model learns the pattern without pollution.

FEW_SHOT_EXAMPLES: list[dict] = []
