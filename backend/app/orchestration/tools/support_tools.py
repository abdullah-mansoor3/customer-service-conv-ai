"""LangChain tool wrappers for support workflow and web-search helpers."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def _import_with_repo_fallback(module_name: str, symbol_name: str):
    """Import helper that adds repository root to sys.path when needed."""
    try:
        module = __import__(module_name, fromlist=[symbol_name])
        return getattr(module, symbol_name)
    except Exception:
        repo_root = Path(__file__).resolve().parents[4]
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        module = __import__(module_name, fromlist=[symbol_name])
        return getattr(module, symbol_name)


def _json_or_error(payload: Any, tool_name: str) -> str:
    if isinstance(payload, dict):
        return json.dumps(payload, ensure_ascii=False)
    return f"{tool_name} returned unsupported payload type: {type(payload).__name__}"


def search_web(query: str, max_results: int = 3) -> str:
    """Return up-to-date web search snippets for factual/time-sensitive queries."""
    try:
        web_search = _import_with_repo_fallback("mcp_server.tools.web_search", "web_search")
    except Exception as exc:  # noqa: BLE001
        return f"Web search unavailable: {exc}"

    result = web_search(query=query, max_results=max_results)
    return _json_or_error(result, "search_web")


MAX_CONTENT_LENGTH = 8000


def get_page_content(url: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
    """Fetch and extract main text content from a web page URL using BeautifulSoup.

    Strips scripts, styles, nav, header, footer, and aside elements,
    then returns the clean body text.  Should be called after search_web
    returns URLs that need deeper reading.
    """
    if not url or not url.strip():
        return '{"url":"","success":false,"error":"URL must be a non-empty string.","content":""}'

    url = url.strip()

    if not url.startswith(("http://", "https://")):
        return json.dumps(
            {"url": url, "success": False, "error": "URL must start with http:// or https://", "content": ""},
            ensure_ascii=False,
        )

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        },
    )

    try:
        with urlopen(request, timeout=15) as response:
            html_bytes = response.read()
            content_encoding = response.headers.get("Content-Encoding", "").lower()

            if content_encoding in ("gzip", "deflate"):
                import gzip

                if content_encoding == "gzip":
                    html_bytes = gzip.decompress(html_bytes)
                else:
                    import zlib

                    html_bytes = zlib.decompress(html_bytes)

            html = html_bytes.decode("utf-8", errors="ignore")

    except Exception as exc:
        logger.warning("get_page_content failed for %s: %s", url, exc)
        return json.dumps(
            {"url": url, "success": False, "error": f"Failed to fetch page: {exc}", "content": ""},
            ensure_ascii=False,
        )

    # ── BeautifulSoup extraction ─────────────────────────────────────────
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("beautifulsoup4 is not installed. Run: pip install beautifulsoup4")
        return json.dumps(
            {"url": url, "success": False, "error": "beautifulsoup4 not installed", "content": ""},
            ensure_ascii=False,
        )

    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup.find_all(["script", "style", "nav", "header", "footer", "aside", "noscript", "iframe"]):
        tag.decompose()

    # Extract clean body text
    body = soup.body
    if body:
        main_content = body.get_text(separator=" ", strip=True)
    else:
        main_content = soup.get_text(separator=" ", strip=True)

    if len(main_content) > max_length:
        main_content = main_content[:max_length].rsplit(" ", 1)[0] + "..."

    return json.dumps(
        {"url": url, "success": True, "content": main_content},
        ensure_ascii=False,
    )


def get_next_best_question(known_state: dict[str, Any]) -> str:
    """Suggest one high-value follow-up question from missing state fields."""
    try:
        next_best_question = _import_with_repo_fallback(
            "mcp_server.tools.support_workflows",
            "next_best_question",
        )
    except Exception as exc:  # noqa: BLE001
        return f"Support workflow tool unavailable: {exc}"

    result = next_best_question(known_state)
    return _json_or_error(result, "get_next_best_question")


def diagnose_connection_issue(known_state: dict[str, Any]) -> str:
    """Return deterministic diagnosis guidance from known state."""
    try:
        diagnose_issue = _import_with_repo_fallback("mcp_server.tools.support_workflows", "diagnose_issue")
    except Exception as exc:  # noqa: BLE001
        return f"Support diagnosis tool unavailable: {exc}"

    result = diagnose_issue(known_state)
    return _json_or_error(result, "diagnose_connection_issue")


def evaluate_escalation(
    known_state: dict[str, Any],
    failed_steps: list[str] | None = None,
    minutes_without_service: int | None = None,
) -> str:
    """Evaluate escalation need and priority from current troubleshooting outcome."""
    try:
        decide_escalation = _import_with_repo_fallback(
            "mcp_server.tools.support_workflows",
            "decide_escalation",
        )
    except Exception as exc:  # noqa: BLE001
        return f"Escalation tool unavailable: {exc}"

    result = decide_escalation(
        known_state=known_state,
        failed_steps=failed_steps or [],
        minutes_without_service=minutes_without_service,
    )
    return _json_or_error(result, "evaluate_escalation")


SEARCH_WEB_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Secondary external lookup tool. Use for time-sensitive/current ISP info when KB is insufficient "
            "or user explicitly asks to search online."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Concise web query with provider and target fact. "
                        "Example: 'PTCL internet package prices PKR'."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to include (1-5 preferred).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


GET_PAGE_CONTENT_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_page_content",
        "description": (
            "Fetch the full content of a web page URL to get detailed information not available in search snippets. "
            "Use after search_web returns relevant URLs - call this tool to get the full page content, "
            "then use that content as context to answer the user's question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The complete URL of the web page to fetch content from.",
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum characters to return (default 8000).",
                    "default": 8000,
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    },
}


GET_NEXT_BEST_QUESTION_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_next_best_question",
        "description": (
            "Workflow tool for troubleshooting conversations only. Returns one best follow-up question based "
            "on missing state fields."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "known_state": {
                    "type": "object",
                    "description": (
                        "State object with keys router_model, lights_status, error_message, connection_type, "
                        "has_restarted."
                    ),
                }
            },
            "required": ["known_state"],
            "additionalProperties": False,
        },
    },
}


DIAGNOSE_CONNECTION_ISSUE_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "diagnose_connection_issue",
        "description": (
            "Workflow diagnosis tool. Use after collecting troubleshooting state to infer likely root cause "
            "and next actions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "known_state": {
                    "type": "object",
                    "description": (
                        "State object with keys router_model, lights_status, error_message, connection_type, "
                        "has_restarted."
                    ),
                }
            },
            "required": ["known_state"],
            "additionalProperties": False,
        },
    },
}


EVALUATE_ESCALATION_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "evaluate_escalation",
        "description": (
            "Workflow escalation tool. Use when troubleshooting has failed or outage is prolonged to decide "
            "escalation necessity and priority."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "known_state": {
                    "type": "object",
                    "description": (
                        "State object with keys router_model, lights_status, error_message, connection_type, "
                        "has_restarted."
                    ),
                },
                "failed_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of troubleshooting steps already attempted and failed.",
                },
                "minutes_without_service": {
                    "type": "integer",
                    "description": "Approximate outage duration in minutes (non-negative integer).",
                },
            },
            "required": ["known_state"],
            "additionalProperties": False,
        },
    },
}
