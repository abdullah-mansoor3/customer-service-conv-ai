"""LangChain tool wrappers for support workflow and web-search helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


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
