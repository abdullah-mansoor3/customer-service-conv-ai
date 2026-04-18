"""Optional bridge calls for teammate-owned tool services (CRM and RAG)."""

from __future__ import annotations

import os
from typing import Any

import httpx


def _post_tool_call(base_url: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    if not base_url:
        return {
            "ok": False,
            "status": "not_configured",
            "message": "Bridge endpoint is not configured.",
        }

    endpoint = f"{base_url.rstrip('/')}/tool/call"
    timeout_sec = float(os.getenv("MCP_BRIDGE_TIMEOUT_SEC", "15"))

    try:
        response = httpx.post(endpoint, json=payload, timeout=timeout_sec)
        response.raise_for_status()
        return {
            "ok": True,
            "status": "ok",
            "endpoint": endpoint,
            "data": response.json(),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status": "error",
            "endpoint": endpoint,
            "message": str(exc),
        }


def call_crm_bridge(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Call teammate CRM tool service if endpoint is configured."""
    base_url = os.getenv("CRM_TOOL_BRIDGE_URL")
    request = {
        "tool": tool_name,
        "payload": payload,
    }
    return _post_tool_call(base_url=base_url, payload=request)


def call_rag_bridge(query: str, top_k: int = 3) -> dict[str, Any]:
    """Call teammate RAG tool service if endpoint is configured."""
    base_url = os.getenv("RAG_TOOL_BRIDGE_URL")
    request = {
        "tool": "rag_retrieve",
        "payload": {
            "query": query,
            "top_k": max(1, min(top_k, 10)),
        },
    }
    return _post_tool_call(base_url=base_url, payload=request)
