"""Bridge tools for teammate-owned CRM and RAG services."""

from __future__ import annotations

from typing import Any

from tools.integrations import call_crm_bridge, call_rag_bridge


def register(mcp) -> None:
    """Register bridge tools into the MCP server."""

    @mcp.tool()
    def crm_tool_call(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Optional bridge for teammate-owned CRM tool-calls via HTTP."""
        return call_crm_bridge(tool_name=tool_name, payload=payload)

    @mcp.tool()
    def rag_tool_call(query: str, top_k: int = 3) -> dict[str, Any]:
        """Optional bridge for teammate-owned RAG tool-calls via HTTP."""
        return call_rag_bridge(query=query, top_k=top_k)
