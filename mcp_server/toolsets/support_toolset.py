"""Core support tools owned by the support-agent workstream."""

from __future__ import annotations

from typing import Any

from tools.support_workflows import decide_escalation, diagnose_issue, next_best_question
from tools.web_search import web_search


def register(mcp) -> None:
    """Register support-native tools into the MCP server."""

    @mcp.tool()
    def search_web(query: str, max_results: int = 5) -> dict[str, Any]:
        """Search the public web for up-to-date troubleshooting information."""
        return web_search(query=query, max_results=max_results)

    @mcp.tool()
    def get_next_best_question(known_state: dict[str, Any]) -> dict[str, Any]:
        """Return one high-value follow-up question based on missing support state."""
        return next_best_question(known_state)

    @mcp.tool()
    def diagnose_connection_issue(known_state: dict[str, Any]) -> dict[str, Any]:
        """Estimate likely root cause and recommended next troubleshooting steps."""
        return diagnose_issue(known_state)

    @mcp.tool()
    def evaluate_escalation(
        known_state: dict[str, Any],
        failed_steps: list[str] | None = None,
        minutes_without_service: int | None = None,
    ) -> dict[str, Any]:
        """Decide whether the ticket should be escalated and with what priority."""
        return decide_escalation(
            known_state=known_state,
            failed_steps=failed_steps or [],
            minutes_without_service=minutes_without_service,
        )
