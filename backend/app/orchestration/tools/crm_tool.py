"""LangChain tool wrappers for CRM profile lookups and updates."""

from __future__ import annotations

from typing import Any

from app.crm.crm_tool import execute_crm_tool


async def get_user_info(user_id: str) -> str:
    """Retrieve CRM profile summary for the current user/session."""
    return await execute_crm_tool("get_user_info", {"user_id": user_id})


async def update_user_info(user_id: str, field: str, value: Any) -> str:
    """Persist one CRM field update for the current user/session."""
    return await execute_crm_tool(
        "update_user_info",
        {
            "user_id": user_id,
            "field": field,
            "value": value,
        },
    )


GET_USER_INFO_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_user_info",
        "description": (
            "Retrieve stored information about the user, including known profile fields and recent interactions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Session or user identifier.",
                }
            },
            "required": ["user_id"],
            "additionalProperties": False,
        },
    },
}


UPDATE_USER_INFO_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "update_user_info",
        "description": (
            "Store or update one user profile field, such as name, contact, or preference values."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Session or user identifier.",
                },
                "field": {
                    "type": "string",
                    "description": "Profile field to update (for example: name, contact, preferred_language).",
                },
                "value": {
                    "type": "string",
                    "description": "New value for the target field.",
                },
            },
            "required": ["user_id", "field", "value"],
            "additionalProperties": False,
        },
    },
}
