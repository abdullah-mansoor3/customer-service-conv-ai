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
            "Profile-memory lookup tool. Use only for personalization/memory requests (name/contact/preferences), "
            "not for ISP factual knowledge."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Session/user identifier for CRM profile lookup.",
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
            "Profile-memory write tool. Use only when user explicitly provides or asks to remember profile data "
            "(name/contact/preferences)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Session/user identifier for CRM profile update.",
                },
                "field": {
                    "type": "string",
                    "description": (
                        "Field to update. Prefer 'name' or 'contact' for top-level fields; other keys are stored "
                        "under preferences."
                    ),
                },
                "value": {
                    "type": "string",
                    "description": "Value to store exactly as provided by the user.",
                },
            },
            "required": ["user_id", "field", "value"],
            "additionalProperties": False,
        },
    },
}
