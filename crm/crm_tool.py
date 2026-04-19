import json
import asyncio
from crm.crm_store import get_user, create_or_get_user, update_user
from mcp_server.tools.integrations import call_crm_bridge

# ── Tool schemas (pass these to your LLM's tool/function list) ───────────────

CRM_TOOLS = [
    {
        "name": "get_user_info",
        "description": (
            "Retrieve stored information about the current user, including their name, "
            "contact details, preferences, and recent interaction history."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "The unique session or user identifier."
                }
            },
            "required": ["user_id"]
        }
    },
    {
        "name": "update_user_info",
        "description": (
            "Store or update a specific piece of information about the user, such as "
            "their name, contact number, or a preference (e.g. preferred language, "
            "appointment time, dietary restriction)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "The unique session or user identifier."
                },
                "field": {
                    "type": "string",
                    "description": (
                        "The field to update. Use 'name' for the user's name, "
                        "'contact' for phone/email, or any custom preference key."
                    )
                },
                "value": {
                    "description": "The value to store for the given field."
                }
            },
            "required": ["user_id", "field", "value"]
        }
    }
]

async def execute_crm_tool(tool_name: str, arguments: dict) -> str:
    """
    Tries the MCP bridge first. Falls back to local store if bridge
    is not configured or returns an error.
    """
    # ── 1. Try the bridge (runs in threadpool since httpx.post is sync) ──
    bridge_result = await asyncio.get_event_loop().run_in_executor(
        None, call_crm_bridge, tool_name, arguments
    )

    if bridge_result["ok"]:
        # Bridge succeeded — return its data as a string for the LLM
        return json.dumps(bridge_result["data"], indent=2)

    if bridge_result["status"] != "not_configured":
        # Bridge is configured but errored — log and fall through to local
        print(f"[CRM bridge error] {bridge_result['message']} — using local fallback")

    # ── 2. Local fallback ─────────────────────────────────────────────────
    return await _execute_local(tool_name, arguments)


async def _execute_local(tool_name: str, arguments: dict) -> str:
    """Local JSON-file CRM — used when bridge is absent or errored."""
    try:
        if tool_name == "get_user_info":
            user_id = arguments["user_id"]
            user = get_user(user_id)
            if user is None:
                user = create_or_get_user(user_id)
                return "No prior record found. A new profile has been created."
            summary = {
                "name": user.get("name"),
                "contact": user.get("contact"),
                "preferences": user.get("preferences", {}),
                "last_seen": user.get("last_seen"),
                "recent_interactions": user.get("interaction_history", [])[-5:],
            }
            return json.dumps(summary, indent=2)

        elif tool_name == "update_user_info":
            user_id = arguments["user_id"]
            field   = arguments["field"]
            value   = arguments["value"]
            updated = update_user(user_id, field, value)
            return f"Updated '{field}' for user {user_id}. Current name: {updated.get('name')}."

        else:
            return f"Unknown CRM tool: {tool_name}"

    except Exception as e:
        return f"CRM tool error: {str(e)}"