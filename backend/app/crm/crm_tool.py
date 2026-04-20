import json
import asyncio

from app.crm.crm_store import create_or_get_user, get_user, update_user

try:
    from mcp_server.tools.integrations import call_crm_bridge
except Exception:  # noqa: BLE001
    call_crm_bridge = None

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

# ── Executor: maps tool name → function ──────────────────────────────────────

async def execute_crm_tool(tool_name: str, arguments: dict) -> str:
    """
    Try bridge integration first when available, then use local fallback.
    """
    if call_crm_bridge is not None:
        bridge_result = await asyncio.get_event_loop().run_in_executor(
            None,
            call_crm_bridge,
            tool_name,
            arguments,
        )

        if bridge_result.get("ok"):
            return json.dumps(bridge_result.get("data", {}), indent=2)

        if bridge_result.get("status") not in {"not_configured", None}:
            print(f"[CRM bridge error] {bridge_result.get('message')} — using local fallback")

    return await _execute_local(tool_name, arguments)


async def _execute_local(tool_name: str, arguments: dict) -> str:
    """Local JSON-file CRM fallback."""
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