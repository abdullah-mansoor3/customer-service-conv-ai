# CRM Module — Customer Relationship Management

## Overview

The CRM module provides persistent user profile management for your conversational AI chatbot. It stores and retrieves user information (name, contact details, preferences, interaction history) across sessions, enabling personalized responses and context-aware conversations.

The module supports both **local persistence** (JSON file) and **remote bridge mode** (HTTP calls to a teammate's service), with automatic fallback if the remote service is unavailable.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Conversation Manager                     │
│  (builds system prompt with user profile context)           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Tool Orchestrator                              │
│  (detects LLM tool calls, routes to appropriate tool)       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            crm/crm_tool.py::execute_crm_tool()              │
│  (bridge-aware CRM executor)                                │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
   [Bridge Mode]            [Local Mode]
   (if URL set)             (fallback)
        │                         │
        ▼                         ▼
   HTTP POST to        crm/crm_store.py
   CRM_TOOL_BRIDGE_URL  (JSON file I/O)
        │
        └────► crm/data/crm_data.json
```

### Components

| File | Purpose |
|------|---------|
| `crm/__init__.py` | Public API exports |
| `crm/crm_tool.py` | Tool executor with bridge-first logic |
| `crm/crm_store.py` | Local JSON persistence layer |
| `crm/data/crm_data.json` | User profile database (auto-created) |

---

## Installation

1. **Copy the CRM module into your project:**
   ```
   your_project/
   ├── crm/
   │   ├── __init__.py
   │   ├── crm_tool.py
   │   ├── crm_store.py
   │   └── data/              # auto-created
   ├── main.py
   ├── conversation_manager.py
   └── integration.py
   ```

2. **No additional dependencies** — uses only Python stdlib (`json`, `datetime`, `os`)

---

## Configuration

### Environment Variables

Set these in your `.env` file:

```env
# Optional: URL of teammate's CRM bridge service
# If unset, the module uses local JSON store
CRM_TOOL_BRIDGE_URL=http://localhost:8001

# Optional: timeout for bridge HTTP calls (seconds)
MCP_BRIDGE_TIMEOUT_SEC=15
```

### Docker Compose

```yaml
services:
  chatbot:
    build: .
    environment:
      # Use local store if your teammate's service isn't running
      - CRM_TOOL_BRIDGE_URL=${CRM_TOOL_BRIDGE_URL:-}
      - MCP_BRIDGE_TIMEOUT_SEC=15
    volumes:
      # Persist CRM data across container restarts
      - ./crm/data:/app/crm/data
```

---

## Usage

### 1. Initialize user profile on session start

In your WebSocket handler (e.g., `main.py`):

```python
from crm import create_or_get_user

@app.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())  # unique per user session

    # Load or create user profile in CRM
    user_profile = create_or_get_user(session_id)

    # Pass to conversation manager
    manager = ConversationManager(
        session_id=session_id,
        user_profile=user_profile
    )
    ...
```

### 2. Inject user context into system prompt

In your `conversation_manager.py`:

```python
def build_system_prompt(self) -> str:
    name = self.user_profile.get("name") or "the user"
    prefs = self.user_profile.get("preferences", {})
    pref_str = ", ".join(f"{k}: {v}" for k, v in prefs.items()) or "none recorded"

    return f"""You are a helpful customer service assistant.

## Current User Profile
- Name: {name}
- Preferences: {pref_str}

## Available Tools
When you need to retrieve or update user information, respond with JSON:

{{
  "tool_call": {{
    "name": "get_user_info" | "update_user_info",
    "arguments": {{ ... }}
  }}
}}
"""
```

### 3. Handle tool calls in message loop

```python
import json
from crm import execute_crm_tool, log_interaction

async def process_message(user_message: str, session_id: str, manager):
    # Log user message
    log_interaction(session_id, "user", user_message)

    # Get LLM response
    prompt = manager.build_prompt(user_message)
    llm_output = await llm_engine.generate(prompt)

    # Check if LLM requested a tool call
    tool_result = await handle_tool_call_if_any(llm_output, session_id)

    if tool_result:
        # Feed tool result back to LLM
        follow_up_prompt = manager.build_tool_result_prompt(tool_result)
        final_response = await llm_engine.generate(follow_up_prompt)
    else:
        final_response = llm_output

    # Log assistant response
    log_interaction(session_id, "assistant", final_response)
    
    return final_response


async def handle_tool_call_if_any(raw_llm_output: str, session_id: str) -> str | None:
    """Returns tool result if a tool call was detected, else None."""
    try:
        # Strip markdown formatting if present
        cleaned = raw_llm_output.strip().strip("```json").strip("```").strip()
        parsed = json.loads(cleaned)
        
        if "tool_call" in parsed:
            tool = parsed["tool_call"]
            name = tool["name"]
            args = tool.get("arguments", {})
            args["user_id"] = session_id  # Always inject session ID

            result = await execute_crm_tool(name, args)
            return result
    except (json.JSONDecodeError, KeyError):
        pass
    return None
```

### 4. Log interactions (optional but recommended)

```python
from crm import log_interaction

# Called automatically in process_message() above,
# but you can also call it directly:

log_interaction(session_id, "user", "What's my name?")
log_interaction(session_id, "assistant", "Your name is Alice.")
```

---

## Tool Schemas

### `get_user_info`

Retrieves stored information about a user.

**Schema:**
```json
{
  "name": "get_user_info",
  "description": "Retrieve stored information about the current user, including their name, contact details, preferences, and recent interaction history.",
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
}
```

**Example LLM call:**
```json
{
  "tool_call": {
    "name": "get_user_info",
    "arguments": {
      "user_id": "session-abc123"
    }
  }
}
```

**Response (JSON):**
```json
{
  "name": "Alice Smith",
  "contact": "alice@example.com",
  "preferences": {
    "preferred_time": "10:00 AM",
    "dietary_restriction": "vegetarian"
  },
  "last_seen": "2024-04-19T14:32:00.123456",
  "recent_interactions": [
    {
      "role": "user",
      "message": "I'd like to book an appointment",
      "timestamp": "2024-04-19T14:30:00.123456"
    }
  ]
}
```

---

### `update_user_info`

Stores or updates a specific field in a user's profile.

**Schema:**
```json
{
  "name": "update_user_info",
  "description": "Store or update a specific piece of information about the user, such as their name, contact number, or a preference.",
  "parameters": {
    "type": "object",
    "properties": {
      "user_id": {
        "type": "string",
        "description": "The unique session or user identifier."
      },
      "field": {
        "type": "string",
        "description": "The field to update. Use 'name' for the user's name, 'contact' for phone/email, or any custom preference key."
      },
      "value": {
        "description": "The value to store for the given field."
      }
    },
    "required": ["user_id", "field", "value"]
  }
}
```

**Example LLM call:**
```json
{
  "tool_call": {
    "name": "update_user_info",
    "arguments": {
      "user_id": "session-abc123",
      "field": "preferred_language",
      "value": "Spanish"
    }
  }
}
```

**Response (string):**
```
Updated 'preferred_language' for user session-abc123. Current name: Alice Smith.
```

---

## Bridge Mode (Teammate Integration)

### What it does

If `CRM_TOOL_BRIDGE_URL` is set, all CRM tool calls are forwarded to your teammate's CRM service via HTTP. If the bridge is unavailable, the system automatically falls back to local JSON storage.

### Setup

1. **Get your teammate's CRM service URL** (e.g., `http://crm-service.herokuapp.com`)

2. **Set environment variable:**
   ```env
   CRM_TOOL_BRIDGE_URL=http://crm-service.herokuapp.com
   ```

3. **Expected bridge contract:**
   
   The bridge expects `POST /tool/call` with this body:
   ```json
   {
     "tool": "get_user_info" | "update_user_info",
     "payload": {
       "user_id": "...",
       "field": "...",
       "value": "..."
     }
   }
   ```

   And returns:
   ```json
   {
     "ok": true | false,
     "status": "ok" | "error" | "not_configured",
     "data": {...},
     "message": "..."
   }
   ```

### Behavior

```python
# When CRM_TOOL_BRIDGE_URL = "http://localhost:8001"
# and the service is running:
execute_crm_tool("get_user_info", {...})
    ↓
    tries HTTP POST to http://localhost:8001/tool/call
    ↓
    gets response: {"ok": true, "data": {...}}
    ↓
    returns stringified data to LLM

# When CRM_TOOL_BRIDGE_URL is unset or bridge is down:
execute_crm_tool("get_user_info", {...})
    ↓
    detects error or not_configured
    ↓
    falls back to local JSON store
    ↓
    returns data from crm/data/crm_data.json
```

---

## Data Storage

### Local Store Format

File: `crm/data/crm_data.json`

```json
{
  "session-abc123": {
    "user_id": "session-abc123",
    "name": "Alice Smith",
    "contact": "alice@example.com",
    "preferences": {
      "preferred_time": "10:00 AM",
      "dietary_restriction": "vegetarian"
    },
    "interaction_history": [
      {
        "role": "user",
        "message": "I'd like to book an appointment",
        "timestamp": "2024-04-19T14:30:00.123456"
      },
      {
        "role": "assistant",
        "message": "Sure, I can help with that...",
        "timestamp": "2024-04-19T14:30:05.123456"
      }
    ],
    "created_at": "2024-04-19T14:30:00.123456",
    "last_seen": "2024-04-19T14:32:00.123456"
  }
}
```

### Storage Limits

- **Interaction history**: Kept to last 20 entries per user to prevent unbounded growth
- **File I/O**: Synchronous; no concurrent write protection (fine for single server)
- **Top-level fields**: `name`, `contact`, `created_at`, `last_seen`, `interaction_history`
- **Custom fields**: Go into `preferences` dict (any key-value pair)

---

## Performance

### Latency

| Operation | Time | Notes |
|-----------|------|-------|
| `get_user_info` (local) | ~5 ms | JSON file read |
| `update_user_info` (local) | ~10 ms | JSON file write |
| Bridge call (HTTP) | 50–500 ms | Depends on network & teammate's service |
| Fallback (on bridge error) | ~5 ms | Automatic, no added latency |

### Benchmarks

Run the included test suite:
```bash
python test_crm.py              # Local only (~50ms for 6 ops)
python test_bridge_mock.py      # Bridge + fallback (~2s total)
```

---

## Error Handling

### Bridge unavailable
→ Automatically falls back to local store. User sees no delay; response may be from local cache instead of remote.

### Invalid tool arguments
→ Returns JSON error string: `"CRM tool error: ..."`

### File I/O error
→ Logs error; returns error message to LLM.

### User not found
→ Creates new profile with defaults; returns creation message.

---

## Examples

### Example 1: Greeting returning user

**User:** "Hi, I'm back"

**LLM system prompt** (includes user profile):
```
Current User Profile:
- Name: Alice Smith
- Preferences: preferred_time: 10:00 AM
```

**LLM behavior:**
1. Recognizes returning user
2. May call `get_user_info` to refresh latest preferences
3. Responds: "Welcome back, Alice! I see you prefer 10:00 AM appointments. How can I help?"

---

### Example 2: Updating preferences

**User:** "I prefer mornings, like 9 AM"

**LLM recognizes preference update:**
```json
{
  "tool_call": {
    "name": "update_user_info",
    "arguments": {
      "user_id": "session-abc123",
      "field": "preferred_time",
      "value": "9:00 AM"
    }
  }
}
```

**Response:** `"Updated 'preferred_time' for user session-abc123. Current name: Alice Smith."`

**LLM continues:** "Got it! I've updated your preference to 9:00 AM. Your next appointment will be scheduled for that time."

---

### Example 3: Multi-turn conversation with CRM context

**Turn 1:**
- User: "What's my phone number?"
- LLM calls `get_user_info` → returns `contact: null`
- LLM: "I don't have a phone number on file. Could you provide one?"

**Turn 2:**
- User: "It's 555-1234"
- LLM calls `update_user_info` with `field: "contact"`, `value: "555-1234"`
- LLM: "Got it, I've saved 555-1234. Is there anything else?"

---

## Limitations & Known Issues

1. **No concurrent write protection**: If multiple users update the same profile simultaneously, the last write wins. For production, consider SQLite or a database.

2. **Interaction history capped at 20**: Prevents `crm_data.json` from growing indefinitely, but means old conversations are pruned.

3. **Bridge timeout fixed at 15s**: Configurable via `MCP_BRIDGE_TIMEOUT_SEC` env var, but no per-call override.

4. **No encryption**: CRM data stored in plain JSON. For sensitive data (passwords, credit cards), add encryption before storing.

5. **No user authentication**: Session ID is assumed to uniquely identify a user. Real systems should use signed JWTs or database sessions.

6. **Preferences are untyped**: Any value (string, number, bool) stored in `preferences` dict. No schema validation.

---

## Extending the CRM

### Add a new top-level field

Edit `crm/crm_store.py::create_or_get_user()`:
```python
data[user_id] = {
    "user_id": user_id,
    "name": None,
    "contact": None,
    "email": None,           # ← NEW
    "preferences": {},
    ...
}
```

Then update `build_system_prompt()` in `conversation_manager.py` to include it.

### Add a new CRM tool

1. Define schema in `crm/crm_tool.py::CRM_TOOLS` list
2. Implement executor function in `_execute_local()`
3. Test with `test_crm.py`
4. Add to LLM's tool list

### Switch to SQLite instead of JSON

Replace `crm/crm_store.py` with:
```python
import sqlite3

def get_user(user_id):
    conn = sqlite3.connect("crm/data/crm.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    # ...
```

No changes needed to `crm_tool.py` — the interface stays the same.

---

## Testing

See separate `test_crm.py` and `test_bridge_mock.py` for unit and integration tests.

Quick check:
```bash
# Test local store
python test_crm.py

# Test bridge fallback
python test_bridge_mock.py
```

---

## FAQ

**Q: Can I use this without the bridge?**
A: Yes. Just leave `CRM_TOOL_BRIDGE_URL` unset. Everything uses the local JSON store.

**Q: What if my teammate's CRM service goes down?**
A: The system automatically falls back to local storage. No conversation interruption.

**Q: How long are conversation histories kept?**
A: Last 20 interactions per user. Older ones are pruned.

**Q: Can I export user data?**
A: Yes, copy `crm/data/crm_data.json` or parse it with Python's `json` module.

**Q: Does this work with cloud deployments?**
A: Yes, but `crm/data/` needs to be a persistent volume (Docker volume or cloud storage). Don't rely on ephemeral filesystems.

---

## Support

For issues:
1. Check that `CRM_TOOL_BRIDGE_URL` is set correctly (or unset for local-only mode)
2. Verify `crm/data/` directory exists and is writable
3. Run `test_crm.py` to isolate local store issues
4. Run `test_bridge_mock.py` to isolate bridge issues
5. Check logs for `[CRM]` tagged messages