# MCP Server for ISP Support Agent

This module provides a standalone MCP server for tool calls.

It is intentionally isolated from existing backend and frontend code so team members can work in parallel with minimal merge conflicts.

Tool registration uses module auto-discovery from `mcp_server/toolsets/`.
Each module can define `register(mcp)` and the server loads it automatically.

## Tool Set

This server exposes six tools:

1. `search_web(query, max_results=5)`
- Finds up-to-date public web information (DuckDuckGo search).

2. `get_next_best_question(known_state)`
- Returns one best follow-up question for missing support state.

3. `diagnose_connection_issue(known_state)`
- Returns likely cause and recommended next troubleshooting steps.

4. `evaluate_escalation(known_state, failed_steps=[], minutes_without_service=None)`
- Decides if escalation is needed and priority level.

5. `crm_tool_call(tool_name, payload)`
- Optional bridge for teammate CRM service (if `CRM_TOOL_BRIDGE_URL` is set).

6. `rag_tool_call(query, top_k=3)`
- Optional bridge for teammate RAG service (if `RAG_TOOL_BRIDGE_URL` is set).

## Why These 3 Core Support Tools

You asked for tools besides CRM and RAG. The three support-native tools are:

1. `get_next_best_question`
2. `diagnose_connection_issue`
3. `evaluate_escalation`

These map directly to a support workflow:
- gather missing facts,
- reason about likely cause,
- decide escalation.

## Quick Start

From repository root:

```bash
cd mcp_server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

The server runs with stdio transport (best for agent/editor MCP integration).

## Environment Setup

Copy env template if you want CRM/RAG bridges:

```bash
cp .env.example .env
```

Then export variables in your shell before starting, for example:

```bash
export CRM_TOOL_BRIDGE_URL=http://127.0.0.1:9101
export RAG_TOOL_BRIDGE_URL=http://127.0.0.1:9102
python server.py
```

## Bridge Contract for Teammates

If teammate services expose HTTP endpoint `POST /tool/call`, this server can call them.

Expected request shape:

```json
{
  "tool": "tool_name",
  "payload": {"any": "json"}
}
```

Expected response shape:

```json
{
  "ok": true,
  "result": {"any": "json"}
}
```

## Parallel Development Guidance

1. Keep MCP-only work inside `mcp_server/`.
2. Avoid editing backend router/orchestration files for tool experimentation.
3. Add new toolset modules under `mcp_server/toolsets/` with a `register(mcp)` function.
4. Keep core logic in `mcp_server/tools/`, and register wrappers in `toolsets/`.
5. Use one file owner per toolset file to avoid overlapping edits.
6. Integrate via environment variables and bridge contracts, not cross-branch imports.

### Teammate Extension Pattern

For CRM teammate:

1. Create `mcp_server/toolsets/crm_toolset.py`
2. Implement `register(mcp)` and add CRM tool wrappers
3. Keep CRM-specific logic in `mcp_server/tools/crm_tools.py`

For RAG teammate:

1. Create `mcp_server/toolsets/rag_toolset.py`
2. Implement `register(mcp)` and add RAG tool wrappers
3. Keep RAG-specific logic in `mcp_server/tools/rag_tools.py`
