# Parallel Team Workflow (Conflict Avoidance)

Use this when CRM, RAG, and support tool tracks are developed in parallel.

## Branch Strategy

1. Create dedicated branches:
- `feature/mcp-support-tools`
- `feature/mcp-crm-tools`
- `feature/mcp-rag-tools`

2. Keep each branch focused on one module only.

3. Rebase daily on `main`:

```bash
git checkout feature/mcp-support-tools
git fetch origin
git rebase origin/main
```

## File Ownership

1. Support owner edits only:
- `mcp_server/toolsets/support_toolset.py`
- `mcp_server/tools/support_workflows.py`
- `mcp_server/tools/web_search.py`

2. CRM owner edits only:
- `mcp_server/toolsets/crm_toolset.py` (recommended new file)
- `mcp_server/tools/crm_tools.py` (recommended new file)

3. RAG owner edits only:
- `mcp_server/toolsets/rag_toolset.py` (recommended new file)
- `mcp_server/tools/rag_tools.py` (recommended new file)

4. Shared file rule:
- Avoid direct edits in `server.py` by all three people at once.
- Instead, each person adds a new `toolsets/*_toolset.py` file with `register(mcp)`.

## Integration Contract

1. Keep tool function names stable once merged.
2. Version payload schemas with a `schema_version` field when changing payload shape.
3. Never break existing fields in place; add new optional fields first.

## Merge Sequence

1. Merge core support branch first.
2. Merge CRM branch second.
3. Merge RAG branch third.
4. Resolve only integration wiring conflicts in `server.py`.
4. In most cases no integration wiring edit is needed because server auto-discovers toolsets.

## Pull Request Checklist

1. PR touches only files in `mcp_server/`.
2. New tool has docstring and example payload in PR description.
3. New tool has at least one test.
4. Branch rebased within 24 hours of merge.
