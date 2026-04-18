"""MCP server for the ISP support agent tool-calling layer.

This server is intentionally isolated from backend/frontend runtime code so
multiple teammates can add tools in parallel with minimal merge conflicts.
"""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("isp-support-agent-tools")


def register_toolsets() -> None:
    """Discover toolset modules and call their register(mcp) function."""
    import toolsets

    for module_info in iter_modules(toolsets.__path__, prefix="toolsets."):
        module = import_module(module_info.name)
        register = getattr(module, "register", None)
        if callable(register):
            register(mcp)


def main() -> None:
    """Run MCP server on stdio transport (best for editor agent integrations)."""
    register_toolsets()
    mcp.run()


if __name__ == "__main__":
    main()
