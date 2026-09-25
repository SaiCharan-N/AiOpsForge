"""AIOpsForge MCP Server.

Thin FastMCP app wiring: creates the server and registers every tool from
tools.ALL_TOOLS. Doesn't know or care what any individual tool does —
adding, removing, or modifying a tool never touches this file (see
tools/__init__.py). Path-safety and per-project isolation live in
workspace.py, shared by every tool.
"""
import os

from mcp.server.fastmcp import FastMCP

from tools import ALL_TOOLS

mcp = FastMCP(
    name="aiopsforge-tools",
    instructions=(
        "Tools for writing files, running shell commands, and running tests "
        "inside an isolated per-project workspace. Always pass the project_id "
        "you were given for the current task."
    ),
    host=os.environ.get("MCP_HOST", "0.0.0.0"),
    port=int(os.environ.get("MCP_PORT", "9000")),
)

for _tool_fn in ALL_TOOLS:
    mcp.tool()(_tool_fn)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
