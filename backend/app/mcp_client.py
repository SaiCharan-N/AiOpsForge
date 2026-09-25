"""MCP client-side discovery and call logic (Week 3 / Days 12-14).

This is what makes AIOpsForge's tool use *dynamic* rather than hardcoded:
the Developer and QA agents don't import a `write_file()` function directly —
they ask the MCP server what tools exist, then call one by name. Swapping or
adding tools later means changing the MCP server, not the agent code.

Every function here opens a fresh MCP session per call. That's simpler and
safer for a student project than holding a long-lived connection open across
agent turns, at some latency cost — a reasonable trade-off to note in your
paper if you ever optimize this further.
"""
import json
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from app.config import settings


@asynccontextmanager
async def mcp_session():
    async with streamablehttp_client(settings.mcp_server_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def discover_tools() -> list[dict[str, Any]]:
    """Lists every tool currently exposed by the MCP server, with its
    description and input schema — this is the 'dynamic tool discovery'
    step the Developer Agent runs before deciding what to call.
    """
    async with mcp_session() as session:
        result = await session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.inputSchema,
            }
            for t in result.tools
        ]


async def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """Calls a single MCP tool by name and returns its text output.
    Raises RuntimeError if the tool itself reported an error.
    """
    async with mcp_session() as session:
        result = await session.call_tool(name, arguments)
        text_parts = [c.text for c in result.content if hasattr(c, "text")]
        output = "\n".join(text_parts)
        if result.isError:
            raise RuntimeError(f"MCP tool '{name}' failed: {output}")
        return output


def tools_catalog_for_prompt(tools: list[dict[str, Any]]) -> str:
    """Formats the discovered tool list as compact text for an LLM prompt."""
    lines = []
    for t in tools:
        params = ", ".join(t["input_schema"].get("properties", {}).keys())
        lines.append(f"- {t['name']}({params}): {t['description']}")
    return "\n".join(lines)
