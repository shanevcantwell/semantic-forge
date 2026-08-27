"""Minimal stdio MCP server stub standing in for semantic-kinematics-mcp in tests.

Speaks just enough of the MCP tool-call protocol to satisfy
`SemanticKinematicsClient`: `model_status`, `analyze_trajectory`, and
`calculate_drift`. Used by the client-lifecycle regression test
(issue #7) to drive real stdio subprocess + anyio task-group teardown,
rather than mocking the transport away.

Run directly as a subprocess: `python stub_sk_server.py`.
"""
import asyncio
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

server = Server("stub-semantic-kinematics-mcp")


_EMPTY_SCHEMA = {"type": "object", "properties": {}}


@server.list_tools()
async def _list_tools() -> list[Tool]:
    return [
        Tool(name="model_status", description="stub", inputSchema=_EMPTY_SCHEMA),
        Tool(name="analyze_trajectory", description="stub", inputSchema=_EMPTY_SCHEMA),
        Tool(name="calculate_drift", description="stub", inputSchema=_EMPTY_SCHEMA),
    ]


@server.call_tool()
async def _call_tool(name: str, arguments: dict) -> CallToolResult:
    if name == "model_status":
        payload = {"backend": "stub", "model_name": "stub-model", "is_loaded": True}
    elif name == "analyze_trajectory":
        payload = {
            "mean_velocity": 0.5,
            "deadpan_score": 0.3,
            "acceleration_spikes": [],
            "torsion": 0.1,
            "curvature": 0.2,
        }
    elif name == "calculate_drift":
        payload = {"drift": 0.4}
    else:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {name}")],
            isError=True,
        )
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(payload))])


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
