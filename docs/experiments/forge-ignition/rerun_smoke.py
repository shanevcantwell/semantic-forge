"""forge-ignition phase 1 smoke: one real permutate_phrasing tool call via the
registered MCP dispatch surface, against live endpoints from semantic_forge_config.json.
Artifact lands at /tmp/semantic_forge_smoke/permutate_phrasing_scope_discipline.json
"""
import asyncio
import json
from pathlib import Path

from mcp import types
from mcp.server import Server
from semantic_forge.handlers import register_handlers


async def main() -> None:
    server = Server("forge-ignition-smoke")
    await register_handlers(server)
    dispatch = server.request_handlers[types.CallToolRequest]
    req = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(
            name="permutate_phrasing",
            arguments={
                "concept": "Do only what the request asks; no extra writes, no side effects beyond scope.",
                "moods": ["imperative", "socratic"],  # minimal: two moods to bound shared-server cost
            },
        ),
    )
    wrapped = await dispatch(req)
    result = getattr(wrapped, "root", wrapped)  # ServerResult wrapper -> CallToolResult
    text = result.content[0].text if getattr(result, "content", None) else str(result)
    data = json.loads(text) if isinstance(text, (str, bytes)) and text.strip().startswith(("{", "[")) else {"raw": text}

    out = Path("/tmp/semantic_forge_smoke")
    out.mkdir(parents=True, exist_ok=True)
    artifact = out / "permutate_phrasing_scope_discipline.json"
    artifact.write_text(json.dumps(data, indent=2) + "\n")
    print("isError:", getattr(result, "isError", None))
    print(f"wrote {artifact}")
    print("first lines:")
    print("\n".join(artifact.read_text().splitlines()[:14]))


if __name__ == "__main__":
    asyncio.run(main())
