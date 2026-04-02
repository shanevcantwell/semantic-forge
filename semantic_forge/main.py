"""Main entry point for semantic-forge MCP server."""

import argparse
import asyncio
import sys
from pathlib import Path

from mcp.server import Server

from semantic_forge.mcp import (
    create_mcp_server,
    get_all_tools,
)
from semantic_forge.handlers import register_handlers
from semantic_forge.concepts import CONCEPT_LIBRARY, get_concept_by_id


async def run_server() -> None:
    """Run the MCP server using stdio transport.

    Note: This server uses stdio transport only (stdio JSON-RPC).
    The --host and --port arguments are not applicable.
    """
    from mcp.server import StdioServerParameters
    from mcp.server.stdio import stdio_server

    server = Server("semantic-forge")

    # Register handlers
    await register_handlers(server)

    # Run the server using stdio transport
    print("Starting semantic-forge MCP server (stdio transport)")

    async with stdio_server() as (read, write):
        await server.run(read, write, StdioServerParameters(command="python", args=["-m", "semantic_forge"]))


def list_concepts() -> None:
    """List all available behavioral concepts."""
    print("Available Behavioral Concepts:")
    print("=" * 60)
    for concept in CONCEPT_LIBRARY:
        print(f"\n{concept.name} (id: {concept.id})")
        print(f"  Core: {concept.core_statement}")
        print(f"  Addresses: {', '.join(concept.addresses)}")
        print(f"  Notes: {concept.notes}")


def show_concept(concept_id: str) -> None:
    """Show details for a specific concept."""
    concept = get_concept_by_id(concept_id)
    if not concept:
        print(f"Concept not found: {concept_id}")
        print(f"Available concepts: {', '.join(c.id for c in CONCEPT_LIBRARY)}")
        return

    print(f"Concept: {concept.name}")
    print("=" * 60)
    print(f"\nID: {concept.id}")
    print(f"\nCore Statement:\n  {concept.core_statement}")
    print(f"\nAddresses:")
    for address in concept.addresses:
        print(f"  - {address}")
    print(f"\nNotes: {concept.notes}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="semantic-forge",
        description="Behavioral Fine-Tuning Data Generation Toolkit",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run as MCP server (stdio transport only)",
    )
    parser.add_argument(
        "--list-concepts",
        action="store_true",
        help="List all available behavioral concepts",
    )
    parser.add_argument(
        "--concept",
        type=str,
        help="Show details for a specific concept by ID",
    )

    args = parser.parse_args()

    if args.list_concepts:
        list_concepts()
        return 0

    if args.concept:
        show_concept(args.concept)
        return 0

    if args.server:
        asyncio.run(run_server())
        return 0

    # Default: show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
