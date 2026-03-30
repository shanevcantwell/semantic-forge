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


async def run_server(host: str = "localhost", port: int = 8080) -> None:
    """Run the MCP server."""
    server = Server("semantic-forge")

    # Register tools
    for tool in get_all_tools():
        # TODO: Register tool handlers
        pass

    # Register handlers
    await register_handlers(server)

    # Run the server
    print(f"Starting semantic-forge MCP server on {host}:{port}")

    # TODO: Implement actual server run
    # await server.run_stdio()


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
        help="Run as MCP server",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to (default: 8080)",
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
        asyncio.run(run_server(args.host, args.port))
        return 0

    # Default: show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
