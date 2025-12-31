"""MASK run command - Run agent interactively or as server.

Usage:
    mask run --interactive
    mask run --server --port 10001
"""

import asyncio
from pathlib import Path
from typing import Optional

import typer


def run_command(
    interactive: bool = typer.Option(
        False,
        "--interactive", "-i",
        help="Run agent in interactive mode",
    ),
    server: bool = typer.Option(
        False,
        "--server", "-s",
        help="Run as A2A server",
    ),
    port: int = typer.Option(
        10001,
        "--port", "-p",
        help="Server port (for --server mode)",
    ),
    config_dir: Path = typer.Option(
        Path("config"),
        "--config", "-c",
        help="Configuration directory",
    ),
    tier: str = typer.Option(
        "thinking",
        "--tier", "-t",
        help="Model tier (fast, thinking, pro)",
    ),
) -> None:
    """Run a MASK agent.

    Either run interactively or start an A2A server.
    """
    if not interactive and not server:
        typer.echo("Specify --interactive or --server mode")
        raise typer.Exit(1)

    if interactive and server:
        typer.echo("Cannot use both --interactive and --server")
        raise typer.Exit(1)

    # Import here to avoid loading dependencies for help
    from mask.agent import create_mask_agent
    from mask.models import ModelTier

    # Parse tier
    tier_map = {
        "fast": ModelTier.FAST,
        "thinking": ModelTier.THINKING,
        "pro": ModelTier.PRO,
    }
    model_tier = tier_map.get(tier.lower(), ModelTier.THINKING)

    # Create agent
    typer.echo(f"Creating agent with tier={tier}...")
    agent = create_mask_agent(
        config_dir=str(config_dir),
        tier=model_tier,
    )

    if interactive:
        asyncio.run(_run_interactive(agent))
    else:
        _run_server(agent, port)


async def _run_interactive(agent) -> None:
    """Run agent in interactive mode."""
    import typer

    typer.echo("Agent ready. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            typer.echo("Goodbye!")
            break

        if not user_input.strip():
            continue

        try:
            response = await agent.invoke(user_input)
            typer.echo(f"Agent: {response}\n")
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)


def _run_server(agent, port: int) -> None:
    """Run agent as A2A server."""
    import typer

    from mask.a2a import MaskA2AServer

    typer.echo(f"Starting A2A server on port {port}...")

    server = MaskA2AServer(
        agent=agent,
        name="mask-agent",
        description="MASK agent server",
    )

    server.run(port=port)
