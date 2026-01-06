"""MASK run command - Run agent interactively or as server.

Usage:
    mask run --interactive
    mask run --server --port 10001
    mask run -i --agent mybot --model claude-sonnet-4-20250514
"""

import asyncio
from pathlib import Path
from typing import Optional

import typer

from mask.cli.config import get_settings, validate_agent_name


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
    agent: str = typer.Option(
        "agent",
        "--agent", "-a",
        help="Agent name to use",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Model name (e.g., claude-sonnet-4-20250514, gpt-4o)",
    ),
    tier: str = typer.Option(
        "thinking",
        "--tier", "-t",
        help="Model tier (fast, thinking, pro) - ignored if --model is set",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="LLM provider override (anthropic, openai, google)",
    ),
    config_dir: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Configuration directory (defaults to agent's config)",
    ),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve",
        help="Auto-approve tool usage (skip confirmations)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output",
    ),
) -> None:
    """Run a MASK agent.

    Either run interactively or start an A2A server.

    Examples:
        mask run -i                           # Interactive with defaults
        mask run -i --agent mybot             # Use specific agent config
        mask run -i --model gpt-4o            # Use specific model
        mask run -s --port 10001              # Start A2A server
    """
    if not interactive and not server:
        typer.echo("Specify --interactive or --server mode")
        raise typer.Exit(1)

    if interactive and server:
        typer.echo("Cannot use both --interactive and --server")
        raise typer.Exit(1)

    # Validate agent name
    if not validate_agent_name(agent):
        typer.echo(f"Invalid agent name: {agent}")
        typer.echo("Agent names must start with a letter and contain only alphanumeric, hyphens, underscores")
        raise typer.Exit(1)

    # Get settings
    settings = get_settings(agent_name=agent)

    # Ensure agent directory exists
    agent_dir = settings.ensure_agent_dir(agent)

    if verbose:
        typer.echo(f"Agent directory: {agent_dir}")
        typer.echo(f"Settings: {settings.to_dict()}")

    # Import here to avoid loading dependencies for help
    from mask.agent import create_mask_agent
    from mask.models import ModelTier

    # Determine model configuration
    if model:
        # Use specific model name
        llm = settings.create_model(model_name=model, provider=provider)
        model_info = model
    else:
        # Use tier-based selection
        tier_map = {
            "fast": ModelTier.FAST,
            "thinking": ModelTier.THINKING,
            "pro": ModelTier.PRO,
        }
        model_tier = tier_map.get(tier.lower(), ModelTier.THINKING)
        llm = settings.create_model(tier=tier, provider=provider)
        model_info = f"tier={tier}"

    # Determine config directory
    if config_dir is None:
        config_dir = agent_dir

    # Load agent-specific system prompt if exists
    agent_md_path = agent_dir / "agent.md"
    system_prompt = None
    if agent_md_path.exists():
        system_prompt = agent_md_path.read_text(encoding="utf-8")
        if verbose:
            typer.echo(f"Loaded agent config from: {agent_md_path}")

    # Get skills directories
    user_skills_dir = settings.get_agent_skills_dir(agent)
    project_skills_dir = settings.get_project_skills_dir()

    # Determine skills directory to use
    skills_dir = None
    if user_skills_dir.exists():
        skills_dir = user_skills_dir
    elif project_skills_dir and project_skills_dir.exists():
        skills_dir = project_skills_dir

    if verbose and skills_dir:
        typer.echo(f"Skills directory: {skills_dir}")

    # Create agent
    typer.echo(f"Creating agent '{agent}' with {model_info}...")

    agent_instance = create_mask_agent(
        model=llm,
        config_dir=str(config_dir),
        system_prompt=system_prompt,
        skills_dir=skills_dir,
        enable_file_access=True,
    )

    if interactive:
        asyncio.run(_run_interactive(agent_instance, auto_approve=auto_approve))
    else:
        _run_server(agent_instance, agent, port)


async def _run_interactive(agent, auto_approve: bool = False) -> None:
    """Run agent in interactive mode."""
    import typer

    typer.echo("Agent ready. Type 'quit' to exit, '/help' for commands.\n")

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nGoodbye!")
            break

        # Handle slash commands
        if user_input.startswith("/"):
            cmd = user_input.lower().strip()
            if cmd in ("/quit", "/exit", "/q"):
                typer.echo("Goodbye!")
                break
            elif cmd == "/help":
                typer.echo("\nCommands:")
                typer.echo("  /quit, /exit, /q - Exit the session")
                typer.echo("  /help            - Show this help")
                typer.echo("  /clear           - Clear conversation (not implemented)")
                typer.echo("")
                continue
            else:
                typer.echo(f"Unknown command: {cmd}")
                continue

        # Handle regular quit
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


def _run_server(agent, agent_name: str, port: int) -> None:
    """Run agent as A2A server."""
    import typer

    from mask.a2a import MaskA2AServer

    typer.echo(f"Starting A2A server for '{agent_name}' on port {port}...")

    server = MaskA2AServer(
        agent=agent,
        name=agent_name,
        description=f"MASK agent: {agent_name}",
    )

    server.run(port=port)
