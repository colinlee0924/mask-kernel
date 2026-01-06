"""MASK CLI main entry point.

This module provides the CLI for MASK Kernel.

Commands:
    mask init <project>     - Initialize a new agent project
    mask run -i             - Run agent interactively
    mask run -s             - Run agent as A2A server
    mask skills list        - List available skills
    mask skills create      - Create a new skill
    mask skills info        - Show skill details
    mask agents list        - List configured agents
"""

import typer

from mask.cli.commands.init import init_command
from mask.cli.commands.run import run_command
from mask.cli.commands.skills import skills_app
from mask.cli.config import get_settings

app = typer.Typer(
    name="mask",
    help="MASK Kernel CLI - Multi-Agent Skill Kit",
    add_completion=False,
)

# Register commands
app.command(name="init")(init_command)
app.command(name="run")(run_command)

# Register skills subcommand
app.add_typer(skills_app, name="skills")


@app.command(name="agents")
def list_agents(
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed information",
    ),
) -> None:
    """List configured agents."""
    settings = get_settings()
    agents = settings.list_agents()

    if not agents:
        typer.echo("No agents configured.")
        typer.echo(f"\nCreate an agent with: mask run -i --agent <name>")
        typer.echo(f"Or initialize a project with: mask init <project>")
        return

    typer.echo("Configured agents:\n")
    for agent_name in agents:
        agent_dir = settings.get_agent_dir(agent_name)
        skills_dir = agent_dir / "skills"

        skill_count = 0
        if skills_dir.exists():
            skill_count = len([
                d for d in skills_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ])

        typer.echo(f"  - {agent_name}")
        if verbose:
            typer.echo(f"    Path: {agent_dir}")
            typer.echo(f"    Skills: {skill_count}")


@app.command(name="config")
def show_config(
    agent: str = typer.Option(
        "agent",
        "--agent", "-a",
        help="Agent name",
    ),
) -> None:
    """Show current configuration."""
    settings = get_settings(agent_name=agent)

    typer.echo("MASK Configuration:\n")
    typer.echo(f"Agent: {settings.agent_name}")
    typer.echo(f"User config directory: {settings.user_config_dir}")
    typer.echo(f"Project root: {settings.project_root or '(not in project)'}")
    typer.echo("")
    typer.echo("API Keys:")
    typer.echo(f"  Anthropic: {'✓' if settings.has_anthropic else '✗'}")
    typer.echo(f"  OpenAI: {'✓' if settings.has_openai else '✗'}")
    typer.echo(f"  Google: {'✓' if settings.has_google else '✗'}")
    typer.echo("")
    typer.echo(f"Default provider: {settings.default_provider}")

    # Show agent-specific paths
    agent_dir = settings.get_agent_dir(agent)
    if agent_dir.exists():
        typer.echo("")
        typer.echo(f"Agent '{agent}' paths:")
        typer.echo(f"  Directory: {agent_dir}")
        typer.echo(f"  Config: {agent_dir / 'agent.md'}")
        typer.echo(f"  Skills: {agent_dir / 'skills'}")


@app.callback()
def main_callback() -> None:
    """MASK Kernel CLI - Multi-Agent Skill Kit."""
    pass


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
