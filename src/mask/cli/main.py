"""MASK CLI main entry point.

This module provides the CLI for MASK Kernel.
"""

import typer

from mask.cli.commands.init import init_command
from mask.cli.commands.run import run_command

app = typer.Typer(
    name="mask",
    help="MASK Kernel CLI - Multi-Agent Skill Kit",
    add_completion=False,
)

# Register commands
app.command(name="init")(init_command)
app.command(name="run")(run_command)


@app.callback()
def main_callback() -> None:
    """MASK Kernel CLI - Multi-Agent Skill Kit."""
    pass


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
