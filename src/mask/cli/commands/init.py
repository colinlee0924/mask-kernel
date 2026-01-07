"""MASK init command - Create new agent project.

Usage:
    mask init my-agent
    mask init my-agent --no-stateless
"""

from pathlib import Path

import typer

from mask.cli.template_engine import TemplateEngine


def init_command(
    project_name: str = typer.Argument(..., help="Name of the project"),
    output_dir: Path = typer.Option(
        Path("."),
        "--output", "-o",
        help="Output directory",
    ),
    stateless: bool = typer.Option(
        True,
        "--stateless/--no-stateless",
        help="Whether agent is stateless by default",
    ),
) -> None:
    """Initialize a new MASK agent project.

    Creates a new directory with the project structure and configuration.
    Uses the native LangChain create_agent API with MASK SkillMiddleware
    for Progressive Disclosure.
    """
    # Handle path input: extract just the final component as project name
    # e.g., "../uat/my-agent" -> project_name = "my-agent", project_dir = "../uat/my-agent"
    project_path = Path(project_name)
    actual_project_name = project_path.name  # Get just the last component

    # Normalize project name
    project_name_normalized = actual_project_name.lower().replace("_", "-")
    module_name = project_name_normalized.replace("-", "_")

    # Determine project directory
    if len(project_path.parts) > 1:
        # User provided a path, use it directly
        project_dir = project_path
    else:
        # User provided just a name, put it in output_dir
        project_dir = output_dir / project_name_normalized

    if project_dir.exists():
        typer.echo(f"Error: Directory '{project_dir}' already exists", err=True)
        raise typer.Exit(1)

    typer.echo(f"Creating MASK agent project: {project_name_normalized}")

    # Template context
    context = {
        "project_name": project_name_normalized,
        "module_name": module_name,
        "stateless": stateless,
    }

    # Render project from template
    try:
        engine = TemplateEngine(template_name="default")
        engine.render_project(project_dir, context)
    except Exception as e:
        typer.echo(f"Error rendering template: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nProject created: {project_dir}")
    typer.echo("\nNext steps:")
    typer.echo(f"  cd {project_dir}")
    typer.echo("  pip install -e .")
    typer.echo(f"  # Edit src/{module_name}/prompts/system.md")
    typer.echo(f"  # Add skills to src/{module_name}/skills/")
    typer.echo(f"  python -m {module_name}.main  # Start A2A server")
