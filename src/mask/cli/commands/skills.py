"""MASK skills command - Manage agent skills.

Usage:
    mask skills list
    mask skills create my-skill
    mask skills info web-research
"""

import re
from pathlib import Path
from typing import Optional

import typer

from mask.cli.config import get_settings, validate_agent_name

# Skill name validation pattern (Agent Skills spec)
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_SKILL_NAME_LENGTH = 64

# SKILL.md template
SKILL_TEMPLATE = '''---
name: {name}
description: {description}
version: 1.0.0
tags: [{tags}]
---

# {title}

## When to Use

- [Describe when this skill should be activated]
- [List relevant user intents or task types]

## Instructions

1. [Step-by-step instructions for the agent]
2. [Best practices for using this skill]
3. [Common patterns and workflows]

## Examples

### Example 1: [Scenario Name]

User: "[Example user request]"

Agent approach:
1. [How the agent should handle this]
2. [Expected actions or responses]

## Supporting Files

This skill directory can include supporting files:
- `scripts/` - Python scripts for automation
- `references/` - Additional reference documentation
- `assets/` - Templates, configurations, data files

Use `read_file` to access these resources when needed.
'''


def validate_skill_name(name: str) -> tuple[bool, str]:
    """Validate skill name per Agent Skills spec.

    Requirements:
    - Max 64 characters
    - Lowercase alphanumeric and hyphens only
    - Cannot start or end with hyphen
    - No consecutive hyphens

    Args:
        name: The skill name to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not name:
        return False, "Skill name is required"

    if len(name) > MAX_SKILL_NAME_LENGTH:
        return False, f"Skill name exceeds {MAX_SKILL_NAME_LENGTH} characters"

    if not SKILL_NAME_PATTERN.match(name):
        return False, "Skill name must be lowercase alphanumeric with single hyphens only"

    return True, ""


def list_skills(
    agent: str = typer.Option(
        "agent",
        "--agent", "-a",
        help="Agent name",
    ),
    project: bool = typer.Option(
        False,
        "--project", "-p",
        help="List only project skills",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed information",
    ),
) -> None:
    """List available skills.

    Shows skills from both user-level and project-level directories.
    """
    settings = get_settings(agent_name=agent)

    user_skills_dir = settings.get_agent_skills_dir(agent)
    project_skills_dir = settings.get_project_skills_dir()

    typer.echo(f"Skills for agent '{agent}':\n")

    # List user skills
    if not project:
        typer.echo(f"**User Skills** ({user_skills_dir}):")
        _list_skills_in_dir(user_skills_dir, verbose)
        typer.echo("")

    # List project skills
    if project_skills_dir:
        typer.echo(f"**Project Skills** ({project_skills_dir}):")
        _list_skills_in_dir(project_skills_dir, verbose)
    elif not project:
        typer.echo("(No project detected - not in a git repository)")


def _list_skills_in_dir(skills_dir: Path, verbose: bool) -> None:
    """List skills in a directory."""
    if not skills_dir.exists():
        typer.echo("  (No skills directory)")
        return

    skills_found = False
    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue
        if skill_dir.name.startswith("."):
            continue

        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue

        skills_found = True

        # Parse skill metadata
        metadata = _parse_skill_metadata(skill_md)
        name = metadata.get("name", skill_dir.name)
        description = metadata.get("description", "No description")

        typer.echo(f"  - **{name}**: {description}")

        if verbose:
            typer.echo(f"    Path: {skill_md}")
            # List subdirectories
            subdirs = []
            for subdir in ["scripts", "references", "assets"]:
                if (skill_dir / subdir).exists():
                    subdirs.append(subdir)
            if subdirs:
                typer.echo(f"    Resources: {', '.join(subdirs)}/")

    if not skills_found:
        typer.echo("  (No skills found)")


def _parse_skill_metadata(skill_md: Path) -> dict:
    """Parse YAML frontmatter from SKILL.md."""
    try:
        import yaml

        content = skill_md.read_text(encoding="utf-8")

        # Match YAML frontmatter
        if not content.startswith("---"):
            return {}

        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}

        frontmatter = parts[1].strip()
        return yaml.safe_load(frontmatter) or {}

    except Exception:
        return {}


def create_skill(
    name: str = typer.Argument(
        ...,
        help="Skill name (lowercase, hyphens allowed)",
    ),
    agent: str = typer.Option(
        "agent",
        "--agent", "-a",
        help="Agent name",
    ),
    project: bool = typer.Option(
        False,
        "--project", "-p",
        help="Create as project skill",
    ),
    description: str = typer.Option(
        "A custom skill for specialized tasks",
        "--description", "-d",
        help="Skill description",
    ),
    tags: str = typer.Option(
        "custom",
        "--tags",
        help="Comma-separated tags",
    ),
) -> None:
    """Create a new skill from template.

    Creates a skill directory with SKILL.md and optional subdirectories.
    """
    # Validate skill name
    is_valid, error = validate_skill_name(name)
    if not is_valid:
        typer.echo(f"Invalid skill name: {error}")
        raise typer.Exit(1)

    settings = get_settings(agent_name=agent)

    # Determine target directory
    if project:
        project_skills_dir = settings.get_project_skills_dir()
        if project_skills_dir is None:
            typer.echo("Error: Not in a git repository. Cannot create project skill.")
            raise typer.Exit(1)
        skills_dir = project_skills_dir
    else:
        skills_dir = settings.get_agent_skills_dir(agent)

    # Ensure skills directory exists
    skills_dir.mkdir(parents=True, exist_ok=True)

    # Create skill directory
    skill_dir = skills_dir / name
    if skill_dir.exists():
        typer.echo(f"Error: Skill '{name}' already exists at {skill_dir}")
        raise typer.Exit(1)

    skill_dir.mkdir()

    # Create subdirectories
    (skill_dir / "scripts").mkdir()
    (skill_dir / "references").mkdir()
    (skill_dir / "assets").mkdir()

    # Create SKILL.md from template
    title = name.replace("-", " ").title()
    skill_md_content = SKILL_TEMPLATE.format(
        name=name,
        description=description,
        tags=tags,
        title=title,
    )

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(skill_md_content, encoding="utf-8")

    # Create placeholder files
    (skill_dir / "scripts" / ".gitkeep").touch()
    (skill_dir / "references" / ".gitkeep").touch()
    (skill_dir / "assets" / ".gitkeep").touch()

    typer.echo(f"Created skill '{name}' at {skill_dir}")
    typer.echo("")
    typer.echo("Directory structure:")
    typer.echo(f"  {skill_dir}/")
    typer.echo("  ├── SKILL.md        # Main instructions")
    typer.echo("  ├── scripts/        # Executable scripts")
    typer.echo("  ├── references/     # Documentation")
    typer.echo("  └── assets/         # Templates, configs")
    typer.echo("")
    typer.echo(f"Edit {skill_md} to customize your skill.")


def info_skill(
    name: str = typer.Argument(
        ...,
        help="Skill name to get info about",
    ),
    agent: str = typer.Option(
        "agent",
        "--agent", "-a",
        help="Agent name",
    ),
    project: bool = typer.Option(
        False,
        "--project", "-p",
        help="Look only in project skills",
    ),
) -> None:
    """Show detailed information about a skill."""
    settings = get_settings(agent_name=agent)

    # Find the skill
    skill_dir = None
    source = None

    if not project:
        user_skills_dir = settings.get_agent_skills_dir(agent)
        user_skill = user_skills_dir / name
        if user_skill.exists() and (user_skill / "SKILL.md").exists():
            skill_dir = user_skill
            source = "user"

    if skill_dir is None:
        project_skills_dir = settings.get_project_skills_dir()
        if project_skills_dir:
            project_skill = project_skills_dir / name
            if project_skill.exists() and (project_skill / "SKILL.md").exists():
                skill_dir = project_skill
                source = "project"

    if skill_dir is None:
        typer.echo(f"Skill '{name}' not found")
        raise typer.Exit(1)

    # Parse metadata
    skill_md = skill_dir / "SKILL.md"
    metadata = _parse_skill_metadata(skill_md)

    typer.echo(f"Skill: {metadata.get('name', name)}")
    typer.echo(f"Source: {source}")
    typer.echo(f"Path: {skill_dir}")
    typer.echo(f"Description: {metadata.get('description', 'No description')}")
    typer.echo(f"Version: {metadata.get('version', '1.0.0')}")
    typer.echo(f"Tags: {', '.join(metadata.get('tags', []))}")
    typer.echo("")

    # List contents
    typer.echo("Contents:")
    for item in sorted(skill_dir.iterdir()):
        if item.name.startswith("."):
            continue
        if item.is_dir():
            file_count = len([f for f in item.iterdir() if not f.name.startswith(".")])
            typer.echo(f"  {item.name}/ ({file_count} files)")
        else:
            size = item.stat().st_size
            typer.echo(f"  {item.name} ({size} bytes)")

    typer.echo("")
    typer.echo(f"Read instructions: read_file(\"{skill_md}\")")


# Create Typer app for skills subcommand
skills_app = typer.Typer(
    name="skills",
    help="Manage agent skills",
    add_completion=False,
)

skills_app.command(name="list")(list_skills)
skills_app.command(name="create")(create_skill)
skills_app.command(name="info")(info_skill)
