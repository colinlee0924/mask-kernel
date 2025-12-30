"""Loader for SKILL.md files.

This module implements parsing and loading of skills from SKILL.md files
following the Anthropic Agent Skills specification.

SKILL.md files have the following structure:
```markdown
---
name: skill-name
description: What the skill does
version: 1.0.0 (optional)
tags: [tag1, tag2] (optional)
license: MIT (optional)
allowed-tools: tool1 tool2 (optional)
---

# Skill Instructions

Detailed instructions for the agent...
```
"""

import logging
import re
from pathlib import Path
from typing import Optional

import yaml

from mask.core.exceptions import SkillLoadError, SkillMetadataError
from mask.core.skill import (
    MAX_SKILL_DESCRIPTION_LENGTH,
    MAX_SKILL_NAME_LENGTH,
    MarkdownSkill,
    SkillMetadata,
)

logger = logging.getLogger(__name__)

# Maximum size for SKILL.md files (10MB)
MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024

# YAML frontmatter pattern
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _is_safe_path(path: Path, base_dir: Path) -> bool:
    """Check if a path is safely contained within base_dir.

    Prevents directory traversal attacks via symlinks or path manipulation.

    Args:
        path: The path to validate.
        base_dir: The base directory that should contain the path.

    Returns:
        True if the path is safely within base_dir.
    """
    try:
        resolved_path = path.resolve()
        resolved_base = base_dir.resolve()
        resolved_path.relative_to(resolved_base)
        return True
    except ValueError:
        return False
    except (OSError, RuntimeError):
        return False


def _validate_skill_name(name: str, directory_name: str) -> tuple[bool, str]:
    """Validate skill name per Agent Skills spec.

    Requirements:
    - Max 64 characters
    - Lowercase alphanumeric and hyphens only
    - Cannot start or end with hyphen
    - No consecutive hyphens
    - Should match parent directory name (warning only)

    Args:
        name: The skill name from YAML frontmatter.
        directory_name: The parent directory name.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not name:
        return False, "name is required"

    if len(name) > MAX_SKILL_NAME_LENGTH:
        return False, f"name exceeds {MAX_SKILL_NAME_LENGTH} characters"

    # Pattern: lowercase alphanumeric, single hyphens between segments
    if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", name):
        return False, "name must be lowercase alphanumeric with single hyphens only"

    if name != directory_name:
        # Warning only, not an error
        logger.warning(
            "Skill name '%s' does not match directory name '%s'",
            name,
            directory_name,
        )

    return True, ""


def parse_skill_md(
    skill_md_path: Path,
    source: str = "local",
) -> tuple[Optional[SkillMetadata], Optional[str]]:
    """Parse a SKILL.md file into metadata and instructions.

    Args:
        skill_md_path: Path to the SKILL.md file.
        source: Source of the skill ('local', 'user', 'project').

    Returns:
        Tuple of (SkillMetadata, instructions_string) or (None, None) on error.

    Raises:
        SkillLoadError: If the file cannot be read or parsed.
    """
    try:
        # Security: Check file size
        file_size = skill_md_path.stat().st_size
        if file_size > MAX_SKILL_FILE_SIZE:
            raise SkillLoadError(
                str(skill_md_path),
                f"file too large ({file_size} bytes, max {MAX_SKILL_FILE_SIZE})",
            )

        content = skill_md_path.read_text(encoding="utf-8")

        # Match YAML frontmatter
        match = FRONTMATTER_PATTERN.match(content)
        if not match:
            raise SkillLoadError(
                str(skill_md_path),
                "no valid YAML frontmatter found",
            )

        frontmatter_str = match.group(1)
        instructions = content[match.end() :].strip()

        # Parse YAML
        try:
            frontmatter_data = yaml.safe_load(frontmatter_str)
        except yaml.YAMLError as e:
            raise SkillLoadError(str(skill_md_path), f"invalid YAML: {e}") from e

        if not isinstance(frontmatter_data, dict):
            raise SkillLoadError(
                str(skill_md_path),
                "frontmatter must be a YAML mapping",
            )

        # Validate required fields
        name = frontmatter_data.get("name")
        description = frontmatter_data.get("description")

        if not name or not description:
            raise SkillLoadError(
                str(skill_md_path),
                "missing required 'name' or 'description' field",
            )

        # Validate name format
        directory_name = skill_md_path.parent.name
        is_valid, error = _validate_skill_name(str(name), directory_name)
        if not is_valid:
            raise SkillMetadataError(error)

        # Validate and possibly truncate description
        description_str = str(description)
        if len(description_str) > MAX_SKILL_DESCRIPTION_LENGTH:
            logger.warning(
                "Description exceeds %d chars in %s, truncating",
                MAX_SKILL_DESCRIPTION_LENGTH,
                skill_md_path,
            )
            description_str = description_str[:MAX_SKILL_DESCRIPTION_LENGTH]

        # Build metadata
        metadata = SkillMetadata(
            name=str(name),
            description=description_str,
            version=str(frontmatter_data.get("version", "1.0.0")),
            tags=frontmatter_data.get("tags", []),
            source=source,
            path=str(skill_md_path),
            license=frontmatter_data.get("license"),
            compatibility=frontmatter_data.get("compatibility"),
            allowed_tools=frontmatter_data.get("allowed-tools"),
        )

        return metadata, instructions

    except (OSError, UnicodeDecodeError) as e:
        raise SkillLoadError(str(skill_md_path), str(e)) from e


def load_markdown_skill(
    skill_dir: Path,
    source: str = "local",
) -> Optional[MarkdownSkill]:
    """Load a MarkdownSkill from a directory containing SKILL.md.

    Args:
        skill_dir: Path to skill directory.
        source: Source of the skill ('local', 'user', 'project').

    Returns:
        MarkdownSkill instance or None if loading fails.
    """
    skill_md_path = skill_dir / "SKILL.md"

    if not skill_md_path.exists():
        logger.debug("No SKILL.md found in %s", skill_dir)
        return None

    try:
        metadata, instructions = parse_skill_md(skill_md_path, source)
        if metadata is None or instructions is None:
            return None

        return MarkdownSkill(
            metadata=metadata,
            instructions=instructions,
            skill_dir=skill_dir,
        )

    except (SkillLoadError, SkillMetadataError) as e:
        logger.warning("Failed to load skill from %s: %s", skill_dir, e)
        return None


def discover_markdown_skills(
    skills_dir: Path,
    source: str = "local",
) -> list[MarkdownSkill]:
    """Discover and load all MarkdownSkills from a directory.

    Scans the skills directory for subdirectories containing SKILL.md files.

    Args:
        skills_dir: Path to the skills directory.
        source: Source of the skills ('local', 'user', 'project').

    Returns:
        List of loaded MarkdownSkill instances.
    """
    skills_dir = skills_dir.expanduser()

    if not skills_dir.exists():
        logger.debug("Skills directory does not exist: %s", skills_dir)
        return []

    try:
        resolved_base = skills_dir.resolve()
    except (OSError, RuntimeError):
        logger.warning("Cannot resolve skills directory: %s", skills_dir)
        return []

    skills: list[MarkdownSkill] = []

    for skill_dir in skills_dir.iterdir():
        # Security: Check path is safe
        if not _is_safe_path(skill_dir, resolved_base):
            logger.warning("Skipping unsafe path: %s", skill_dir)
            continue

        if not skill_dir.is_dir():
            continue

        skill = load_markdown_skill(skill_dir, source)
        if skill is not None:
            skills.append(skill)

    return skills
