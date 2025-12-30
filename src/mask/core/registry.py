"""Skill Registry for MASK.

This module implements the SkillRegistry, which manages skill registration,
discovery, and Progressive Disclosure.

Progressive Disclosure Pattern:
1. Initially, only loader tools are visible to the agent
2. When a loader tool is invoked, the skill becomes "active"
3. Active skills expose their capability tools to the agent

Example:
    registry = SkillRegistry()
    registry.discover_from_directory(Path("skills"))

    # Get only loader tools (for initial agent state)
    loader_tools = registry.get_all_loader_tools()

    # After skill activation, get all relevant tools
    active_tools = registry.get_tools_for_active_skills(["pdf-processing"])
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional

from langchain_core.tools import BaseTool

from mask.core.exceptions import SkillAlreadyRegisteredError, SkillNotFoundError
from mask.core.skill import BaseSkill, MarkdownSkill

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Registry for managing skills and Progressive Disclosure.

    The registry maintains a collection of skills and provides methods for:
    - Registering individual skills
    - Discovering skills from directories
    - Getting loader tools (always visible)
    - Getting capability tools for active skills

    Attributes:
        _skills: Dictionary mapping skill names to BaseSkill instances.
    """

    def __init__(self) -> None:
        """Initialize an empty skill registry."""
        self._skills: dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill) -> None:
        """Register a skill in the registry.

        Args:
            skill: The skill to register.

        Raises:
            SkillAlreadyRegisteredError: If a skill with the same name
                is already registered.
        """
        name = skill.metadata.name
        if name in self._skills:
            raise SkillAlreadyRegisteredError(name)

        self._skills[name] = skill
        logger.debug("Registered skill: %s", name)

    def unregister(self, skill_name: str) -> None:
        """Unregister a skill from the registry.

        Args:
            skill_name: Name of the skill to unregister.

        Raises:
            SkillNotFoundError: If the skill is not registered.
        """
        if skill_name not in self._skills:
            raise SkillNotFoundError(skill_name)

        del self._skills[skill_name]
        logger.debug("Unregistered skill: %s", skill_name)

    def get(self, skill_name: str) -> BaseSkill:
        """Get a skill by name.

        Args:
            skill_name: Name of the skill to retrieve.

        Returns:
            The requested BaseSkill instance.

        Raises:
            SkillNotFoundError: If the skill is not registered.
        """
        if skill_name not in self._skills:
            raise SkillNotFoundError(skill_name)
        return self._skills[skill_name]

    def has(self, skill_name: str) -> bool:
        """Check if a skill is registered.

        Args:
            skill_name: Name of the skill to check.

        Returns:
            True if the skill is registered.
        """
        return skill_name in self._skills

    def list_skills(
        self,
        filter_fn: Optional[Callable[[BaseSkill], bool]] = None,
    ) -> List[str]:
        """List registered skill names.

        Args:
            filter_fn: Optional function to filter skills.
                Returns True for skills to include.

        Returns:
            List of skill names matching the filter.
        """
        if filter_fn is None:
            return list(self._skills.keys())

        return [
            name for name, skill in self._skills.items()
            if filter_fn(skill)
        ]

    def get_all_skills(self) -> List[BaseSkill]:
        """Get all registered skills.

        Returns:
            List of all BaseSkill instances.
        """
        return list(self._skills.values())

    # =========================================================================
    # Progressive Disclosure Methods
    # =========================================================================

    def get_all_loader_tools(self) -> List[BaseTool]:
        """Get loader tools for all registered skills.

        Loader tools are always visible to the agent and serve as the
        entry point for skill activation.

        Returns:
            List of loader tools from all registered skills.
        """
        tools: List[BaseTool] = []
        for skill in self._skills.values():
            if skill.metadata.enabled:
                loader_tool = skill.get_loader_tool()
                tools.append(loader_tool)
        return tools

    def get_tools_for_active_skills(
        self,
        active_skills: List[str],
    ) -> List[BaseTool]:
        """Get tools for active skills plus all loader tools.

        This implements the Progressive Disclosure pattern:
        - Loader tools are always included (for activating more skills)
        - Capability tools are only included for active skills

        Args:
            active_skills: List of skill names that have been activated.

        Returns:
            List of tools (loader tools + capability tools for active skills).
        """
        tools: List[BaseTool] = []

        for name, skill in self._skills.items():
            if not skill.metadata.enabled:
                continue

            # Always include loader tool
            tools.append(skill.get_loader_tool())

            # Include capability tools only for active skills
            if name in active_skills:
                capability_tools = skill.get_tools()
                tools.extend(capability_tools)

        return tools

    def get_skill_instructions(self, skill_name: str) -> str:
        """Get instructions for a specific skill.

        Args:
            skill_name: Name of the skill.

        Returns:
            Instructions string for the skill.

        Raises:
            SkillNotFoundError: If the skill is not registered.
        """
        skill = self.get(skill_name)
        return skill.get_instructions()

    def get_active_skill_instructions(
        self,
        active_skills: List[str],
    ) -> str:
        """Get combined instructions for all active skills.

        Args:
            active_skills: List of active skill names.

        Returns:
            Combined instructions string.
        """
        instructions_parts: List[str] = []

        for skill_name in active_skills:
            if skill_name in self._skills:
                skill = self._skills[skill_name]
                if skill.metadata.enabled:
                    instructions = skill.get_instructions()
                    if instructions:
                        instructions_parts.append(
                            f"## {skill.metadata.name}\n\n{instructions}"
                        )

        if not instructions_parts:
            return ""

        return "\n\n---\n\n".join(instructions_parts)

    # =========================================================================
    # Discovery Methods
    # =========================================================================

    def discover_from_directory(
        self,
        skills_dir: Path,
        source: str = "local",
    ) -> int:
        """Discover and register skills from a directory.

        Supports both SKILL.md (markdown) and skill.py (Python) skills.
        Python skills take precedence if both exist in the same directory.

        Args:
            skills_dir: Path to directory containing skill subdirectories.
            source: Source identifier ('local', 'user', 'project').

        Returns:
            Number of skills successfully registered.
        """
        from mask.loader.python_loader import load_python_skill
        from mask.loader.skill_md_loader import load_markdown_skill

        skills_dir = Path(skills_dir).expanduser()

        if not skills_dir.exists():
            logger.debug("Skills directory does not exist: %s", skills_dir)
            return 0

        count = 0

        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            # Skip hidden directories
            if skill_dir.name.startswith("."):
                continue

            skill: Optional[BaseSkill] = None

            # Try Python skill first (takes precedence)
            if (skill_dir / "skill.py").exists():
                try:
                    skill = load_python_skill(skill_dir, source)
                except Exception as e:
                    logger.warning(
                        "Failed to load Python skill from %s: %s",
                        skill_dir, e,
                    )

            # Fall back to markdown skill
            if skill is None and (skill_dir / "SKILL.md").exists():
                try:
                    skill = load_markdown_skill(skill_dir, source)
                except Exception as e:
                    logger.warning(
                        "Failed to load markdown skill from %s: %s",
                        skill_dir, e,
                    )

            if skill is not None:
                try:
                    self.register(skill)
                    count += 1
                except SkillAlreadyRegisteredError:
                    logger.warning(
                        "Skill '%s' already registered, skipping",
                        skill.metadata.name,
                    )

        logger.info("Discovered %d skills from %s", count, skills_dir)
        return count

    def discover_from_multiple_directories(
        self,
        directories: List[tuple[Path, str]],
    ) -> int:
        """Discover skills from multiple directories with different sources.

        Args:
            directories: List of (path, source) tuples.

        Returns:
            Total number of skills registered.
        """
        total = 0
        for path, source in directories:
            total += self.discover_from_directory(path, source)
        return total

    # =========================================================================
    # Metadata Methods
    # =========================================================================

    def get_skills_summary(self) -> List[dict]:
        """Get summary information for all registered skills.

        Useful for displaying available skills to users or agents.

        Returns:
            List of dictionaries with skill metadata.
        """
        return [
            {
                "name": skill.metadata.name,
                "description": skill.metadata.description,
                "version": skill.metadata.version,
                "tags": skill.metadata.tags,
                "source": skill.metadata.source,
                "enabled": skill.metadata.enabled,
                "type": "python" if not isinstance(skill, MarkdownSkill) else "markdown",
            }
            for skill in self._skills.values()
        ]

    def __len__(self) -> int:
        """Return the number of registered skills."""
        return len(self._skills)

    def __contains__(self, skill_name: str) -> bool:
        """Check if a skill is registered."""
        return skill_name in self._skills

    def __iter__(self):
        """Iterate over skill names."""
        return iter(self._skills)
