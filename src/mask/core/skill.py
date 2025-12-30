"""Core skill definitions for MASK.

This module defines the base classes and data structures for skills:
- SkillMetadata: Metadata for a skill (name, description, version, etc.)
- BaseSkill: Abstract base class for all skills
- MarkdownSkill: Skill loaded from SKILL.md files
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


# Skill specification constraints (per Anthropic Agent Skills spec)
MAX_SKILL_NAME_LENGTH = 64
MAX_SKILL_DESCRIPTION_LENGTH = 1024


@dataclass
class SkillMetadata:
    """Metadata for a skill.

    Follows the Anthropic Agent Skills specification with additional fields
    for MASK-specific functionality.

    Attributes:
        name: Skill identifier (max 64 chars, lowercase alphanumeric and hyphens).
        description: What the skill does (max 1024 chars).
        version: Semantic version string.
        tags: Optional list of tags for categorization.
        source: Where the skill was loaded from ('local', 'user', 'project').
        path: Path to the SKILL.md file if applicable.
        enabled: Whether the skill is enabled.
        allowed_tools: Optional space-delimited list of pre-approved tools.
        license: Optional license name or reference.
        compatibility: Optional environment requirements.
    """

    name: str
    description: str
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    source: str = "local"
    path: Optional[str] = None
    enabled: bool = True
    allowed_tools: Optional[str] = None
    license: Optional[str] = None
    compatibility: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        if len(self.name) > MAX_SKILL_NAME_LENGTH:
            from mask.core.exceptions import SkillMetadataError

            raise SkillMetadataError(
                f"Skill name exceeds {MAX_SKILL_NAME_LENGTH} characters: {self.name}"
            )

        if len(self.description) > MAX_SKILL_DESCRIPTION_LENGTH:
            # Truncate description with warning instead of raising error
            self.description = self.description[:MAX_SKILL_DESCRIPTION_LENGTH]


class BaseSkill(ABC):
    """Abstract base class for all skills.

    A skill represents a capability that can be loaded on-demand by an agent.
    Skills follow the Progressive Disclosure pattern:
    1. Initially only the loader tool is visible
    2. When the agent calls the loader, the skill's instructions are returned
    3. After activation, the skill's capability tools become available

    Subclasses must implement:
    - metadata property: Returns skill metadata
    - get_tools(): Returns capability tools provided by this skill
    - get_loader_tool(): Returns the loader tool for Progressive Disclosure

    Optionally override:
    - get_instructions(): Returns usage instructions for the agent
    """

    @property
    @abstractmethod
    def metadata(self) -> SkillMetadata:
        """Return the skill's metadata.

        Returns:
            SkillMetadata instance with skill information.
        """
        pass

    @abstractmethod
    def get_tools(self) -> List["BaseTool"]:
        """Return the capability tools provided by this skill.

        These tools are made available to the agent after the skill is activated.

        Returns:
            List of LangChain BaseTool instances.
        """
        pass

    @abstractmethod
    def get_loader_tool(self) -> "BaseTool":
        """Return the loader tool for Progressive Disclosure.

        The loader tool is always visible and allows the agent to "activate"
        this skill. When called, it returns the skill's instructions and
        marks the skill as loaded.

        Returns:
            LangChain BaseTool that activates this skill.
        """
        pass

    def get_instructions(self) -> str:
        """Return usage instructions for the agent.

        Override this method to provide detailed instructions on how to use
        this skill. The instructions are returned when the loader tool is called.

        Returns:
            Markdown-formatted instructions string.
        """
        return f"# {self.metadata.name}\n\n{self.metadata.description}"


class MarkdownSkill(BaseSkill):
    """A skill loaded from a SKILL.md file.

    Markdown skills follow the Anthropic Agent Skills specification:
    - YAML frontmatter with name and description (required)
    - Markdown body with instructions for the agent
    - No custom tools (only loader tool is provided)

    This is useful for skills that provide instructions/prompts without
    custom functionality.
    """

    def __init__(
        self,
        metadata: SkillMetadata,
        instructions: str,
        skill_dir: Optional[Path] = None,
    ) -> None:
        """Initialize MarkdownSkill.

        Args:
            metadata: Skill metadata parsed from YAML frontmatter.
            instructions: Markdown instructions from SKILL.md body.
            skill_dir: Optional path to skill directory for resolving relative paths.
        """
        self._metadata = metadata
        self._instructions = instructions
        self._skill_dir = skill_dir

    @property
    def metadata(self) -> SkillMetadata:
        """Return the skill's metadata."""
        return self._metadata

    def get_tools(self) -> List["BaseTool"]:
        """Return empty list - MarkdownSkill has no capability tools."""
        return []

    def get_loader_tool(self) -> "BaseTool":
        """Return the loader tool for this skill.

        The loader tool returns the skill's instructions when called.
        """
        from langchain_core.tools import tool

        skill_name = self.metadata.name
        instructions = self._instructions

        @tool(name=f"use_{skill_name.replace('-', '_')}")
        def loader() -> str:
            """Load and activate this skill.

            Returns the skill's instructions and usage guidelines.
            """
            return instructions

        # Update tool description
        loader.description = f"Activate the {skill_name} skill. {self.metadata.description}"

        return loader

    def get_instructions(self) -> str:
        """Return the full instructions from SKILL.md."""
        return self._instructions

    @property
    def skill_dir(self) -> Optional[Path]:
        """Return the skill directory path."""
        return self._skill_dir
