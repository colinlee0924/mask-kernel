"""Prompt loading utilities.

This module provides utilities for loading agent prompts from configuration
files, typically stored in config/prompts/ directory.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

from mask.core.exceptions import PromptNotFoundError

logger = logging.getLogger(__name__)


class PromptLoader:
    """Load prompts from a configuration directory.

    Prompts are stored as Markdown files (.md) in the prompts directory.
    Supports optional YAML frontmatter for metadata.

    Usage:
        loader = PromptLoader("config/prompts")
        system_prompt = loader.load("system")  # loads system.md
        persona = loader.load("persona")  # loads persona.md
    """

    def __init__(self, prompts_dir: str | Path) -> None:
        """Initialize prompt loader.

        Args:
            prompts_dir: Path to prompts directory (e.g., config/prompts).
        """
        self.prompts_dir = Path(prompts_dir)

    def load(self, name: str, default: Optional[str] = None) -> str:
        """Load a prompt by name.

        Args:
            name: Prompt name (without .md extension).
            default: Default value if file not found.

        Returns:
            Prompt content as string.

        Raises:
            PromptNotFoundError: If prompt file not found and no default.
        """
        prompt_path = self.prompts_dir / f"{name}.md"

        if not prompt_path.exists():
            if default is not None:
                logger.debug("Using default for prompt '%s'", name)
                return default
            raise PromptNotFoundError(name, str(prompt_path))

        content = prompt_path.read_text(encoding="utf-8")

        # Strip YAML frontmatter if present
        content = self._strip_frontmatter(content)

        logger.debug("Loaded prompt '%s' from %s", name, prompt_path)
        return content

    def _strip_frontmatter(self, content: str) -> str:
        """Strip YAML frontmatter from content.

        Args:
            content: Raw file content.

        Returns:
            Content with frontmatter removed.
        """
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                # Return everything after the second ---
                return parts[2].strip()
        return content

    def load_all(self) -> Dict[str, str]:
        """Load all prompts from directory.

        Returns:
            Dict mapping prompt names to content.
        """
        prompts: Dict[str, str] = {}

        if not self.prompts_dir.exists():
            logger.debug("Prompts directory does not exist: %s", self.prompts_dir)
            return prompts

        for path in self.prompts_dir.glob("*.md"):
            name = path.stem
            try:
                prompts[name] = self.load(name)
            except PromptNotFoundError:
                # Should not happen, but handle gracefully
                continue

        logger.debug("Loaded %d prompts from %s", len(prompts), self.prompts_dir)
        return prompts

    def exists(self, name: str) -> bool:
        """Check if a prompt file exists.

        Args:
            name: Prompt name.

        Returns:
            True if the prompt file exists.
        """
        prompt_path = self.prompts_dir / f"{name}.md"
        return prompt_path.exists()


def load_prompts(config_dir: str | Path = "config") -> Dict[str, str]:
    """Convenience function to load all prompts from config directory.

    Args:
        config_dir: Base config directory (prompts are in config/prompts/).

    Returns:
        Dict mapping prompt names to content.
    """
    loader = PromptLoader(Path(config_dir) / "prompts")
    return loader.load_all()


def get_prompt(
    config_dir: str | Path = "config",
    name: str = "system",
    default: Optional[str] = None,
) -> str:
    """Convenience function to load a specific prompt.

    Args:
        config_dir: Base config directory.
        name: Prompt name.
        default: Default value if not found.

    Returns:
        Prompt content.
    """
    loader = PromptLoader(Path(config_dir) / "prompts")
    return loader.load(name, default)
