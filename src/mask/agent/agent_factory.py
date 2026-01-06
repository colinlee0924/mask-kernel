"""Agent factory functions.

This module provides factory functions for creating MASK agents with
common configurations.

Features:
- Auto-discovery of skills from configured directories
- Tier-based model selection (FAST, THINKING, PRO)
- Automatic inclusion of filesystem tools for Level 3 progressive disclosure
- Session storage integration for stateful agents
"""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from mask.agent.base_agent import BaseAgent, SimpleAgent
from mask.agent.prompt_loader import PromptLoader
from mask.core.registry import SkillRegistry
from mask.models.llm_factory import LLMFactory, ModelTier
from mask.storage.base import SessionStore

logger = logging.getLogger(__name__)


def create_mask_agent(
    model: Optional[BaseChatModel] = None,
    skill_registry: Optional[SkillRegistry] = None,
    system_prompt: Optional[str] = None,
    config_dir: str | Path = "config",
    *,
    tier: ModelTier = ModelTier.THINKING,
    provider: Optional[str] = None,
    stateless: bool = True,
    session_store: Optional[SessionStore] = None,
    additional_tools: Optional[List[BaseTool]] = None,
    skills_dir: Optional[str | Path] = None,
    enable_file_access: bool = True,
    file_access_paths: Optional[List[Path]] = None,
) -> SimpleAgent:
    """Create a MASK agent with common configuration.

    This factory function simplifies agent creation by:
    - Auto-loading prompts from config directory
    - Auto-discovering skills from skills directory
    - Creating LLM from tier specification
    - Setting up session storage if needed
    - Optionally including filesystem tools for Level 3 progressive disclosure

    Args:
        model: Optional pre-configured model. If not provided, created from tier.
        skill_registry: Optional skill registry. Auto-discovers if not provided.
        system_prompt: System prompt. Loaded from config/prompts/system.md if not provided.
        config_dir: Configuration directory path.
        tier: Model capability tier (FAST, THINKING, PRO).
        provider: LLM provider override.
        stateless: Whether agent is stateless (default True).
        session_store: Storage backend for stateful operation.
        additional_tools: Non-skill tools to include.
        skills_dir: Skills directory. Defaults to {config_dir}/skills or src/*/skills.
        enable_file_access: Whether to include read_file tool for Level 3 progressive
            disclosure. Defaults to True.
        file_access_paths: Optional list of allowed paths for file access. If not
            provided, defaults to the skills directory only (for security).

    Returns:
        Configured SimpleAgent instance.

    Example:
        # Minimal usage - uses defaults
        agent = create_mask_agent()

        # With custom configuration
        agent = create_mask_agent(
            tier=ModelTier.PRO,
            stateless=False,
            session_store=RedisSessionStore("redis://localhost:6379"),
        )

        # With restricted file access
        agent = create_mask_agent(
            enable_file_access=True,
            file_access_paths=[Path("/app/skills"), Path("/app/data")],
        )
    """
    config_path = Path(config_dir)

    # Create or use provided model
    if model is None:
        factory = LLMFactory()
        model = factory.get_model(tier=tier, provider=provider)
        logger.debug("Created model: tier=%s, provider=%s", tier, provider)

    # Load system prompt
    if system_prompt is None:
        prompt_loader = PromptLoader(config_path / "prompts")
        system_prompt = prompt_loader.load(
            "system",
            default="You are a helpful assistant.",
        )

    # Setup skill registry and determine skills path
    skills_path: Optional[Path] = None

    if skill_registry is None:
        skill_registry = SkillRegistry()

        # Auto-discover skills
        if skills_dir:
            skills_path = Path(skills_dir)
        else:
            # Try common locations
            skills_path = config_path / "skills"
            if not skills_path.exists():
                # Try src/*/skills pattern
                src_skills = list(Path("src").glob("*/skills"))
                if src_skills:
                    skills_path = src_skills[0]

        if skills_path and skills_path.exists():
            count = skill_registry.discover_from_directory(skills_path)
            logger.debug("Discovered %d skills from %s", count, skills_path)

    # Prepare additional tools list
    all_additional_tools = list(additional_tools or [])

    # Add filesystem tools for Level 3 progressive disclosure
    if enable_file_access:
        from mask.tools.filesystem import create_read_file_tool

        # Determine allowed paths for security
        allowed_paths = file_access_paths
        if allowed_paths is None and skills_path and skills_path.exists():
            # Default: only allow access to skills directory
            allowed_paths = [skills_path]

        read_file_tool = create_read_file_tool(
            allowed_prefixes=allowed_paths,
        )
        all_additional_tools.append(read_file_tool)
        logger.debug(
            "Added read_file tool with allowed paths: %s",
            [str(p) for p in (allowed_paths or [])],
        )

    # Create agent
    agent = SimpleAgent(
        model=model,
        skill_registry=skill_registry,
        system_prompt=system_prompt,
        stateless=stateless,
        session_store=session_store,
        additional_tools=all_additional_tools if all_additional_tools else None,
    )

    logger.info(
        "Created MASK agent: stateless=%s, skills=%d, file_access=%s",
        stateless,
        len(skill_registry),
        enable_file_access,
    )

    return agent


def create_stateful_agent(
    session_store: SessionStore,
    **kwargs,
) -> SimpleAgent:
    """Create a stateful MASK agent.

    Convenience function for creating stateful agents.

    Args:
        session_store: Storage backend for sessions.
        **kwargs: Additional arguments passed to create_mask_agent.

    Returns:
        Configured stateful SimpleAgent instance.
    """
    return create_mask_agent(
        stateless=False,
        session_store=session_store,
        **kwargs,
    )


def create_minimal_agent(
    model: BaseChatModel,
    system_prompt: str = "You are a helpful assistant.",
) -> SimpleAgent:
    """Create a minimal agent without skills.

    Useful for simple use cases that don't need Progressive Disclosure.

    Args:
        model: The LLM model to use.
        system_prompt: The system prompt.

    Returns:
        Minimal SimpleAgent instance.
    """
    return SimpleAgent(
        model=model,
        skill_registry=SkillRegistry(),
        system_prompt=system_prompt,
        stateless=True,
    )
