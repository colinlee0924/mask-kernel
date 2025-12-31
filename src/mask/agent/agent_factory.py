"""Agent factory functions.

This module provides factory functions for creating MASK agents with
common configurations.
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
) -> SimpleAgent:
    """Create a MASK agent with common configuration.

    This factory function simplifies agent creation by:
    - Auto-loading prompts from config directory
    - Auto-discovering skills from skills directory
    - Creating LLM from tier specification
    - Setting up session storage if needed

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

    # Setup skill registry
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

        if skills_path.exists():
            count = skill_registry.discover_from_directory(skills_path)
            logger.debug("Discovered %d skills from %s", count, skills_path)

    # Create agent
    agent = SimpleAgent(
        model=model,
        skill_registry=skill_registry,
        system_prompt=system_prompt,
        stateless=stateless,
        session_store=session_store,
        additional_tools=additional_tools,
    )

    logger.info(
        "Created MASK agent: stateless=%s, skills=%d",
        stateless,
        len(skill_registry),
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
