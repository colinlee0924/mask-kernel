"""MASK Kernel - Multi-Agent Skill Kit.

A framework for building expertise agents with progressive skill disclosure.

Quick Start:
    from mask.agent import create_mask_agent
    from mask.models import ModelTier

    agent = create_mask_agent(tier=ModelTier.THINKING)
    response = await agent.invoke("Hello!")

Modules:
    - mask.core: Core abstractions (Skill, Registry, State)
    - mask.agent: Agent classes and factories
    - mask.models: LLM factory with tier-based selection
    - mask.middleware: Skill middleware for Progressive Disclosure
    - mask.loader: Skill loaders (SKILL.md, Python)
    - mask.session: Session management
    - mask.storage: Storage backends (Memory, Redis, PostgreSQL)
    - mask.a2a: A2A Protocol integration
    - mask.mcp: MCP integration
    - mask.observability: OpenInference tracing
    - mask.cli: CLI commands (mask init, mask run)
"""

__version__ = "0.1.0"

# Core exports for convenience
from mask.core import (
    BaseSkill,
    MarkdownSkill,
    SkillMetadata,
    SkillRegistry,
    SkillState,
)
from mask.agent import (
    BaseAgent,
    SimpleAgent,
    create_mask_agent,
    load_prompts,
)
from mask.models import (
    LLMFactory,
    ModelTier,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "BaseSkill",
    "MarkdownSkill",
    "SkillMetadata",
    "SkillRegistry",
    "SkillState",
    # Agent
    "BaseAgent",
    "SimpleAgent",
    "create_mask_agent",
    "load_prompts",
    # Models
    "LLMFactory",
    "ModelTier",
]
