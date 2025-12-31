"""MASK agent abstraction layer.

This module provides the agent foundation for MASK:
- BaseAgent: Abstract base class for agents
- SimpleAgent: Basic agent implementation
- Factory functions for easy agent creation
- Prompt loading utilities
"""

from mask.agent.agent_factory import (
    create_mask_agent,
    create_minimal_agent,
    create_stateful_agent,
)
from mask.agent.base_agent import BaseAgent, SimpleAgent
from mask.agent.prompt_loader import (
    PromptLoader,
    get_prompt,
    load_prompts,
)

__all__ = [
    # Agent classes
    "BaseAgent",
    "SimpleAgent",
    # Factory functions
    "create_mask_agent",
    "create_stateful_agent",
    "create_minimal_agent",
    # Prompt utilities
    "PromptLoader",
    "load_prompts",
    "get_prompt",
]
