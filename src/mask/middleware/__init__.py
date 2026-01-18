"""MASK middleware components.

This module provides middleware for agent execution:
- SkillMiddleware: Progressive Disclosure of skills
- A2AStreamingMiddleware: Real-time event propagation for A2A protocol
"""

from mask.middleware.a2a_streaming import A2AStreamingMiddleware
from mask.middleware.skill_middleware import (
    SkillMiddleware,
    build_skills_system_prompt,
    create_loader_tool_with_activation,
    filter_tools_for_state,
    inject_skills_into_messages,
)

__all__ = [
    # Skill middleware
    "SkillMiddleware",
    "build_skills_system_prompt",
    "inject_skills_into_messages",
    "filter_tools_for_state",
    "create_loader_tool_with_activation",
    # A2A streaming middleware
    "A2AStreamingMiddleware",
]
