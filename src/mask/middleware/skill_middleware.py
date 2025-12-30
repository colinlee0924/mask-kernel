"""Skill Middleware for Progressive Disclosure.

This module implements the middleware layer that enables Progressive Disclosure
of skills in the MASK framework.

Key responsibilities:
1. Inject skill metadata into system prompts
2. Filter available tools based on active skills
3. Handle skill activation through loader tools

The middleware intercepts model calls and:
- Prepends skill information to the system prompt
- Filters tools to show loader tools + active skill tools
"""

import logging
from typing import Any, Callable, List, Optional, Sequence

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool

from mask.core.registry import SkillRegistry
from mask.core.state import SkillState

logger = logging.getLogger(__name__)


def build_skills_system_prompt(
    registry: SkillRegistry,
    active_skills: List[str],
) -> str:
    """Build system prompt section describing available skills.

    Args:
        registry: The skill registry containing available skills.
        active_skills: List of currently active skill names.

    Returns:
        System prompt section describing skills.
    """
    lines: List[str] = []

    # List available skills
    skills_summary = registry.get_skills_summary()
    if skills_summary:
        lines.append("## Available Skills")
        lines.append("")
        lines.append("You have access to the following skills. Use the corresponding ")
        lines.append("`use_<skill_name>` tool to activate a skill and receive detailed ")
        lines.append("instructions for its use.")
        lines.append("")

        for skill_info in skills_summary:
            if skill_info["enabled"]:
                name = skill_info["name"]
                desc = skill_info["description"]
                status = "ACTIVE" if name in active_skills else "available"
                lines.append(f"- **{name}** ({status}): {desc}")

        lines.append("")

    # Include active skill instructions
    if active_skills:
        lines.append("## Active Skill Instructions")
        lines.append("")
        instructions = registry.get_active_skill_instructions(active_skills)
        if instructions:
            lines.append(instructions)
        lines.append("")

    return "\n".join(lines)


def inject_skills_into_messages(
    messages: Sequence[BaseMessage],
    skills_prompt: str,
) -> List[BaseMessage]:
    """Inject skills information into message sequence.

    If the first message is a SystemMessage, the skills prompt is prepended.
    Otherwise, a new SystemMessage is added at the beginning.

    Args:
        messages: Original message sequence.
        skills_prompt: Skills prompt to inject.

    Returns:
        Modified message list with skills information.
    """
    if not skills_prompt:
        return list(messages)

    messages_list = list(messages)

    if messages_list and isinstance(messages_list[0], SystemMessage):
        # Prepend to existing system message
        original_content = messages_list[0].content
        if isinstance(original_content, str):
            new_content = f"{skills_prompt}\n\n---\n\n{original_content}"
        else:
            # Handle non-string content (list of content blocks)
            new_content = f"{skills_prompt}\n\n---\n\n{str(original_content)}"

        messages_list[0] = SystemMessage(content=new_content)
    else:
        # Add new system message at beginning
        messages_list.insert(0, SystemMessage(content=skills_prompt))

    return messages_list


def filter_tools_for_state(
    registry: SkillRegistry,
    state: SkillState,
    additional_tools: Optional[List[BaseTool]] = None,
) -> List[BaseTool]:
    """Filter tools based on current skill state.

    Implements Progressive Disclosure:
    - Loader tools are always included
    - Capability tools only for active skills
    - Additional tools (non-skill) are always included

    Args:
        registry: The skill registry.
        state: Current skill state with active skills.
        additional_tools: Non-skill tools to always include.

    Returns:
        Filtered list of tools.
    """
    active_skills = state.get("skills_loaded", [])

    # Get skill-related tools
    skill_tools = registry.get_tools_for_active_skills(active_skills)

    # Combine with additional tools
    all_tools = list(skill_tools)
    if additional_tools:
        all_tools.extend(additional_tools)

    return all_tools


class SkillMiddleware:
    """Middleware for Progressive Disclosure of skills.

    This middleware wraps model calls to:
    1. Inject skill metadata into system prompts
    2. Filter available tools based on active skills

    Usage:
        middleware = SkillMiddleware(registry)

        # In your agent graph node
        def call_model(state: SkillState):
            messages = middleware.prepare_messages(state)
            tools = middleware.get_tools(state, additional_tools=[...])
            response = model.bind_tools(tools).invoke(messages)
            return {"messages": [response]}
    """

    def __init__(
        self,
        registry: SkillRegistry,
        include_skill_instructions: bool = True,
    ) -> None:
        """Initialize the middleware.

        Args:
            registry: Skill registry for tool management.
            include_skill_instructions: Whether to include skill instructions
                in the system prompt when skills are active.
        """
        self.registry = registry
        self.include_skill_instructions = include_skill_instructions

    def prepare_messages(
        self,
        state: SkillState,
        messages: Optional[Sequence[BaseMessage]] = None,
    ) -> List[BaseMessage]:
        """Prepare messages with skill information injected.

        Args:
            state: Current skill state.
            messages: Original messages. If None, uses state["messages"].

        Returns:
            Messages with skills prompt injected.
        """
        if messages is None:
            messages = state.get("messages", [])

        active_skills = state.get("skills_loaded", [])

        # Build skills prompt
        skills_prompt = build_skills_system_prompt(
            self.registry,
            active_skills,
        )

        # Inject into messages
        return inject_skills_into_messages(messages, skills_prompt)

    def get_tools(
        self,
        state: SkillState,
        additional_tools: Optional[List[BaseTool]] = None,
    ) -> List[BaseTool]:
        """Get tools for the current state.

        Args:
            state: Current skill state.
            additional_tools: Non-skill tools to include.

        Returns:
            List of available tools.
        """
        return filter_tools_for_state(
            self.registry,
            state,
            additional_tools,
        )

    def create_skill_activation_callback(
        self,
    ) -> Callable[[str], dict[str, Any]]:
        """Create a callback for skill activation.

        The callback is called when a loader tool is invoked,
        adding the skill to the active skills list.

        Returns:
            Callback function that returns state update dict.
        """

        def activate_skill(skill_name: str) -> dict[str, Any]:
            """Activate a skill and return state update."""
            if self.registry.has(skill_name):
                logger.info("Activating skill: %s", skill_name)
                return {"skills_loaded": [skill_name]}
            else:
                logger.warning("Attempted to activate unknown skill: %s", skill_name)
                return {}

        return activate_skill


def create_loader_tool_with_activation(
    registry: SkillRegistry,
    skill_name: str,
) -> BaseTool:
    """Create a loader tool that activates the skill and returns instructions.

    This creates a tool that when invoked:
    1. Returns the skill's instructions
    2. The state update (skill activation) is handled by the agent

    Args:
        registry: The skill registry.
        skill_name: Name of the skill.

    Returns:
        A BaseTool for loading/activating the skill.

    Note:
        This is an alternative to using the skill's built-in get_loader_tool().
        Use this when you need custom activation logic.
    """
    from langchain_core.tools import tool

    skill = registry.get(skill_name)
    description = skill.metadata.description

    @tool(name=f"use_{skill_name.replace('-', '_')}")
    def loader() -> str:
        """Activate the skill and get instructions."""
        return skill.get_instructions()

    loader.description = f"Activate the {skill_name} skill. {description}"
    return loader
