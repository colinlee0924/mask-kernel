"""Skill Middleware for Progressive Disclosure.

This module implements the middleware layer that enables Progressive Disclosure
of skills in the MASK framework using LangChain 1.x AgentMiddleware.

Key responsibilities:
1. Intercept each model call via wrap_model_call()
2. Dynamically filter tools based on skills_loaded state
3. Inject skill metadata into system prompts

The middleware uses request.override(tools=...) to dynamically change
which tools the model sees on each invocation. This enables true Progressive
Disclosure within a single invoke() call.

Progressive Disclosure Levels:
- Level 1 (Metadata): Skill name + description injected into system prompt
- Level 2 (Instructions): Full SKILL.md loaded when loader tool is called
- Level 3 (Resources): Agent uses read_file to access scripts/, references/, assets/

When a loader tool is called:
1. It returns a Command with skills_loaded update
2. The state is updated by LangGraph
3. Next model call goes through wrap_model_call() again
4. Middleware filters tools based on new skills_loaded
5. Model now sees the newly activated skill's tools
"""

import logging
from typing import Any, Callable, List, Optional, Sequence

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool

try:
    from langchain.agents.middleware import (
        AgentMiddleware,
        ModelRequest,
        ModelResponse,
    )

    HAS_AGENT_MIDDLEWARE = True
except ImportError:
    # Fallback for older LangChain versions
    HAS_AGENT_MIDDLEWARE = False
    AgentMiddleware = object
    ModelRequest = Any
    ModelResponse = Any

from mask.core.registry import SkillRegistry
from mask.core.state import SkillState

logger = logging.getLogger(__name__)


def build_skills_system_prompt(
    registry: SkillRegistry,
    active_skills: List[str],
    include_paths: bool = True,
) -> str:
    """Build system prompt section describing available skills.

    Implements Progressive Disclosure Level 1 by injecting skill metadata
    (name, description, path) into the system prompt.

    Args:
        registry: The skill registry containing available skills.
        active_skills: List of currently active skill names.
        include_paths: Whether to include skill paths for Level 3 access.

    Returns:
        System prompt section describing skills.
    """
    lines: List[str] = []

    # List available skills
    skills_summary = registry.get_skills_summary()
    all_skills = registry.get_all_skills()

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

                # Include path information for Level 3 access
                if include_paths:
                    # Find the skill to get its directory
                    skill = next(
                        (s for s in all_skills if s.metadata.name == name),
                        None,
                    )
                    if skill and skill.skill_dir:
                        skill_dir = skill.skill_dir
                        lines.append(f"  - Path: `{skill_dir / 'SKILL.md'}`")

                        # List available resource subdirectories
                        subdirs = []
                        for subdir in ["scripts", "references", "assets"]:
                            subdir_path = skill_dir / subdir
                            if subdir_path.exists() and subdir_path.is_dir():
                                subdirs.append(f"`{subdir}/`")
                        if subdirs:
                            lines.append(f"  - Resources: {', '.join(subdirs)}")

        lines.append("")

        # Add Level 3 usage instructions if paths are included
        if include_paths:
            lines.append("**How to Access Skill Resources (Progressive Disclosure):**")
            lines.append("")
            lines.append("1. **Activate a skill**: Call `use_<skill_name>()` to load instructions")
            lines.append("2. **Read additional resources**: Use `read_file(path)` for detailed docs in:")
            lines.append("   - `scripts/`: Executable Python or shell scripts")
            lines.append("   - `references/`: Documentation, API specs, examples")
            lines.append("   - `assets/`: Templates, configurations, data files")
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


class SkillMiddleware(AgentMiddleware if HAS_AGENT_MIDDLEWARE else object):
    """Middleware for Progressive Disclosure of skills.

    Implements LangChain 1.x AgentMiddleware to intercept each model call
    and dynamically filter tools based on skills_loaded state.

    Key method: wrap_model_call()
    - Called before each model invocation
    - Reads skills_loaded from request.state
    - Filters tools via registry.get_tools_for_active_skills()
    - Uses request.override(tools=...) to replace tool list
    - Calls handler(filtered_request) to continue

    This enables true Progressive Disclosure within a single invoke():
    1. Initial call: only loader tools visible
    2. Agent calls use_calculator → returns Command with skills_loaded update
    3. State updated, next model call triggers wrap_model_call again
    4. Now calculator tools are visible

    Usage:
        middleware = SkillMiddleware(registry)
        agent = create_agent(
            model=model,
            tools=all_tools,
            middleware=[middleware],
            state_schema=SkillState,
        )
    """

    def __init__(
        self,
        registry: SkillRegistry,
        include_skill_instructions: bool = True,
        include_skill_paths: bool = True,
        verbose: bool = False,
        additional_tools: Optional[List[BaseTool]] = None,
    ) -> None:
        """Initialize the middleware.

        Args:
            registry: Skill registry for tool management.
            include_skill_instructions: Whether to include skill instructions
                in the system prompt when skills are active.
            include_skill_paths: Whether to include skill paths in system prompt
                for Level 3 progressive disclosure (read_file access).
            verbose: Whether to log tool filtering details.
            additional_tools: Non-skill tools to always include.
        """
        if HAS_AGENT_MIDDLEWARE:
            super().__init__()
        self.registry = registry
        self.include_skill_instructions = include_skill_instructions
        self.include_skill_paths = include_skill_paths
        self.verbose = verbose
        self.additional_tools = additional_tools or []

    # =========================================================================
    # AgentMiddleware Implementation (LangChain 1.x)
    # =========================================================================

    def wrap_model_call(
        self,
        request: "ModelRequest",
        handler: Callable[["ModelRequest"], "ModelResponse"],
    ) -> "ModelResponse":
        """Intercept model call and dynamically filter tools.

        This is the core method for Progressive Disclosure. It:
        1. Reads skills_loaded from request.state
        2. Gets filtered tools from registry
        3. Overrides the request's tools
        4. Passes to next handler

        Args:
            request: Model request with state and tools.
            handler: Next handler in the chain.

        Returns:
            Model response from the handler.
        """
        # Extract skills_loaded from state
        skills_loaded = self._get_skills_loaded(request)

        # Get filtered tools (loaders + active skill capability tools)
        relevant_tools = self.registry.get_tools_for_active_skills(skills_loaded)

        # Add additional non-skill tools
        if self.additional_tools:
            relevant_tools = list(relevant_tools) + list(self.additional_tools)

        if self.verbose:
            logger.info("[SkillMiddleware] skills_loaded: %s", skills_loaded)
            logger.info(
                "[SkillMiddleware] tools (%d): %s",
                len(relevant_tools),
                [t.name for t in relevant_tools],
            )

        # Override tools in request
        filtered_request = request.override(tools=relevant_tools)

        # Call next handler
        return handler(filtered_request)

    async def awrap_model_call(
        self,
        request: "ModelRequest",
        handler: Callable[["ModelRequest"], "ModelResponse"],
    ) -> "ModelResponse":
        """Async version of wrap_model_call."""
        skills_loaded = self._get_skills_loaded(request)
        relevant_tools = self.registry.get_tools_for_active_skills(skills_loaded)

        if self.additional_tools:
            relevant_tools = list(relevant_tools) + list(self.additional_tools)

        if self.verbose:
            logger.info("[SkillMiddleware] (async) skills_loaded: %s", skills_loaded)
            logger.info(
                "[SkillMiddleware] (async) tools (%d): %s",
                len(relevant_tools),
                [t.name for t in relevant_tools],
            )

        filtered_request = request.override(tools=relevant_tools)
        return await handler(filtered_request)

    def _get_skills_loaded(self, request: "ModelRequest") -> List[str]:
        """Extract skills_loaded from request state.

        Args:
            request: Model request with state.

        Returns:
            List of loaded skill names.
        """
        skills_loaded = []
        if hasattr(request, "state") and request.state is not None:
            if isinstance(request.state, dict):
                skills_loaded = request.state.get("skills_loaded", [])
            else:
                skills_loaded = getattr(request.state, "skills_loaded", [])
        return skills_loaded

    # =========================================================================
    # Legacy Methods (for backward compatibility)
    # =========================================================================

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

        # Build skills prompt with optional path information
        skills_prompt = build_skills_system_prompt(
            self.registry,
            active_skills,
            include_paths=self.include_skill_paths,
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
    1. Returns a Command with skills_loaded update
    2. Returns the skill's instructions as a ToolMessage

    Args:
        registry: The skill registry.
        skill_name: Name of the skill.

    Returns:
        A BaseTool for loading/activating the skill.

    Note:
        This is an alternative to using the skill's built-in get_loader_tool().
        Use this when you need custom activation logic.
    """
    from typing import Any

    from langchain_core.messages import ToolMessage
    from langchain_core.tools import tool
    from langgraph.types import Command

    skill = registry.get(skill_name)
    description = skill.metadata.description

    @tool(name=f"use_{skill_name.replace('-', '_')}")
    def loader(runtime: Any = None) -> Command:
        """Activate the skill and get instructions."""
        instructions = skill.get_instructions()

        # Get tool_call_id from runtime if available
        tool_call_id = "unknown"
        if runtime is not None and hasattr(runtime, "tool_call_id"):
            tool_call_id = runtime.tool_call_id

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=instructions,
                        tool_call_id=tool_call_id,
                    )
                ],
                "skills_loaded": [skill_name],
            }
        )

    loader.description = f"Activate the {skill_name} skill. {description}"
    return loader
