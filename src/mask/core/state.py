"""State management for MASK skills.

This module defines the state structures and reducers used for tracking
skill activation in the Progressive Disclosure pattern.

State Scopes (for Multi-Agent support):
- NONE: No state persistence (pure stateless, no Progressive Disclosure)
- REQUEST: State persists within single invoke() - default for Progressive Disclosure
- TASK: State persists within a task (for multi-agent handoffs)
- CONVERSATION: Full session persistence across invokes

Three state modes are available:
1. ACCUMULATE (default): Skills accumulate as they are loaded
2. REPLACE: Each new skill activation replaces all previous skills
3. FIFO: Only the most recent N skills are kept (configurable)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Callable, Dict, List, Optional

from langgraph.graph import MessagesState


class StateScope(Enum):
    """State persistence scope for agents.

    Defines how long state (skills_loaded, context) is maintained:
    - NONE: No state at all (pure stateless)
    - REQUEST: State only within single invoke() call (default)
    - TASK: State within a task/handoff chain (multi-agent)
    - CONVERSATION: Full session persistence
    """

    NONE = "none"
    REQUEST = "request"
    TASK = "task"
    CONVERSATION = "conversation"


class SkillStateMode(Enum):
    """Skill state management modes."""

    REPLACE = "replace"
    ACCUMULATE = "accumulate"
    FIFO = "fifo"


@dataclass
class HandoffContext:
    """Context passed between agents in multi-agent handoffs.

    Enables Orchestrator-Worker pattern with context isolation:
    - Parent can pre-activate skills for child agents
    - Pass task-specific context without polluting conversation history
    - Maintain skill state across agent boundaries

    Attributes:
        initial_skills: Skills to pre-activate in the child agent.
        context_data: Arbitrary data to pass to child agent.
        parent_agent: Name of the parent agent for tracing.
        task_id: Task identifier for grouping related agents.
    """

    initial_skills: List[str] = field(default_factory=list)
    context_data: Dict[str, Any] = field(default_factory=dict)
    parent_agent: Optional[str] = None
    task_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "initial_skills": self.initial_skills,
            "context_data": self.context_data,
            "parent_agent": self.parent_agent,
            "task_id": self.task_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HandoffContext":
        """Create from dictionary."""
        return cls(
            initial_skills=data.get("initial_skills", []),
            context_data=data.get("context_data", {}),
            parent_agent=data.get("parent_agent"),
            task_id=data.get("task_id"),
        )


# =============================================================================
# State Reducers
# =============================================================================


def skill_list_replace(current: List[str], new: List[str]) -> List[str]:
    """Replace mode reducer: New skills replace all previous skills.

    Use case: Simple tasks that only need one skill at a time.

    Args:
        current: Currently activated skills (ignored if new is non-empty).
        new: Newly activated skills.

    Returns:
        The new list if non-empty, otherwise current.

    Example:
        >>> skill_list_replace(["pdf"], ["web-search"])
        ["web-search"]
    """
    return new if new else current


def skill_list_accumulate(current: List[str], new: List[str]) -> List[str]:
    """Accumulate mode reducer: Skills accumulate as they are loaded.

    This is the default behavior. Once a skill is activated, it stays
    activated until the session ends.

    Args:
        current: Currently activated skills.
        new: Newly activated skills to add.

    Returns:
        Combined list of unique activated skills.

    Example:
        >>> skill_list_accumulate(["pdf"], ["web-search"])
        ["pdf", "web-search"]
        >>> skill_list_accumulate(["pdf", "web-search"], ["pdf"])  # no duplicate
        ["pdf", "web-search"]
    """
    if not current:
        return new
    seen = set(current)
    result = list(current)
    for skill in new:
        if skill not in seen:
            seen.add(skill)
            result.append(skill)
    return result


def skill_list_fifo(max_skills: int = 3) -> Callable[[List[str], List[str]], List[str]]:
    """FIFO mode reducer factory: Keep only the most recent N skills.

    Use case: Cost control, limiting concurrent tool exposure.

    Args:
        max_skills: Maximum number of skills to keep (default: 3).

    Returns:
        A reducer function that enforces the FIFO limit.

    Example:
        >>> fifo_reducer = skill_list_fifo(2)
        >>> fifo_reducer(["a", "b"], ["c"])
        ["b", "c"]
    """

    def reducer(current: List[str], new: List[str]) -> List[str]:
        if not current:
            return new[:max_skills]
        combined = list(current)
        seen = set(current)
        for skill in new:
            if skill not in seen:
                seen.add(skill)
                combined.append(skill)
        return combined[-max_skills:]

    return reducer


# Backward compatibility alias
skill_list_reducer = skill_list_accumulate


# =============================================================================
# State Classes
# =============================================================================


class SkillState(MessagesState):
    """Agent state with skill tracking (Accumulate mode - default).

    Extends LangGraph's MessagesState to include skill activation state.
    Uses the accumulate reducer to ensure skills remain activated once loaded.

    Attributes:
        skills_loaded: List of activated skill names. Skills are added via
            the loader tools and persist across conversation turns.

    Example:
        Initial state: skills_loaded = []
        After loading pdf skill: skills_loaded = ["pdf"]
        After loading web-search: skills_loaded = ["pdf", "web-search"]
    """

    skills_loaded: Annotated[List[str], skill_list_accumulate] = []


class SkillStateReplace(MessagesState):
    """Agent state with skill tracking (Replace mode).

    Each new skill activation replaces all previous skills.
    """

    skills_loaded: Annotated[List[str], skill_list_replace] = []


class SkillStateFIFO(MessagesState):
    """Agent state with skill tracking (FIFO mode, max 3 skills).

    Only the most recent 3 skills are kept.
    """

    skills_loaded: Annotated[List[str], skill_list_fifo(3)] = []


def create_fifo_state(max_skills: int = 3) -> type:
    """Create a custom FIFO state class with specified max skills.

    Args:
        max_skills: Maximum number of skills to keep.

    Returns:
        A new state class with the FIFO reducer.

    Example:
        >>> FiveSkillState = create_fifo_state(5)
        >>> state = FiveSkillState(messages=[], skills_loaded=["a", "b", "c", "d", "e", "f"])
        >>> # After reduction: skills_loaded = ["b", "c", "d", "e", "f"]
    """

    class CustomFIFOState(MessagesState):
        skills_loaded: Annotated[List[str], skill_list_fifo(max_skills)] = []

    CustomFIFOState.__name__ = f"SkillStateFIFO{max_skills}"
    return CustomFIFOState


class SkillStateUpdate:
    """State update for skill activation.

    Used by the SkillMiddleware to update the agent state when
    skills are discovered or activated.
    """

    def __init__(self, skills_loaded: List[str] | None = None) -> None:
        """Initialize state update.

        Args:
            skills_loaded: List of skill names to mark as loaded.
        """
        self.skills_loaded = skills_loaded or []

    def to_dict(self) -> dict:
        """Convert to dictionary for state update.

        Returns:
            Dictionary with skills_loaded key.
        """
        return {"skills_loaded": self.skills_loaded}
