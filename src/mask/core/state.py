"""State management for MASK skills.

This module defines the state structures and reducers used for tracking
skill activation in the Progressive Disclosure pattern.
"""

from typing import Annotated, List

from langgraph.graph import MessagesState


def skill_list_reducer(current: List[str], new: List[str]) -> List[str]:
    """Reducer for skill activation state.

    This reducer implements cumulative skill activation:
    - Once a skill is activated, it stays activated
    - New skills are added to the existing list
    - Duplicates are automatically removed

    Args:
        current: Currently activated skills.
        new: Newly activated skills to add.

    Returns:
        Combined list of unique activated skills.

    Example:
        >>> skill_list_reducer(["pdf"], ["web-search"])
        ["pdf", "web-search"]
        >>> skill_list_reducer(["pdf", "web-search"], ["pdf"])  # no duplicate
        ["pdf", "web-search"]
    """
    # Combine and deduplicate while preserving order
    seen = set(current)
    result = list(current)
    for skill in new:
        if skill not in seen:
            seen.add(skill)
            result.append(skill)
    return result


class SkillState(MessagesState):
    """Agent state with skill tracking.

    Extends LangGraph's MessagesState to include skill activation state.
    Uses the skill_list_reducer to ensure skills remain activated once loaded.

    Attributes:
        skills_loaded: List of activated skill names. Skills are added via
            the loader tools and persist across conversation turns.

    Example:
        Initial state: skills_loaded = []
        After loading pdf skill: skills_loaded = ["pdf"]
        After loading web-search: skills_loaded = ["pdf", "web-search"]
    """

    skills_loaded: Annotated[List[str], skill_list_reducer] = []


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
