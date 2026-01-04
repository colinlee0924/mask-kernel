"""MASK core module.

This module provides the core abstractions for the MASK framework:
- Skill definitions and metadata
- State management for Progressive Disclosure
- Custom exceptions
"""

from mask.core.exceptions import (
    MaskError,
    SkillError,
    SkillNotFoundError,
    SkillLoadError,
    SkillAlreadyRegisteredError,
    SkillMetadataError,
)
from mask.core.skill import (
    SkillMetadata,
    BaseSkill,
    MarkdownSkill,
    MAX_SKILL_NAME_LENGTH,
    MAX_SKILL_DESCRIPTION_LENGTH,
)
from mask.core.state import (
    HandoffContext,
    SkillState,
    SkillStateFIFO,
    SkillStateMode,
    SkillStateReplace,
    SkillStateUpdate,
    StateScope,
    create_fifo_state,
    skill_list_accumulate,
    skill_list_fifo,
    skill_list_reducer,
    skill_list_replace,
)
from mask.core.registry import SkillRegistry

__all__ = [
    # Exceptions
    "MaskError",
    "SkillError",
    "SkillNotFoundError",
    "SkillLoadError",
    "SkillAlreadyRegisteredError",
    "SkillMetadataError",
    # Skill classes
    "SkillMetadata",
    "BaseSkill",
    "MarkdownSkill",
    "MAX_SKILL_NAME_LENGTH",
    "MAX_SKILL_DESCRIPTION_LENGTH",
    # State & Scope
    "StateScope",
    "SkillState",
    "SkillStateReplace",
    "SkillStateFIFO",
    "SkillStateMode",
    "SkillStateUpdate",
    "HandoffContext",
    "create_fifo_state",
    "skill_list_reducer",
    "skill_list_replace",
    "skill_list_accumulate",
    "skill_list_fifo",
    # Registry
    "SkillRegistry",
]
