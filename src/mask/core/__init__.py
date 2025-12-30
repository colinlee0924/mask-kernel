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
    SkillState,
    SkillStateUpdate,
    skill_list_reducer,
)

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
    # State
    "SkillState",
    "SkillStateUpdate",
    "skill_list_reducer",
]
