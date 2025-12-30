"""MASK skill loaders.

This module provides loaders for different skill formats:
- SKILL.md files (Markdown-based skills)
- Python modules (skill.py with BaseSkill implementations)
"""

from mask.loader.python_loader import (
    discover_python_skills,
    load_python_skill,
)
from mask.loader.skill_md_loader import (
    discover_markdown_skills,
    load_markdown_skill,
    parse_skill_md,
)

__all__ = [
    "parse_skill_md",
    "load_markdown_skill",
    "discover_markdown_skills",
    "load_python_skill",
    "discover_python_skills",
]
