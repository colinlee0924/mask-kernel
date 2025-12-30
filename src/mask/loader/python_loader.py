"""Loader for Python-based skills.

This module implements loading of skills from Python modules (skill.py files)
that contain BaseSkill implementations.

Python skills provide more flexibility than SKILL.md files, allowing
custom tool implementations and complex logic.

Directory structure for Python skills:
```
my-skill/
├── SKILL.md          # Required: metadata and instructions
└── skill.py          # Required: BaseSkill implementation
```
"""

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional

from mask.core.exceptions import SkillLoadError
from mask.core.skill import BaseSkill

logger = logging.getLogger(__name__)


def _is_safe_path(path: Path, base_dir: Path) -> bool:
    """Check if a path is safely contained within base_dir.

    Prevents directory traversal attacks via symlinks or path manipulation.

    Args:
        path: The path to validate.
        base_dir: The base directory that should contain the path.

    Returns:
        True if the path is safely within base_dir.
    """
    try:
        resolved_path = path.resolve()
        resolved_base = base_dir.resolve()
        resolved_path.relative_to(resolved_base)
        return True
    except ValueError:
        return False
    except (OSError, RuntimeError):
        return False


def load_python_skill(
    skill_dir: Path,
    source: str = "local",
) -> Optional[BaseSkill]:
    """Load a Python skill from a directory containing skill.py.

    The skill.py module must define a class that inherits from BaseSkill
    and can be instantiated without arguments, OR define a `create_skill()`
    factory function that returns a BaseSkill instance.

    Args:
        skill_dir: Path to skill directory containing skill.py.
        source: Source of the skill ('local', 'user', 'project').

    Returns:
        BaseSkill instance or None if loading fails.

    Raises:
        SkillLoadError: If the module cannot be loaded or doesn't contain
            a valid skill implementation.
    """
    skill_py_path = skill_dir / "skill.py"

    if not skill_py_path.exists():
        logger.debug("No skill.py found in %s", skill_dir)
        return None

    # Security: Validate path
    try:
        skill_py_path = skill_py_path.resolve()
    except (OSError, RuntimeError) as e:
        raise SkillLoadError(str(skill_dir), f"cannot resolve path: {e}") from e

    module_name = f"mask_skills.{skill_dir.name}"

    try:
        # Load module from file path
        spec = importlib.util.spec_from_file_location(module_name, skill_py_path)
        if spec is None or spec.loader is None:
            raise SkillLoadError(str(skill_dir), "failed to create module spec")

        module = importlib.util.module_from_spec(spec)

        # Add to sys.modules before executing to handle circular imports
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            # Clean up on failure
            sys.modules.pop(module_name, None)
            raise SkillLoadError(str(skill_dir), f"module execution failed: {e}") from e

        # Try factory function first
        if hasattr(module, "create_skill"):
            create_fn = getattr(module, "create_skill")
            if callable(create_fn):
                try:
                    skill = create_fn()
                    if isinstance(skill, BaseSkill):
                        logger.debug(
                            "Loaded skill '%s' via create_skill()",
                            skill.metadata.name,
                        )
                        return skill
                except Exception as e:
                    raise SkillLoadError(
                        str(skill_dir),
                        f"create_skill() failed: {e}",
                    ) from e

        # Fall back to finding BaseSkill subclass
        skill_class = _find_skill_class(module)
        if skill_class is None:
            raise SkillLoadError(
                str(skill_dir),
                "no BaseSkill subclass or create_skill() found",
            )

        try:
            skill = skill_class()
            logger.debug(
                "Loaded skill '%s' via class instantiation",
                skill.metadata.name,
            )
            return skill
        except Exception as e:
            raise SkillLoadError(
                str(skill_dir),
                f"skill instantiation failed: {e}",
            ) from e

    except SkillLoadError:
        raise
    except Exception as e:
        raise SkillLoadError(str(skill_dir), f"unexpected error: {e}") from e


def _find_skill_class(module) -> Optional[type]:
    """Find a BaseSkill subclass in the given module.

    Searches for classes that directly inherit from BaseSkill.
    If multiple are found, returns the first one.

    Args:
        module: The Python module to search.

    Returns:
        A BaseSkill subclass or None if not found.
    """
    for name in dir(module):
        obj = getattr(module, name)

        # Skip non-classes
        if not isinstance(obj, type):
            continue

        # Skip BaseSkill itself
        if obj is BaseSkill:
            continue

        # Check if it's a subclass of BaseSkill
        try:
            if issubclass(obj, BaseSkill):
                return obj
        except TypeError:
            # issubclass can raise TypeError for some edge cases
            continue

    return None


def discover_python_skills(
    skills_dir: Path,
    source: str = "local",
) -> list[BaseSkill]:
    """Discover and load all Python skills from a directory.

    Scans the skills directory for subdirectories containing skill.py files.

    Args:
        skills_dir: Path to the skills directory.
        source: Source of the skills ('local', 'user', 'project').

    Returns:
        List of loaded BaseSkill instances.
    """
    skills_dir = skills_dir.expanduser()

    if not skills_dir.exists():
        logger.debug("Skills directory does not exist: %s", skills_dir)
        return []

    try:
        resolved_base = skills_dir.resolve()
    except (OSError, RuntimeError):
        logger.warning("Cannot resolve skills directory: %s", skills_dir)
        return []

    skills: list[BaseSkill] = []

    for skill_dir in skills_dir.iterdir():
        # Security: Check path is safe
        if not _is_safe_path(skill_dir, resolved_base):
            logger.warning("Skipping unsafe path: %s", skill_dir)
            continue

        if not skill_dir.is_dir():
            continue

        # Only load if skill.py exists
        if not (skill_dir / "skill.py").exists():
            continue

        try:
            skill = load_python_skill(skill_dir, source)
            if skill is not None:
                skills.append(skill)
        except SkillLoadError as e:
            logger.warning("Failed to load Python skill from %s: %s", skill_dir, e)

    return skills
