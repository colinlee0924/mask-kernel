"""Filesystem tools for MASK agents.

This module provides filesystem access tools for Progressive Disclosure Level 3,
allowing agents to read skill resources (scripts/, references/, assets/).

Security:
- Path traversal prevention (no .. or ~ allowed)
- Optional prefix restrictions to limit access to specific directories
- Safe file reading with proper error handling

Usage:
    from mask.tools import create_read_file_tool, create_filesystem_tools

    # Create read_file tool restricted to skills directory
    read_file = create_read_file_tool(
        allowed_prefixes=[Path("/app/skills")]
    )

    # Create all filesystem tools
    tools = create_filesystem_tools(
        allowed_prefixes=[Path("/app/skills")],
        include_list_dir=True,
    )
"""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.tools import BaseTool, tool

logger = logging.getLogger(__name__)

# Constants
MAX_LINE_LENGTH = 2000
DEFAULT_READ_LIMIT = 500
LINE_NUMBER_WIDTH = 6

# Empty file warning
EMPTY_CONTENT_WARNING = "<system-reminder>Warning: File exists but has empty contents</system-reminder>"


def _validate_path(
    file_path: str,
    allowed_prefixes: Optional[List[Path]] = None,
) -> tuple[bool, str, Optional[Path]]:
    """Validate file path for security.

    Args:
        file_path: The path to validate.
        allowed_prefixes: Optional list of allowed directory prefixes.

    Returns:
        Tuple of (is_valid, error_message, resolved_path).
    """
    # Check for path traversal attempts
    if ".." in file_path:
        return False, "Error: Path traversal (..) not allowed", None

    if file_path.startswith("~"):
        return False, "Error: Home directory expansion (~) not allowed", None

    # Resolve the path
    try:
        path = Path(file_path).resolve()
    except (OSError, RuntimeError) as e:
        return False, f"Error: Invalid path - {e}", None

    # Check allowed prefixes
    if allowed_prefixes:
        allowed = any(
            path.is_relative_to(prefix.resolve())
            for prefix in allowed_prefixes
        )
        if not allowed:
            allowed_str = ", ".join(str(p) for p in allowed_prefixes)
            return False, f"Error: Access denied. Allowed directories: {allowed_str}", None

    return True, "", path


def _format_content_with_line_numbers(
    lines: List[str],
    start_line: int = 1,
) -> str:
    """Format file content with line numbers (cat -n style).

    Long lines are truncated with an indicator.

    Args:
        lines: List of line strings.
        start_line: Starting line number.

    Returns:
        Formatted content with line numbers.
    """
    result = []

    for i, line in enumerate(lines, start=start_line):
        # Truncate long lines
        if len(line) > MAX_LINE_LENGTH:
            line = line[:MAX_LINE_LENGTH] + "..."

        result.append(f"{i:{LINE_NUMBER_WIDTH}d}\t{line}")

    return "\n".join(result)


def create_read_file_tool(
    allowed_prefixes: Optional[List[Path]] = None,
    max_lines: int = DEFAULT_READ_LIMIT,
    tool_name: str = "read_file",
) -> BaseTool:
    """Create a read_file tool for accessing skill resources.

    This tool enables Progressive Disclosure Level 3 by allowing agents
    to read additional files from skill directories (scripts/, references/, assets/).

    Args:
        allowed_prefixes: Optional list of allowed directory prefixes.
            If not set, all readable paths are allowed.
        max_lines: Default maximum lines to read.
        tool_name: Name for the tool.

    Returns:
        Configured read_file tool.

    Example:
        # Restrict to skills directory
        read_file = create_read_file_tool(
            allowed_prefixes=[Path("/app/skills")]
        )

        # Add to agent
        agent = create_mask_agent(
            additional_tools=[read_file]
        )
    """

    @tool(name=tool_name)
    def read_file(
        file_path: str,
        offset: int = 0,
        limit: int = max_lines,
    ) -> str:
        """Read a file from the filesystem.

        Use this tool to read skill resources like SKILL.md, scripts,
        reference documents, or configuration files.

        Args:
            file_path: Absolute path to the file.
            offset: Line number to start reading from (0-indexed).
            limit: Maximum number of lines to read (default: 500).

        Returns:
            File content with line numbers, or error message.

        Example:
            # Read a skill's SKILL.md
            read_file("/app/skills/pdf-processing/SKILL.md")

            # Read a reference document
            read_file("/app/skills/pdf-processing/references/formats.md")

            # Read with pagination
            read_file("/app/skills/large-skill/SKILL.md", offset=100, limit=50)
        """
        # Validate path
        is_valid, error, resolved_path = _validate_path(file_path, allowed_prefixes)
        if not is_valid:
            return error

        # Check file exists
        if not resolved_path.exists():
            return f"Error: File '{file_path}' not found"

        if not resolved_path.is_file():
            return f"Error: '{file_path}' is not a file"

        try:
            content = resolved_path.read_text(encoding="utf-8")

            # Check for empty content
            if not content.strip():
                return EMPTY_CONTENT_WARNING

            lines = content.splitlines()

            # Validate offset
            if offset >= len(lines):
                return f"Error: Offset {offset} exceeds file length ({len(lines)} lines)"

            # Apply offset and limit
            end = min(offset + limit, len(lines))
            selected = lines[offset:end]

            # Format with line numbers
            formatted = _format_content_with_line_numbers(selected, start_line=offset + 1)

            # Add pagination hint if truncated
            if end < len(lines):
                remaining = len(lines) - end
                formatted += f"\n\n[... {remaining} more lines. Use offset={end} to continue reading]"

            return formatted

        except UnicodeDecodeError:
            return f"Error: Cannot read '{file_path}' - not a text file"
        except PermissionError:
            return f"Error: Permission denied reading '{file_path}'"
        except Exception as e:
            logger.warning("Error reading file %s: %s", file_path, e)
            return f"Error reading file: {e}"

    return read_file


def create_list_directory_tool(
    allowed_prefixes: Optional[List[Path]] = None,
    tool_name: str = "list_directory",
) -> BaseTool:
    """Create a list_directory tool for exploring skill directories.

    Args:
        allowed_prefixes: Optional list of allowed directory prefixes.
        tool_name: Name for the tool.

    Returns:
        Configured list_directory tool.
    """

    @tool(name=tool_name)
    def list_directory(directory_path: str) -> str:
        """List contents of a directory.

        Use this tool to explore skill directories and find available
        resources like scripts, reference documents, or templates.

        Args:
            directory_path: Absolute path to the directory.

        Returns:
            Directory listing showing files and subdirectories.

        Example:
            # List a skill's directory
            list_directory("/app/skills/pdf-processing")

            # List scripts subdirectory
            list_directory("/app/skills/pdf-processing/scripts")
        """
        # Validate path
        is_valid, error, resolved_path = _validate_path(directory_path, allowed_prefixes)
        if not is_valid:
            return error

        if not resolved_path.exists():
            return f"Error: Directory '{directory_path}' not found"

        if not resolved_path.is_dir():
            return f"Error: '{directory_path}' is not a directory"

        try:
            entries = []

            for entry in sorted(resolved_path.iterdir()):
                # Skip hidden files
                if entry.name.startswith("."):
                    continue

                if entry.is_dir():
                    entries.append(f"  {entry.name}/")
                else:
                    # Show file size
                    size = entry.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f}KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f}MB"
                    entries.append(f"  {entry.name} ({size_str})")

            if not entries:
                return f"Directory '{directory_path}' is empty"

            result = f"Contents of {directory_path}:\n\n"
            result += "\n".join(entries)
            return result

        except PermissionError:
            return f"Error: Permission denied accessing '{directory_path}'"
        except Exception as e:
            logger.warning("Error listing directory %s: %s", directory_path, e)
            return f"Error listing directory: {e}"

    return list_directory


def create_filesystem_tools(
    allowed_prefixes: Optional[List[Path]] = None,
    include_list_dir: bool = True,
    max_lines: int = DEFAULT_READ_LIMIT,
) -> List[BaseTool]:
    """Create a set of filesystem tools for skill resource access.

    This is a convenience function that creates commonly used filesystem
    tools with consistent security settings.

    Args:
        allowed_prefixes: Optional list of allowed directory prefixes.
        include_list_dir: Whether to include list_directory tool.
        max_lines: Default maximum lines for read_file.

    Returns:
        List of configured filesystem tools.

    Example:
        # Create tools for a skills-only agent
        tools = create_filesystem_tools(
            allowed_prefixes=[Path("/app/skills")],
            include_list_dir=True,
        )

        agent = create_mask_agent(additional_tools=tools)
    """
    tools = [
        create_read_file_tool(
            allowed_prefixes=allowed_prefixes,
            max_lines=max_lines,
        )
    ]

    if include_list_dir:
        tools.append(
            create_list_directory_tool(allowed_prefixes=allowed_prefixes)
        )

    return tools


def create_skill_resource_tool(
    skill_dir: Path,
    skill_name: str,
) -> BaseTool:
    """Create a tool for reading resources from a specific skill.

    This is a more restrictive version that only allows reading from
    a single skill's directory.

    Args:
        skill_dir: Path to the skill directory.
        skill_name: Name of the skill (for tool naming).

    Returns:
        Configured resource reading tool.
    """
    safe_name = skill_name.replace("-", "_")

    @tool(name=f"read_{safe_name}_resource")
    def read_skill_resource(relative_path: str) -> str:
        """Read a resource file from this skill's directory.

        Args:
            relative_path: Path relative to the skill directory.
                Examples: "references/api.md", "scripts/helper.py"

        Returns:
            File content or error message.
        """
        # Build full path
        full_path = skill_dir / relative_path

        # Security: ensure path stays within skill directory
        try:
            resolved = full_path.resolve()
            if not resolved.is_relative_to(skill_dir.resolve()):
                return "Error: Path outside skill directory"
        except (OSError, RuntimeError):
            return "Error: Invalid path"

        if not resolved.exists():
            return f"Error: Resource '{relative_path}' not found"

        if not resolved.is_file():
            return f"Error: '{relative_path}' is not a file"

        try:
            content = resolved.read_text(encoding="utf-8")
            lines = content.splitlines()
            return _format_content_with_line_numbers(lines)
        except Exception as e:
            return f"Error reading resource: {e}"

    read_skill_resource.__doc__ = f"""Read a resource file from the {skill_name} skill directory.

    Available subdirectories:
    - scripts/: Executable scripts
    - references/: Documentation and reference materials
    - assets/: Templates, configurations, and other assets

    Args:
        relative_path: Path relative to the skill directory.
            Examples: "references/api.md", "scripts/helper.py"

    Returns:
        File content with line numbers, or error message.
    """

    return read_skill_resource
