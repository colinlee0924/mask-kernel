"""Template engine for MASK CLI project scaffolding.

This module provides a Jinja2-based template engine for rendering
project templates. Templates are stored in the templates/ directory.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)

# Default template directory (relative to this module)
DEFAULT_TEMPLATES_DIR = Path(__file__).parent / "templates"


class TemplateEngine:
    """Jinja2-based template engine for project scaffolding.

    Renders templates from a template directory, supporting:
    - Variable directory names (e.g., {{ module_name }})
    - .jinja file extension for template files
    - Preserving non-template files as-is

    Example:
        engine = TemplateEngine()
        engine.render_project(
            output_dir=Path("my-project"),
            context={
                "project_name": "my-project",
                "module_name": "my_project",
                "stateless": True,
            },
        )
    """

    def __init__(
        self,
        template_name: str = "default",
        templates_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the template engine.

        Args:
            template_name: Name of the template directory to use.
            templates_dir: Base directory containing templates.
                Defaults to the templates/ directory next to this module.
        """
        self.templates_dir = templates_dir or DEFAULT_TEMPLATES_DIR
        self.template_name = template_name
        self.template_path = self.templates_dir / template_name

        if not self.template_path.exists():
            raise ValueError(
                f"Template '{template_name}' not found at {self.template_path}"
            )

        # Create Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_path)),
            autoescape=select_autoescape(["html", "xml"]),
            keep_trailing_newline=True,
            trim_blocks=False,
            lstrip_blocks=False,
        )

        logger.debug("Initialized TemplateEngine with template: %s", template_name)

    def render_project(
        self,
        output_dir: Path,
        context: Dict[str, Any],
    ) -> None:
        """Render the entire project template to output directory.

        Args:
            output_dir: Directory to create the project in.
            context: Template context variables (project_name, module_name, etc.)
        """
        if output_dir.exists():
            raise ValueError(f"Output directory already exists: {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Walk through template directory
        for root, dirs, files in os.walk(self.template_path):
            # Calculate relative path from template root
            rel_root = Path(root).relative_to(self.template_path)

            # Render directory name (may contain {{ module_name }})
            rendered_rel_root = self._render_path(rel_root, context)
            target_dir = output_dir / rendered_rel_root

            # Create target directory
            target_dir.mkdir(parents=True, exist_ok=True)

            # Process files
            for filename in files:
                src_file = Path(root) / filename

                # Render filename (may contain {{ module_name }})
                rendered_filename = self._render_path(Path(filename), context)

                # Check if it's a template file
                if rendered_filename.suffix == ".jinja":
                    # Remove .jinja extension and render content
                    final_filename = rendered_filename.stem
                    target_file = target_dir / final_filename

                    # Get relative path for Jinja2 loader
                    template_rel_path = str(rel_root / filename)
                    self._render_template_file(template_rel_path, target_file, context)
                else:
                    # Copy non-template file as-is
                    target_file = target_dir / rendered_filename
                    shutil.copy2(src_file, target_file)

        logger.info("Rendered project to: %s", output_dir)

    def _render_path(self, path: Path, context: Dict[str, Any]) -> Path:
        """Render a path that may contain Jinja2 variables.

        Args:
            path: Path that may contain {{ variable }} patterns.
            context: Template context variables.

        Returns:
            Rendered path with variables substituted.
        """
        path_str = str(path)

        # Simple variable substitution for paths
        for key, value in context.items():
            path_str = path_str.replace("{{ " + key + " }}", str(value))
            path_str = path_str.replace("{{" + key + "}}", str(value))

        return Path(path_str)

    def _render_template_file(
        self,
        template_path: str,
        output_path: Path,
        context: Dict[str, Any],
    ) -> None:
        """Render a single template file.

        Args:
            template_path: Relative path to template file (for Jinja2 loader).
            output_path: Path to write rendered content.
            context: Template context variables.
        """
        try:
            template = self.env.get_template(template_path)
            content = template.render(**context)

            output_path.write_text(content, encoding="utf-8")
            logger.debug("Rendered template: %s -> %s", template_path, output_path)
        except Exception as e:
            logger.error("Failed to render template %s: %s", template_path, e)
            raise

    def render_string(self, template_string: str, context: Dict[str, Any]) -> str:
        """Render a template string.

        Args:
            template_string: Jinja2 template string.
            context: Template context variables.

        Returns:
            Rendered string.
        """
        template = self.env.from_string(template_string)
        return template.render(**context)

    def list_templates(self) -> list:
        """List available template names.

        Returns:
            List of template directory names.
        """
        if not self.templates_dir.exists():
            return []

        return [
            d.name
            for d in self.templates_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
