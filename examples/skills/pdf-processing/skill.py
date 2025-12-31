"""PDF Processing skill implementation.

This is an example skill showing how to create Python-based skills
with custom tools.

Note: This is a demo implementation. For production use, you would
integrate with actual PDF libraries like PyMuPDF or pdfplumber.
"""

from pathlib import Path
from typing import List, Optional

from langchain_core.tools import BaseTool, tool

from mask.core.skill import BaseSkill, SkillMetadata


class PDFProcessingSkill(BaseSkill):
    """Skill for processing PDF documents."""

    def __init__(self) -> None:
        """Initialize the PDF processing skill."""
        self._metadata = SkillMetadata(
            name="pdf-processing",
            description="Extract and analyze content from PDF documents",
            version="1.0.0",
            tags=["document", "pdf", "extraction"],
            source="local",
        )

        # Load instructions from SKILL.md
        skill_md_path = Path(__file__).parent / "SKILL.md"
        if skill_md_path.exists():
            content = skill_md_path.read_text(encoding="utf-8")
            # Skip YAML frontmatter
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    self._instructions = parts[2].strip()
                else:
                    self._instructions = content
            else:
                self._instructions = content
        else:
            self._instructions = "PDF Processing skill for document analysis."

    @property
    def metadata(self) -> SkillMetadata:
        """Return skill metadata."""
        return self._metadata

    def get_tools(self) -> List[BaseTool]:
        """Return the skill's tools."""
        return [
            self._create_extract_text_tool(),
            self._create_extract_tables_tool(),
            self._create_get_page_count_tool(),
        ]

    def get_instructions(self) -> str:
        """Return skill instructions."""
        return self._instructions

    def _create_extract_text_tool(self) -> BaseTool:
        """Create the text extraction tool."""

        @tool
        def pdf_extract_text(file_path: str, page_numbers: Optional[str] = None) -> str:
            """Extract text content from a PDF file.

            Args:
                file_path: Path to the PDF file.
                page_numbers: Optional comma-separated page numbers (e.g., "1,2,3").
                             If not provided, extracts from all pages.

            Returns:
                Extracted text content.
            """
            # Demo implementation - returns simulated content
            path = Path(file_path)
            if not path.suffix.lower() == ".pdf":
                return f"Error: {file_path} is not a PDF file"

            # In production, you would use PyMuPDF, pdfplumber, etc.
            pages_info = f" (pages: {page_numbers})" if page_numbers else ""
            return (
                f"[Demo] Extracted text from {path.name}{pages_info}:\n\n"
                "This is simulated PDF content. In production, integrate with "
                "PyMuPDF (fitz) or pdfplumber for actual PDF text extraction."
            )

        return pdf_extract_text

    def _create_extract_tables_tool(self) -> BaseTool:
        """Create the table extraction tool."""

        @tool
        def pdf_extract_tables(file_path: str, output_format: str = "markdown") -> str:
            """Extract tables from a PDF file.

            Args:
                file_path: Path to the PDF file.
                output_format: Output format ('markdown' or 'json').

            Returns:
                Extracted tables in the specified format.
            """
            path = Path(file_path)
            if not path.suffix.lower() == ".pdf":
                return f"Error: {file_path} is not a PDF file"

            # Demo implementation
            if output_format == "json":
                return (
                    '[{"headers": ["Column A", "Column B"], '
                    '"rows": [["Value 1", "Value 2"]]}]'
                )
            else:
                return (
                    f"[Demo] Tables from {path.name}:\n\n"
                    "| Column A | Column B |\n"
                    "|----------|----------|\n"
                    "| Value 1  | Value 2  |\n"
                )

        return pdf_extract_tables

    def _create_get_page_count_tool(self) -> BaseTool:
        """Create the page count tool."""

        @tool
        def pdf_get_page_count(file_path: str) -> str:
            """Get the number of pages in a PDF file.

            Args:
                file_path: Path to the PDF file.

            Returns:
                Number of pages.
            """
            path = Path(file_path)
            if not path.suffix.lower() == ".pdf":
                return f"Error: {file_path} is not a PDF file"

            # Demo implementation
            return f"[Demo] {path.name} has 10 pages (simulated)"

        return pdf_get_page_count


def create_skill() -> PDFProcessingSkill:
    """Factory function to create the skill instance."""
    return PDFProcessingSkill()
