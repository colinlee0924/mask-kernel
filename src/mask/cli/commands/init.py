"""MASK init command - Create new agent project.

Usage:
    mask init my-agent
    mask init my-agent --no-stateless
"""

from pathlib import Path

import typer

# Skills README content
SKILLS_README = """# Skills Directory

Place your skill implementations here.

## Skill Structure

Each skill should be in its own directory with at least a SKILL.md file:

```
skills/
├── my-skill/
│   ├── SKILL.md      # Required: metadata and instructions
│   └── skill.py      # Optional: Python implementation
```

## SKILL.md Format

```markdown
---
name: my-skill
description: What this skill does
version: 1.0.0
tags: [category, another-tag]
---

# My Skill

Instructions for using this skill...
```

## Python Skills

For skills with custom tools, create a `skill.py`:

```python
from mask.core import BaseSkill, SkillMetadata
from langchain_core.tools import tool

class MySkill(BaseSkill):
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="my-skill",
            description="What this skill does",
        )

    def get_tools(self):
        @tool
        def my_tool(param: str) -> str:
            \"\"\"Tool description.\"\"\"
            return f"Result: {param}"
        return [my_tool]
```

See the MASK documentation for more details.
"""


def init_command(
    project_name: str = typer.Argument(..., help="Name of the project"),
    output_dir: Path = typer.Option(
        Path("."),
        "--output", "-o",
        help="Output directory",
    ),
    stateless: bool = typer.Option(
        True,
        "--stateless/--no-stateless",
        help="Whether agent is stateless by default",
    ),
    with_mcp: bool = typer.Option(
        False,
        "--with-mcp",
        help="Include MCP server configuration",
    ),
    with_a2a: bool = typer.Option(
        True,
        "--with-a2a/--no-a2a",
        help="Include A2A server setup",
    ),
) -> None:
    """Initialize a new MASK agent project.

    Creates a new directory with the project structure and configuration.
    """
    # Handle path input: extract just the final component as project name
    # e.g., "../uat/my-agent" -> project_name = "my-agent", project_dir = "../uat/my-agent"
    project_path = Path(project_name)
    actual_project_name = project_path.name  # Get just the last component

    # Normalize project name
    project_name_normalized = actual_project_name.lower().replace("_", "-")
    module_name = project_name_normalized.replace("-", "_")

    # Determine project directory
    if len(project_path.parts) > 1:
        # User provided a path, use it directly
        project_dir = project_path
    else:
        # User provided just a name, put it in output_dir
        project_dir = output_dir / project_name_normalized

    if project_dir.exists():
        typer.echo(f"Error: Directory '{project_dir}' already exists", err=True)
        raise typer.Exit(1)

    typer.echo(f"Creating MASK agent project: {project_name_normalized}")

    # Create directory structure
    project_dir.mkdir(parents=True)

    # Create subdirectories
    (project_dir / "config" / "prompts").mkdir(parents=True)
    (project_dir / "src" / module_name / "skills").mkdir(parents=True)
    (project_dir / "tests").mkdir(parents=True)

    # Template context
    context = {
        "project_name": project_name_normalized,
        "module_name": module_name,
        "stateless": stateless,
        "with_mcp": with_mcp,
        "with_a2a": with_a2a,
    }

    # Generate files
    _write_pyproject_toml(project_dir, context)
    _write_readme(project_dir, context)
    _write_env_example(project_dir, context)
    _write_agent_py(project_dir, context)
    _write_main_py(project_dir, context)
    _write_init_py(project_dir, context)
    _write_system_prompt(project_dir, context)
    _write_skills_readme(project_dir)
    _write_test_agent(project_dir, context)

    if with_mcp:
        _write_mcp_config(project_dir)

    typer.echo(f"\nProject created: {project_dir}")
    typer.echo("\nNext steps:")
    typer.echo(f"  cd {project_dir}")
    typer.echo("  pip install -e .")
    typer.echo("  # Edit config/prompts/system.md")
    typer.echo("  # Add skills to src/{module_name}/skills/")
    if with_a2a:
        typer.echo("  python -m {module_name}.main  # Start A2A server")


def _write_pyproject_toml(project_dir: Path, context: dict) -> None:
    """Write pyproject.toml."""
    # Use GitHub URL since mask-kernel is not on PyPI yet
    mask_kernel_dep = "mask-kernel[phoenix,anthropic] @ git+https://github.com/colinlee0924/mask-kernel.git"

    content = f'''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{context["project_name"]}"
version = "0.1.0"
description = "MASK agent: {context["project_name"]}"
readme = "README.md"
requires-python = ">=3.10"

dependencies = [
    "{mask_kernel_dep}",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24.0",
    "httpx>=0.27.0",
    "a2a-sdk>=0.3.0",
]

[tool.hatch.metadata]
allow-direct-references = true

[tool.hatch.build.targets.wheel]
packages = ["src/{context["module_name"]}"]
'''
    (project_dir / "pyproject.toml").write_text(content, encoding="utf-8")


def _write_readme(project_dir: Path, context: dict) -> None:
    """Write README.md."""
    content = f'''# {context["project_name"]}

A MASK agent project.

## Installation

```bash
pip install -e .
```

## Configuration

1. Copy `.env.example` to `.env` and configure your API keys
2. Edit `config/prompts/system.md` with your agent's system prompt
3. Add skills to `src/{context["module_name"]}/skills/`

## Running

```bash
python -m {context["module_name"]}.main
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
'''
    (project_dir / "README.md").write_text(content, encoding="utf-8")


def _write_env_example(project_dir: Path, context: dict) -> None:
    """Write .env.example."""
    content = '''# LLM Provider API Keys
ANTHROPIC_API_KEY=your-anthropic-key
# OPENAI_API_KEY=your-openai-key
# GOOGLE_API_KEY=your-google-key

# MASK Configuration
MASK_LLM_PROVIDER=anthropic

# Phoenix Observability (recommended - default)
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006

# Langfuse Observability (optional alternative)
# LANGFUSE_SECRET_KEY=your-langfuse-project-secret-key
# LANGFUSE_PUBLIC_KEY=your-langfuse-project-public-key
# LANGFUSE_BASE_URL=http://localhost:3001
'''
    if context["with_mcp"]:
        content += '''
# MCP Server Configuration (example for Jira)
# JIRA_URL=https://your-org.atlassian.net
# JIRA_EMAIL=your-email@example.com
# JIRA_API_TOKEN=your-jira-token
'''
    (project_dir / ".env.example").write_text(content, encoding="utf-8")


def _write_agent_py(project_dir: Path, context: dict) -> None:
    """Write agent.py."""
    content = f'''"""Agent implementation using MASK kernel."""

from pathlib import Path
from typing import Optional

from mask.agent import BaseAgent, SimpleAgent, load_prompts
from mask.core import SkillRegistry
from mask.models import LLMFactory, ModelTier


class {context["module_name"].title().replace("_", "")}Agent(SimpleAgent):
    """Custom agent implementation."""

    def __init__(
        self,
        config_dir: str = "config",
        tier: ModelTier = ModelTier.THINKING,
    ):
        """Initialize the agent.

        Args:
            config_dir: Path to configuration directory.
            tier: Model capability tier to use.
        """
        # Load prompts from config/prompts/
        prompts = load_prompts(config_dir)
        system_prompt = prompts.get("system", "You are a helpful assistant.")

        # Initialize LLM
        factory = LLMFactory()
        model = factory.get_model(tier=tier)

        # Initialize skill registry
        registry = SkillRegistry()

        # Discover skills from skills directory
        skills_dir = Path(__file__).parent / "skills"
        if skills_dir.exists():
            registry.discover_from_directory(skills_dir)

        super().__init__(
            model=model,
            skill_registry=registry,
            system_prompt=system_prompt,
            stateless={context["stateless"]},
        )


def create_agent(
    config_dir: str = "config",
    tier: ModelTier = ModelTier.THINKING,
) -> {context["module_name"].title().replace("_", "")}Agent:
    """Create and return the agent instance.

    Args:
        config_dir: Path to configuration directory.
        tier: Model capability tier.

    Returns:
        Configured agent instance.
    """
    return {context["module_name"].title().replace("_", "")}Agent(
        config_dir=config_dir,
        tier=tier,
    )
'''
    (project_dir / "src" / context["module_name"] / "agent.py").write_text(
        content, encoding="utf-8"
    )


def _write_main_py(project_dir: Path, context: dict) -> None:
    """Write main.py."""
    if context["with_a2a"]:
        content = f'''"""Main entry point for {context["project_name"]} A2A server."""

import os
from pathlib import Path

from dotenv import load_dotenv

from mask.a2a import MaskA2AServer
from mask.observability import setup_openinference_tracing

from {context["module_name"]}.agent import create_agent

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


def main():
    """Start the A2A server."""
    # Setup Phoenix tracing (filters A2A noise by default)
    setup_openinference_tracing(
        project_name="{context["project_name"]}",
        endpoint=os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"),
    )

    agent = create_agent()

    server = MaskA2AServer(
        agent=agent,
        name="{context["project_name"]}",
        description="MASK agent: {context["project_name"]}",
    )

    print(f"Starting {context["project_name"]} on port 10001...")
    server.run(port=10001)


if __name__ == "__main__":
    main()
'''
    else:
        content = f'''"""Main entry point for {context["project_name"]}."""

import asyncio

from {context["module_name"]}.agent import create_agent


async def main():
    """Run the agent interactively."""
    agent = create_agent()

    print("Agent ready. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        response = await agent.invoke(user_input)
        print(f"Agent: {{response}}")


if __name__ == "__main__":
    asyncio.run(main())
'''
    (project_dir / "src" / context["module_name"] / "main.py").write_text(
        content, encoding="utf-8"
    )


def _write_init_py(project_dir: Path, context: dict) -> None:
    """Write __init__.py."""
    content = f'''"""{context["project_name"]} - A MASK agent."""

from {context["module_name"]}.agent import create_agent

__all__ = ["create_agent"]
'''
    (project_dir / "src" / context["module_name"] / "__init__.py").write_text(
        content, encoding="utf-8"
    )


def _write_system_prompt(project_dir: Path, context: dict) -> None:
    """Write system.md prompt."""
    content = f'''# System Prompt

You are a helpful assistant powered by the {context["project_name"]} agent.

## Guidelines

- Provide clear, actionable responses
- When uncertain, ask clarifying questions
- Use available skills when appropriate

## Available Capabilities

[Add your agent's capabilities here]
'''
    (project_dir / "config" / "prompts" / "system.md").write_text(
        content, encoding="utf-8"
    )


def _write_skills_readme(project_dir: Path) -> None:
    """Write skills README."""
    # This goes in the skills subdirectory
    pass  # Skills dir will have the README from the global SKILLS_README


def _write_mcp_config(project_dir: Path) -> None:
    """Write MCP configuration."""
    content = '''{
  "mcpServers": {
    "example": {
      "command": "uvx",
      "args": ["mcp-example"],
      "env": {
        "API_KEY": "${EXAMPLE_API_KEY}"
      }
    }
  }
}
'''
    (project_dir / "config" / "mcp_servers.json").write_text(
        content, encoding="utf-8"
    )


def _write_test_agent(project_dir: Path, context: dict) -> None:
    """Write test file."""
    content = f'''"""Tests for {context["project_name"]} agent."""

import pytest

from {context["module_name"]}.agent import create_agent


def test_agent_creation():
    """Test that agent can be created."""
    # Note: This will fail without proper API keys
    # Use mocking for actual tests
    pass


@pytest.mark.asyncio
async def test_agent_invoke():
    """Test agent invocation."""
    # Note: This will fail without proper API keys
    # Use mocking for actual tests
    pass
'''
    (project_dir / "tests" / "test_agent.py").write_text(content, encoding="utf-8")
    (project_dir / "tests" / "__init__.py").write_text("", encoding="utf-8")

    # Also write A2A test script if with_a2a
    if context.get("with_a2a", True):
        _write_a2a_test_script(project_dir, context)


def _write_a2a_test_script(project_dir: Path, context: dict) -> None:
    """Write A2A integration test script."""
    content = f'''"""A2A Integration Test for {context["project_name"]}.

Usage:
    1. Start the agent server: python -m {context["module_name"]}.main
    2. Run this test: python tests/test_a2a.py

Prerequisites:
    - Agent server running on http://localhost:10001
    - Phoenix running on http://localhost:6006 (for observability)
"""

import asyncio
import logging
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest


async def main() -> None:
    """Send test messages to {context["project_name"]}."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    base_url = "http://localhost:10001"

    async with httpx.AsyncClient() as httpx_client:
        # Initialize resolver and get agent card
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )

        try:
            agent_card = await resolver.get_agent_card()
            logger.info(f"Agent card fetched: {{agent_card.name}}")
        except Exception as e:
            logger.error(f"Failed to fetch agent card: {{e}}")
            logger.error("Make sure the agent server is running!")
            raise

        # Initialize client
        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)

        # Test questions
        test_questions = [
            "What is 2 + 2?",
            "Explain what Python is in one sentence.",
            "What is the capital of France?",
        ]

        # Use same contextId for all messages in this session
        # This groups all traces under one session in Phoenix
        context_id = uuid4().hex

        logger.info("\\n" + "=" * 60)
        logger.info("{context["project_name"]} A2A Test")
        logger.info(f"Session (contextId): {{context_id}}")
        logger.info("=" * 60)

        for i, question in enumerate(test_questions, 1):
            logger.info(f"\\n--- Test {{i}}/{{len(test_questions)}} ---")
            logger.info(f"Question: {{question}}")

            # Prepare request with contextId for session grouping
            send_message_payload = {{
                "message": {{
                    "role": "user",
                    "parts": [{{"kind": "text", "text": question}}],
                    "messageId": uuid4().hex,
                    "contextId": context_id,  # Same contextId = same session
                }},
            }}

            request = SendMessageRequest(
                id=str(uuid4()), params=MessageSendParams(**send_message_payload)
            )

            try:
                response = await client.send_message(request)

                # Extract response text
                if hasattr(response, "message") and response.message:
                    result_text = response.message.parts[0].text
                elif hasattr(response, "data") and response.data:
                    result_text = str(response.data)
                else:
                    result_text = str(response.model_dump())

                logger.info(f"Response: {{result_text[:200]}}...")
                logger.info("✅ Request successful")
            except Exception as e:
                logger.error(f"❌ Request failed: {{e}}", exc_info=True)

        logger.info("\\n" + "=" * 60)
        logger.info("Test Complete!")
        logger.info("=" * 60)
        logger.info("\\nCheck Phoenix UI (http://localhost:6006):")
        logger.info(f"  - Project: {context["project_name"]}")
        logger.info("  - Trace structure: Agent → model → ChatAnthropic")
        logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
'''
    (project_dir / "tests" / "test_a2a.py").write_text(content, encoding="utf-8")
