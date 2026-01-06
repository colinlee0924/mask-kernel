# CLAUDE.md - AI Assistant Guide for MASK Kernel

## Project Overview

MASK (Multi-Agent Skill Kit) Kernel is a Python framework for building expertise agents with **Progressive Skill Disclosure**. It provides:

- **Progressive Disclosure**: Skills are discovered and loaded on-demand, reducing cognitive load
- **Tier-based LLM Selection**: Abstracted model selection (FAST/THINKING/PRO) across providers
- **A2A Protocol Integration**: Agent-to-Agent communication for multi-agent ecosystems
- **MCP Integration**: Connect to Model Context Protocol servers for external tools
- **Built-in Observability**: Phoenix/Langfuse tracing with A2A noise filtering

## Architecture

### Core Concepts

**Progressive Disclosure Pattern** (follows Anthropic Agent Skills Specification):

Three levels of skill information disclosure:

- **Level 1 - Metadata**: Skill name and description shown in system prompt
- **Level 2 - Instructions**: Full SKILL.md content loaded when skill is activated
- **Level 3 - Resources**: Additional files (scripts/, references/, assets/) accessed via `read_file` tool

Flow:
1. Agent starts with only loader tools visible (`use_<skill>` tools)
2. User asks to use a skill → Agent calls the loader tool
3. Loader returns `Command(update={"skills_loaded": [skill_name]})` to update state
4. LangGraph updates state → SkillMiddleware filters tools based on new state
5. Skill's capability tools become visible for subsequent model calls
6. Agent can use `read_file` tool to access Level 3 resources (scripts, references, assets)

**Model Tiers** (`src/mask/models/llm_factory.py`):
- `ModelTier.FAST`: Quick responses (Haiku, GPT-4o-mini, Gemini Flash)
- `ModelTier.THINKING`: Balanced reasoning (Sonnet, GPT-4o)
- `ModelTier.PRO`: Complex analysis (Opus, o1, Gemini Pro)

**State Scopes** (`src/mask/core/state.py`):
- `NONE`: Pure stateless
- `REQUEST`: State within single invoke() - default
- `TASK`: State across multi-agent handoffs
- `CONVERSATION`: Full session persistence

**Skill State Modes**:
- `ACCUMULATE` (default): Skills persist once loaded
- `REPLACE`: Each new skill replaces previous
- `FIFO`: Only N most recent skills kept

## Directory Structure

```
mask-kernel/
├── src/mask/
│   ├── __init__.py          # Package exports
│   ├── agent/                # Agent implementations
│   │   ├── base_agent.py     # BaseAgent abstract class, SimpleAgent
│   │   ├── agent_factory.py  # create_mask_agent() factory
│   │   └── prompt_loader.py  # Load prompts from config/
│   ├── a2a/                  # A2A Protocol integration
│   │   ├── executor.py       # MaskAgentExecutor bridges MASK to A2A
│   │   └── server.py         # MaskA2AServer for exposing agents
│   ├── checkpointer/         # LangGraph checkpointers for persistence
│   │   ├── __init__.py       # Package exports
│   │   └── postgres.py       # PostgreSQL checkpointer wrapper
│   ├── cli/                  # CLI commands
│   │   ├── main.py           # Typer app entry point
│   │   ├── config.py         # Settings and configuration management
│   │   └── commands/
│   │       ├── init.py       # `mask init` project scaffolding
│   │       ├── run.py        # `mask run` interactive agent
│   │       └── skills.py     # `mask skills` management commands
│   ├── core/                 # Core abstractions
│   │   ├── skill.py          # BaseSkill, MarkdownSkill, SkillMetadata
│   │   ├── registry.py       # SkillRegistry for managing skills
│   │   ├── state.py          # SkillState, StateScope, HandoffContext
│   │   ├── events.py         # AgentEvent for structured streaming
│   │   └── exceptions.py     # Custom exceptions
│   ├── loader/               # Skill loaders
│   │   ├── skill_md_loader.py    # Parse SKILL.md files
│   │   └── python_loader.py      # Load Python skill modules
│   ├── middleware/           # Agent middleware
│   │   └── skill_middleware.py   # SkillMiddleware for Progressive Disclosure
│   ├── models/               # LLM abstraction
│   │   ├── llm_factory.py    # LLMFactory with tier-based selection
│   │   └── config.py         # Model configuration
│   ├── mcp/                  # MCP integration
│   │   ├── client.py         # MCP client wrapper
│   │   └── integration.py    # Tool conversion utilities
│   ├── observability/        # Tracing and monitoring
│   │   ├── setup.py          # setup_openinference_tracing(), etc.
│   │   └── attributes.py     # Span attribute utilities
│   ├── session/              # Session management
│   │   └── session.py        # Session class
│   ├── storage/              # Storage backends
│   │   ├── base.py           # SessionStore interface
│   │   ├── memory_store.py   # In-memory storage
│   │   ├── redis_store.py    # Redis backend
│   │   └── postgresql_store.py
│   └── tools/                # Reusable tools for agents
│       ├── __init__.py       # Package exports
│       └── filesystem.py     # read_file, list_directory tools
├── examples/
│   ├── demo_progressive_disclosure.py
│   ├── multi_turn_postgres.py    # Multi-turn with PostgreSQL
│   ├── a2a_server_postgres.py    # A2A server with persistence
│   └── skills/               # Example skills
│       ├── pdf-processing/
│       │   ├── SKILL.md
│       │   ├── scripts/      # Executable scripts (Level 3)
│       │   ├── references/   # Reference documents (Level 3)
│       │   └── assets/       # Templates, configs (Level 3)
│       └── data-analysis/
│           └── SKILL.md
├── tests/
│   ├── test_observability.py
│   └── integration/
├── pyproject.toml            # Dependencies and build config
└── README.md
```

## Development Workflow

### Setup

```bash
# Install with all dependencies
pip install -e ".[dev,phoenix,anthropic]"

# Or specific extras
pip install -e ".[anthropic]"      # Anthropic models
pip install -e ".[phoenix]"        # Phoenix observability
pip install -e ".[mcp]"            # MCP integration
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mask

# Run specific test
pytest tests/test_observability.py -v
```

### Linting and Type Checking

```bash
# Ruff for linting
ruff check src/

# Ruff for formatting
ruff format src/

# Mypy for type checking
mypy src/mask/
```

### Creating a New Agent Project

```bash
mask init my-agent
cd my-agent
pip install -e .
python -m my_agent.main
```

## Code Conventions

### Imports

Order: stdlib → third-party → local, with TYPE_CHECKING for type-only imports:

```python
import logging
from typing import TYPE_CHECKING, Optional, List

from langchain_core.tools import BaseTool
from langgraph.types import Command

from mask.core.skill import BaseSkill
from mask.core.state import SkillState

if TYPE_CHECKING:
    from mask.agent.base_agent import BaseAgent
```

### Logging

Use module-level loggers:

```python
logger = logging.getLogger(__name__)
logger.debug("Operation details: %s", value)
logger.info("Important event: %s", event)
logger.warning("Potential issue: %s", issue)
```

### Docstrings

Google-style docstrings:

```python
def function(param: str, optional: int = 10) -> Result:
    """Brief description of function.

    Longer description if needed.

    Args:
        param: Description of param.
        optional: Description of optional param.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param is invalid.

    Example:
        >>> result = function("test")
    """
```

### Type Hints

- Use type hints everywhere
- Use `Optional[T]` for nullable types
- Use `List`, `Dict` from typing for Python 3.10 compatibility
- Prefer `Annotated` for LangGraph state reducers

### Error Handling

Use custom exceptions from `mask.core.exceptions`:

```python
from mask.core.exceptions import SkillNotFoundError, SkillLoadError

if skill_name not in self._skills:
    raise SkillNotFoundError(skill_name)
```

## Key Patterns

### Creating a Skill

**Markdown Skill** (`skills/my-skill/SKILL.md`):
```markdown
---
name: my-skill
description: What this skill does
version: 1.0.0
tags: [category]
---

# My Skill Instructions

Detailed instructions for the agent...
```

**Python Skill** (`skills/my-skill/skill.py`):
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

    def get_tools(self) -> list:
        @tool
        def my_tool(param: str) -> str:
            """Tool description."""
            return f"Result: {param}"
        return [my_tool]

    def get_loader_tool(self):
        # Use default from MarkdownSkill or implement custom
        from langchain_core.tools import StructuredTool
        from langgraph.types import Command
        from langchain_core.messages import ToolMessage

        skill_name = self.metadata.name
        skill_instance = self

        def loader(runtime=None):
            instructions = skill_instance.get_instructions()
            tool_call_id = getattr(runtime, "tool_call_id", "unknown")
            return Command(
                update={
                    "messages": [ToolMessage(content=instructions, tool_call_id=tool_call_id)],
                    "skills_loaded": [skill_name],
                }
            )

        return StructuredTool.from_function(
            func=loader,
            name=f"use_{skill_name.replace('-', '_')}",
            description=f"Activate {skill_name}. {self.metadata.description}",
        )
```

### Using the LLM Factory

```python
from mask.models import LLMFactory, ModelTier

factory = LLMFactory()  # Defaults to anthropic

# Get model by tier
fast_model = factory.get_model(tier=ModelTier.FAST)
thinking_model = factory.get_model(tier=ModelTier.THINKING)
pro_model = factory.get_model(tier=ModelTier.PRO)

# Override provider
openai_model = factory.get_model(tier=ModelTier.THINKING, provider="openai")
```

### Setting Up Observability

```python
from mask.observability import setup_openinference_tracing

# Phoenix (recommended for development)
setup_openinference_tracing(
    project_name="my-agent",
    endpoint="http://localhost:6006",
    filter_a2a_noise=True,  # Filters out A2A SDK traces
)

# Dual tracing (Phoenix + Langfuse)
from mask.observability import setup_dual_tracing
setup_dual_tracing(
    project_name="my-agent",
    phoenix_endpoint="http://localhost:6006",
)
```

### Creating an A2A Server

```python
from mask.a2a import MaskA2AServer
from mask.agent import create_mask_agent

agent = create_mask_agent(tier=ModelTier.THINKING)

server = MaskA2AServer(
    agent=agent,
    name="my-agent",
    description="Agent description",
)

server.run(port=10001)
```

### Multi-Agent Handoffs

```python
from mask.core.state import HandoffContext

# Parent agent creates handoff context
handoff = HandoffContext(
    initial_skills=["pdf-processing"],  # Pre-activate skills
    context_data={"task": "analyze document"},
    parent_agent="orchestrator",
    task_id="task-123",
)

# Child agent receives context
response = await child_agent.invoke(
    message="Analyze this PDF",
    handoff_context=handoff,
)
```

### Filesystem Tools (Level 3 Progressive Disclosure)

Enable agents to read skill resources beyond SKILL.md:

```python
from mask.tools import create_read_file_tool, create_filesystem_tools
from pathlib import Path

# Create read_file tool restricted to skills directory
read_file = create_read_file_tool(
    allowed_prefixes=[Path("/app/skills")]
)

# Create all filesystem tools (read_file + list_directory)
tools = create_filesystem_tools(
    allowed_prefixes=[Path("/app/skills")],
    include_list_dir=True,
)

# Add to agent
agent = create_mask_agent(additional_tools=tools)
```

The `create_mask_agent()` factory includes `read_file` by default:

```python
# Enabled by default with skills directory access
agent = create_mask_agent(
    enable_file_access=True,  # Default
    file_access_paths=[Path("/custom/skills")],  # Override paths
)

# Disable filesystem access
agent = create_mask_agent(enable_file_access=False)
```

### Multi-Turn Conversations with PostgreSQL

Use LangGraph checkpointers for conversation persistence:

```python
from mask.checkpointer import create_postgres_checkpointer, create_thread_config
from mask.agent import create_mask_agent

# Create and initialize checkpointer
checkpointer = await create_postgres_checkpointer(
    "postgresql://user:pass@localhost:5432/mask_db"
)

# Create agent
agent = create_mask_agent()
graph = agent.build_graph(checkpointer=checkpointer.checkpointer)

# Create config with thread_id for conversation tracking
config = create_thread_config("user-123-session-456")

# First turn
response1 = await graph.ainvoke(
    {"messages": [HumanMessage(content="Hello, my name is Alice")]},
    config=config,
)

# Later turn (same thread_id) - agent remembers context
response2 = await graph.ainvoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    config=config,
)
# Agent responds: "Your name is Alice"

# Cleanup
await checkpointer.close()
```

Using as async context manager:

```python
async with PostgresCheckpointer(connection_string) as checkpointer:
    graph = agent.build_graph(checkpointer=checkpointer.checkpointer)
    # Use graph...
```

### CLI Commands

**Run an interactive agent session:**

```bash
# Basic usage
mask run

# With specific model
mask run --model claude-sonnet-4-20250514

# With custom agent name and skills directory
mask run --agent my-agent --skills-dir ./my-skills

# Verbose logging
mask run --verbose
```

**Manage skills:**

```bash
# List available skills
mask skills list
mask skills list --skills-dir ./custom-skills

# Get skill information
mask skills info pdf-processing

# Create a new skill from template
mask skills create my-new-skill
mask skills create my-new-skill --skills-dir ./my-skills
```

**Initialize a new agent project:**

```bash
mask init my-agent
cd my-agent
pip install -e .
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `GOOGLE_API_KEY` | Google API key | - |
| `MASK_LLM_PROVIDER` | Default LLM provider | `anthropic` |
| `MASK_SKILLS_DIR` | Default skills directory | `./skills` |
| `MASK_CONFIG_DIR` | User config directory | `~/.mask` |
| `PHOENIX_COLLECTOR_ENDPOINT` | Phoenix endpoint | `http://localhost:6006` |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key | - |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key | - |
| `LANGFUSE_BASE_URL` | Langfuse URL | `https://cloud.langfuse.com` |
| `MASK_{TIER}_{PROVIDER}_MODEL` | Override model for tier | - |
| `POSTGRES_CONNECTION_STRING` | PostgreSQL connection URL | - |

## Common Tasks

### Adding a New Skill

1. Create directory under `skills/` (or use `mask skills create`):
   ```
   skills/new-skill/
   ├── SKILL.md       # Required: skill metadata and instructions
   ├── skill.py       # Optional: Python skill with custom tools
   ├── scripts/       # Optional: Level 3 executable scripts
   ├── references/    # Optional: Level 3 reference documents
   └── assets/        # Optional: Level 3 templates, configs
   ```

2. Register in SkillRegistry:
   ```python
   registry = SkillRegistry()
   registry.discover_from_directory(Path("skills"))
   ```

3. Using CLI to create a skill from template:
   ```bash
   mask skills create my-skill
   # Creates skills/my-skill/ with SKILL.md template and subdirectories
   ```

### Adding a New Storage Backend

1. Create class extending `SessionStore` in `src/mask/storage/`
2. Implement `get_or_create()`, `save()`, `delete()` methods
3. Add to optional dependencies in `pyproject.toml`

### Adding Observability to Custom Spans

```python
from mask.observability.attributes import set_span_io, set_span_metadata

with tracer.start_as_current_span("my-operation") as span:
    set_span_io(span, input_value="input", output_value="output")
    set_span_metadata(span, agent_name="my-agent")
```

## Testing Patterns

### Async Tests

```python
import pytest

@pytest.mark.asyncio
async def test_agent_invoke():
    agent = create_mask_agent()
    response = await agent.invoke("Hello")
    assert response
```

### Mocking LLM Responses

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_with_mock():
    with patch("mask.models.LLMFactory.get_model") as mock:
        mock.return_value = AsyncMock()
        mock.return_value.ainvoke.return_value = "mocked response"
        # Test code here
```

## Troubleshooting

### Missing Traces in Phoenix

1. Ensure `setup_openinference_tracing()` is called before creating agents
2. Check that `filter_a2a_noise=True` isn't hiding expected spans
3. Verify Phoenix is running at configured endpoint

### Skill Not Loading

1. Check SKILL.md has valid YAML frontmatter with `name` and `description`
2. Verify skill name matches directory name (warning if different)
3. Check logs for `SkillLoadError` or `SkillMetadataError`

### Progressive Disclosure Not Working

1. Ensure using `SimpleAgent` or custom agent with `SkillMiddleware`
2. Verify loader tool returns `Command` with `skills_loaded` update
3. Check that `state_schema` includes `SkillState` or equivalent

### Filesystem Access Denied

1. Check `allowed_prefixes` configuration in `create_read_file_tool()`
2. Ensure paths are absolute (no relative paths with `..` or `~`)
3. Verify the skills directory exists and is readable
4. Check logs for security validation errors

### PostgreSQL Checkpointer Issues

1. Ensure dependencies installed: `pip install mask-kernel[checkpointer-postgres]`
2. Verify PostgreSQL connection string format: `postgresql://user:pass@host:port/db`
3. Check that `setup()` or `setup_sync()` was called before using checkpointer
4. For connection pool issues, verify `pool_size` parameter
5. Use `auto_setup=True` (default) to automatically create required tables

### Multi-Turn Conversation Not Persisting

1. Verify using same `thread_id` in `create_thread_config()` across invocations
2. Check PostgreSQL checkpointer is properly initialized
3. Ensure graph is compiled with `checkpointer=checkpointer.checkpointer`
4. Verify database connection is active (not closed between invocations)

### CLI Configuration Issues

1. Check environment variables: `MASK_SKILLS_DIR`, `MASK_CONFIG_DIR`
2. Verify API keys are set: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.
3. Use `--verbose` flag for detailed logging
4. Check `~/.mask/` directory for agent-specific configurations
