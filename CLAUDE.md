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

**Progressive Disclosure Pattern**:
1. Agent starts with only loader tools visible (`use_<skill>` tools)
2. User asks to use a skill → Agent calls the loader tool
3. Loader returns `Command(update={"skills_loaded": [skill_name]})` to update state
4. LangGraph updates state → SkillMiddleware filters tools based on new state
5. Skill's capability tools become visible for subsequent model calls

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
│   │   ├── helpers.py        # create_a2a_executor() helper (recommended)
│   │   ├── executor.py       # MaskAgentExecutor bridges MASK to A2A
│   │   ├── openai_compat.py  # OpenAI-compatible API wrapper for Open WebUI
│   │   └── server.py         # MaskA2AServer for exposing agents (deprecated)
│   ├── cli/                  # CLI commands
│   │   ├── main.py           # Typer app entry point
│   │   ├── template_engine.py # Jinja2-based project scaffolding
│   │   ├── templates/        # Project templates
│   │   │   └── default/      # Default template for mask init
│   │   └── commands/
│   │       └── init.py       # `mask init` project scaffolding
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
│   └── storage/              # Storage backends
│       ├── base.py           # SessionStore interface
│       ├── memory_store.py   # In-memory storage
│       ├── redis_store.py    # Redis backend
│       └── postgresql_store.py
├── examples/
│   ├── demo_progressive_disclosure.py
│   └── skills/               # Example skills
│       ├── pdf-processing/
│       │   └── SKILL.md
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

# Phoenix (recommended for development) - reads from environment variables
setup_openinference_tracing()

# Or with explicit configuration
setup_openinference_tracing(
    project_name="my-agent",
    endpoint="http://localhost:6006",
    api_key="your-phoenix-api-key",  # Optional for cloud Phoenix
    filter_a2a_noise=True,  # Filters out A2A SDK traces
)

# Dual tracing (Phoenix + Langfuse)
from mask.observability import setup_dual_tracing
setup_dual_tracing(
    project_name="my-agent",
    phoenix_endpoint="http://localhost:6006",
)
```

### Creating an A2A Server (Recommended)

Use the native A2A SDK with MASK helpers:

```python
from a2a import A2AServer
from mask.a2a.helpers import create_a2a_executor

# Create your agent (using native LangChain or MASK agent)
agent = create_my_agent()

# Create executor using MASK helper
executor = create_a2a_executor(agent, server_name="my-agent")

# Use native A2A SDK
server = A2AServer(executor)
server.run(port=10001)
```

### Creating an A2A Server (Legacy - Deprecated)

> **Note**: `MaskA2AServer` is deprecated. Use the native A2A SDK pattern above.

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

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `GOOGLE_API_KEY` | Google API key | - |
| `MASK_LLM_PROVIDER` | Default LLM provider | `anthropic` |
| `PHOENIX_COLLECTOR_ENDPOINT` | Phoenix endpoint | `http://localhost:6006` |
| `PHOENIX_PROJECT_NAME` | Phoenix project name | `mask-agent` |
| `PHOENIX_API_KEY` | Phoenix API key (for cloud) | - |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key | - |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key | - |
| `LANGFUSE_BASE_URL` | Langfuse URL | `https://cloud.langfuse.com` |
| `MASK_{TIER}_{PROVIDER}_MODEL` | Override model for tier | - |

## Common Tasks

### Adding a New Skill

1. Create directory under `skills/`:
   ```
   skills/new-skill/
   ├── SKILL.md       # Required for markdown skills
   └── skill.py       # Optional for Python skills
   ```

2. Register in SkillRegistry:
   ```python
   registry = SkillRegistry()
   registry.discover_from_directory(Path("skills"))
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

---

## Critical Implementation Requirements

> **IMPORTANT**: These are strict requirements discovered during implementation testing. AI assistants MUST follow these patterns to avoid common pitfalls.

### 1. LangChain Agent API

**Use LangChain 1.x `create_agent`, NOT LangGraph's `create_react_agent`:**

```python
# CORRECT - LangChain 1.x API
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=prompt,
    middleware=[middleware],
)

# WRONG - Don't use LangGraph directly for basic agents
from langgraph.prebuilt import create_react_agent  # Avoid this
```

Reference: [LangChain create_agent documentation](https://docs.langchain.com/oss/python/langchain/overview)

### 2. SkillMiddleware MUST Receive `additional_tools`

**Without this, tools are filtered out and the agent won't use them:**

```python
# CORRECT - Pass core tools as additional_tools
core_tools = [*get_custom_tools(), *mcp_tools]
all_tools = [*registry.get_all_tools(), *core_tools]

middleware = SkillMiddleware(registry, additional_tools=core_tools)

agent = create_agent(model=model, tools=all_tools, middleware=[middleware])

# WRONG - Tools will be filtered out!
middleware = SkillMiddleware(registry)  # Missing additional_tools
```

### 3. Avoid Duplicate Tool Names

**Separate core_tools from skill tools to prevent "Tool names must be unique" error:**

```python
# CORRECT - Separate core tools from skill tools
core_tools = [*get_custom_tools(), *mcp_tools]  # Always visible
skill_tools = registry.get_all_tools()          # Controlled by middleware
all_tools = [*skill_tools, *core_tools]

# Pass ONLY core_tools to middleware (skill tools come from registry)
middleware = SkillMiddleware(registry, additional_tools=core_tools)

# WRONG - Passing all tools causes duplicates
middleware = SkillMiddleware(registry, additional_tools=all_tools)  # Duplicates!
```

### 4. A2A SDK Correct Imports

**Use the correct import paths from a2a-sdk:**

```python
# CORRECT imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

# WRONG - These don't exist
from a2a import A2AServer  # Wrong!
from a2a.server import A2AServer  # Wrong!
```

### 5. AgentSkill Requires `tags` Field

```python
# CORRECT - Include tags
AgentSkill(
    id="general",
    name="General Assistant",
    description="General purpose assistance",
    tags=["general"],  # Required!
)

# WRONG - Missing tags causes validation error
AgentSkill(id="general", name="General", description="...")  # Missing tags!
```

### 6. Async/Sync Separation for Uvicorn

**MCP tools require async loading, but uvicorn.run() is sync:**

```python
# CORRECT - Async setup before sync uvicorn.run
async def setup_agent():
    mcp_tools = await load_mcp_tools_from_config(config_path)  # Async
    return await create_agent(...)

def main():
    agent = asyncio.run(setup_agent())  # Run async setup
    executor = create_a2a_executor(agent)
    uvicorn.run(app.build(), host="0.0.0.0", port=10001)  # Sync

# WRONG - Can't await inside sync function
def main():
    mcp_tools = await load_mcp_tools(...)  # SyntaxError!
```

### 7. Skill Names Use Hyphens

**Skill names must use hyphens, not underscores:**

```python
# CORRECT
SkillMetadata(name="data-analysis", description="...")

# WRONG - Underscores cause loader tool name issues
SkillMetadata(name="data_analysis", description="...")  # use_data_analysis conflicts
```

### 8. Streaming Artifact ID Reuse

**For append operations, reuse the same artifact_id:**

```python
# CORRECT - Reuse artifact_id for append
artifact_id = str(uuid.uuid4())  # Generate once

# First chunk
artifact = _create_text_artifact(artifact_id, "response", chunk)
await event_queue.enqueue_event(TaskArtifactUpdateEvent(artifact=artifact, append=False))

# Subsequent chunks - SAME artifact_id
artifact = _create_text_artifact(artifact_id, "response", next_chunk)
await event_queue.enqueue_event(TaskArtifactUpdateEvent(artifact=artifact, append=True))

# WRONG - new_text_artifact() generates new ID each time
from a2a.utils import new_text_artifact
artifact = new_text_artifact("response", chunk)  # Different ID each time!
```

### 9. Environment Variable Patterns

**Use consistent naming for server configuration:**

```bash
# A2A Server
A2A_HOST=0.0.0.0
A2A_PORT=10001

# OpenAI-Compatible Wrapper
OPENAI_COMPAT_HOST=0.0.0.0
OPENAI_COMPAT_PORT=11434

# Phoenix Observability
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
PHOENIX_PROJECT_NAME=my-agent
PHOENIX_API_KEY=optional-for-cloud
```

### 10. Open WebUI Docker Networking

**When Open WebUI runs in Docker, use `host.docker.internal`:**

```
# Open WebUI in Docker connecting to host services
URL: http://host.docker.internal:11434/v1

# Open WebUI on host connecting to local services
URL: http://localhost:11434/v1
```

---

## OpenAI-Compatible Wrapper

For testing agents with Open WebUI or other OpenAI-compatible clients:

### Architecture

```
Open WebUI → OpenAI Wrapper (:11434) → A2A Agent (:10001)
            (mask.a2a.openai_compat)    (a2a-sdk)
```

### Usage

```python
from mask.a2a import run_openai_compat_server

# Start wrapper (requires A2A agent running on :10001)
run_openai_compat_server(
    a2a_base_url="http://localhost:10001",
    model_name="my-agent",
    port=11434,
)
```

### Generated Project Entry Points

```bash
# Start A2A agent
python -m my_agent.main

# Start OpenAI wrapper (separate terminal)
python -m my_agent.main_openai
```

## Tips for 啟動 virtual environment
在 "/Users/lizonglin/colin_ws/workspaces/ws_agentic_system/PROJECT_A/uat_v2" 目錄下
```bash
source ~/.venv/bin/activate
uv pip install -e "/Users/lizonglin/colin_ws/workspaces/ws_agentic_system/PROJECT_A/mask-kernel"
```
我們一律都要用 uv venv 建立 virtual environment 並使用 uv pip install 安裝套件 啟動 python 應用程式 也一定要 cd 到對應目錄下 source .venv/bin/activate 再執行