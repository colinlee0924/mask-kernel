# MASK Kernel

**M**ulti-**A**gent **S**kill **K**it - A framework for building expertise agents with progressive skill disclosure.

## Philosophy

MASK Kernel is designed to **enhance, not replace** native SDKs. Developers use:

- **Native LangChain** `create_agent` API for agent creation
- **Native A2A SDK** for Agent-to-Agent protocol
- **MASK helpers** for Progressive Disclosure, observability, and MCP integration

## Features

- **Progressive Disclosure**: Skills are discovered and loaded on-demand via `SkillMiddleware`
- **LLM Factory**: Tier-based model selection (FAST/THINKING/PRO) across providers
- **A2A Protocol**: Native A2A SDK integration with `create_a2a_executor()` helper
- **MCP Integration**: Load MCP tools with `load_mcp_tools_from_config()`
- **Real-time Streaming**: Default `stream=True` for Open WebUI integration
- **OpenAI-Compatible API**: Wrapper for A2A agents to work with Open WebUI
- **CLI Scaffold**: Generate new agent projects with `mask init`
- **Built-in Observability**: Phoenix/Langfuse tracing with A2A noise filtering

## Installation

```bash
# From GitHub (recommended)
pip install "mask-kernel[phoenix,anthropic] @ git+https://github.com/colinlee0924/mask-kernel.git"
```

With optional dependencies:

```bash
# With Anthropic LLM support
pip install "mask-kernel[anthropic] @ git+https://github.com/colinlee0924/mask-kernel.git"

# With Phoenix observability (recommended)
pip install "mask-kernel[phoenix] @ git+https://github.com/colinlee0924/mask-kernel.git"

# With MCP integration
pip install "mask-kernel[mcp] @ git+https://github.com/colinlee0924/mask-kernel.git"

# All optional dependencies
pip install "mask-kernel[all] @ git+https://github.com/colinlee0924/mask-kernel.git"
```

## Quick Start

### 1. Create a new agent project

```bash
mask init my-agent
cd my-agent
```

### 2. Setup environment

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 4. Start the agent server

```bash
python -m my_agent.main
```

### 5. Test the agent

```bash
curl -X POST http://localhost:10001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "id": "1",
    "params": {
      "message": {
        "messageId": "msg-001",
        "role": "user",
        "parts": [{"text": "Hello!"}]
      }
    }
  }'
```

## Architecture

### Agent Creation (using LangChain 1.x `create_agent`)

MASK uses [LangChain's `create_agent` API](https://docs.langchain.com/oss/python/langchain/overview) (not LangGraph's `create_react_agent`). This provides a simpler, more intuitive interface built on top of LangGraph:

```python
# src/my_agent/agent.py
from langchain.agents import create_agent
from mask.core import SkillRegistry
from mask.middleware import SkillMiddleware
from mask.models import LLMFactory, ModelTier
from mask.mcp import load_mcp_tools_from_config

async def create_agent_instance():
    model = LLMFactory().get_model(tier=ModelTier.THINKING)

    # Setup skills for Progressive Disclosure
    registry = SkillRegistry()
    registry.discover_from_directory(Path("skills"))

    # Load MCP tools
    mcp_tools = await load_mcp_tools_from_config(Path("config/mcp_servers.json"))

    tools = [
        *registry.get_all_tools(),
        *get_custom_tools(),
        *mcp_tools,
    ]

    # CRITICAL: Pass tools as additional_tools for visibility
    middleware = SkillMiddleware(registry, additional_tools=tools)

    # Returns CompiledStateGraph - pass directly to create_a2a_executor()
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=load_system_prompt(),
        middleware=[middleware],
    )
```

### A2A Server (using native A2A SDK + MASK helper)

```python
# src/my_agent/main.py
import asyncio
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from mask.a2a import create_a2a_executor
from mask.observability import setup_openinference_tracing

def main():
    setup_openinference_tracing(project_name="my-agent")

    # Create agent (async for MCP tools)
    agent = asyncio.run(create_agent_instance())

    # MASK helper - wraps CompiledStateGraph for A2A
    executor = create_a2a_executor(agent, server_name="my-agent", stream=True)

    # Native A2A SDK
    agent_card = AgentCard(
        name="my-agent",
        description="My awesome agent",
        url="http://localhost:10001/",
        version="1.0.0",
        skills=[AgentSkill(id="general", name="General", description="...", tags=["general"])],
        capabilities=AgentCapabilities(streaming=True),
    )

    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)

    uvicorn.run(app.build(), host="0.0.0.0", port=10001)
```

## Generated Project Structure

```
my-agent/
├── .env.example              # Environment template
├── pyproject.toml            # Dependencies
├── README.md
├── src/my_agent/
│   ├── __init__.py
│   ├── agent.py              # Agent creation (LangChain + SkillMiddleware)
│   ├── main.py               # A2A server entry point
│   ├── main_openai.py        # OpenAI-compatible wrapper for Open WebUI
│   ├── prompts/
│   │   └── system.md         # System prompt
│   ├── skills/               # Progressive Disclosure skills
│   ├── tools/
│   │   ├── __init__.py
│   │   └── example.py        # Custom LangChain tools
│   └── config/
│       └── mcp_servers.json  # MCP server configuration
└── tests/
    ├── __init__.py
    └── test_agent.py
```

## Open WebUI Integration

MASK provides an OpenAI-compatible wrapper to test your A2A agent with Open WebUI:

### Architecture

```
Open WebUI → OpenAI Wrapper (:11434) → A2A Agent (:10001)
```

### Usage

```bash
# Terminal 1: Start A2A agent
python -m my_agent.main

# Terminal 2: Start OpenAI wrapper
python -m my_agent.main_openai
```

### Open WebUI Configuration

1. Settings → Connections → OpenAI API
2. Click "+" to add connection:
   - URL: `http://localhost:11434/v1` (or `http://host.docker.internal:11434/v1` if Open WebUI runs in Docker)
   - API Key: `sk-dummy` (any value)
3. Select your agent model in the chat

### Programmatic Usage

```python
from mask.a2a import create_openai_compat_app, run_openai_compat_server

# Option 1: Get FastAPI app for custom setup
app = create_openai_compat_app(
    a2a_base_url="http://localhost:10001",
    model_name="my-agent",
)

# Option 2: Run server directly
run_openai_compat_server(
    a2a_base_url="http://localhost:10001",
    model_name="my-agent",
    port=11434,
)
```

## Key Concepts

### Progressive Disclosure

Skills are loaded on-demand to reduce cognitive load:

1. Agent starts with only `use_<skill>` loader tools visible
2. User asks for a skill → Agent calls the loader
3. Loader returns `Command(update={"skills_loaded": [skill_name]})`
4. `SkillMiddleware` filters tools based on new state
5. Skill's capability tools become visible

### Model Tiers

```python
from mask.models import LLMFactory, ModelTier

factory = LLMFactory()
fast_model = factory.get_model(tier=ModelTier.FAST)      # Haiku, GPT-4o-mini
thinking_model = factory.get_model(tier=ModelTier.THINKING)  # Sonnet, GPT-4o
pro_model = factory.get_model(tier=ModelTier.PRO)        # Opus, o1
```

### MCP Integration

```python
from mask.mcp import load_mcp_tools_from_config

# config/mcp_servers.json
{
  "mcpServers": {
    "my-server": {
      "command": "npx",
      "args": ["-y", "@example/mcp-server"]
    }
  }
}

# Load as LangChain tools
tools = await load_mcp_tools_from_config(Path("config/mcp_servers.json"))
```

## Observability

Phoenix tracing with A2A noise filtering:

```python
from mask.observability import setup_openinference_tracing

setup_openinference_tracing(
    project_name="my-agent",
    endpoint="http://localhost:6006",
    api_key=os.environ.get("PHOENIX_API_KEY"),  # Optional for cloud Phoenix
    filter_a2a_noise=True,
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
| `PHOENIX_PROJECT_NAME` | Phoenix project name | - |
| `PHOENIX_API_KEY` | Phoenix API key (cloud) | - |

## What MASK Provides vs Native SDKs

| Component | MASK Provides | Native SDK |
|-----------|---------------|------------|
| Agent Creation | `SkillMiddleware` | LangChain `create_agent` |
| A2A Integration | `create_a2a_executor()` | A2A SDK classes |
| MCP Tools | `load_mcp_tools_from_config()` | - |
| Model Selection | `LLMFactory` with tiers | Provider-specific |
| Observability | `setup_openinference_tracing()` | OpenTelemetry |
| Project Scaffold | `mask init` | - |

## License

MIT
