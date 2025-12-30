# MASK Kernel

**M**ulti-**A**gent **S**kill **K**it - A framework for building expertise agents with progressive skill disclosure.

## Features

- **Progressive Disclosure**: Skills are discovered and loaded on-demand
- **LLM Factory**: Tier-based model selection (fast/thinking/pro)
- **A2A Protocol**: Agent-to-Agent communication for multi-agent ecosystems
- **MCP Integration**: Connect to MCP servers for external tools
- **CLI Scaffold**: Generate new agent projects with `mask init`

## Installation

```bash
pip install mask-kernel
```

With optional dependencies:

```bash
# With Anthropic LLM support
pip install mask-kernel[anthropic]

# With MCP integration
pip install mask-kernel[mcp]

# With observability
pip install mask-kernel[observability]

# All optional dependencies
pip install mask-kernel[all]
```

## Quick Start

### Create a new agent project

```bash
mask init my-agent
cd my-agent
pip install -e .
```

### Use MASK in your code

```python
from mask.agent import BaseAgent, load_prompts
from mask.models import LLMFactory, ModelTier
from mask.core import SkillRegistry

# Initialize LLM with tier-based selection
factory = LLMFactory()
model = factory.get_model(tier=ModelTier.THINKING)

# Load prompts from config
prompts = load_prompts("config")
system_prompt = prompts.get("system", "You are a helpful assistant.")

# Create agent with skills
registry = SkillRegistry()
agent = create_mask_agent(
    model=model,
    skill_registry=registry,
    system_prompt=system_prompt,
)
```

## Documentation

See the [full documentation](docs/) for more details.

## License

MIT
