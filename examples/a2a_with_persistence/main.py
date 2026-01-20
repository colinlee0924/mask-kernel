#!/usr/bin/env python3
"""Example: A2A Server with PostgreSQL Persistence.

This example demonstrates how to create an A2A server with full persistence:
- LangGraph checkpointing for conversation state
- DatabaseTaskStore for A2A task persistence
- Session history synchronization with frontends like Open WebUI

Prerequisites:
    1. PostgreSQL database running
    2. Install dependencies:
       pip install mask-kernel[anthropic,postgres-checkpointer]
       # Or for local dev with SQLite:
       pip install mask-kernel[anthropic,sqlite]

    3. Set environment variables:
       # For production (PostgreSQL):
       export ENV=production
       export DATABASE_URL=postgresql://user:pass@localhost:5432/mask_kernel
       export ANTHROPIC_API_KEY=your-api-key

       # For local dev (SQLite - default):
       export ENV=local  # or just don't set it
       export ANTHROPIC_API_KEY=your-api-key

Usage:
    python examples/a2a_with_persistence/main.py

    # Or with custom settings
    ENV=production DATABASE_URL=postgresql://... python main.py

Architecture:
    Three-layer storage:
    1. DatabaseTaskStore (A2A) - A2A task state & artifacts
    2. PostgresSaver/SqliteSaver (LangGraph) - Conversation checkpoints
    3. SessionStore (MASK) - Session metadata (optional)

    ID Mapping:
    Open WebUI chat_id = A2A context_id = LangGraph thread_id

Message Synchronization:
    The executor automatically detects Open WebUI retry/delete operations:
    - When fullHistory is provided in request metadata, the executor compares
      it with checkpoint state to detect changes
    - Retry: Rolls back to previous checkpoint, regenerates response
    - Delete: Rolls back to appropriate checkpoint
"""

import asyncio
import logging
import os

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_agent(checkpointer=None):
    """Create a simple LangGraph agent for demonstration.

    Args:
        checkpointer: Optional checkpoint saver for persistence.

    Returns:
        CompiledStateGraph instance.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import create_react_agent

    # Create model
    model = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
    )

    # Create some demo tools
    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"The weather in {city} is sunny and 22°C."

    @tool
    def search_knowledge(query: str) -> str:
        """Search the knowledge base for information."""
        return f"Knowledge base results for '{query}': This is a demo response."

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            allowed = set("0123456789+-*/.()")
            if not all(c in allowed or c.isspace() for c in expression):
                return "Invalid expression"
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"

    tools = [get_weather, search_knowledge, calculate]

    system_prompt = """You are a helpful assistant with access to various tools.

Available tools:
- get_weather: Check weather for any city
- search_knowledge: Search the knowledge base
- calculate: Evaluate mathematical expressions

Be helpful and use tools when appropriate."""

    # Use provided checkpointer or default to MemorySaver
    if checkpointer is None:
        checkpointer = MemorySaver()

    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=system_prompt,
        checkpointer=checkpointer,
    )

    return agent


def main():
    """Main entry point."""
    from mask.a2a import create_a2a_executor, create_database_task_store
    from mask.checkpoints import get_checkpointer
    from mask.observability import setup_openinference_tracing

    # Setup observability (optional but recommended)
    try:
        setup_openinference_tracing(
            project_name="mask-persistent-demo",
            filter_a2a_noise=True,
        )
        logger.info("Phoenix tracing enabled")
    except Exception as e:
        logger.warning("Failed to setup tracing: %s", e)

    # Get configuration
    host = os.environ.get("A2A_HOST", "0.0.0.0")
    port = int(os.environ.get("A2A_PORT", "10001"))
    env = os.environ.get("ENV", "local")

    logger.info("Environment: %s", env)

    # Get checkpointer based on environment (SQLite for local, PostgreSQL for production)
    checkpointer = get_checkpointer()
    if checkpointer:
        logger.info("Checkpointer created: %s", type(checkpointer).__name__)
    else:
        logger.warning("No checkpointer available, using in-memory only")

    # Create agent with checkpointer
    logger.info("Creating agent...")
    agent = create_agent(checkpointer=checkpointer)

    # Create executor
    # The executor automatically handles message sync when fullHistory is provided
    executor = create_a2a_executor(
        agent,
        server_name="mask-persistent-demo",
        stream=True,
        checkpointer=checkpointer,  # Pass checkpointer for sync support
    )

    # Create task store (use database for production, in-memory for local)
    database_url = os.environ.get("DATABASE_URL")
    if env == "production" and database_url:
        task_store = create_database_task_store(database_url)
        logger.info("Using DatabaseTaskStore for A2A tasks")
    else:
        task_store = InMemoryTaskStore()
        logger.info("Using InMemoryTaskStore for A2A tasks")

    # Build A2A server using native SDK
    agent_card = AgentCard(
        name="mask-persistent-demo",
        description="Demo agent with checkpoint persistence and message sync",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        skills=[
            AgentSkill(
                id="general",
                name="General Assistant",
                description="General purpose assistance with persistence",
                tags=["general"],
            )
        ],
        capabilities=AgentCapabilities(streaming=True),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
    )

    handler = DefaultRequestHandler(agent_executor=executor, task_store=task_store)
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)

    # Print status
    checkpointer_type = type(checkpointer).__name__ if checkpointer else "None"
    task_store_type = type(task_store).__name__
    logger.info("Starting A2A server on %s:%d", host, port)
    logger.info("Checkpointer: %s", checkpointer_type)
    logger.info("TaskStore: %s", task_store_type)
    logger.info("Agent Card URL: http://%s:%d/.well-known/agent.json", host, port)

    if env == "local":
        logger.info("Note: Using SQLite for checkpoints (local development)")
        logger.info("Set ENV=production and DATABASE_URL for PostgreSQL")

    # Run server
    uvicorn.run(
        app.build(),
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
