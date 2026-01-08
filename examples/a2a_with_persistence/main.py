#!/usr/bin/env python3
"""Example: A2A Server with PostgreSQL Persistence.

This example demonstrates how to create an A2A server with full persistence:
- LangGraph checkpointing for conversation state
- DatabaseTaskStore for A2A task persistence
- Session history synchronization with frontends like Open WebUI

Prerequisites:
    1. PostgreSQL database running
    2. Install dependencies:
       pip install mask-kernel[anthropic]
       pip install langgraph-checkpoint-postgres
       pip install psycopg[binary]

    3. Set environment variables:
       export DATABASE_URL=postgresql://user:pass@localhost:5432/mask_kernel
       export ANTHROPIC_API_KEY=your-api-key

Usage:
    python examples/a2a_with_persistence/main.py

    # Or with custom settings
    DATABASE_URL=postgresql://... A2A_PORT=10001 python main.py

Architecture:
    Three-layer storage:
    1. DatabaseTaskStore (A2A) - A2A task state & artifacts
    2. PostgresSaver (LangGraph) - Conversation checkpoints
    3. SessionStore (MASK) - Session metadata (optional)

    ID Mapping:
    Open WebUI chat_id = A2A context_id = LangGraph thread_id
"""

import asyncio
import logging
import os
import sys
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """Get database URL from environment."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        logger.warning(
            "DATABASE_URL not set. Using default: postgresql://postgres:postgres@localhost:5432/mask_kernel"
        )
        url = "postgresql://postgres:postgres@localhost:5432/mask_kernel"
    return url


def create_agent():
    """Create a simple LangGraph agent for demonstration.

    Returns:
        CompiledStateGraph instance.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.tools import tool
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
        # Demo implementation
        return f"The weather in {city} is sunny and 22°C."

    @tool
    def search_knowledge(query: str) -> str:
        """Search the knowledge base for information."""
        # Demo implementation
        return f"Knowledge base results for '{query}': This is a demo response."

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            # Safe eval for simple math
            allowed = set("0123456789+-*/.()")
            if not all(c in allowed or c.isspace() for c in expression):
                return "Invalid expression"
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"

    tools = [get_weather, search_knowledge, calculate]

    # Create agent
    system_prompt = """You are a helpful assistant with access to various tools.

Available tools:
- get_weather: Check weather for any city
- search_knowledge: Search the knowledge base
- calculate: Evaluate mathematical expressions

Be helpful and use tools when appropriate."""

    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=system_prompt,
    )

    return agent


def main():
    """Main entry point."""
    import uvicorn

    from mask.a2a import create_persistent_a2a_server
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
    database_url = get_database_url()
    host = os.environ.get("A2A_HOST", "0.0.0.0")
    port = int(os.environ.get("A2A_PORT", "10001"))

    logger.info("Creating agent...")
    agent = create_agent()

    logger.info("Creating persistent A2A server...")
    logger.info("Database URL: %s", database_url.split("@")[-1])  # Hide credentials

    # Create server with persistence
    app = create_persistent_a2a_server(
        agent=agent,
        name="mask-persistent-demo",
        description="Demo agent with PostgreSQL persistence",
        database_url=database_url,
        url=f"http://{host}:{port}/",
        stream=True,
    )

    logger.info("Starting A2A server on %s:%d", host, port)
    logger.info("Agent Card URL: http://%s:%d/.well-known/agent.json", host, port)

    # Run server
    uvicorn.run(
        app.build(),
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
