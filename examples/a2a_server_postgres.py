"""A2A Server with PostgreSQL persistence.

This example demonstrates how to create an A2A server that uses PostgreSQL
for conversation persistence, suitable for production deployments.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Open WebUI / Client                          │
    └───────────────────────────────┬─────────────────────────────────┘
                                    │ HTTP/SSE
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    A2A Server (:10001)                           │
    │  ┌─────────────────────────────────────────────────────────────┐│
    │  │                    MASK Agent                                ││
    │  │  - Progressive Disclosure Skills                             ││
    │  │  - read_file tool for Level 3                                ││
    │  └─────────────────────────────────────────────────────────────┘│
    │                              │                                   │
    │  ┌─────────────────────────────────────────────────────────────┐│
    │  │           PostgreSQL Checkpointer                            ││
    │  │  - Conversation persistence                                  ││
    │  │  - Skill state preservation                                  ││
    │  └─────────────────────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   PostgreSQL Database                            │
    │  - checkpoints table                                             │
    │  - checkpoint_writes table                                       │
    └─────────────────────────────────────────────────────────────────┘

Requirements:
    pip install mask-kernel[postgresql,a2a]
    pip install langgraph-checkpoint-postgres psycopg[binary,pool]

    # Start PostgreSQL
    docker run -d --name mask-postgres \
        -e POSTGRES_PASSWORD=postgres \
        -e POSTGRES_DB=mask_db \
        -p 5432:5432 \
        postgres:15

Environment variables:
    ANTHROPIC_API_KEY=your-key
    POSTGRES_URL=postgresql://postgres:postgres@localhost:5432/mask_db

Usage:
    python examples/a2a_server_postgres.py
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_server(
    agent_name: str = "mask-agent",
    port: int = 10001,
    postgres_url: Optional[str] = None,
    skills_dir: Optional[Path] = None,
):
    """Create an A2A server with PostgreSQL persistence.

    Args:
        agent_name: Name for the agent (used in A2A discovery).
        port: Port to run the server on.
        postgres_url: PostgreSQL connection URL.
        skills_dir: Path to skills directory.

    Returns:
        Configured MaskA2AServer instance.
    """
    from mask.a2a import MaskA2AServer
    from mask.agent import create_mask_agent
    from mask.core import SkillRegistry
    from mask.models import LLMFactory, ModelTier

    # Get PostgreSQL URL from environment if not provided
    if postgres_url is None:
        postgres_url = os.environ.get(
            "POSTGRES_URL",
            "postgresql://postgres:postgres@localhost:5432/mask_db"
        )

    logger.info(f"Creating agent: {agent_name}")

    # Create skill registry
    registry = SkillRegistry()

    # Discover skills
    if skills_dir is None:
        skills_dir = Path(__file__).parent / "skills"

    if skills_dir.exists():
        count = registry.discover_from_directory(skills_dir)
        logger.info(f"Discovered {count} skills from {skills_dir}")
    else:
        logger.warning(f"Skills directory not found: {skills_dir}")

    # Create LLM
    factory = LLMFactory()
    model = factory.get_model(tier=ModelTier.THINKING)

    # Create agent with file access enabled
    agent = create_mask_agent(
        model=model,
        skill_registry=registry,
        system_prompt=f"""You are {agent_name}, an AI assistant with specialized skills.

You can access skill resources using the read_file tool to get detailed
documentation, scripts, and reference materials.

Remember context from previous messages in the conversation.""",
        skills_dir=skills_dir,
        enable_file_access=True,
    )

    # Create A2A server
    server = MaskA2AServer(
        agent=agent,
        name=agent_name,
        description=f"MASK agent with PostgreSQL persistence: {agent_name}",
        stream=True,  # Enable streaming for better UX
    )

    logger.info(f"Server created: {agent_name} on port {port}")
    logger.info(f"PostgreSQL URL: {postgres_url.split('@')[0]}@...")  # Hide password

    return server, postgres_url


async def run_with_checkpointer():
    """Run the server with PostgreSQL checkpointer.

    This demonstrates the full pattern with checkpointer integration.
    """
    from mask.checkpointer import create_postgres_checkpointer

    agent_name = os.environ.get("AGENT_NAME", "mask-agent")
    port = int(os.environ.get("PORT", "10001"))
    postgres_url = os.environ.get(
        "POSTGRES_URL",
        "postgresql://postgres:postgres@localhost:5432/mask_db"
    )

    logger.info("=" * 60)
    logger.info("MASK A2A Server with PostgreSQL")
    logger.info("=" * 60)

    # Initialize checkpointer
    logger.info("Connecting to PostgreSQL...")
    checkpointer = await create_postgres_checkpointer(postgres_url)
    logger.info("PostgreSQL checkpointer initialized")

    try:
        # Create server
        server, _ = create_server(
            agent_name=agent_name,
            port=port,
            postgres_url=postgres_url,
        )

        # Run server
        logger.info(f"Starting A2A server on port {port}...")
        logger.info(f"Agent: {agent_name}")
        logger.info(f"Streaming: enabled")
        logger.info("")
        logger.info("Endpoints:")
        logger.info(f"  POST http://localhost:{port}/  - Send task")
        logger.info(f"  GET  http://localhost:{port}/.well-known/agent.json - Agent info")
        logger.info("")

        server.run(port=port, log_level="info")

    finally:
        await checkpointer.close()
        logger.info("PostgreSQL connection closed")


def run_simple():
    """Run the server without checkpointer (simpler setup).

    Use this for testing or when persistence isn't needed.
    """
    agent_name = os.environ.get("AGENT_NAME", "mask-agent")
    port = int(os.environ.get("PORT", "10001"))

    logger.info("=" * 60)
    logger.info("MASK A2A Server (Simple Mode)")
    logger.info("=" * 60)

    server, _ = create_server(
        agent_name=agent_name,
        port=port,
    )

    logger.info(f"Starting A2A server on port {port}...")
    logger.info(f"Agent: {agent_name}")
    logger.info("")
    logger.info("Note: Running without PostgreSQL persistence")
    logger.info("      Conversations will not persist across restarts")
    logger.info("")

    server.run(port=port, log_level="info")


# Docker Compose configuration for reference
DOCKER_COMPOSE_EXAMPLE = """
# docker-compose.yml for MASK A2A Server with PostgreSQL

version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: mask_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  mask-agent:
    build: .
    ports:
      - "10001:10001"
    environment:
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      POSTGRES_URL: postgresql://postgres:postgres@postgres:5432/mask_db
      AGENT_NAME: mask-agent
      PORT: 10001
    depends_on:
      postgres:
        condition: service_healthy
    volumes:
      - ./skills:/app/skills:ro

volumes:
  postgres_data:
"""

# Dockerfile example for reference
DOCKERFILE_EXAMPLE = """
# Dockerfile for MASK A2A Server

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install -e ".[postgresql,a2a]"
RUN pip install langgraph-checkpoint-postgres psycopg[binary,pool]

# Copy application code
COPY src/ src/
COPY examples/ examples/

# Copy skills
COPY skills/ skills/

# Expose port
EXPOSE 10001

# Run server
CMD ["python", "examples/a2a_server_postgres.py"]
"""


if __name__ == "__main__":
    import sys

    # Check command line args
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        # Run without checkpointer
        run_simple()
    else:
        # Run with checkpointer
        asyncio.run(run_with_checkpointer())
