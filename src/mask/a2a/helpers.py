"""A2A Helper Functions.

This module provides helper functions that return native A2A SDK types.
Developers can use these helpers with the native A2A SDK directly.

Usage:
    from langchain.agents import create_agent
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from mask.a2a import create_a2a_executor

    graph = create_agent(model, tools, system_prompt)
    executor = create_a2a_executor(graph, server_name="my-agent")
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card, http_handler=handler)

For persistent storage with PostgreSQL:
    # Use mask.checkpoints for LangGraph checkpointer
    from mask.checkpoints import setup_postgres_tables, create_async_checkpointer
    # Use mask.a2a for A2A task store
    from mask.a2a import create_a2a_executor, create_database_task_store

    # Initialize tables and create checkpointer
    setup_postgres_tables(database_url)
    checkpointer = await create_async_checkpointer(database_url)

    # Create agent with checkpointer
    agent = await create_agent(checkpointer=checkpointer)

    # Create executor and task store
    executor = create_a2a_executor(agent, server_name="my-agent")
    task_store = create_database_task_store(database_url)

    # Build server with native SDK
    handler = DefaultRequestHandler(agent_executor=executor, task_store=task_store)
    app = A2AStarletteApplication(agent_card, http_handler=handler)
"""

import logging
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from a2a.server.tasks import TaskStore
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from mask.a2a.executor import MaskAgentExecutor
    from mask.agent.base_agent import BaseAgent
    from mask.storage.base import SessionStore

logger = logging.getLogger(__name__)


def create_a2a_executor(
    agent: Union["BaseAgent", "CompiledStateGraph"],
    stream: bool = True,
    server_name: Optional[str] = None,
    checkpointer: Optional["BaseCheckpointSaver"] = None,
    session_store: Optional["SessionStore"] = None,
) -> "MaskAgentExecutor":
    """Create an A2A executor from a LangChain CompiledStateGraph or MASK agent.

    This helper function creates a MaskAgentExecutor that bridges your agent
    to the A2A protocol. It supports both:
    - LangChain CompiledStateGraph from create_agent() (recommended)
    - MASK BaseAgent (legacy)

    Features:
    - Real-time streaming via TaskArtifactUpdateEvent (default enabled)
    - Multi-agent handoffs with context isolation
    - OpenTelemetry tracing integration
    - PostgreSQL persistence via LangGraph checkpointer (optional)
    - Session history synchronization with frontend (optional)

    Args:
        agent: LangChain CompiledStateGraph or MASK BaseAgent instance.
        stream: Whether to use real-time streaming (default True for Open WebUI).
        server_name: Optional server name for trace display in Phoenix/Langfuse.
            If not provided, falls back to the agent name attribute.
        checkpointer: Optional LangGraph checkpointer for persistence.
            Use PostgresSaver.from_conn_string(DATABASE_URL) for PostgreSQL.
        session_store: Optional MASK SessionStore for session metadata.

    Returns:
        MaskAgentExecutor instance compatible with A2A SDK.

    Example:
        from langchain.agents import create_agent
        from a2a.server.apps import A2AStarletteApplication
        from a2a.server.request_handlers import DefaultRequestHandler
        from a2a.server.tasks import InMemoryTaskStore
        from a2a.types import AgentCapabilities, AgentCard, AgentSkill
        from mask.a2a import create_a2a_executor

        # Create agent using native LangChain API
        graph = create_agent(model, tools, system_prompt)

        # Create executor (with optional persistence)
        executor = create_a2a_executor(
            graph,
            server_name="my-agent",
            # checkpointer=PostgresSaver.from_conn_string(DATABASE_URL),
        )

        # Build A2A server with native SDK
        agent_card = AgentCard(
            name="my-agent",
            skills=[AgentSkill(id="general", name="General", description="...", tags=["general"])],
            capabilities=AgentCapabilities(streaming=True),
            ...
        )
        handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
        app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)

        import uvicorn
        uvicorn.run(app.build(), host="0.0.0.0", port=10001)
    """
    from mask.a2a.executor import MaskAgentExecutor

    return MaskAgentExecutor(
        agent=agent,
        stream=stream,
        server_name=server_name,
        checkpointer=checkpointer,
        session_store=session_store,
    )


def create_database_task_store(database_url: str) -> "TaskStore":
    """Create DatabaseTaskStore for A2A task persistence.

    This helper creates a DatabaseTaskStore backed by PostgreSQL for
    persisting A2A tasks across server restarts. Falls back to
    InMemoryTaskStore if dependencies are not available.

    Args:
        database_url: PostgreSQL connection URL.
            Example: "postgresql://user:pass@localhost:5432/my_agent"

    Returns:
        DatabaseTaskStore instance or InMemoryTaskStore as fallback.

    Example:
        from mask.a2a import create_database_task_store

        task_store = create_database_task_store(
            "postgresql://user:pass@localhost:5432/my_agent"
        )

        handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=task_store,
        )
    """
    try:
        from a2a.server.tasks import DatabaseTaskStore
        from sqlalchemy.ext.asyncio import create_async_engine

        # Convert postgresql:// to postgresql+asyncpg:// for async
        async_url = database_url
        if async_url.startswith("postgresql://"):
            async_url = async_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif async_url.startswith("postgres://"):
            async_url = async_url.replace("postgres://", "postgresql+asyncpg://", 1)

        engine = create_async_engine(async_url)
        task_store = DatabaseTaskStore(engine=engine, create_table=True)
        logger.info("Created DatabaseTaskStore for A2A tasks")
        return task_store
    except ImportError as e:
        logger.warning(
            "DatabaseTaskStore dependencies not available: %s. "
            "Install with: pip install sqlalchemy asyncpg. "
            "Using InMemoryTaskStore as fallback.",
            e,
        )
        from a2a.server.tasks import InMemoryTaskStore

        return InMemoryTaskStore()
    except Exception as e:
        logger.warning(
            "Failed to create DatabaseTaskStore: %s. Using InMemoryTaskStore.",
            e,
        )
        from a2a.server.tasks import InMemoryTaskStore

        return InMemoryTaskStore()
