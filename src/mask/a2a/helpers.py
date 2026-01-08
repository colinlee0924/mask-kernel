"""A2A Helper Functions.

This module provides helper functions that return native A2A SDK types.
Developers can use these helpers with the native A2A SDK directly.

Usage:
    from langchain.agents import create_agent
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from mask.a2a import create_a2a_executor

    graph = create_agent(model, tools, system_prompt)
    executor = create_a2a_executor(graph, server_name="my-agent")
    handler = DefaultRequestHandler(agent_executor=executor, task_store=...)
    app = A2AStarletteApplication(agent_card, http_handler=handler)

For persistent storage with PostgreSQL:
    from mask.a2a import create_a2a_executor, create_persistent_a2a_server

    # Create executor with persistence
    executor = create_a2a_executor(
        graph,
        server_name="my-agent",
        checkpointer=PostgresSaver.from_conn_string(DATABASE_URL),
    )

    # Or use the all-in-one helper
    app = create_persistent_a2a_server(
        agent=graph,
        name="my-agent",
        database_url="postgresql://user:pass@localhost:5432/mask_kernel",
    )
"""

from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from a2a.server.apps import A2AStarletteApplication
    from a2a.types import AgentSkill
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from mask.a2a.executor import MaskAgentExecutor
    from mask.agent.base_agent import BaseAgent
    from mask.storage.base import SessionStore


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


def create_persistent_a2a_server(
    agent: Union["BaseAgent", "CompiledStateGraph"],
    name: str,
    description: str,
    database_url: str,
    url: Optional[str] = None,
    version: str = "1.0.0",
    skills: Optional[List["AgentSkill"]] = None,
    stream: bool = True,
    default_input_modes: Optional[List[str]] = None,
    default_output_modes: Optional[List[str]] = None,
) -> "A2AStarletteApplication":
    """Create an A2A server with PostgreSQL persistence.

    This helper creates a complete A2A server with:
    - DatabaseTaskStore for A2A task persistence
    - PostgresSaver for LangGraph checkpoint persistence
    - Automatic schema creation

    Three-layer storage architecture:
    1. DatabaseTaskStore (A2A tasks) - schema: a2a
    2. PostgresSaver (LangGraph checkpoints) - schema: langgraph
    3. SessionStore (MASK sessions) - schema: mask

    Args:
        agent: LangChain CompiledStateGraph or MASK BaseAgent instance.
        name: Agent name for AgentCard.
        description: Agent description for AgentCard.
        database_url: PostgreSQL connection URL.
            Example: "postgresql://user:pass@localhost:5432/mask_kernel"
        url: Agent URL (auto-generated from host:port if not provided).
        version: Agent version string.
        skills: List of AgentSkill for AgentCard. If not provided, creates
            a default "general" skill.
        stream: Whether to enable streaming responses (default True).
        default_input_modes: Supported input modes (default: ["text"]).
        default_output_modes: Supported output modes (default: ["text"]).

    Returns:
        A2AStarletteApplication instance ready for uvicorn.run().

    Example:
        from mask.a2a import create_persistent_a2a_server
        from langchain.agents import create_agent
        import uvicorn

        graph = create_agent(model, tools, system_prompt)

        app = create_persistent_a2a_server(
            agent=graph,
            name="my-agent",
            description="My persistent agent",
            database_url="postgresql://user:pass@localhost:5432/mask_kernel",
        )

        uvicorn.run(app.build(), host="0.0.0.0", port=10001)
    """
    import logging

    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.types import AgentCapabilities, AgentCard, AgentSkill

    logger = logging.getLogger(__name__)

    # Create PostgresSaver for LangGraph checkpoints
    # Note: from_conn_string() returns a context manager, so for long-running servers
    # we use psycopg_pool.ConnectionPool directly
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg import connect
        from psycopg_pool import ConnectionPool

        # First, run setup with autocommit=True to allow CREATE INDEX CONCURRENTLY
        # This needs to be outside a transaction block
        with connect(database_url, autocommit=True) as setup_conn:
            temp_saver = PostgresSaver(setup_conn)
            temp_saver.setup()
        logger.info("PostgresSaver tables initialized (schema created)")

        # Create connection pool for long-running server
        pool = ConnectionPool(conninfo=database_url)
        checkpointer = PostgresSaver(pool)
        logger.info("Created PostgresSaver checkpointer for LangGraph")
    except ImportError as e:
        logger.warning(
            "langgraph-checkpoint-postgres or psycopg not installed. "
            "Install with: pip install langgraph-checkpoint-postgres psycopg[pool]. Error: %s",
            e,
        )
        checkpointer = None
    except Exception as e:
        logger.warning("Failed to create PostgresSaver: %s", e)
        checkpointer = None

    # Create DatabaseTaskStore for A2A tasks
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
    except ImportError as e:
        logger.warning(
            "a2a DatabaseTaskStore dependencies not available: %s. "
            "Using InMemoryTaskStore. Tasks will not persist across restarts.",
            e,
        )
        from a2a.server.tasks import InMemoryTaskStore

        task_store = InMemoryTaskStore()
    except Exception as e:
        logger.warning("Failed to create DatabaseTaskStore: %s. Using InMemory.", e)
        from a2a.server.tasks import InMemoryTaskStore

        task_store = InMemoryTaskStore()

    # Create executor with persistence
    executor = create_a2a_executor(
        agent=agent,
        stream=stream,
        server_name=name,
        checkpointer=checkpointer,
    )

    # Create default skill if not provided
    if not skills:
        skills = [
            AgentSkill(
                id="general",
                name="General Assistant",
                description=description,
                tags=["general"],
            )
        ]

    # Create agent card
    agent_card = AgentCard(
        name=name,
        description=description,
        url=url or "http://localhost:10001/",
        version=version,
        skills=skills,
        capabilities=AgentCapabilities(streaming=stream),
        defaultInputModes=default_input_modes or ["text"],
        defaultOutputModes=default_output_modes or ["text"],
    )

    # Create request handler
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )

    # Create application
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )

    logger.info(
        "Created persistent A2A server: name=%s, url=%s, persistence=%s",
        name,
        agent_card.url,
        "enabled" if checkpointer else "disabled",
    )

    return app
