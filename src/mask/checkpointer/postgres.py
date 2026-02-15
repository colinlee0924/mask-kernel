"""PostgreSQL Checkpointer for LangGraph.

This module provides a PostgreSQL-backed checkpointer for LangGraph agents,
enabling multi-turn conversation persistence across API calls and sessions.

Features:
- Persistent conversation state storage
- Thread-based session management
- Skill state preservation (skills_loaded)
- Automatic table creation
- Connection pooling

Requirements:
    pip install langgraph-checkpoint-postgres psycopg[binary,pool]

Usage:
    from mask.checkpointer import create_postgres_checkpointer

    # Create checkpointer
    checkpointer = await create_postgres_checkpointer(
        "postgresql://user:pass@localhost/db"
    )

    # Use with agent
    from mask.agent import create_mask_agent

    agent = create_mask_agent()
    graph = agent.build_graph(checkpointer=checkpointer)

    # Invoke with thread_id
    config = {"configurable": {"thread_id": "user-123-session-456"}}
    response = await graph.ainvoke({"messages": [...]}, config=config)

    # Resume later with same thread_id
    response = await graph.ainvoke({"messages": [...]}, config=config)
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PostgresCheckpointer:
    """PostgreSQL-backed checkpointer wrapper.

    This is a wrapper around langgraph-checkpoint-postgres that provides
    a simplified interface for MASK agents.

    Attributes:
        connection_string: PostgreSQL connection URL.
        pool: Connection pool (if using sync mode).
        checkpointer: The underlying LangGraph checkpointer.
    """

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
        auto_setup: bool = True,
    ) -> None:
        """Initialize PostgreSQL checkpointer.

        Args:
            connection_string: PostgreSQL connection URL.
            pool_size: Connection pool size.
            auto_setup: Whether to auto-create tables.
        """
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.auto_setup = auto_setup
        self._checkpointer = None
        self._pool = None

    async def setup(self) -> None:
        """Initialize the checkpointer (async).

        Creates the connection pool and sets up tables if auto_setup is True.
        """
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from psycopg_pool import AsyncConnectionPool
        except ImportError:
            raise ImportError(
                "PostgreSQL checkpointer requires additional dependencies. "
                "Install with: pip install langgraph-checkpoint-postgres psycopg[binary,pool]"
            )

        # Create async connection pool
        self._pool = AsyncConnectionPool(
            self.connection_string,
            min_size=1,
            max_size=self.pool_size,
            open=False,
        )
        await self._pool.open()

        # Create checkpointer
        self._checkpointer = AsyncPostgresSaver(self._pool)

        # Setup tables if needed
        if self.auto_setup:
            await self._checkpointer.setup()
            logger.debug("PostgreSQL checkpointer tables created/verified")

        logger.info("PostgreSQL checkpointer initialized")

    def setup_sync(self) -> None:
        """Initialize the checkpointer (sync).

        Creates the connection pool and sets up tables if auto_setup is True.
        """
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg_pool import ConnectionPool
        except ImportError:
            raise ImportError(
                "PostgreSQL checkpointer requires additional dependencies. "
                "Install with: pip install langgraph-checkpoint-postgres psycopg[binary,pool]"
            )

        # Create sync connection pool
        self._pool = ConnectionPool(
            self.connection_string,
            min_size=1,
            max_size=self.pool_size,
            open=False,
        )
        self._pool.open()

        # Create checkpointer
        self._checkpointer = PostgresSaver(self._pool)

        # Setup tables if needed
        if self.auto_setup:
            self._checkpointer.setup()
            logger.debug("PostgreSQL checkpointer tables created/verified")

        logger.info("PostgreSQL checkpointer initialized (sync)")

    @property
    def checkpointer(self):
        """Get the underlying LangGraph checkpointer."""
        if self._checkpointer is None:
            raise RuntimeError(
                "Checkpointer not initialized. Call setup() or setup_sync() first."
            )
        return self._checkpointer

    async def close(self) -> None:
        """Close the connection pool (async)."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._checkpointer = None
            logger.debug("PostgreSQL checkpointer closed")

    def close_sync(self) -> None:
        """Close the connection pool (sync)."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None
            self._checkpointer = None
            logger.debug("PostgreSQL checkpointer closed (sync)")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def __enter__(self):
        """Sync context manager entry."""
        self.setup_sync()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit."""
        self.close_sync()


async def create_postgres_checkpointer(
    connection_string: str,
    pool_size: int = 10,
    auto_setup: bool = True,
) -> "PostgresCheckpointer":
    """Create and initialize a PostgreSQL checkpointer.

    This is a convenience function that creates and sets up a PostgreSQL
    checkpointer in one call.

    Args:
        connection_string: PostgreSQL connection URL.
            Format: postgresql://user:password@host:port/database
        pool_size: Connection pool size.
        auto_setup: Whether to auto-create tables.

    Returns:
        Initialized PostgresCheckpointer instance.

    Example:
        checkpointer = await create_postgres_checkpointer(
            "postgresql://user:pass@localhost:5432/mask_db"
        )

        # Use the underlying checkpointer with LangGraph
        graph = graph_builder.compile(checkpointer=checkpointer.checkpointer)
    """
    wrapper = PostgresCheckpointer(
        connection_string=connection_string,
        pool_size=pool_size,
        auto_setup=auto_setup,
    )
    await wrapper.setup()
    return wrapper


def create_thread_config(
    thread_id: str,
    checkpoint_ns: str = "",
    checkpoint_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a LangGraph config dict for a conversation thread.

    This helper creates the proper config structure for using
    checkpointers with LangGraph.

    Args:
        thread_id: Unique identifier for the conversation thread.
            Use a combination of user_id and session_id for multi-user apps.
        checkpoint_ns: Optional namespace for organizing checkpoints.
        checkpoint_id: Optional specific checkpoint to resume from.

    Returns:
        Config dict for LangGraph invoke/stream calls.

    Example:
        config = create_thread_config("user-123-session-456")
        response = await graph.ainvoke(
            {"messages": [HumanMessage(content="Hello")]},
            config=config,
        )
    """
    configurable = {
        "thread_id": thread_id,
    }

    if checkpoint_ns:
        configurable["checkpoint_ns"] = checkpoint_ns

    if checkpoint_id:
        configurable["checkpoint_id"] = checkpoint_id

    return {"configurable": configurable}


# SQL for manual table setup (if not using auto_setup)
SETUP_SQL = """
-- LangGraph checkpoint tables (reference)
-- These are automatically created by langgraph-checkpoint-postgres

-- Main checkpoint table
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

-- Checkpoint writes table (for pending writes)
CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS checkpoints_thread_id_idx ON checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS checkpoint_writes_thread_id_idx ON checkpoint_writes(thread_id);
"""
