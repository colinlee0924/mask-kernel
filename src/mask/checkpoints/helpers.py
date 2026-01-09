"""Helper functions for LangGraph PostgreSQL checkpoints.

This module provides helper functions for setting up PostgreSQL-based
checkpointing for LangGraph agents. These helpers are separate from
the A2A module because checkpointing is a LangGraph concern, not A2A.

Usage:
    from mask.checkpoints import setup_postgres_tables, create_async_checkpointer

    # 1. Initialize tables (sync, call once at startup)
    setup_postgres_tables(database_url)

    # 2. Create async checkpointer (must be in async context)
    checkpointer = await create_async_checkpointer(database_url)

    # 3. Pass to agent creation
    agent = await create_agent(checkpointer=checkpointer)
"""

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

logger = logging.getLogger(__name__)


def setup_postgres_tables(database_url: str) -> bool:
    """Initialize PostgreSQL tables for LangGraph checkpoints.

    Must be called before creating checkpointer. Uses autocommit=True
    for DDL operations (CREATE INDEX CONCURRENTLY requires being outside
    a transaction block).

    Args:
        database_url: PostgreSQL connection URL.
            Example: "postgresql://user:pass@localhost:5432/my_db"

    Returns:
        True if setup succeeded, False otherwise.

    Example:
        >>> setup_postgres_tables("postgresql://user:pass@localhost:5432/my_agent")
        True
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg import connect

        with connect(database_url, autocommit=True) as setup_conn:
            temp_saver = PostgresSaver(setup_conn)
            temp_saver.setup()
        logger.info("PostgresSaver tables initialized")
        return True
    except ImportError as e:
        logger.warning(
            "langgraph-checkpoint-postgres or psycopg not installed: %s. "
            "Install with: pip install langgraph-checkpoint-postgres psycopg[pool]",
            e,
        )
        return False
    except Exception as e:
        logger.warning("Failed to setup PostgresSaver tables: %s", e)
        return False


async def create_async_checkpointer(
    database_url: str,
    min_size: int = 2,
    max_size: int = 20,
    timeout: float = 60.0,
) -> Optional["AsyncPostgresSaver"]:
    """Create AsyncPostgresSaver for LangGraph checkpoints.

    MUST be called in async context (inside async function).
    Call setup_postgres_tables() first to initialize the schema.

    IMPORTANT: This must be called within the same event loop where
    the checkpointer will be used (e.g., inside uvicorn's lifespan).
    Creating the pool in one event loop and using it in another will
    cause PoolTimeout errors.

    Args:
        database_url: PostgreSQL connection URL.
            Example: "postgresql://user:pass@localhost:5432/my_db"
        min_size: Minimum number of connections in the pool.
        max_size: Maximum number of connections in the pool.
        timeout: Connection timeout in seconds.

    Returns:
        AsyncPostgresSaver instance or None if creation fails.

    Example:
        >>> @asynccontextmanager
        ... async def lifespan(app):
        ...     checkpointer = await create_async_checkpointer(database_url)
        ...     yield
        ...     if checkpointer:
        ...         await checkpointer.conn.close()
    """
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool

        # Create pool with open=False to avoid deprecation warning and race conditions
        # Then explicitly open with wait=True to ensure connections are ready
        async_pool = AsyncConnectionPool(
            conninfo=database_url,
            min_size=min_size,
            max_size=max_size,
            timeout=timeout,
            open=False,
        )
        await async_pool.open(wait=True)
        checkpointer = AsyncPostgresSaver(async_pool)
        logger.info(
            "Created AsyncPostgresSaver checkpointer (pool: min=%d, max=%d)",
            min_size,
            max_size,
        )
        return checkpointer
    except ImportError as e:
        logger.warning(
            "langgraph-checkpoint-postgres or psycopg_pool not installed: %s. "
            "Install with: pip install langgraph-checkpoint-postgres psycopg[pool]",
            e,
        )
        return None
    except Exception as e:
        logger.warning("Failed to create AsyncPostgresSaver: %s", e)
        return None
