"""Helper functions for LangGraph checkpoints.

This module provides helper functions for setting up SQLite (local dev) and
PostgreSQL (production) based checkpointing for LangGraph agents. These helpers
are separate from the A2A module because checkpointing is a LangGraph concern.

Environment-based checkpointer selection:
    - ENV=local (default): Use SQLite for local development
    - ENV=production: Use PostgreSQL for K8s deployment

Usage (environment-based):
    from mask.checkpoints import get_checkpointer

    # Automatically selects SQLite or PostgreSQL based on ENV
    checkpointer = get_checkpointer()

Usage (explicit PostgreSQL):
    from mask.checkpoints import setup_postgres_tables, create_async_checkpointer

    # 1. Initialize tables (sync, call once at startup)
    setup_postgres_tables(database_url)

    # 2. Create async checkpointer (must be in async context)
    checkpointer = await create_async_checkpointer(database_url)

    # 3. Pass to agent creation
    agent = await create_agent(checkpointer=checkpointer)

Usage (explicit SQLite):
    from mask.checkpoints import create_sqlite_checkpointer

    # Create SQLite checkpointer for local development
    checkpointer = create_sqlite_checkpointer("checkpoints.db")

    # Or use async version
    checkpointer = await create_async_sqlite_checkpointer("checkpoints.db")
"""

import logging
import os
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

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


# =============================================================================
# SQLite Checkpointer (Local Development)
# =============================================================================


def create_sqlite_checkpointer(
    db_path: str = "checkpoints.db",
) -> Optional["SqliteSaver"]:
    """Create SQLite checkpointer for local development.

    This is a synchronous checkpointer suitable for local development
    and testing. The schema is compatible with PostgresSaver.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        SqliteSaver instance or None if creation fails.

    Example:
        >>> checkpointer = create_sqlite_checkpointer("checkpoints.db")
        >>> graph = create_agent(model, tools, checkpointer=checkpointer)
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        checkpointer = SqliteSaver.from_conn_string(db_path)
        # Setup creates the required tables
        checkpointer.setup()
        logger.info("Created SqliteSaver checkpointer: %s", db_path)
        return checkpointer
    except ImportError as e:
        logger.warning(
            "langgraph-checkpoint-sqlite not installed: %s. "
            "Install with: pip install langgraph-checkpoint-sqlite",
            e,
        )
        return None
    except Exception as e:
        logger.warning("Failed to create SqliteSaver: %s", e)
        return None


async def create_async_sqlite_checkpointer(
    db_path: str = "checkpoints.db",
) -> Optional["AsyncSqliteSaver"]:
    """Create async SQLite checkpointer for local development.

    Async version of SQLite checkpointer, useful when your agent
    uses async patterns but you want SQLite for local development.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        AsyncSqliteSaver instance or None if creation fails.

    Example:
        >>> checkpointer = await create_async_sqlite_checkpointer("checkpoints.db")
        >>> graph = create_agent(model, tools, checkpointer=checkpointer)
    """
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        checkpointer = AsyncSqliteSaver.from_conn_string(db_path)
        await checkpointer.setup()
        logger.info("Created AsyncSqliteSaver checkpointer: %s", db_path)
        return checkpointer
    except ImportError as e:
        logger.warning(
            "langgraph-checkpoint-sqlite not installed: %s. "
            "Install with: pip install langgraph-checkpoint-sqlite",
            e,
        )
        return None
    except Exception as e:
        logger.warning("Failed to create AsyncSqliteSaver: %s", e)
        return None


# =============================================================================
# Environment-based Checkpointer Selection
# =============================================================================


def get_checkpointer(
    database_url: str | None = None,
    db_path: str = "checkpoints.db",
) -> Optional["BaseCheckpointSaver"]:
    """Get checkpointer based on environment.

    Automatically selects SQLite or PostgreSQL based on the ENV
    environment variable:
    - ENV=local (default): Use SQLite
    - ENV=production: Use PostgreSQL

    For PostgreSQL, you must also set DATABASE_URL environment variable
    or pass database_url parameter.

    Args:
        database_url: Optional PostgreSQL URL. Falls back to DATABASE_URL env var.
        db_path: Path to SQLite database (used when ENV=local).

    Returns:
        Checkpointer instance or None if creation fails.

    Example:
        >>> # Automatic selection based on ENV
        >>> checkpointer = get_checkpointer()
        >>>
        >>> # Force PostgreSQL with explicit URL
        >>> checkpointer = get_checkpointer(database_url="postgresql://...")
    """
    env = os.getenv("ENV", "local").lower()

    if env == "production":
        # Use PostgreSQL for production
        db_url = database_url or os.getenv("DATABASE_URL")
        if not db_url:
            logger.warning(
                "ENV=production but DATABASE_URL not set. "
                "Falling back to SQLite checkpointer."
            )
            return create_sqlite_checkpointer(db_path)

        # Setup tables first (sync operation)
        if not setup_postgres_tables(db_url):
            logger.warning("Failed to setup PostgreSQL tables, falling back to SQLite")
            return create_sqlite_checkpointer(db_path)

        # Create sync PostgresSaver
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg_pool import ConnectionPool

            pool = ConnectionPool(
                conninfo=db_url,
                min_size=1,
                max_size=5,
            )
            checkpointer = PostgresSaver(pool)
            logger.info("Created PostgresSaver checkpointer for production")
            return checkpointer
        except ImportError as e:
            logger.warning(
                "PostgreSQL dependencies not available: %s. Falling back to SQLite.", e
            )
            return create_sqlite_checkpointer(db_path)
        except Exception as e:
            logger.warning("Failed to create PostgresSaver: %s. Falling back to SQLite.", e)
            return create_sqlite_checkpointer(db_path)
    else:
        # Use SQLite for local development
        return create_sqlite_checkpointer(db_path)


async def get_async_checkpointer(
    database_url: str | None = None,
    db_path: str = "checkpoints.db",
) -> Union["AsyncPostgresSaver", "AsyncSqliteSaver"] | None:
    """Get async checkpointer based on environment.

    Async version of get_checkpointer(). Automatically selects
    async SQLite or PostgreSQL based on the ENV environment variable.

    Args:
        database_url: Optional PostgreSQL URL. Falls back to DATABASE_URL env var.
        db_path: Path to SQLite database (used when ENV=local).

    Returns:
        Async checkpointer instance or None if creation fails.

    Example:
        >>> # Automatic selection based on ENV
        >>> checkpointer = await get_async_checkpointer()
        >>>
        >>> # Force PostgreSQL with explicit URL
        >>> checkpointer = await get_async_checkpointer(database_url="postgresql://...")
    """
    env = os.getenv("ENV", "local").lower()

    if env == "production":
        # Use PostgreSQL for production
        db_url = database_url or os.getenv("DATABASE_URL")
        if not db_url:
            logger.warning(
                "ENV=production but DATABASE_URL not set. "
                "Falling back to async SQLite checkpointer."
            )
            return await create_async_sqlite_checkpointer(db_path)

        # Setup tables first (sync operation)
        if not setup_postgres_tables(db_url):
            logger.warning("Failed to setup PostgreSQL tables, falling back to SQLite")
            return await create_async_sqlite_checkpointer(db_path)

        # Create async PostgresSaver
        return await create_async_checkpointer(db_url)
    else:
        # Use SQLite for local development
        return await create_async_sqlite_checkpointer(db_path)
