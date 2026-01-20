"""LangGraph Checkpoint helpers for PostgreSQL and SQLite persistence.

This module provides helper functions for setting up checkpointing for
LangGraph agents. It supports:
- SQLite for local development
- PostgreSQL for production (K8s deployment)

Environment-based selection:
    from mask.checkpoints import get_checkpointer
    checkpointer = get_checkpointer()  # Auto-selects based on ENV

Explicit PostgreSQL:
    from mask.checkpoints import setup_postgres_tables, create_async_checkpointer
    setup_postgres_tables(database_url)
    checkpointer = await create_async_checkpointer(database_url)

Explicit SQLite:
    from mask.checkpoints import create_sqlite_checkpointer
    checkpointer = create_sqlite_checkpointer("checkpoints.db")
"""

from mask.checkpoints.helpers import (
    create_async_checkpointer,
    create_async_sqlite_checkpointer,
    create_sqlite_checkpointer,
    get_async_checkpointer,
    get_checkpointer,
    setup_postgres_tables,
)

__all__ = [
    # Environment-based selection (recommended)
    "get_checkpointer",
    "get_async_checkpointer",
    # SQLite helpers (local development)
    "create_sqlite_checkpointer",
    "create_async_sqlite_checkpointer",
    # PostgreSQL helpers (production)
    "setup_postgres_tables",
    "create_async_checkpointer",
]
