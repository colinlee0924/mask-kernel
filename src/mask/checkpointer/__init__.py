"""MASK Checkpointer - LangGraph checkpointer integrations.

This module provides checkpointer implementations for persisting
LangGraph agent state across conversations.

Checkpointers enable:
- Multi-turn conversation persistence
- Session resumption across API calls
- Skill state preservation (skills_loaded)

Available checkpointers:
- PostgresCheckpointer: PostgreSQL-backed persistence (recommended for production)
- AsyncPostgresCheckpointer: Async version of PostgresCheckpointer

Usage:
    from mask.checkpointer import create_postgres_checkpointer

    # Create checkpointer
    checkpointer = await create_postgres_checkpointer(
        "postgresql://user:pass@localhost/db"
    )

    # Use with agent
    agent = create_mask_agent(
        checkpointer=checkpointer,
    )

    # Invoke with thread_id for persistence
    response = await agent.invoke(
        "Hello",
        config={"configurable": {"thread_id": "session-123"}}
    )
"""

from mask.checkpointer.postgres import (
    create_postgres_checkpointer,
    create_thread_config,
    PostgresCheckpointer,
)

__all__ = [
    "create_postgres_checkpointer",
    "create_thread_config",
    "PostgresCheckpointer",
]
