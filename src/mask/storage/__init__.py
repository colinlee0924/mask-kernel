"""MASK session storage backends.

This module provides storage backends for session persistence:
- MemorySessionStore: In-memory storage (default)
- RedisSessionStore: Redis-backed storage (requires mask-kernel[redis])
- PostgreSQLSessionStore: PostgreSQL-backed storage (requires mask-kernel[postgresql])
"""

from mask.storage.base import SessionStore
from mask.storage.memory_store import MemorySessionStore

# Lazy imports for optional backends
__all__ = [
    "SessionStore",
    "MemorySessionStore",
    "RedisSessionStore",
    "PostgreSQLSessionStore",
]


def __getattr__(name: str):
    """Lazy import optional storage backends."""
    if name == "RedisSessionStore":
        from mask.storage.redis_store import RedisSessionStore
        return RedisSessionStore
    elif name == "PostgreSQLSessionStore":
        from mask.storage.postgresql_store import PostgreSQLSessionStore
        return PostgreSQLSessionStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
