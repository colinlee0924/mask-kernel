"""PostgreSQL session storage implementation.

This storage backend provides persistent session storage using PostgreSQL,
suitable for applications requiring durable storage and complex queries.

Requires: pip install mask-kernel[postgresql]
"""

import json
import logging
from typing import Optional

from mask.core.exceptions import StorageConnectionError
from mask.session.session import Session
from mask.storage.base import SessionStore

logger = logging.getLogger(__name__)

# SQL for creating the sessions table
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS mask_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255),
    data JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE,
    messages JSONB NOT NULL DEFAULT '[]',
    skills_loaded JSONB NOT NULL DEFAULT '[]',
    pagination_cursor INTEGER
);

CREATE INDEX IF NOT EXISTS idx_mask_sessions_user_id ON mask_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_mask_sessions_expires_at ON mask_sessions(expires_at);
"""


class PostgreSQLSessionStore(SessionStore):
    """PostgreSQL-backed session storage.

    This implementation uses PostgreSQL for durable session storage.
    Suitable for:
    - Applications requiring persistent storage
    - Complex session queries
    - Integration with existing PostgreSQL infrastructure

    Session data is stored with JSONB columns for flexible schema.

    Example:
        store = PostgreSQLSessionStore(
            "postgresql://user:pass@localhost/db"
        )
        async with store:
            await store.save(session)
            session = await store.get(session_id)
    """

    def __init__(
        self,
        connection_string: str,
        table_name: str = "mask_sessions",
        auto_create_table: bool = True,
    ) -> None:
        """Initialize PostgreSQL session store.

        Args:
            connection_string: PostgreSQL connection URL.
            table_name: Name of the sessions table.
            auto_create_table: Whether to create table on first connect.
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self.auto_create_table = auto_create_table
        self._pool = None
        self._table_created = False

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            try:
                import asyncpg
            except ImportError:
                raise ImportError(
                    "PostgreSQL support requires the 'asyncpg' package. "
                    "Install with: pip install mask-kernel[postgresql]"
                )

            try:
                self._pool = await asyncpg.create_pool(
                    self.connection_string,
                    min_size=1,
                    max_size=10,
                )
            except Exception as e:
                raise StorageConnectionError("postgresql", str(e)) from e

            if self.auto_create_table and not self._table_created:
                async with self._pool.acquire() as conn:
                    await conn.execute(CREATE_TABLE_SQL)
                    self._table_created = True
                    logger.debug("Created sessions table")

        return self._pool

    async def save(self, session: Session) -> None:
        """Save a session to PostgreSQL.

        Args:
            session: The Session to save.
        """
        pool = await self._get_pool()

        # Prepare data
        session_dict = session.to_dict()

        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.table_name} (
                    session_id, user_id, data, created_at, updated_at,
                    expires_at, messages, skills_loaded, pagination_cursor
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (session_id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    data = EXCLUDED.data,
                    updated_at = EXCLUDED.updated_at,
                    expires_at = EXCLUDED.expires_at,
                    messages = EXCLUDED.messages,
                    skills_loaded = EXCLUDED.skills_loaded,
                    pagination_cursor = EXCLUDED.pagination_cursor
                """,
                session.session_id,
                session.user_id,
                json.dumps(session_dict["data"]),
                session.created_at,
                session.updated_at,
                session.expires_at,
                json.dumps(session_dict["messages"]),
                json.dumps(session.skills_loaded),
                session.pagination_cursor,
            )

        logger.debug("Saved session to PostgreSQL: %s", session.session_id)

    async def get(self, session_id: str) -> Optional[Session]:
        """Retrieve a session from PostgreSQL.

        Args:
            session_id: The session ID to retrieve.

        Returns:
            The Session if found and not expired, None otherwise.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT session_id, user_id, data, created_at, updated_at,
                       expires_at, messages, skills_loaded, pagination_cursor
                FROM {self.table_name}
                WHERE session_id = $1
                """,
                session_id,
            )

        if row is None:
            return None

        # Reconstruct session
        session_dict = {
            "session_id": row["session_id"],
            "user_id": row["user_id"],
            "data": json.loads(row["data"]) if row["data"] else {},
            "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat(),
            "expires_at": row["expires_at"].isoformat() if row["expires_at"] else None,
            "messages": json.loads(row["messages"]) if row["messages"] else [],
            "skills_loaded": json.loads(row["skills_loaded"]) if row["skills_loaded"] else [],
            "pagination_cursor": row["pagination_cursor"],
        }

        session = Session.from_dict(session_dict)

        # Check expiration
        if session.is_expired():
            await self.delete(session_id)
            return None

        return session

    async def delete(self, session_id: str) -> None:
        """Delete a session from PostgreSQL.

        Args:
            session_id: The session ID to delete.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self.table_name} WHERE session_id = $1",
                session_id,
            )

        logger.debug("Deleted session from PostgreSQL: %s", session_id)

    async def exists(self, session_id: str) -> bool:
        """Check if a session exists in PostgreSQL.

        Args:
            session_id: The session ID to check.

        Returns:
            True if the session exists.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.fetchval(
                f"SELECT 1 FROM {self.table_name} WHERE session_id = $1",
                session_id,
            )

        return result is not None

    async def cleanup_expired(self) -> int:
        """Remove expired sessions from database.

        Returns:
            Number of sessions removed.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {self.table_name}
                WHERE expires_at IS NOT NULL AND expires_at < NOW()
                """
            )

        # Parse result to get count
        count = int(result.split()[-1]) if result else 0
        if count > 0:
            logger.debug("Cleaned up %d expired sessions", count)
        return count

    async def count(self) -> int:
        """Get the number of stored sessions.

        Returns:
            Number of sessions.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            return await conn.fetchval(
                f"SELECT COUNT(*) FROM {self.table_name}"
            )

    async def find_by_user(self, user_id: str) -> list[Session]:
        """Find all sessions for a user.

        Args:
            user_id: The user ID to search for.

        Returns:
            List of sessions for the user.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT session_id, user_id, data, created_at, updated_at,
                       expires_at, messages, skills_loaded, pagination_cursor
                FROM {self.table_name}
                WHERE user_id = $1
                ORDER BY updated_at DESC
                """,
                user_id,
            )

        sessions = []
        for row in rows:
            session_dict = {
                "session_id": row["session_id"],
                "user_id": row["user_id"],
                "data": json.loads(row["data"]) if row["data"] else {},
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
                "expires_at": row["expires_at"].isoformat() if row["expires_at"] else None,
                "messages": json.loads(row["messages"]) if row["messages"] else [],
                "skills_loaded": json.loads(row["skills_loaded"]) if row["skills_loaded"] else [],
                "pagination_cursor": row["pagination_cursor"],
            }
            session = Session.from_dict(session_dict)
            if not session.is_expired():
                sessions.append(session)

        return sessions

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.debug("Closed PostgreSQL connection pool")
