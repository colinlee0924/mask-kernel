"""Base session storage interface.

This module defines the abstract interface for session storage backends.
Following the pattern from A2A SDK's TaskStore.
"""

from abc import ABC, abstractmethod
from typing import Optional

from mask.session.session import Session


class SessionStore(ABC):
    """Abstract interface for session storage.

    Implementations must provide methods for persisting and retrieving
    Session objects. This follows the Repository pattern.

    Example implementations:
    - MemorySessionStore: In-memory storage (default)
    - RedisSessionStore: Redis-backed storage for distributed systems
    - PostgreSQLSessionStore: PostgreSQL-backed for persistence
    """

    @abstractmethod
    async def save(self, session: Session) -> None:
        """Save or update a session in the store.

        Args:
            session: The Session object to persist.

        Raises:
            StorageError: If the save operation fails.
        """
        pass

    @abstractmethod
    async def get(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID.

        Args:
            session_id: The unique session identifier.

        Returns:
            The Session object if found, None otherwise.

        Raises:
            StorageError: If the retrieval fails due to storage issues.
        """
        pass

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Delete a session from the store.

        Args:
            session_id: The unique session identifier.

        Raises:
            StorageError: If the deletion fails.
        """
        pass

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        """Check if a session exists.

        Args:
            session_id: The unique session identifier.

        Returns:
            True if the session exists, False otherwise.
        """
        pass

    async def get_or_create(
        self,
        session_id: str,
        **defaults,
    ) -> Session:
        """Get an existing session or create a new one.

        Args:
            session_id: The session ID.
            **defaults: Default values for new session creation.

        Returns:
            Existing or newly created Session.
        """
        session = await self.get(session_id)
        if session is None:
            session = Session(session_id=session_id, **defaults)
            await self.save(session)
        return session

    async def close(self) -> None:
        """Close the storage connection.

        Override this method to clean up resources like database
        connections or connection pools.
        """
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
