"""In-memory session storage implementation.

This is the default storage backend, suitable for single-process deployments
or development/testing scenarios.
"""

import asyncio
import logging
from typing import Dict, Optional

from mask.session.session import Session
from mask.storage.base import SessionStore

logger = logging.getLogger(__name__)


class MemorySessionStore(SessionStore):
    """In-memory session storage.

    This implementation stores sessions in a dictionary and is suitable for:
    - Development and testing
    - Single-process deployments
    - Stateless agent configurations (no persistence needed)

    Note: Sessions are lost when the process terminates.

    Example:
        store = MemorySessionStore()
        async with store:
            await store.save(session)
            session = await store.get(session_id)
    """

    def __init__(self) -> None:
        """Initialize the in-memory store."""
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def save(self, session: Session) -> None:
        """Save a session to memory.

        Args:
            session: The Session to save.
        """
        async with self._lock:
            self._sessions[session.session_id] = session
            logger.debug("Saved session: %s", session.session_id)

    async def get(self, session_id: str) -> Optional[Session]:
        """Retrieve a session from memory.

        Args:
            session_id: The session ID to retrieve.

        Returns:
            The Session if found and not expired, None otherwise.
        """
        async with self._lock:
            session = self._sessions.get(session_id)

            if session is None:
                return None

            # Check expiration
            if session.is_expired():
                logger.debug("Session expired: %s", session_id)
                del self._sessions[session_id]
                return None

            return session

    async def delete(self, session_id: str) -> None:
        """Delete a session from memory.

        Args:
            session_id: The session ID to delete.
        """
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.debug("Deleted session: %s", session_id)

    async def exists(self, session_id: str) -> bool:
        """Check if a session exists and is not expired.

        Args:
            session_id: The session ID to check.

        Returns:
            True if the session exists and is valid.
        """
        session = await self.get(session_id)
        return session is not None

    async def cleanup_expired(self) -> int:
        """Remove expired sessions.

        Returns:
            Number of sessions removed.
        """
        count = 0
        async with self._lock:
            expired_ids = [
                sid for sid, session in self._sessions.items()
                if session.is_expired()
            ]
            for sid in expired_ids:
                del self._sessions[sid]
                count += 1

        if count > 0:
            logger.debug("Cleaned up %d expired sessions", count)
        return count

    async def count(self) -> int:
        """Get the number of stored sessions.

        Returns:
            Number of sessions.
        """
        async with self._lock:
            return len(self._sessions)

    async def clear(self) -> None:
        """Remove all sessions."""
        async with self._lock:
            self._sessions.clear()
            logger.debug("Cleared all sessions")

    async def list_session_ids(self) -> list[str]:
        """List all session IDs.

        Returns:
            List of session IDs.
        """
        async with self._lock:
            return list(self._sessions.keys())
