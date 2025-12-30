"""Redis session storage implementation.

This storage backend is suitable for distributed deployments where sessions
need to be shared across multiple processes or servers.

Requires: pip install mask-kernel[redis]
"""

import json
import logging
from typing import Optional

from mask.core.exceptions import StorageConnectionError
from mask.session.session import Session
from mask.storage.base import SessionStore

logger = logging.getLogger(__name__)


class RedisSessionStore(SessionStore):
    """Redis-backed session storage.

    This implementation uses Redis for persistent, distributed session storage.
    Suitable for:
    - Multi-process deployments
    - Distributed systems
    - High-availability requirements

    Session data is stored as JSON with optional TTL support.

    Example:
        store = RedisSessionStore("redis://localhost:6379")
        async with store:
            await store.save(session)
            session = await store.get(session_id)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "mask:session:",
        default_ttl: Optional[int] = None,
    ) -> None:
        """Initialize Redis session store.

        Args:
            redis_url: Redis connection URL.
            key_prefix: Prefix for Redis keys.
            default_ttl: Default TTL in seconds for sessions (None = no expiry).
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self._client = None

    def _get_key(self, session_id: str) -> str:
        """Generate Redis key for a session ID."""
        return f"{self.key_prefix}{session_id}"

    async def _get_client(self):
        """Get or create Redis client."""
        if self._client is None:
            try:
                import redis.asyncio as aioredis
            except ImportError:
                raise ImportError(
                    "Redis support requires the 'redis' package. "
                    "Install with: pip install mask-kernel[redis]"
                )

            try:
                self._client = await aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                # Test connection
                await self._client.ping()
            except Exception as e:
                raise StorageConnectionError("redis", str(e)) from e

        return self._client

    async def save(self, session: Session) -> None:
        """Save a session to Redis.

        Args:
            session: The Session to save.
        """
        client = await self._get_client()
        key = self._get_key(session.session_id)

        # Serialize session to JSON
        data = json.dumps(session.to_dict())

        # Determine TTL
        ttl = self.default_ttl
        if session.expires_at:
            from datetime import datetime
            ttl = int((session.expires_at - datetime.now()).total_seconds())
            if ttl <= 0:
                # Session already expired, don't save
                return

        if ttl:
            await client.setex(key, ttl, data)
        else:
            await client.set(key, data)

        logger.debug("Saved session to Redis: %s", session.session_id)

    async def get(self, session_id: str) -> Optional[Session]:
        """Retrieve a session from Redis.

        Args:
            session_id: The session ID to retrieve.

        Returns:
            The Session if found, None otherwise.
        """
        client = await self._get_client()
        key = self._get_key(session_id)

        data = await client.get(key)
        if data is None:
            return None

        try:
            session_dict = json.loads(data)
            session = Session.from_dict(session_dict)

            # Check expiration (belt and suspenders with Redis TTL)
            if session.is_expired():
                await self.delete(session_id)
                return None

            return session
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(
                "Failed to deserialize session %s: %s",
                session_id, e,
            )
            return None

    async def delete(self, session_id: str) -> None:
        """Delete a session from Redis.

        Args:
            session_id: The session ID to delete.
        """
        client = await self._get_client()
        key = self._get_key(session_id)
        await client.delete(key)
        logger.debug("Deleted session from Redis: %s", session_id)

    async def exists(self, session_id: str) -> bool:
        """Check if a session exists in Redis.

        Args:
            session_id: The session ID to check.

        Returns:
            True if the session exists.
        """
        client = await self._get_client()
        key = self._get_key(session_id)
        return await client.exists(key) > 0

    async def set_ttl(self, session_id: str, ttl: int) -> None:
        """Set TTL on an existing session.

        Args:
            session_id: The session ID.
            ttl: Time to live in seconds.
        """
        client = await self._get_client()
        key = self._get_key(session_id)
        await client.expire(key, ttl)

    async def get_ttl(self, session_id: str) -> Optional[int]:
        """Get remaining TTL for a session.

        Args:
            session_id: The session ID.

        Returns:
            Remaining TTL in seconds, or None if no TTL set.
        """
        client = await self._get_client()
        key = self._get_key(session_id)
        ttl = await client.ttl(key)
        return ttl if ttl > 0 else None

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.debug("Closed Redis connection")

    async def count(self) -> int:
        """Get approximate number of sessions.

        Note: Uses SCAN which may not be exact count.

        Returns:
            Approximate number of sessions.
        """
        client = await self._get_client()
        pattern = f"{self.key_prefix}*"
        count = 0
        async for _ in client.scan_iter(match=pattern):
            count += 1
        return count
