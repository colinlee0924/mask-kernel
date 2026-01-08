"""Task Sync Metadata Storage.

This module provides storage for A2A task synchronization metadata,
tracking the relationship between A2A tasks and LangGraph checkpoints.

Used for:
- Tracking which A2A tasks are linked to which checkpoints
- Marking tasks as cancelled when forking occurs
- Supporting garbage collection of abandoned branches

SQL Schema:
    CREATE TABLE IF NOT EXISTS mask_task_sync_metadata (
        id UUID PRIMARY KEY,
        a2a_task_id UUID NOT NULL,
        session_id VARCHAR(255) NOT NULL,
        linked_checkpoint_id TEXT,
        superseded_by UUID,
        status VARCHAR(20) DEFAULT 'active',
        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
        CONSTRAINT fk_superseded FOREIGN KEY (superseded_by)
            REFERENCES mask_task_sync_metadata(id)
    );

    CREATE INDEX IF NOT EXISTS idx_task_sync_session ON mask_task_sync_metadata(session_id);
    CREATE INDEX IF NOT EXISTS idx_task_sync_a2a_task ON mask_task_sync_metadata(a2a_task_id);
    CREATE INDEX IF NOT EXISTS idx_task_sync_status ON mask_task_sync_metadata(status);
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TaskSyncMetadata:
    """Metadata for tracking A2A task sync state.

    Attributes:
        id: Unique identifier for this metadata record.
        a2a_task_id: The A2A task ID this metadata relates to.
        session_id: The session/context/thread ID.
        linked_checkpoint_id: The LangGraph checkpoint ID this task created.
        superseded_by: ID of the metadata record that supersedes this one.
        status: Status of this record ('active', 'cancelled').
        created_at: When this record was created.
        updated_at: When this record was last updated.
    """

    id: str
    a2a_task_id: str
    session_id: str
    linked_checkpoint_id: Optional[str] = None
    superseded_by: Optional[str] = None
    status: str = "active"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def create(
        cls,
        a2a_task_id: str,
        session_id: str,
        linked_checkpoint_id: Optional[str] = None,
    ) -> "TaskSyncMetadata":
        """Create a new TaskSyncMetadata instance.

        Args:
            a2a_task_id: The A2A task ID.
            session_id: The session/context ID.
            linked_checkpoint_id: Optional checkpoint ID.

        Returns:
            New TaskSyncMetadata instance.
        """
        now = datetime.now(timezone.utc)
        return cls(
            id=str(uuid.uuid4()),
            a2a_task_id=a2a_task_id,
            session_id=session_id,
            linked_checkpoint_id=linked_checkpoint_id,
            created_at=now,
            updated_at=now,
        )

    def mark_cancelled(self, superseded_by: Optional[str] = None) -> None:
        """Mark this task as cancelled.

        Args:
            superseded_by: Optional ID of the superseding record.
        """
        self.status = "cancelled"
        self.superseded_by = superseded_by
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "a2a_task_id": self.a2a_task_id,
            "session_id": self.session_id,
            "linked_checkpoint_id": self.linked_checkpoint_id,
            "superseded_by": self.superseded_by,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskSyncMetadata":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            a2a_task_id=data["a2a_task_id"],
            session_id=data["session_id"],
            linked_checkpoint_id=data.get("linked_checkpoint_id"),
            superseded_by=data.get("superseded_by"),
            status=data.get("status", "active"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


# SQL for creating the task_sync_metadata table
CREATE_TASK_SYNC_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS mask_task_sync_metadata (
    id UUID PRIMARY KEY,
    a2a_task_id UUID NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    linked_checkpoint_id TEXT,
    superseded_by UUID,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    CONSTRAINT fk_superseded FOREIGN KEY (superseded_by)
        REFERENCES mask_task_sync_metadata(id)
);

CREATE INDEX IF NOT EXISTS idx_task_sync_session ON mask_task_sync_metadata(session_id);
CREATE INDEX IF NOT EXISTS idx_task_sync_a2a_task ON mask_task_sync_metadata(a2a_task_id);
CREATE INDEX IF NOT EXISTS idx_task_sync_status ON mask_task_sync_metadata(status);
CREATE INDEX IF NOT EXISTS idx_task_sync_checkpoint ON mask_task_sync_metadata(linked_checkpoint_id);
"""


class TaskSyncStore:
    """Storage for TaskSyncMetadata using PostgreSQL.

    This store tracks the relationship between A2A tasks and LangGraph
    checkpoints, supporting sync operations like fork detection and
    garbage collection.

    Example:
        store = TaskSyncStore("postgresql://user:pass@localhost/db")
        async with store:
            # Create metadata for a new task
            metadata = TaskSyncMetadata.create(
                a2a_task_id="task-123",
                session_id="session-456",
                linked_checkpoint_id="ckpt-789",
            )
            await store.save(metadata)

            # Find all active tasks for a session
            active = await store.get_active_by_session("session-456")

            # Cancel tasks linked to abandoned checkpoints
            await store.bulk_cancel_by_checkpoints(["ckpt-old-1", "ckpt-old-2"])
    """

    def __init__(
        self,
        connection_string: str,
        table_name: str = "mask_task_sync_metadata",
        auto_create_table: bool = True,
    ) -> None:
        """Initialize TaskSyncStore.

        Args:
            connection_string: PostgreSQL connection URL.
            table_name: Name of the table.
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
                    "Install with: pip install asyncpg"
                )

            try:
                self._pool = await asyncpg.create_pool(
                    self.connection_string,
                    min_size=1,
                    max_size=10,
                )
            except Exception as e:
                from mask.core.exceptions import StorageConnectionError

                raise StorageConnectionError("postgresql", str(e)) from e

            if self.auto_create_table and not self._table_created:
                async with self._pool.acquire() as conn:
                    await conn.execute(CREATE_TASK_SYNC_TABLE_SQL)
                    self._table_created = True
                    logger.debug("Created task_sync_metadata table")

        return self._pool

    async def save(self, metadata: TaskSyncMetadata) -> None:
        """Save or update task sync metadata.

        Args:
            metadata: The TaskSyncMetadata to save.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.table_name} (
                    id, a2a_task_id, session_id, linked_checkpoint_id,
                    superseded_by, status, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (id) DO UPDATE SET
                    linked_checkpoint_id = EXCLUDED.linked_checkpoint_id,
                    superseded_by = EXCLUDED.superseded_by,
                    status = EXCLUDED.status,
                    updated_at = EXCLUDED.updated_at
                """,
                uuid.UUID(metadata.id),
                uuid.UUID(metadata.a2a_task_id),
                metadata.session_id,
                metadata.linked_checkpoint_id,
                uuid.UUID(metadata.superseded_by) if metadata.superseded_by else None,
                metadata.status,
                metadata.created_at,
                metadata.updated_at,
            )

        logger.debug("Saved task sync metadata: %s", metadata.id)

    async def get(self, metadata_id: str) -> Optional[TaskSyncMetadata]:
        """Get task sync metadata by ID.

        Args:
            metadata_id: The metadata ID.

        Returns:
            TaskSyncMetadata if found, None otherwise.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id, a2a_task_id, session_id, linked_checkpoint_id,
                       superseded_by, status, created_at, updated_at
                FROM {self.table_name}
                WHERE id = $1
                """,
                uuid.UUID(metadata_id),
            )

        if row is None:
            return None

        return self._row_to_metadata(row)

    async def get_by_task_id(self, a2a_task_id: str) -> Optional[TaskSyncMetadata]:
        """Get task sync metadata by A2A task ID.

        Args:
            a2a_task_id: The A2A task ID.

        Returns:
            TaskSyncMetadata if found, None otherwise.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id, a2a_task_id, session_id, linked_checkpoint_id,
                       superseded_by, status, created_at, updated_at
                FROM {self.table_name}
                WHERE a2a_task_id = $1
                ORDER BY created_at DESC
                LIMIT 1
                """,
                uuid.UUID(a2a_task_id),
            )

        if row is None:
            return None

        return self._row_to_metadata(row)

    async def get_active_by_session(self, session_id: str) -> List[TaskSyncMetadata]:
        """Get all active task metadata for a session.

        Args:
            session_id: The session ID.

        Returns:
            List of active TaskSyncMetadata.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, a2a_task_id, session_id, linked_checkpoint_id,
                       superseded_by, status, created_at, updated_at
                FROM {self.table_name}
                WHERE session_id = $1 AND status = 'active'
                ORDER BY created_at DESC
                """,
                session_id,
            )

        return [self._row_to_metadata(row) for row in rows]

    async def get_by_checkpoint(
        self, checkpoint_id: str
    ) -> Optional[TaskSyncMetadata]:
        """Get task sync metadata by checkpoint ID.

        Args:
            checkpoint_id: The LangGraph checkpoint ID.

        Returns:
            TaskSyncMetadata if found, None otherwise.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id, a2a_task_id, session_id, linked_checkpoint_id,
                       superseded_by, status, created_at, updated_at
                FROM {self.table_name}
                WHERE linked_checkpoint_id = $1
                """,
                checkpoint_id,
            )

        if row is None:
            return None

        return self._row_to_metadata(row)

    async def bulk_cancel_by_checkpoints(
        self,
        checkpoint_ids: List[str],
        superseded_by: Optional[str] = None,
    ) -> int:
        """Cancel all tasks linked to the given checkpoints.

        Used when forking creates abandoned branches.

        Args:
            checkpoint_ids: List of checkpoint IDs to cancel.
            superseded_by: Optional ID of superseding metadata.

        Returns:
            Number of tasks cancelled.
        """
        if not checkpoint_ids:
            return 0

        pool = await self._get_pool()
        now = datetime.now(timezone.utc)

        async with pool.acquire() as conn:
            result = await conn.execute(
                f"""
                UPDATE {self.table_name}
                SET status = 'cancelled',
                    superseded_by = $1,
                    updated_at = $2
                WHERE linked_checkpoint_id = ANY($3)
                AND status = 'active'
                """,
                uuid.UUID(superseded_by) if superseded_by else None,
                now,
                checkpoint_ids,
            )

        # Parse result to get count
        count = int(result.split()[-1]) if result else 0
        if count > 0:
            logger.info("Cancelled %d tasks linked to abandoned checkpoints", count)
        return count

    async def cancel_task(
        self,
        a2a_task_id: str,
        superseded_by: Optional[str] = None,
    ) -> bool:
        """Cancel a specific task.

        Args:
            a2a_task_id: The A2A task ID to cancel.
            superseded_by: Optional ID of superseding metadata.

        Returns:
            True if a task was cancelled.
        """
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)

        async with pool.acquire() as conn:
            result = await conn.execute(
                f"""
                UPDATE {self.table_name}
                SET status = 'cancelled',
                    superseded_by = $1,
                    updated_at = $2
                WHERE a2a_task_id = $3
                AND status = 'active'
                """,
                uuid.UUID(superseded_by) if superseded_by else None,
                now,
                uuid.UUID(a2a_task_id),
            )

        return "UPDATE 1" in result if result else False

    async def cleanup_old_cancelled(self, days: int = 7) -> int:
        """Remove cancelled tasks older than N days.

        Garbage collection for storage cleanup.

        Args:
            days: Age threshold in days.

        Returns:
            Number of records deleted.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {self.table_name}
                WHERE status = 'cancelled'
                AND updated_at < NOW() - INTERVAL '{days} days'
                """
            )

        count = int(result.split()[-1]) if result else 0
        if count > 0:
            logger.info("Cleaned up %d old cancelled task records", count)
        return count

    def _row_to_metadata(self, row) -> TaskSyncMetadata:
        """Convert database row to TaskSyncMetadata."""
        return TaskSyncMetadata(
            id=str(row["id"]),
            a2a_task_id=str(row["a2a_task_id"]),
            session_id=row["session_id"],
            linked_checkpoint_id=row["linked_checkpoint_id"],
            superseded_by=str(row["superseded_by"]) if row["superseded_by"] else None,
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.debug("Closed TaskSyncStore connection pool")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
