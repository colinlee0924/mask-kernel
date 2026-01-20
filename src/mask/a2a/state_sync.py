"""State Synchronization for Open WebUI and LangGraph Checkpoints.

This module provides the StateSynchronizer class that handles synchronization
between Open WebUI's message history and LangGraph checkpoints, supporting:
- Retry detection (user regenerates a response)
- Delete detection (user deletes messages)
- Checkpoint rollback (restore to previous state)

The synchronization follows the "Frontend Source of Truth" pattern where
Open WebUI maintains the authoritative message history.

Usage:
    from mask.a2a.state_sync import StateSynchronizer, SyncResult

    # Create synchronizer with a LangGraph CompiledStateGraph
    synchronizer = StateSynchronizer(graph)

    # Analyze sync state
    result = await synchronizer.analyze(thread_id, incoming_messages)

    if result.action == "rollback":
        # Get config with checkpoint_id for rollback
        config = synchronizer.get_invoke_config(thread_id, result)
        # Execute with rollback config
        response = await graph.ainvoke(input, config)
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of synchronization analysis.

    Attributes:
        action: The recommended action:
            - "continue": Normal execution, no sync needed
            - "rollback": Rollback to a previous checkpoint
            - "reset": State is inconsistent, need to reset
        target_checkpoint_id: Checkpoint ID to rollback to (if action is "rollback")
        rollback_to_index: Message index to rollback to (for debugging)
        message: Human-readable description of what was detected
        frontend_count: Number of messages from frontend
        backend_count: Number of messages in checkpoint
    """

    action: str
    target_checkpoint_id: str | None = None
    rollback_to_index: int | None = None
    message: str | None = None
    frontend_count: int = 0
    backend_count: int = 0


class StateSynchronizer:
    """Synchronize Open WebUI messages with LangGraph checkpoints.

    This class handles the detection and resolution of state differences
    between the frontend (Open WebUI) and backend (LangGraph checkpoints).

    Supported operations:
    - Retry: User regenerates a response (same user message, different response)
    - Delete: User removes one or more messages
    - Branch: User starts a new conversation branch from a previous point

    Example:
        from mask.a2a.state_sync import StateSynchronizer

        synchronizer = StateSynchronizer(graph)

        # In your executor's execute method:
        result = await synchronizer.analyze(thread_id, frontend_messages)

        if result.action == "rollback":
            config = synchronizer.get_invoke_config(thread_id, result)
        else:
            config = {"configurable": {"thread_id": thread_id}}

        # Execute with appropriate config
        await graph.ainvoke({"messages": [user_message]}, config)
    """

    def __init__(
        self,
        graph: "CompiledStateGraph",
        checkpointer: Optional["BaseCheckpointSaver"] = None,
    ) -> None:
        """Initialize StateSynchronizer.

        Args:
            graph: LangGraph CompiledStateGraph instance.
            checkpointer: Optional checkpointer (will use graph's checkpointer if not provided).
        """
        self.graph = graph
        self.checkpointer = checkpointer

    def _hash_message(self, message: Any) -> str:
        """Compute hash of a single message for comparison.

        Supports both dict and LangChain BaseMessage formats.

        Args:
            message: Message dict or BaseMessage.

        Returns:
            8-character hash of the message.
        """
        if isinstance(message, dict):
            content = message.get("content", "")
            role = message.get("role", "")
        else:
            # LangChain BaseMessage
            content = getattr(message, "content", "")
            role = getattr(message, "type", "")

        return hashlib.md5(f"{role}:{content}".encode()).hexdigest()[:8]

    def _hash_messages(self, messages: list[Any]) -> list[str]:
        """Compute hashes for all messages.

        Args:
            messages: List of message dicts or BaseMessages.

        Returns:
            List of 8-character hashes.
        """
        return [self._hash_message(msg) for msg in messages]

    async def get_checkpoint_messages(self, thread_id: str) -> list[dict[str, Any]]:
        """Get messages from the current checkpoint.

        Args:
            thread_id: The thread ID to query.

        Returns:
            List of message dicts from the checkpoint.
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = await self.graph.aget_state(config)

            if not state or not state.values:
                return []

            messages = state.values.get("messages", [])

            # Convert LangChain messages to dicts
            result = []
            for msg in messages:
                if isinstance(msg, dict):
                    result.append(msg)
                elif hasattr(msg, "content"):
                    result.append({
                        "role": getattr(msg, "type", "unknown"),
                        "content": msg.content,
                        "id": getattr(msg, "id", None),
                    })

            return result

        except Exception as e:
            logger.debug("Failed to get checkpoint messages: %s", e)
            return []

    async def analyze(
        self,
        thread_id: str,
        frontend_messages: list[dict[str, Any]],
    ) -> SyncResult:
        """Analyze sync state between frontend and checkpoint.

        Compares frontend messages with checkpoint state to detect:
        - Retry: User regenerated a response
        - Delete: User deleted messages
        - Normal: Standard new message flow

        Args:
            thread_id: The thread/session ID.
            frontend_messages: Messages from Open WebUI (source of truth).

        Returns:
            SyncResult with recommended action.
        """
        # Get current checkpoint state
        backend_messages = await self.get_checkpoint_messages(thread_id)

        frontend_count = len(frontend_messages)
        backend_count = len(backend_messages)

        # No checkpoint exists - new conversation
        if backend_count == 0:
            return SyncResult(
                action="continue",
                message="New conversation, no checkpoint exists",
                frontend_count=frontend_count,
                backend_count=backend_count,
            )

        # Compute hashes for comparison
        frontend_hashes = self._hash_messages(frontend_messages)
        backend_hashes = self._hash_messages(backend_messages)

        logger.debug(
            "[SYNC] Thread: %s | Frontend: %d msgs | Backend: %d msgs",
            thread_id,
            frontend_count,
            backend_count,
        )
        logger.debug("[SYNC] Frontend hashes (last 5): %s", frontend_hashes[-5:])
        logger.debug("[SYNC] Backend hashes (last 5): %s", backend_hashes[-5:])

        # Case 1: Normal new message (frontend has one more message)
        if frontend_count == backend_count + 1:
            # Check if all previous messages match
            if frontend_hashes[:backend_count] == backend_hashes:
                return SyncResult(
                    action="continue",
                    message="Normal new message",
                    frontend_count=frontend_count,
                    backend_count=backend_count,
                )

        # Case 2: Retry detection (same count but last message different)
        if frontend_count == backend_count and frontend_count > 0:
            if frontend_hashes[-1] != backend_hashes[-1]:
                # This is a retry - find checkpoint before last user message
                target_index = self._find_last_user_message_index(backend_messages)
                checkpoint_id = await self._find_checkpoint_at_index(
                    thread_id, target_index
                )
                return SyncResult(
                    action="rollback",
                    target_checkpoint_id=checkpoint_id,
                    rollback_to_index=target_index,
                    message=f"Retry detected: rollback to index {target_index}",
                    frontend_count=frontend_count,
                    backend_count=backend_count,
                )

        # Case 3: Delete detection (frontend has fewer messages)
        if frontend_count < backend_count:
            # Find the divergence point
            diverge_index = self._find_diverge_point(frontend_hashes, backend_hashes)
            checkpoint_id = await self._find_checkpoint_at_index(
                thread_id, diverge_index
            )
            return SyncResult(
                action="rollback",
                target_checkpoint_id=checkpoint_id,
                rollback_to_index=diverge_index,
                message=f"Delete detected: rollback to index {diverge_index}",
                frontend_count=frontend_count,
                backend_count=backend_count,
            )

        # Case 4: Frontend has significantly more messages (state inconsistency)
        if frontend_count > backend_count + 1:
            return SyncResult(
                action="reset",
                message="State inconsistency: frontend has too many messages",
                frontend_count=frontend_count,
                backend_count=backend_count,
            )

        # Default: continue normally
        return SyncResult(
            action="continue",
            message="State synchronized",
            frontend_count=frontend_count,
            backend_count=backend_count,
        )

    def _find_diverge_point(
        self,
        frontend_hashes: list[str],
        backend_hashes: list[str],
    ) -> int:
        """Find the index where frontend and backend diverge.

        Args:
            frontend_hashes: Hashes from frontend messages.
            backend_hashes: Hashes from backend messages.

        Returns:
            Index of first difference (or length of frontend if all match).
        """
        for i, (f_hash, b_hash) in enumerate(zip(frontend_hashes, backend_hashes)):
            if f_hash != b_hash:
                return i
        return len(frontend_hashes)

    def _find_last_user_message_index(self, messages: list[Any]) -> int:
        """Find the index of the last user message.

        Used for retry detection - we want to rollback to before
        the assistant's response so we can regenerate it.

        Args:
            messages: List of messages.

        Returns:
            Index of last user message, or 0 if none found.
        """
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, dict):
                role = msg.get("role", "")
            else:
                role = getattr(msg, "type", "")

            if role in ("user", "human"):
                return i

        return 0

    async def _find_checkpoint_at_index(
        self,
        thread_id: str,
        target_index: int,
    ) -> str | None:
        """Find checkpoint with message count <= target_index.

        Iterates through checkpoint history to find a checkpoint
        that can be used as the starting point for rollback.

        Args:
            thread_id: The thread ID.
            target_index: Target message count.

        Returns:
            Checkpoint ID if found, None otherwise.
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}

            async for state in self.graph.aget_state_history(config):
                msg_count = len(state.values.get("messages", []))
                if msg_count <= target_index:
                    checkpoint_id = state.config.get("configurable", {}).get(
                        "checkpoint_id"
                    )
                    logger.debug(
                        "[SYNC] Found checkpoint at index %d: %s",
                        msg_count,
                        checkpoint_id,
                    )
                    return checkpoint_id

        except Exception as e:
            logger.warning("Failed to find checkpoint at index %d: %s", target_index, e)

        return None

    def get_invoke_config(
        self,
        thread_id: str,
        sync_result: SyncResult,
    ) -> dict[str, Any]:
        """Generate invoke config based on sync result.

        Creates the config dict to pass to graph.ainvoke() with
        appropriate thread_id and checkpoint_id.

        Args:
            thread_id: The thread ID.
            sync_result: Result from analyze().

        Returns:
            Config dict for graph.ainvoke().
        """
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

        if sync_result.action == "rollback" and sync_result.target_checkpoint_id:
            config["configurable"]["checkpoint_id"] = sync_result.target_checkpoint_id
            logger.info(
                "[SYNC] Using rollback config: thread=%s, checkpoint=%s",
                thread_id,
                sync_result.target_checkpoint_id,
            )

        return config

    async def sync_to_frontend(
        self,
        thread_id: str,
        frontend_messages: list[dict[str, Any]],
    ) -> str | None:
        """Force sync checkpoint to match frontend state.

        Use this when you need to explicitly reshape the checkpoint
        to match the frontend's message history.

        Args:
            thread_id: The thread ID.
            frontend_messages: Messages from frontend (source of truth).

        Returns:
            New checkpoint ID if sync succeeded, None otherwise.
        """
        try:
            from langchain_core.messages import AIMessage, HumanMessage

            # Convert frontend messages to LangChain format
            lc_messages = []
            for msg in frontend_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role in ("user", "human"):
                    lc_messages.append(HumanMessage(content=content))
                elif role in ("assistant", "ai"):
                    lc_messages.append(AIMessage(content=content))

            # Update checkpoint state
            config = {"configurable": {"thread_id": thread_id}}
            await self.graph.aupdate_state(
                config,
                values={"messages": lc_messages},
            )

            # Get new checkpoint ID
            new_state = await self.graph.aget_state(config)
            if new_state and new_state.config:
                checkpoint_id = new_state.config.get("configurable", {}).get(
                    "checkpoint_id"
                )
                logger.info(
                    "[SYNC] Synced checkpoint to frontend: thread=%s, checkpoint=%s",
                    thread_id,
                    checkpoint_id,
                )
                return checkpoint_id

        except Exception as e:
            logger.exception("Failed to sync checkpoint to frontend: %s", e)

        return None

    async def get_checkpoint_history(
        self,
        thread_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get checkpoint history for debugging.

        Args:
            thread_id: The thread ID.
            limit: Maximum number of checkpoints to return.

        Returns:
            List of checkpoint info dicts.
        """
        history = []
        try:
            config = {"configurable": {"thread_id": thread_id}}
            count = 0

            async for state in self.graph.aget_state_history(config):
                if count >= limit:
                    break

                checkpoint_id = state.config.get("configurable", {}).get(
                    "checkpoint_id", "unknown"
                )
                msg_count = len(state.values.get("messages", []))

                history.append({
                    "checkpoint_id": checkpoint_id,
                    "message_count": msg_count,
                    "index": count,
                })
                count += 1

        except Exception as e:
            logger.debug("Failed to get checkpoint history: %s", e)

        return history
