"""A2A Agent Executor.

This module bridges MASK BaseAgent to A2A AgentExecutor interface,
following patterns from a2a-python-samples.
"""

import logging
from typing import TYPE_CHECKING

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import TaskState, TaskStatus
from a2a.utils import new_agent_text_message

if TYPE_CHECKING:
    from mask.agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class MaskAgentExecutor(AgentExecutor):
    """Bridge MASK BaseAgent to A2A AgentExecutor.

    This executor wraps a MASK agent and handles the conversion between
    A2A protocol messages and agent inputs/outputs.

    Following a2a-python-samples pattern:
    - Extract user message from RequestContext
    - Execute agent (with optional streaming)
    - Enqueue results to EventQueue

    Example:
        from mask.a2a import MaskAgentExecutor

        executor = MaskAgentExecutor(my_agent)
        # Used by A2A server internally
    """

    def __init__(
        self,
        agent: "BaseAgent",
        stream: bool = False,
    ) -> None:
        """Initialize executor with MASK agent.

        Args:
            agent: The BaseAgent instance to execute.
            stream: Whether to use streaming responses.
        """
        self.agent = agent
        self.stream = stream

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute agent and push results to event queue.

        Args:
            context: A2A request context containing user message.
            event_queue: Queue for sending events back to client.
        """
        # Extract user message from A2A request
        user_message = self._extract_user_message(context)

        if not user_message:
            logger.warning("No user message found in request context")
            await event_queue.enqueue_event(
                new_agent_text_message("No message provided.")
            )
            return

        # Extract session ID for observability trace grouping
        session_id = self._extract_session_id(context)

        logger.debug(
            "Executing agent with message: %s... (session: %s)",
            user_message[:50],
            session_id or "none",
        )

        # Use using_session context for Phoenix session tracking
        # This sets session.id on all spans created within this context
        try:
            if session_id:
                try:
                    from openinference.instrumentation import using_session

                    with using_session(session_id):
                        await self._execute_with_error_handling(
                            user_message, event_queue, session_id
                        )
                    return
                except ImportError:
                    logger.debug("openinference.instrumentation not available")

            await self._execute_with_error_handling(
                user_message, event_queue, session_id
            )
        except Exception as e:
            logger.exception("Agent execution failed: %s", e)
            await event_queue.enqueue_event(
                new_agent_text_message(f"Error: {str(e)}")
            )

    async def _execute_with_error_handling(
        self,
        message: str,
        event_queue: EventQueue,
        session_id: str = None,
    ) -> None:
        """Execute agent with proper error handling.

        Args:
            message: User message.
            event_queue: Event queue for responses.
            session_id: Optional session ID for trace grouping.
        """
        if self.stream:
            await self._execute_streaming(message, event_queue, session_id)
        else:
            await self._execute_non_streaming(message, event_queue, session_id)

    async def _execute_non_streaming(
        self,
        message: str,
        event_queue: EventQueue,
        session_id: str = None,
    ) -> None:
        """Execute agent without streaming.

        Args:
            message: User message.
            event_queue: Event queue for responses.
            session_id: Optional session ID for trace grouping.
        """
        response = await self.agent.invoke(message, session_id=session_id)
        await event_queue.enqueue_event(
            new_agent_text_message(response)
        )

    async def _execute_streaming(
        self,
        message: str,
        event_queue: EventQueue,
        session_id: str = None,
    ) -> None:
        """Execute agent with streaming.

        Args:
            message: User message.
            event_queue: Event queue for responses.
            session_id: Optional session ID for trace grouping.
        """
        full_response = ""
        async for chunk in self.agent.stream(message, session_id=session_id):
            full_response += chunk
            # Note: A2A streaming events could be sent here
            # For now, we collect and send final response

        await event_queue.enqueue_event(
            new_agent_text_message(full_response)
        )

    def _extract_user_message(self, context: RequestContext) -> str:
        """Extract text message from A2A request context.

        Args:
            context: The request context.

        Returns:
            Extracted user message text.
        """
        message = context.message
        if message and message.parts:
            for part in message.parts:
                # Handle different part types
                if hasattr(part, "root") and hasattr(part.root, "text"):
                    return part.root.text
                if hasattr(part, "text"):
                    return part.text

        return ""

    def _extract_session_id(self, context: RequestContext) -> str | None:
        """Extract session ID from A2A request context.

        A2A uses context_id (contextId in JSON) as the session/conversation identifier.
        This allows multiple traces to be grouped under one session in Phoenix.

        Args:
            context: The request context.

        Returns:
            Session ID if found, None otherwise.
        """
        # Check message for context_id (from A2A Message.contextId)
        message = context.message
        if message:
            if hasattr(message, "context_id") and message.context_id:
                return message.context_id

        # Fallback: check RequestContext for context_id
        if hasattr(context, "context_id") and context.context_id:
            return context.context_id

        return None

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Handle task cancellation.

        Args:
            context: Request context.
            event_queue: Event queue.
        """
        logger.info("Task cancellation requested")
        # MASK agents don't currently support cancellation
        # Just acknowledge the cancellation
        pass
