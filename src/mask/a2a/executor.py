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
        server_name: str = None,
    ) -> None:
        """Initialize executor with MASK agent.

        Args:
            agent: The BaseAgent instance to execute.
            stream: Whether to use streaming responses.
            server_name: A2A server name for trace display (e.g., "phase1-agent-github").
                        If not provided, falls back to agent name.
        """
        self.agent = agent
        self.stream = stream
        self.server_name = server_name

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

        # Create a user-friendly root span for Phoenix display
        # This wraps the A2A infrastructure spans with a readable agent name
        try:
            await self._execute_with_tracing(user_message, event_queue, session_id)
        except Exception as e:
            logger.exception("Agent execution failed: %s", e)
            await event_queue.enqueue_event(
                new_agent_text_message(f"Error: {str(e)}")
            )

    async def _execute_with_tracing(
        self,
        message: str,
        event_queue: EventQueue,
        session_id: str = None,
    ) -> None:
        """Execute agent with OpenTelemetry tracing.

        Creates a clean root span with user-friendly name and OpenInference
        attributes for better Phoenix display. This span becomes the primary
        trace root, replacing the verbose A2A SDK span names.
        """
        # Use server_name for root span (distinguishes from LangGraph agent name)
        # Falls back to agent name if server_name not provided
        agent_name = getattr(self.agent, "name", "MaskAgent")
        span_name = self.server_name or agent_name

        try:
            from opentelemetry import trace
            from opentelemetry.context import Context

            tracer = trace.get_tracer("mask.a2a")

            # Create a NEW root span by passing empty context (no parent)
            # This breaks the link to A2A's parent span, making ours the root
            with tracer.start_as_current_span(
                name=span_name,
                context=Context(),  # Empty context = no parent = root span
                attributes={
                    "input.value": message,
                    "input.mime_type": "text/plain",
                    "mask.agent.name": agent_name,
                    "mask.server.name": span_name,
                    "openinference.span.kind": "AGENT",
                },
            ) as span:
                # Set session.id for Phoenix session linking
                if session_id:
                    span.set_attribute("session.id", session_id)

                # Execute with session context for child spans
                if session_id:
                    try:
                        from openinference.instrumentation import using_session

                        with using_session(session_id):
                            response_text = await self._execute_and_capture(
                                message, event_queue, session_id
                            )
                    except ImportError:
                        response_text = await self._execute_and_capture(
                            message, event_queue, session_id
                        )
                else:
                    response_text = await self._execute_and_capture(
                        message, event_queue, session_id
                    )

                # Set output after execution
                if response_text:
                    span.set_attribute("output.value", response_text)
                    span.set_attribute("output.mime_type", "text/plain")

        except ImportError:
            logger.debug("OpenTelemetry not available, executing without tracing")
            await self._execute_and_capture(message, event_queue, session_id)
        except Exception as e:
            logger.warning("Tracing setup failed: %s, executing without tracing", e)
            await self._execute_and_capture(message, event_queue, session_id)

    async def _execute_and_capture(
        self,
        message: str,
        event_queue: EventQueue,
        session_id: str = None,
    ) -> str:
        """Execute agent and capture the response text.

        Returns:
            The response text from the agent.
        """
        if self.stream:
            return await self._execute_streaming_capture(message, event_queue, session_id)
        else:
            return await self._execute_non_streaming_capture(message, event_queue, session_id)

    async def _execute_non_streaming_capture(
        self,
        message: str,
        event_queue: EventQueue,
        session_id: str = None,
    ) -> str:
        """Execute agent without streaming and capture response."""
        response = await self.agent.invoke(message, session_id=session_id)
        await event_queue.enqueue_event(new_agent_text_message(response))
        return response

    async def _execute_streaming_capture(
        self,
        message: str,
        event_queue: EventQueue,
        session_id: str = None,
    ) -> str:
        """Execute agent with streaming and capture response."""
        full_response = ""
        async for chunk in self.agent.stream(message, session_id=session_id):
            full_response += chunk
        await event_queue.enqueue_event(new_agent_text_message(full_response))
        return full_response

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
