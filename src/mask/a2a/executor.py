"""A2A Agent Executor.

This module bridges MASK BaseAgent to A2A AgentExecutor interface,
following patterns from a2a-python-samples.

Supports multi-agent handoffs with context isolation:
- HandoffContext for passing initial_skills and context_data
- Task-scoped sessions for agent coordination
- Parent-child relationship tracking for observability
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import TaskState, TaskStatus
from a2a.utils import new_agent_text_message

from mask.core.state import HandoffContext
from mask.observability.attributes import (
    set_span_io,
    set_span_metadata,
    set_span_session,
)

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

        # Extract handoff context for multi-agent coordination
        handoff_context = self._extract_handoff_context(context)

        logger.debug(
            "Executing agent with message: %s... (session: %s, handoff: %s)",
            user_message[:50],
            session_id or "none",
            handoff_context.parent_agent if handoff_context else "none",
        )

        # Create a user-friendly root span for Phoenix display
        # This wraps the A2A infrastructure spans with a readable agent name
        try:
            await self._execute_with_tracing(
                user_message, event_queue, session_id, handoff_context
            )
        except Exception as e:
            logger.exception("Agent execution failed: %s", e)
            await event_queue.enqueue_event(
                new_agent_text_message(f"Error: {str(e)}")
            )

    async def _execute_with_tracing(
        self,
        message: str,
        event_queue: EventQueue,
        session_id: Optional[str] = None,
        handoff_context: Optional[HandoffContext] = None,
    ) -> None:
        """Execute agent with OpenTelemetry tracing.

        Creates a clean root span with user-friendly name and multi-backend
        compatible attributes (Phoenix/OpenInference, Langfuse, GenAI).
        This span becomes the primary trace root, replacing the verbose
        A2A SDK span names.
        """
        # Use server_name for root span (distinguishes from LangGraph agent name)
        # Falls back to agent name if server_name not provided
        agent_name = getattr(self.agent, "name", "MaskAgent")
        span_name = self.server_name or agent_name

        try:
            from opentelemetry import trace
            from opentelemetry.context import Context

            tracer = trace.get_tracer("mask.a2a")

            # Build span attributes
            span_attributes: Dict[str, Any] = {
                "openinference.span.kind": "AGENT",
            }

            # Add handoff context attributes for tracing
            if handoff_context:
                if handoff_context.parent_agent:
                    span_attributes["mask.handoff.parent_agent"] = (
                        handoff_context.parent_agent
                    )
                if handoff_context.task_id:
                    span_attributes["mask.handoff.task_id"] = handoff_context.task_id
                if handoff_context.initial_skills:
                    span_attributes["mask.handoff.initial_skills"] = ",".join(
                        handoff_context.initial_skills
                    )

            # Create a NEW root span by passing empty context (no parent)
            # This breaks the link to A2A's parent span, making ours the root
            with tracer.start_as_current_span(
                name=span_name,
                context=Context(),  # Empty context = no parent = root span
                attributes=span_attributes,
            ) as span:
                # Use multi-backend attribute utilities for compatibility
                # with Phoenix, Langfuse, and OpenTelemetry GenAI
                set_span_io(span, input_value=message)
                set_span_session(span, session_id=session_id, trace_name=span_name)
                set_span_metadata(span, agent_name=agent_name, server_name=span_name)

                # Execute with session context for child spans
                if session_id:
                    try:
                        from openinference.instrumentation import using_session

                        with using_session(session_id):
                            response_text = await self._execute_and_capture(
                                message, event_queue, session_id, handoff_context
                            )
                    except ImportError:
                        response_text = await self._execute_and_capture(
                            message, event_queue, session_id, handoff_context
                        )
                else:
                    response_text = await self._execute_and_capture(
                        message, event_queue, session_id, handoff_context
                    )

                # Set output after execution (multi-backend compatible)
                if response_text:
                    set_span_io(span, output_value=response_text)

        except ImportError:
            logger.debug("OpenTelemetry not available, executing without tracing")
            await self._execute_and_capture(
                message, event_queue, session_id, handoff_context
            )
        except Exception as e:
            logger.warning("Tracing setup failed: %s, executing without tracing", e)
            await self._execute_and_capture(
                message, event_queue, session_id, handoff_context
            )

    async def _execute_and_capture(
        self,
        message: str,
        event_queue: EventQueue,
        session_id: Optional[str] = None,
        handoff_context: Optional[HandoffContext] = None,
    ) -> str:
        """Execute agent and capture the response text.

        Returns:
            The response text from the agent.
        """
        if self.stream:
            return await self._execute_streaming_capture(
                message, event_queue, session_id, handoff_context
            )
        else:
            return await self._execute_non_streaming_capture(
                message, event_queue, session_id, handoff_context
            )

    async def _execute_non_streaming_capture(
        self,
        message: str,
        event_queue: EventQueue,
        session_id: Optional[str] = None,
        handoff_context: Optional[HandoffContext] = None,
    ) -> str:
        """Execute agent without streaming and capture response."""
        response = await self.agent.invoke(
            message, session_id=session_id, handoff_context=handoff_context
        )
        await event_queue.enqueue_event(new_agent_text_message(response))
        return response

    async def _execute_streaming_capture(
        self,
        message: str,
        event_queue: EventQueue,
        session_id: Optional[str] = None,
        handoff_context: Optional[HandoffContext] = None,
    ) -> str:
        """Execute agent with streaming and capture response."""
        full_response = ""
        async for chunk in self.agent.stream(
            message, session_id=session_id, handoff_context=handoff_context
        ):
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

    def _extract_session_id(self, context: RequestContext) -> Optional[str]:
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

    def _extract_handoff_context(
        self, context: RequestContext
    ) -> Optional[HandoffContext]:
        """Extract handoff context from A2A request context.

        Handoff context is passed via A2A message metadata for multi-agent
        coordination. It allows parent agents to:
        - Pre-activate skills in child agents (initial_skills)
        - Pass task-specific data without polluting conversation (context_data)
        - Track parent-child relationships (parent_agent, task_id)

        Args:
            context: The request context.

        Returns:
            HandoffContext if found, None otherwise.
        """
        message = context.message
        if not message:
            return None

        # Check message metadata for handoff context
        # A2A supports metadata field on messages
        metadata: Optional[Dict[str, Any]] = None
        if hasattr(message, "metadata") and message.metadata:
            metadata = message.metadata
        elif hasattr(message, "root") and hasattr(message.root, "metadata"):
            metadata = message.root.metadata

        if not metadata:
            return None

        # Extract handoff context from metadata
        handoff_data = metadata.get("handoff_context") or metadata.get("handoff")
        if not handoff_data:
            return None

        # Parse handoff context
        if isinstance(handoff_data, dict):
            return HandoffContext.from_dict(handoff_data)

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
