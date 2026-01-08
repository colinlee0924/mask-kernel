"""A2A Agent Executor.

This module bridges MASK agents and LangChain CompiledStateGraph to A2A
AgentExecutor interface, following patterns from a2a-python-samples.

Supports:
- LangChain CompiledStateGraph from create_agent() (recommended)
- MASK BaseAgent (legacy)
- Real-time streaming via TaskArtifactUpdateEvent
- Multi-agent handoffs with context isolation
- Frontend Source of Truth pattern for Open WebUI integration

Usage:
    from langchain.agents import create_agent
    from mask.a2a import create_a2a_executor

    graph = create_agent(model, tools, system_prompt)
    executor = create_a2a_executor(graph, server_name="my-agent")
"""

import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Artifact, Part, TaskArtifactUpdateEvent, TextPart
from a2a.utils import new_agent_text_message

from mask.core.state import HandoffContext
from mask.observability.attributes import (
    set_span_io,
    set_span_metadata,
    set_span_session,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from mask.agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


def _create_text_artifact(
    artifact_id: str,
    name: str,
    text: str,
) -> Artifact:
    """Create a text artifact with a specific artifact_id.

    Unlike new_text_artifact() which generates a new ID each time,
    this function allows reusing the same ID for streaming append operations.

    Args:
        artifact_id: The artifact ID to use.
        name: Human-readable name for the artifact.
        text: The text content.

    Returns:
        Artifact with the specified ID.
    """
    return Artifact(
        artifact_id=artifact_id,
        name=name,
        parts=[Part(root=TextPart(text=text))],
    )


class MaskAgentExecutor(AgentExecutor):
    """Bridge LangChain CompiledStateGraph or MASK BaseAgent to A2A AgentExecutor.

    This executor supports two agent types:
    - LangChain CompiledStateGraph from create_agent() (recommended)
    - MASK BaseAgent (legacy)

    Features:
    - Real-time streaming via TaskArtifactUpdateEvent (default enabled)
    - Multi-agent handoffs with context isolation
    - OpenTelemetry tracing integration

    Example:
        from langchain.agents import create_agent
        from mask.a2a import create_a2a_executor

        graph = create_agent(model, tools, system_prompt)
        executor = create_a2a_executor(graph, server_name="my-agent")
    """

    def __init__(
        self,
        agent: Union["BaseAgent", "CompiledStateGraph"],
        stream: bool = True,
        server_name: str = None,
    ) -> None:
        """Initialize executor with agent.

        Args:
            agent: LangChain CompiledStateGraph or MASK BaseAgent instance.
            stream: Whether to use real-time streaming (default True for Open WebUI).
            server_name: A2A server name for trace display (e.g., "my-agent").
                        If not provided, falls back to agent name attribute.
        """
        self.agent = agent
        self.stream = stream
        self.server_name = server_name
        # Detect agent type: CompiledStateGraph has ainvoke but not invoke with session
        self._is_graph = hasattr(agent, "ainvoke") and not hasattr(
            agent, "invoke_with_session"
        )

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

        # Extract context_id and task_id for streaming events
        context_id = self._extract_context_id(context)
        task_id = self._extract_task_id(context)

        # Extract frontend history for "Frontend Source of Truth" pattern
        # This allows Open WebUI to control the conversation state
        frontend_history = self._extract_frontend_history(context)

        logger.debug(
            "Executing agent with message: %s... (session: %s, handoff: %s, history: %d msgs)",
            user_message[:50],
            session_id or "none",
            handoff_context.parent_agent if handoff_context else "none",
            len(frontend_history) if frontend_history else 0,
        )

        # Create a user-friendly root span for Phoenix display
        # This wraps the A2A infrastructure spans with a readable agent name
        try:
            await self._execute_with_tracing(
                user_message,
                event_queue,
                session_id,
                handoff_context,
                context_id,
                task_id,
                frontend_history,
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
        context_id: Optional[str] = None,
        task_id: Optional[str] = None,
        frontend_history: Optional[List[Dict[str, Any]]] = None,
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
                # Log trace ID for debugging
                trace_id = format(span.get_span_context().trace_id, '032x')
                logger.info("[TRACING] Created span: name=%s, trace_id=%s, session_id=%s",
                           span_name, trace_id, session_id)

                # Use multi-backend attribute utilities for compatibility
                # with Phoenix, Langfuse, and OpenTelemetry GenAI
                set_span_io(span, input_value=message)
                set_span_session(span, session_id=session_id, trace_name=span_name)
                set_span_metadata(span, agent_name=agent_name, server_name=span_name)

                # Log that session was set
                if session_id:
                    logger.info("[TRACING] Set session.id attribute: %s", session_id)

                # Execute with session context for child spans
                if session_id:
                    try:
                        from openinference.instrumentation import using_session

                        with using_session(session_id):
                            response_text = await self._execute_and_capture(
                                message,
                                event_queue,
                                session_id,
                                handoff_context,
                                context_id,
                                task_id,
                                frontend_history,
                            )
                    except ImportError:
                        response_text = await self._execute_and_capture(
                            message,
                            event_queue,
                            session_id,
                            handoff_context,
                            context_id,
                            task_id,
                            frontend_history,
                        )
                else:
                    response_text = await self._execute_and_capture(
                        message,
                        event_queue,
                        session_id,
                        handoff_context,
                        context_id,
                        task_id,
                        frontend_history,
                    )

                # Set output after execution (multi-backend compatible)
                if response_text:
                    set_span_io(span, output_value=response_text)

        except ImportError:
            logger.debug("OpenTelemetry not available, executing without tracing")
            await self._execute_and_capture(
                message,
                event_queue,
                session_id,
                handoff_context,
                context_id,
                task_id,
                frontend_history,
            )
        except Exception as e:
            logger.warning("Tracing setup failed: %s, executing without tracing", e)
            await self._execute_and_capture(
                message,
                event_queue,
                session_id,
                handoff_context,
                context_id,
                task_id,
                frontend_history,
            )

    async def _execute_and_capture(
        self,
        message: str,
        event_queue: EventQueue,
        session_id: Optional[str] = None,
        handoff_context: Optional[HandoffContext] = None,
        context_id: Optional[str] = None,
        task_id: Optional[str] = None,
        frontend_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Execute agent and capture the response text.

        Args:
            message: User message to process.
            event_queue: A2A event queue for sending responses.
            session_id: Session ID for tracing.
            handoff_context: Multi-agent handoff context.
            context_id: A2A context ID for streaming events.
            task_id: A2A task ID for streaming events.
            frontend_history: Full conversation history from frontend (Open WebUI).

        Returns:
            The response text from the agent.
        """
        if self.stream:
            return await self._execute_streaming_capture(
                message,
                event_queue,
                session_id,
                handoff_context,
                context_id,
                task_id,
                frontend_history,
            )
        else:
            return await self._execute_non_streaming_capture(
                message, event_queue, session_id, handoff_context, context_id, frontend_history
            )

    async def _execute_non_streaming_capture(
        self,
        message: str,
        event_queue: EventQueue,
        session_id: Optional[str] = None,
        handoff_context: Optional[HandoffContext] = None,
        context_id: Optional[str] = None,
        frontend_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Execute agent without streaming and capture response.

        Supports both CompiledStateGraph and BaseAgent.

        When frontend_history is provided (Frontend Source of Truth pattern),
        it is used as the conversation history instead of relying on
        LangGraph checkpoints.
        """
        if self._is_graph:
            from langchain_core.messages import HumanMessage

            # Build messages to send to the agent
            if frontend_history:
                # Frontend Source of Truth: use provided history
                # Convert to LangChain messages and add current user message
                langchain_messages = self._convert_frontend_to_langchain_messages(
                    frontend_history
                )
                # The current message should already be in frontend_history
                # but ensure it's there
                if not langchain_messages or (
                    hasattr(langchain_messages[-1], "content")
                    and langchain_messages[-1].content != message
                ):
                    langchain_messages.append(HumanMessage(content=message))

                # Use a unique thread_id per request to avoid checkpoint conflicts
                # Since we're using frontend as source of truth, we don't need
                # to persist state in LangGraph checkpoints
                thread_id = str(uuid.uuid4())
                config = {"configurable": {"thread_id": thread_id}}

                logger.debug(
                    "Using frontend history with %d messages (stateless mode)",
                    len(langchain_messages),
                )
            else:
                # No frontend history - use checkpoint-based mode
                # (original behavior for backward compatibility)
                langchain_messages = [HumanMessage(content=message)]
                thread_id = context_id or str(uuid.uuid4())
                config = {"configurable": {"thread_id": thread_id}}

            result = await self.agent.ainvoke(
                {"messages": langchain_messages},
                config=config,
            )
            # Extract last AI message content
            messages = result.get("messages", [])
            response = ""
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    response = last_msg.content
        else:
            # BaseAgent: invoke(message, session_id=..., handoff_context=...)
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
        context_id: Optional[str] = None,
        task_id: Optional[str] = None,
        frontend_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Execute agent with real-time streaming via TaskArtifactUpdateEvent.

        Sends each chunk immediately as it arrives, providing real-time feedback
        for UI clients like Open WebUI.

        Supports both CompiledStateGraph and BaseAgent.

        When frontend_history is provided (Frontend Source of Truth pattern),
        it is used as the conversation history instead of relying on
        LangGraph checkpoints.
        """
        full_response = ""
        artifact_id: Optional[str] = None

        # Generate IDs if not provided
        context_id = context_id or str(uuid.uuid4())
        task_id = task_id or str(uuid.uuid4())

        if self._is_graph:
            # CompiledStateGraph: use astream with stream_mode="messages"
            from langchain_core.messages import HumanMessage

            # Build messages to send to the agent
            if frontend_history:
                # Frontend Source of Truth: use provided history
                langchain_messages = self._convert_frontend_to_langchain_messages(
                    frontend_history
                )
                # Ensure current message is included
                if not langchain_messages or (
                    hasattr(langchain_messages[-1], "content")
                    and langchain_messages[-1].content != message
                ):
                    langchain_messages.append(HumanMessage(content=message))

                # Use context_id as thread_id for session grouping consistency
                # Since we're in stateless mode (no checkpointer), same thread_id
                # across requests is safe - each request builds fresh history
                thread_id = context_id or str(uuid.uuid4())
                config = {"configurable": {"thread_id": thread_id}}

                logger.debug(
                    "Streaming with frontend history: %d messages (stateless mode)",
                    len(langchain_messages),
                )
            else:
                # No frontend history - use checkpoint-based mode
                langchain_messages = [HumanMessage(content=message)]
                thread_id = context_id or str(uuid.uuid4())
                config = {"configurable": {"thread_id": thread_id}}

            async for event in self.agent.astream(
                {"messages": langchain_messages},
                config=config,
                stream_mode="messages",
            ):
                # Extract content from streaming events
                if isinstance(event, tuple) and len(event) == 2:
                    msg, metadata = event
                    if hasattr(msg, "content") and msg.content:
                        content = msg.content
                        # Handle both str and list content types
                        if isinstance(content, list):
                            # Content blocks format: [{'text': '...', 'type': 'text', 'index': 0}]
                            chunk = "".join(
                                item.get("text", "") if isinstance(item, dict) else str(item)
                                for item in content
                            )
                        else:
                            chunk = str(content)

                        if not chunk:
                            continue

                        full_response += chunk

                        # Send chunk immediately via TaskArtifactUpdateEvent
                        if artifact_id is None:
                            # First chunk - generate artifact ID and create artifact
                            artifact_id = str(uuid.uuid4())
                            artifact = _create_text_artifact(artifact_id, "response", chunk)
                            await event_queue.enqueue_event(
                                TaskArtifactUpdateEvent(
                                    artifact=artifact,
                                    contextId=context_id,
                                    taskId=task_id,
                                    append=False,
                                )
                            )
                        else:
                            # Subsequent chunks - append using SAME artifact_id
                            artifact = _create_text_artifact(artifact_id, "response", chunk)
                            await event_queue.enqueue_event(
                                TaskArtifactUpdateEvent(
                                    artifact=artifact,
                                    contextId=context_id,
                                    taskId=task_id,
                                    append=True,
                                )
                            )

            # Send final chunk marker with same artifact_id
            if artifact_id:
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        artifact=_create_text_artifact(artifact_id, "response", ""),
                        contextId=context_id,
                        taskId=task_id,
                        append=True,
                        lastChunk=True,
                    )
                )
        else:
            # BaseAgent: use stream() method
            async for chunk in self.agent.stream(
                message, session_id=session_id, handoff_context=handoff_context
            ):
                full_response += chunk

                # Send chunk immediately via TaskArtifactUpdateEvent
                if artifact_id is None:
                    # First chunk - generate artifact ID and create artifact
                    artifact_id = str(uuid.uuid4())
                    artifact = _create_text_artifact(artifact_id, "response", chunk)
                    await event_queue.enqueue_event(
                        TaskArtifactUpdateEvent(
                            artifact=artifact,
                            contextId=context_id,
                            taskId=task_id,
                            append=False,
                        )
                    )
                else:
                    # Subsequent chunks - append using SAME artifact_id
                    artifact = _create_text_artifact(artifact_id, "response", chunk)
                    await event_queue.enqueue_event(
                        TaskArtifactUpdateEvent(
                            artifact=artifact,
                            contextId=context_id,
                            taskId=task_id,
                            append=True,
                        )
                    )

            # Send final chunk marker with same artifact_id
            if artifact_id:
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        artifact=_create_text_artifact(artifact_id, "response", ""),
                        contextId=context_id,
                        taskId=task_id,
                        append=True,
                        lastChunk=True,
                    )
                )

        # Also send final message for non-streaming clients
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
            context_id_value = getattr(message, "context_id", None)
            logger.info(
                "[SESSION] Extracting from message: context_id=%s, type=%s",
                context_id_value,
                type(context_id_value).__name__,
            )
            if context_id_value:
                logger.info("[SESSION] Using message.context_id: %s", context_id_value)
                return context_id_value

        # Fallback: check RequestContext for context_id
        if hasattr(context, "context_id") and context.context_id:
            logger.info("[SESSION] Using context.context_id: %s", context.context_id)
            return context.context_id

        logger.warning("[SESSION] No session_id found in context!")
        return None

    def _extract_context_id(self, context: RequestContext) -> Optional[str]:
        """Extract context ID from A2A request context for streaming events.

        Args:
            context: The request context.

        Returns:
            Context ID if found, generates a UUID otherwise.
        """
        message = context.message
        if message:
            if hasattr(message, "context_id") and message.context_id:
                return message.context_id

        if hasattr(context, "context_id") and context.context_id:
            return context.context_id

        return str(uuid.uuid4())

    def _extract_task_id(self, context: RequestContext) -> Optional[str]:
        """Extract task ID from A2A request context for streaming events.

        Args:
            context: The request context.

        Returns:
            Task ID if found, generates a UUID otherwise.
        """
        if hasattr(context, "task_id") and context.task_id:
            return context.task_id

        message = context.message
        if message and hasattr(message, "task_id") and message.task_id:
            return message.task_id

        return str(uuid.uuid4())

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

    def _extract_frontend_history(
        self, context: RequestContext
    ) -> Optional[List[Dict[str, Any]]]:
        """Extract frontend message history from A2A request.

        This implements the "Frontend Source of Truth" pattern where Open WebUI
        sends the full conversation history, and we use that instead of relying
        on LangGraph checkpoints.

        The history is expected in message.metadata.history as a list of messages
        in OpenAI format:
        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]

        Args:
            context: The request context.

        Returns:
            List of message dicts if found, None otherwise.
        """
        message = context.message
        if not message:
            return None

        # Check message metadata for frontend history
        metadata: Optional[Dict[str, Any]] = None
        if hasattr(message, "metadata") and message.metadata:
            metadata = message.metadata
        elif hasattr(message, "root") and hasattr(message.root, "metadata"):
            metadata = message.root.metadata

        if not metadata:
            return None

        # Extract history from metadata
        history = metadata.get("history") or metadata.get("messages")
        if history and isinstance(history, list):
            logger.debug(
                "Extracted frontend history with %d messages", len(history)
            )
            return history

        return None

    def _convert_frontend_to_langchain_messages(
        self, frontend_messages: List[Dict[str, Any]]
    ) -> List[Any]:
        """Convert frontend (OpenAI format) messages to LangChain messages.

        This converts messages from Open WebUI format to LangChain format,
        filtering out tool calls since Open WebUI doesn't preserve them.

        Args:
            frontend_messages: List of messages in OpenAI format.

        Returns:
            List of LangChain message objects.
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        langchain_messages = []

        for msg in frontend_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            elif role == "system":
                langchain_messages.append(SystemMessage(content=content))
            # Skip tool messages - Open WebUI doesn't preserve them

        return langchain_messages

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
