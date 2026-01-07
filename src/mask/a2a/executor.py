"""A2A Agent Executor.

This module bridges MASK agents and LangChain CompiledStateGraph to A2A
AgentExecutor interface, following patterns from a2a-python-samples.

Supports:
- LangChain CompiledStateGraph from create_agent() (recommended)
- MASK BaseAgent (legacy)
- Real-time streaming via TaskArtifactUpdateEvent
- Multi-agent handoffs with context isolation

Usage:
    from langchain.agents import create_agent
    from mask.a2a import create_a2a_executor

    graph = create_agent(model, tools, system_prompt)
    executor = create_a2a_executor(graph, server_name="my-agent")
"""

import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import TaskArtifactUpdateEvent
from a2a.utils import new_agent_text_message, new_text_artifact

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
                user_message,
                event_queue,
                session_id,
                handoff_context,
                context_id,
                task_id,
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
                                message,
                                event_queue,
                                session_id,
                                handoff_context,
                                context_id,
                                task_id,
                            )
                    except ImportError:
                        response_text = await self._execute_and_capture(
                            message,
                            event_queue,
                            session_id,
                            handoff_context,
                            context_id,
                            task_id,
                        )
                else:
                    response_text = await self._execute_and_capture(
                        message,
                        event_queue,
                        session_id,
                        handoff_context,
                        context_id,
                        task_id,
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
            )

    async def _execute_and_capture(
        self,
        message: str,
        event_queue: EventQueue,
        session_id: Optional[str] = None,
        handoff_context: Optional[HandoffContext] = None,
        context_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """Execute agent and capture the response text.

        Args:
            message: User message to process.
            event_queue: A2A event queue for sending responses.
            session_id: Session ID for tracing.
            handoff_context: Multi-agent handoff context.
            context_id: A2A context ID for streaming events.
            task_id: A2A task ID for streaming events.

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
        """Execute agent without streaming and capture response.

        Supports both CompiledStateGraph and BaseAgent.
        """
        if self._is_graph:
            # CompiledStateGraph: invoke({"messages": [HumanMessage(...)]})
            from langchain_core.messages import HumanMessage

            result = await self.agent.ainvoke(
                {"messages": [HumanMessage(content=message)]}
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
    ) -> str:
        """Execute agent with real-time streaming via TaskArtifactUpdateEvent.

        Sends each chunk immediately as it arrives, providing real-time feedback
        for UI clients like Open WebUI.

        Supports both CompiledStateGraph and BaseAgent.
        """
        full_response = ""
        artifact_id: Optional[str] = None

        # Generate IDs if not provided
        context_id = context_id or str(uuid.uuid4())
        task_id = task_id or str(uuid.uuid4())

        if self._is_graph:
            # CompiledStateGraph: use astream with stream_mode="messages"
            from langchain_core.messages import HumanMessage

            async for event in self.agent.astream(
                {"messages": [HumanMessage(content=message)]},
                stream_mode="messages",
            ):
                # Extract content from streaming events
                if isinstance(event, tuple) and len(event) == 2:
                    msg, metadata = event
                    if hasattr(msg, "content") and msg.content:
                        content = msg.content
                        # Handle both str and list content types
                        if isinstance(content, list):
                            # Join list content (e.g., from tool calls)
                            chunk = "".join(
                                str(item) if not isinstance(item, dict) else ""
                                for item in content
                            )
                        else:
                            chunk = str(content)

                        if not chunk:
                            continue

                        full_response += chunk

                        # Send chunk immediately via TaskArtifactUpdateEvent
                        if artifact_id is None:
                            # First chunk - create new artifact
                            artifact = new_text_artifact("response", chunk)
                            artifact_id = artifact.artifact_id
                            await event_queue.enqueue_event(
                                TaskArtifactUpdateEvent(
                                    artifact=artifact,
                                    contextId=context_id,
                                    taskId=task_id,
                                    append=False,
                                )
                            )
                        else:
                            # Subsequent chunks - append
                            await event_queue.enqueue_event(
                                TaskArtifactUpdateEvent(
                                    artifact=new_text_artifact("response", chunk),
                                    contextId=context_id,
                                    taskId=task_id,
                                    append=True,
                                )
                            )

            # Send final chunk marker
            if artifact_id:
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        artifact=new_text_artifact("response", ""),
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
                    artifact = new_text_artifact("response", chunk)
                    artifact_id = artifact.artifact_id
                    await event_queue.enqueue_event(
                        TaskArtifactUpdateEvent(
                            artifact=artifact,
                            contextId=context_id,
                            taskId=task_id,
                            append=False,
                        )
                    )
                else:
                    await event_queue.enqueue_event(
                        TaskArtifactUpdateEvent(
                            artifact=new_text_artifact("response", chunk),
                            contextId=context_id,
                            taskId=task_id,
                            append=True,
                        )
                    )

            # Send final chunk marker
            if artifact_id:
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        artifact=new_text_artifact("response", ""),
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
            if hasattr(message, "context_id") and message.context_id:
                return message.context_id

        # Fallback: check RequestContext for context_id
        if hasattr(context, "context_id") and context.context_id:
            return context.context_id

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
