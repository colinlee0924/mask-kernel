"""A2A Agent Executor.

This module bridges MASK agents and LangChain CompiledStateGraph to A2A
AgentExecutor interface, following patterns from a2a-python-samples.

Supports:
- LangChain CompiledStateGraph from create_agent() (recommended)
- MASK BaseAgent (legacy)
- Real-time streaming via TaskArtifactUpdateEvent
- Multi-agent handoffs with context isolation
- PostgreSQL persistence via LangGraph checkpointer
- Session history synchronization with frontend (Open WebUI)

Usage:
    from langchain.agents import create_agent
    from mask.a2a import create_a2a_executor

    graph = create_agent(model, tools, system_prompt)
    executor = create_a2a_executor(graph, server_name="my-agent")
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

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
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from mask.a2a.delegation import DelegationToolFactory
    from mask.a2a.state_sync import StateSynchronizer, SyncResult
    from mask.agent.base_agent import BaseAgent
    from mask.middleware.a2a_streaming import A2AStreamingMiddleware
    from mask.storage.base import SessionStore

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
        agent: BaseAgent | CompiledStateGraph,
        stream: bool = True,
        server_name: str = None,
        checkpointer: BaseCheckpointSaver | None = None,
        session_store: SessionStore | None = None,
        streaming_middleware: A2AStreamingMiddleware | None = None,
        delegation_factory: DelegationToolFactory | None = None,
    ) -> None:
        """Initialize executor with agent.

        Args:
            agent: LangChain CompiledStateGraph or MASK BaseAgent instance.
            stream: Whether to use real-time streaming (default True for Open WebUI).
            server_name: A2A server name for trace display (e.g., "my-agent").
                        If not provided, falls back to agent name attribute.
            checkpointer: Optional LangGraph checkpointer for persistence.
                         If provided, enables session history persistence.
            session_store: Optional MASK SessionStore for session metadata.
            streaming_middleware: Optional A2AStreamingMiddleware for event propagation.
                         If provided, the middleware's event_queue is set dynamically
                         before each execution to enable real-time event streaming.
            delegation_factory: Optional DelegationToolFactory for multi-agent delegation.
                         If provided, the factory's event_queue is set dynamically
                         before each execution to enable sub-agent event propagation.
        """
        self.agent = agent
        self.stream = stream
        self.server_name = server_name
        self.checkpointer = checkpointer
        self.session_store = session_store
        self.streaming_middleware = streaming_middleware
        self.delegation_factory = delegation_factory
        # Track last checkpoint_id for metadata injection
        self._last_checkpoint_id: str | None = None
        # Detect agent type: CompiledStateGraph has ainvoke but not invoke with session
        self._is_graph = hasattr(agent, "ainvoke") and not hasattr(
            agent, "invoke_with_session"
        )
        # Initialize StateSynchronizer for message sync (only for graph-based agents)
        self._synchronizer: StateSynchronizer | None = None
        if self._is_graph:
            from mask.a2a.state_sync import StateSynchronizer
            self._synchronizer = StateSynchronizer(agent, checkpointer)

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

        # Extract full history for sync detection (Open WebUI sends this)
        full_history = self._extract_full_history(context)

        # Analyze sync state if we have history and a synchronizer
        sync_result: SyncResult | None = None
        if full_history and self._synchronizer and context_id:
            sync_result = await self._synchronizer.analyze(context_id, full_history)
            if sync_result:
                logger.info(
                    "[SYNC] Thread: %s | Action: %s | Message: %s",
                    context_id,
                    sync_result.action,
                    sync_result.message,
                )

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
                sync_result,
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
        session_id: str | None = None,
        handoff_context: HandoffContext | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
        sync_result: SyncResult | None = None,
    ) -> None:
        """Execute agent with OpenTelemetry tracing.

        Creates a clean root span with user-friendly name and multi-backend
        compatible attributes (Phoenix/OpenInference, Langfuse, GenAI).
        This span becomes the primary trace root, replacing the verbose
        A2A SDK span names.

        Args:
            message: User message to process.
            event_queue: A2A event queue for sending responses.
            session_id: Session ID for tracing.
            handoff_context: Multi-agent handoff context.
            context_id: A2A context ID (used as thread_id).
            task_id: A2A task ID for streaming events.
            sync_result: Result from StateSynchronizer.analyze() for rollback handling.
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
            span_attributes: dict[str, Any] = {
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
                                sync_result,
                            )
                    except ImportError:
                        response_text = await self._execute_and_capture(
                            message,
                            event_queue,
                            session_id,
                            handoff_context,
                            context_id,
                            task_id,
                            sync_result,
                        )
                else:
                    response_text = await self._execute_and_capture(
                        message,
                        event_queue,
                        session_id,
                        handoff_context,
                        context_id,
                        task_id,
                        sync_result,
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
                sync_result,
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
                sync_result,
            )

    async def _execute_and_capture(
        self,
        message: str,
        event_queue: EventQueue,
        session_id: str | None = None,
        handoff_context: HandoffContext | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
        sync_result: SyncResult | None = None,
    ) -> str:
        """Execute agent and capture the response text.

        Args:
            message: User message to process.
            event_queue: A2A event queue for sending responses.
            session_id: Session ID for tracing.
            handoff_context: Multi-agent handoff context.
            context_id: A2A context ID for streaming events.
            task_id: A2A task ID for streaming events.
            sync_result: Result from StateSynchronizer.analyze() for rollback handling.

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
                sync_result,
            )
        else:
            return await self._execute_non_streaming_capture(
                message, event_queue, session_id, handoff_context, context_id, sync_result
            )

    async def _execute_non_streaming_capture(
        self,
        message: str,
        event_queue: EventQueue,
        session_id: str | None = None,
        handoff_context: HandoffContext | None = None,
        context_id: str | None = None,
        sync_result: SyncResult | None = None,
    ) -> str:
        """Execute agent without streaming and capture response.

        Supports both CompiledStateGraph and BaseAgent.

        Args:
            message: User message to process.
            event_queue: A2A event queue for sending responses.
            session_id: Session ID for tracing.
            handoff_context: Multi-agent handoff context.
            context_id: A2A context ID (used as thread_id).
            sync_result: Result from StateSynchronizer.analyze() for rollback handling.

        Returns:
            The response text from the agent.
        """
        if self._is_graph:
            # CompiledStateGraph: invoke({"messages": [HumanMessage(...)]})
            from langchain_core.messages import HumanMessage

            # Use context_id as thread_id for multi-turn conversation memory
            thread_id = context_id or str(uuid.uuid4())

            # Build config with sync result (may include checkpoint_id for rollback)
            if sync_result and self._synchronizer:
                config = self._synchronizer.get_invoke_config(thread_id, sync_result)
            else:
                config = {"configurable": {"thread_id": thread_id}}

            result = await self.agent.ainvoke(
                {"messages": [HumanMessage(content=message)]},
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
        session_id: str | None = None,
        handoff_context: HandoffContext | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
        sync_result: SyncResult | None = None,
    ) -> str:
        """Execute agent with real-time streaming via TaskArtifactUpdateEvent.

        Sends each chunk immediately as it arrives, providing real-time feedback
        for UI clients like Open WebUI.

        Supports both CompiledStateGraph and BaseAgent.
        Uses astream_events() for rich event streaming (thinking, tool calls, etc.)

        Args:
            message: User message to process.
            event_queue: A2A event queue for sending responses.
            session_id: Session ID for tracing.
            handoff_context: Multi-agent handoff context.
            context_id: A2A context ID (used as thread_id).
            task_id: A2A task ID for streaming events.
            sync_result: Result from StateSynchronizer.analyze() for rollback handling.

        Returns:
            The response text from the agent.
        """
        full_response = ""

        # Generate IDs if not provided
        context_id = context_id or str(uuid.uuid4())
        task_id = task_id or str(uuid.uuid4())

        if self._is_graph:
            # CompiledStateGraph: use astream_events for rich streaming
            full_response = await self._execute_rich_streaming(
                message, event_queue, context_id, task_id, sync_result
            )
        else:
            # BaseAgent: use stream() method
            artifact_id: str | None = None
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

    async def _execute_rich_streaming(
        self,
        message: str,
        event_queue: EventQueue,
        context_id: str,
        task_id: str,
        sync_result: SyncResult | None = None,
    ) -> str:
        """Execute with rich streaming using astream_events().

        Streams different event types for UI rendering:
        - thinking: Model reasoning process (extended thinking)
        - tool_call: Tool invocation with arguments
        - tool_result: Tool execution result
        - response: Final text response

        Each event type uses a unique artifact_id for proper UI rendering.

        If streaming_middleware is configured, its event_queue is set to enable
        real-time event propagation from middleware hooks.

        Args:
            message: User message to process.
            event_queue: A2A event queue for sending responses.
            context_id: A2A context ID (used as thread_id).
            task_id: A2A task ID for streaming events.
            sync_result: Result from StateSynchronizer.analyze() for rollback handling.

        Returns:
            The response text from the agent.
        """
        from langchain_core.messages import HumanMessage

        # Set up streaming middleware with event_queue and context for this execution
        if self.streaming_middleware:
            self.streaming_middleware.event_queue = event_queue
            self.streaming_middleware.context_id = context_id
            self.streaming_middleware.task_id = task_id
            self.streaming_middleware.reset()

        # Set up delegation factory with event_queue and context for sub-agent events
        if self.delegation_factory:
            self.delegation_factory.event_queue = event_queue
            self.delegation_factory.context_id = context_id
            self.delegation_factory.task_id = task_id

        full_response = ""
        thread_id = context_id or str(uuid.uuid4())

        # Build config with sync result (may include checkpoint_id for rollback)
        if sync_result and self._synchronizer:
            config = self._synchronizer.get_invoke_config(thread_id, sync_result)
        else:
            config = {"configurable": {"thread_id": thread_id}}

        # Artifact IDs for different event types (reused for appending)
        response_artifact_id: str | None = None
        thinking_artifact_id: str | None = None
        current_tool_artifact_id: str | None = None

        # Track tool calls for proper result matching
        active_tool_calls: dict[str, str] = {}  # run_id -> tool_name

        async for event in self.agent.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            version="v2",
        ):
            kind = event.get("event", "")
            data = event.get("data", {})

            # ===== Extended Thinking / Reasoning =====
            if kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content"):
                    content = chunk.content

                    # Check for thinking blocks (Claude extended thinking)
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                block_type = block.get("type", "")

                                # Thinking block (Claude extended thinking)
                                if block_type == "thinking":
                                    thinking_text = block.get("thinking", "")
                                    if thinking_text:
                                        if thinking_artifact_id is None:
                                            thinking_artifact_id = str(uuid.uuid4())
                                            artifact = _create_text_artifact(
                                                thinking_artifact_id, "thinking", thinking_text
                                            )
                                            await event_queue.enqueue_event(
                                                TaskArtifactUpdateEvent(
                                                    artifact=artifact,
                                                    contextId=context_id,
                                                    taskId=task_id,
                                                    append=False,
                                                )
                                            )
                                        else:
                                            artifact = _create_text_artifact(
                                                thinking_artifact_id, "thinking", thinking_text
                                            )
                                            await event_queue.enqueue_event(
                                                TaskArtifactUpdateEvent(
                                                    artifact=artifact,
                                                    contextId=context_id,
                                                    taskId=task_id,
                                                    append=True,
                                                )
                                            )

                                # Text block (normal response)
                                elif block_type == "text" or "text" in block:
                                    text = block.get("text", "")
                                    if text:
                                        full_response += text
                                        if response_artifact_id is None:
                                            response_artifact_id = str(uuid.uuid4())
                                            artifact = _create_text_artifact(
                                                response_artifact_id, "response", text
                                            )
                                            await event_queue.enqueue_event(
                                                TaskArtifactUpdateEvent(
                                                    artifact=artifact,
                                                    contextId=context_id,
                                                    taskId=task_id,
                                                    append=False,
                                                )
                                            )
                                        else:
                                            artifact = _create_text_artifact(
                                                response_artifact_id, "response", text
                                            )
                                            await event_queue.enqueue_event(
                                                TaskArtifactUpdateEvent(
                                                    artifact=artifact,
                                                    contextId=context_id,
                                                    taskId=task_id,
                                                    append=True,
                                                )
                                            )
                    elif isinstance(content, str) and content:
                        # Simple string content
                        full_response += content
                        if response_artifact_id is None:
                            response_artifact_id = str(uuid.uuid4())
                            artifact = _create_text_artifact(
                                response_artifact_id, "response", content
                            )
                            await event_queue.enqueue_event(
                                TaskArtifactUpdateEvent(
                                    artifact=artifact,
                                    contextId=context_id,
                                    taskId=task_id,
                                    append=False,
                                )
                            )
                        else:
                            artifact = _create_text_artifact(
                                response_artifact_id, "response", content
                            )
                            await event_queue.enqueue_event(
                                TaskArtifactUpdateEvent(
                                    artifact=artifact,
                                    contextId=context_id,
                                    taskId=task_id,
                                    append=True,
                                )
                            )

            # ===== Tool Call Start =====
            elif kind == "on_tool_start":
                run_id = event.get("run_id", "")
                tool_name = event.get("name", "unknown")
                tool_input = data.get("input", {})

                # Track this tool call
                active_tool_calls[run_id] = tool_name

                # Create tool_call artifact with input
                # Extract only JSON-serializable parts of tool_input
                current_tool_artifact_id = str(uuid.uuid4())

                def extract_serializable_input(obj):
                    """Extract only JSON-serializable data from tool input."""
                    if obj is None or isinstance(obj, (str, int, float, bool)):
                        return obj
                    elif isinstance(obj, dict):
                        result = {}
                        for k, v in obj.items():
                            # Skip runtime/internal keys
                            if k in ("runtime", "config", "callbacks", "state"):
                                continue
                            try:
                                serialized = extract_serializable_input(v)
                                if serialized is not None:
                                    result[k] = serialized
                            except (TypeError, ValueError):
                                pass
                        return result if result else None
                    elif isinstance(obj, (list, tuple)):
                        result = []
                        for item in obj:
                            try:
                                serialized = extract_serializable_input(item)
                                if serialized is not None:
                                    result.append(serialized)
                            except (TypeError, ValueError):
                                pass
                        return result if result else None
                    else:
                        # Try to get a simple string representation
                        try:
                            json.dumps(obj)
                            return obj
                        except (TypeError, ValueError):
                            return None

                try:
                    serializable_input = extract_serializable_input(tool_input)
                    if serializable_input is None:
                        serializable_input = {}
                    tool_info = json.dumps(
                        {"tool": tool_name, "input": serializable_input, "status": "running"},
                        ensure_ascii=False,
                        indent=2,
                    )
                except Exception:
                    tool_info = json.dumps(
                        {"tool": tool_name, "input": {}, "status": "running"},
                        ensure_ascii=False,
                        indent=2,
                    )
                artifact = _create_text_artifact(
                    current_tool_artifact_id, "tool_call", tool_info
                )
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        artifact=artifact,
                        contextId=context_id,
                        taskId=task_id,
                        append=False,
                    )
                )
                logger.debug("Tool started: %s with input: %s", tool_name, tool_input)

            # ===== Tool Call End =====
            elif kind == "on_tool_end":
                run_id = event.get("run_id", "")
                tool_name = active_tool_calls.pop(run_id, "unknown")
                output = data.get("output", "")

                # Send tool_result artifact
                result_artifact_id = str(uuid.uuid4())

                # Handle different output types
                if hasattr(output, "content"):
                    output_str = str(output.content)
                else:
                    output_str = str(output) if output else ""

                result_info = json.dumps(
                    {"tool": tool_name, "output": output_str, "status": "completed"},
                    ensure_ascii=False,
                    indent=2,
                )
                artifact = _create_text_artifact(
                    result_artifact_id, "tool_result", result_info
                )
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        artifact=artifact,
                        contextId=context_id,
                        taskId=task_id,
                        append=False,
                    )
                )
                logger.debug("Tool completed: %s with output: %s...", tool_name, output_str[:100])

        # Send final markers for active artifacts
        if thinking_artifact_id:
            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    artifact=_create_text_artifact(thinking_artifact_id, "thinking", ""),
                    contextId=context_id,
                    taskId=task_id,
                    append=True,
                    lastChunk=True,
                )
            )

        if response_artifact_id:
            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    artifact=_create_text_artifact(response_artifact_id, "response", ""),
                    contextId=context_id,
                    taskId=task_id,
                    append=True,
                    lastChunk=True,
                )
            )

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

    def _extract_context_id(self, context: RequestContext) -> str | None:
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

    def _extract_task_id(self, context: RequestContext) -> str | None:
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
    ) -> HandoffContext | None:
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
        # Check context.metadata first (A2A SDK property from MessageSendParams)
        # This is where metadata is passed in message/send params
        metadata: dict[str, Any] | None = None

        # Primary: context.metadata (from MessageSendParams.metadata)
        if hasattr(context, "metadata") and context.metadata:
            metadata = context.metadata
            logger.debug("Found metadata on context: %s", list(metadata.keys()))

        # Fallback: check message.metadata
        if not metadata:
            message = context.message
            if message:
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

        logger.debug("Found handoff_context in metadata: %s", handoff_data)

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

    # =========================================================================
    # Persistence & Sync Methods (Frontend Source of Truth Pattern)
    # =========================================================================

    def _extract_full_history(
        self, context: RequestContext
    ) -> list[dict[str, Any]] | None:
        """Extract full message history from A2A request configuration.

        The frontend (Open WebUI via Pipe Function) sends complete message
        history in the configuration.fullHistory field. This is the "Frontend
        Source of Truth" pattern.

        Args:
            context: The request context.

        Returns:
            List of message dictionaries if found, None otherwise.
        """
        message = context.message
        if not message:
            return None

        # Check message metadata for fullHistory (from A2A configuration)
        metadata: dict[str, Any] | None = None
        if hasattr(message, "metadata") and message.metadata:
            metadata = message.metadata
        elif hasattr(message, "root") and hasattr(message.root, "metadata"):
            metadata = message.root.metadata

        if not metadata:
            return None

        # Extract fullHistory from configuration
        config = metadata.get("configuration") or metadata
        full_history = config.get("fullHistory") or config.get("full_history")

        if isinstance(full_history, list):
            return full_history

        return None

    def _extract_metadata_from_messages(
        self, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Extract LangGraph/A2A metadata from frontend messages.

        Extracts checkpoint_id, thread_id, and task_id from message metadata
        for sync detection and forking.

        Args:
            messages: List of message dictionaries from frontend.

        Returns:
            Dictionary with extracted metadata:
            - last_checkpoint_id: Most recent checkpoint ID
            - thread_id: Thread/session ID
            - task_ids: List of task IDs
        """
        result = {
            "last_checkpoint_id": None,
            "thread_id": None,
            "task_ids": [],
        }

        if not messages:
            return result

        # Iterate through messages to find metadata
        for msg in reversed(messages):
            metadata = msg.get("metadata", {})
            if not metadata:
                continue

            # Extract checkpoint_id from assistant messages
            if not result["last_checkpoint_id"]:
                ckpt_id = metadata.get("langgraph_checkpoint_id")
                if ckpt_id:
                    result["last_checkpoint_id"] = ckpt_id

            # Extract thread_id
            if not result["thread_id"]:
                tid = metadata.get("langgraph_thread_id")
                if tid:
                    result["thread_id"] = tid

            # Collect task_ids
            task_id = metadata.get("a2a_task_id")
            if task_id and task_id not in result["task_ids"]:
                result["task_ids"].append(task_id)

        return result

    def _inject_metadata_to_response(
        self,
        response_text: str,
        checkpoint_id: str | None = None,
        thread_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Create response with injected metadata for frontend tracking.

        Injects LangGraph checkpoint_id, thread_id, and A2A task_id into
        the response metadata. Frontend can use these IDs for sync operations.

        Args:
            response_text: The response content.
            checkpoint_id: LangGraph checkpoint ID.
            thread_id: LangGraph thread ID.
            task_id: A2A task ID.

        Returns:
            Response dictionary with metadata.
        """
        metadata = {}
        if checkpoint_id:
            metadata["langgraph_checkpoint_id"] = checkpoint_id
        if thread_id:
            metadata["langgraph_thread_id"] = thread_id
        if task_id:
            metadata["a2a_task_id"] = task_id

        return {
            "role": "assistant",
            "content": response_text,
            "metadata": metadata if metadata else None,
        }

    def _compute_messages_hash(self, messages: list[dict[str, Any]]) -> str:
        """Compute hash of message contents for change detection.

        Args:
            messages: List of message dictionaries.

        Returns:
            SHA-256 hash of message contents.
        """
        # Extract content from messages for hashing
        contents = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")
            contents.append(f"{role}:{content}")

        hash_input = "\n".join(contents)
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    def _detect_regenerate(
        self,
        frontend_messages: list[dict[str, Any]],
        last_checkpoint_id: str | None,
    ) -> bool:
        """Detect if this is a regenerate (retry) request.

        A regenerate is detected when:
        1. Frontend sends same user message as before
        2. The last message is a user message (assistant response was deleted)

        Args:
            frontend_messages: Messages from frontend.
            last_checkpoint_id: Last checkpoint ID from frontend metadata.

        Returns:
            True if this appears to be a regenerate request.
        """
        if not frontend_messages:
            return False

        # If frontend provides a checkpoint_id that's not the latest,
        # and the last message is from user, it's likely a regenerate
        last_msg = frontend_messages[-1]
        if last_msg.get("role") == "user":
            # Check if we have a checkpoint_id that indicates time travel
            if last_checkpoint_id:
                logger.debug(
                    "Potential regenerate detected: user message with checkpoint_id %s",
                    last_checkpoint_id,
                )
                return True

        return False

    def _detect_deletion(
        self,
        frontend_messages: list[dict[str, Any]],
        backend_messages: list[dict[str, Any]],
    ) -> list[str]:
        """Detect deleted messages by comparing frontend and backend.

        Args:
            frontend_messages: Messages from frontend (source of truth).
            backend_messages: Messages from LangGraph checkpoint.

        Returns:
            List of message IDs that were deleted.
        """
        if not backend_messages:
            return []

        # Build set of frontend message hashes
        frontend_hashes = set()
        for msg in frontend_messages:
            content = msg.get("content", "")
            role = msg.get("role", "")
            frontend_hashes.add(f"{role}:{content[:100]}")

        # Find backend messages not in frontend
        deleted_ids = []
        for msg in backend_messages:
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            role = msg.get("role", "") if isinstance(msg, dict) else ""
            msg_hash = f"{role}:{content[:100]}"
            if msg_hash not in frontend_hashes:
                msg_id = msg.get("id") if isinstance(msg, dict) else None
                if msg_id:
                    deleted_ids.append(msg_id)

        return deleted_ids

    async def _get_checkpoint_messages(
        self, thread_id: str
    ) -> list[dict[str, Any]]:
        """Get messages from LangGraph checkpoint.

        Args:
            thread_id: The thread ID to query.

        Returns:
            List of messages from checkpoint.
        """
        if not self._is_graph or not self.checkpointer:
            return []

        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = await self.agent.aget_state(config)
            if state and state.values:
                messages = state.values.get("messages", [])
                # Convert LangChain messages to dicts
                result = []
                for msg in messages:
                    if hasattr(msg, "content"):
                        result.append({
                            "role": getattr(msg, "type", "unknown"),
                            "content": msg.content,
                            "id": getattr(msg, "id", None),
                        })
                return result
        except Exception as e:
            logger.warning("Failed to get checkpoint messages: %s", e)

        return []

    async def _sync_to_checkpoint(
        self,
        thread_id: str,
        frontend_messages: list[dict[str, Any]],
        parent_checkpoint_id: str | None = None,
    ) -> str | None:
        """Sync frontend messages to LangGraph checkpoint.

        Uses update_state to reshape checkpoint to match frontend state.

        Args:
            thread_id: The thread ID.
            frontend_messages: Messages from frontend.
            parent_checkpoint_id: Optional checkpoint to fork from.

        Returns:
            New checkpoint ID if sync was performed, None otherwise.
        """
        if not self._is_graph:
            return None

        try:
            from langchain_core.messages import AIMessage, HumanMessage

            # Convert frontend messages to LangChain format
            lc_messages = []
            for msg in frontend_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user" or role == "human":
                    lc_messages.append(HumanMessage(content=content))
                elif role == "assistant" or role == "ai":
                    lc_messages.append(AIMessage(content=content))

            # Build config with optional checkpoint_id for forking
            config = {"configurable": {"thread_id": thread_id}}
            if parent_checkpoint_id:
                config["configurable"]["checkpoint_id"] = parent_checkpoint_id

            # Update state to match frontend
            await self.agent.aupdate_state(
                config,
                values={"messages": lc_messages},
            )

            logger.info(
                "Synced checkpoint for thread %s (parent: %s)",
                thread_id,
                parent_checkpoint_id,
            )

            # Get new checkpoint ID
            new_state = await self.agent.aget_state(config)
            if new_state and new_state.config:
                return new_state.config.get("configurable", {}).get("checkpoint_id")

        except Exception as e:
            logger.exception("Failed to sync checkpoint: %s", e)

        return None
