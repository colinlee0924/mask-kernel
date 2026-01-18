"""Delegation tool factory for multi-agent orchestration.

This module provides tools for delegating tasks from an orchestrator agent
to sub-agents via A2A protocol, using the native A2A SDK.

The delegation tools return LangGraph Command objects to:
1. Inject ToolMessage with sub-agent results
2. Update agent state (e.g., delegation_history)

Following the MASK skill loader pattern with Command support.

IMPORTANT: This module uses the native A2A SDK's ClientFactory and Client
classes for reliable communication, avoiding the event loop issues we
encountered with custom SSE parsing in StreamingA2AClient.
"""

import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Tuple
from uuid import uuid4

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command

from mask.a2a.remote_agent import NativeRemoteAgentFactory
from mask.core.events import AgentEvent

if TYPE_CHECKING:
    from a2a.server.events import EventQueue

logger = logging.getLogger(__name__)


class DelegationToolFactory:
    """Factory for creating delegation tools using native A2A SDK.

    Creates tools that delegate tasks to sub-agents. Uses the official
    A2A SDK's ClientFactory and Client for reliable communication.

    Example:
        factory = DelegationToolFactory()
        await factory.register_agent("http://localhost:10001", "jira-agent")
        await factory.register_agent("http://localhost:10002", "faq-agent")

        # Get tools for LLM routing
        tools = factory.get_tools()

        # Or send directly (for parameter routing)
        result = await factory.send_message_direct("jira-agent", "Hello")

    Attributes:
        event_queue: A2A EventQueue for streaming events (set dynamically).
        track_delegation_history: Whether to track delegation in agent state.
    """

    def __init__(
        self,
        event_queue: Optional["EventQueue"] = None,
        track_delegation_history: bool = True,
    ) -> None:
        """Initialize delegation tool factory.

        Args:
            event_queue: A2A EventQueue for streaming events (optional, can be set later).
            track_delegation_history: Whether to include delegation_history in Command update.
        """
        self.event_queue = event_queue
        self.track_delegation_history = track_delegation_history

        # Use native A2A SDK factory
        self._native_factory = NativeRemoteAgentFactory()
        self._descriptions: Dict[str, str] = {}

        # A2A context for streaming events (set by executor before each invocation)
        self.context_id: Optional[str] = None
        self.task_id: Optional[str] = None

    @property
    def connections(self) -> Dict[str, Any]:
        """Get all registered connections."""
        return self._native_factory.connections

    @property
    def cards(self) -> Dict[str, Any]:
        """Get all registered agent cards."""
        return self._native_factory.cards

    async def register_agent(
        self,
        url: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        """Register a sub-agent and return its name.

        Args:
            url: Base URL of the A2A sub-agent.
            name: Optional override for agent name.
            description: Optional description for the delegation tool.

        Returns:
            The registered agent name.

        Raises:
            httpx.HTTPError: If connection to agent fails.
        """
        agent_name = await self._native_factory.register_agent(url, name=name)

        # Use provided description or extract from agent card
        if description:
            self._descriptions[agent_name] = description
        else:
            card = self._native_factory.cards.get(agent_name)
            if card and card.description:
                self._descriptions[agent_name] = card.description
            else:
                self._descriptions[agent_name] = f"Delegate tasks to {agent_name}"

        logger.info("Registered sub-agent: %s at %s", agent_name, url)
        return agent_name

    def get_tools(self) -> List[BaseTool]:
        """Get all delegation tools for registered agents.

        Returns:
            List of delegation tools.
        """
        return [
            self._create_delegation_tool(name)
            for name in self._native_factory.get_agent_names()
        ]

    def get_agent_names(self) -> List[str]:
        """Get names of all registered agents.

        Returns:
            List of agent names.
        """
        return self._native_factory.get_agent_names()

    async def send_message_direct(
        self,
        agent_name: str,
        message: str,
        context_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """Send message directly to an agent (for parameter routing).

        This method bypasses LLM routing and sends directly to the specified
        agent. Used by OrchestratorExecutor for parameter-based routing.

        Args:
            agent_name: Name of the target agent.
            message: Message text to send.
            context_id: Optional context ID.
            task_id: Optional task ID.

        Returns:
            Text response from the agent.
        """
        return await self._native_factory.send_message_direct(
            agent_name=agent_name,
            text=message,
            context_id=context_id,
            task_id=task_id,
        )

    async def send_message_streaming(
        self,
        agent_name: str,
        message: str,
        context_id: Optional[str] = None,
    ) -> AsyncGenerator[Tuple[str, Any], None]:
        """Send message with streaming events propagation.

        This method yields all streaming events from the sub-agent, allowing
        the orchestrator to re-emit them to the parent event queue.

        Events are also automatically emitted to self.event_queue if set.

        Args:
            agent_name: Name of the target agent.
            message: Message text to send.
            context_id: Optional context ID.

        Yields:
            Tuple of (event_type, event_data):
            - ("status_update", event) - status/thinking/tool events
            - ("artifact_update", event) - content streaming
            - ("task", Task) - intermediate task state
            - ("final", result) - final result
            - ("final_text", str) - extracted text from final result
        """
        final_result = None

        async for event_type, event_data in self._native_factory.send_message_streaming(
            agent_name=agent_name,
            text=message,
            context_id=context_id,
        ):
            # Emit events to A2A event queue if available
            if self.event_queue and event_type in ("status_update", "artifact_update"):
                await self._propagate_event_to_queue(
                    event_type=event_type,
                    event_data=event_data,
                    source_agent=agent_name,
                )

            # Yield for caller to handle
            yield (event_type, event_data)

            # Track final result
            if event_type == "final":
                final_result = event_data

        # Yield extracted text from final result
        if final_result:
            final_text = self._native_factory._extract_response_text(final_result)
            yield ("final_text", final_text)

    async def _propagate_event_to_queue(
        self,
        event_type: str,
        event_data: Any,
        source_agent: str,
    ) -> None:
        """Propagate a streaming event to the A2A event queue.

        Converts sub-agent events to orchestrator events with proper metadata.

        Args:
            event_type: Type of event ("status_update" or "artifact_update").
            event_data: The event data (TaskStatusUpdateEvent, dict, etc.).
            source_agent: Name of the source sub-agent.
        """
        if not self.event_queue:
            return

        try:
            from a2a.types import (
                Message,
                Part,
                Role,
                TaskState,
                TaskStatus,
                TaskStatusUpdateEvent,
                TextPart,
            )

            # Use stored context IDs
            ctx_id = self.context_id or str(uuid4())
            t_id = self.task_id or str(uuid4())

            if event_type == "status_update":
                # Extract status info from event
                status_text = self._extract_status_text(event_data, source_agent)
                event_metadata = self._extract_event_metadata(event_data, source_agent)

                if status_text:
                    # Create TextPart with metadata for filtering
                    text_part = TextPart(
                        text=status_text,
                        metadata=event_metadata,
                    )

                    await self.event_queue.enqueue_event(
                        TaskStatusUpdateEvent(
                            contextId=ctx_id,
                            taskId=t_id,
                            final=False,
                            status=TaskStatus(
                                state=TaskState.working,
                                message=Message(
                                    messageId=str(uuid4()),
                                    role=Role.agent,
                                    parts=[Part(root=text_part)],
                                ),
                            ),
                        )
                    )

        except Exception as e:
            logger.warning("Failed to propagate event to queue: %s", e)

    def _extract_status_text(self, event_data: Any, source_agent: str) -> Optional[str]:
        """Extract display text from status event.

        Note: We do NOT add [agent] prefix here because:
        1. The metadata already contains source_agent for tracking
        2. The pipe function's _render_trajectory() handles formatting
        3. Adding prefix here would cause duplication

        Args:
            event_data: Status event data.
            source_agent: Name of the source agent (used only for logging).

        Returns:
            Human-readable status text or None.
        """
        # Handle TaskStatusUpdateEvent
        if hasattr(event_data, "status") and event_data.status:
            status = event_data.status
            if hasattr(status, "message") and status.message:
                parts = status.message.parts or []
                for part in parts:
                    # Try to get text from Part
                    if hasattr(part, "root") and part.root:
                        root = part.root
                        if hasattr(root, "text") and root.text:
                            return root.text  # Return raw text without prefix
                    elif hasattr(part, "text") and part.text:
                        return part.text  # Return raw text without prefix

        # Handle dict-like event
        if isinstance(event_data, dict):
            status = event_data.get("status", {})
            message = status.get("message", {})
            parts = message.get("parts", [])
            for part in parts:
                text = None
                if isinstance(part, dict):
                    text = part.get("text")
                    if not text and "root" in part:
                        text = part["root"].get("text")
                if text:
                    return text  # Return raw text without prefix

        return None

    def _extract_event_metadata(self, event_data: Any, source_agent: str) -> Dict[str, Any]:
        """Extract metadata from status event for filtering.

        Preserves key fields from original event for proper formatting:
        - event_type: For categorizing the event
        - tool_name: For tool_start/tool_end events
        - input: For tool_start events (tool arguments)
        - output: For tool_end events (tool result preview)
        - duration_ms: For tool_end events
        - agent_name: For agent_start events

        Args:
            event_data: Status event data.
            source_agent: Name of the source agent.

        Returns:
            Metadata dict with event_type, agent info, etc.
        """
        metadata: Dict[str, Any] = {
            "source_agent": source_agent,
            "is_propagated": True,  # Mark as propagated from sub-agent
        }

        # Try to extract original event_type from sub-agent's metadata
        if hasattr(event_data, "status") and event_data.status:
            status = event_data.status
            if hasattr(status, "message") and status.message:
                parts = status.message.parts or []
                for part in parts:
                    if hasattr(part, "root") and part.root:
                        root = part.root
                        if hasattr(root, "metadata") and root.metadata:
                            orig_meta = root.metadata
                            if isinstance(orig_meta, dict):
                                metadata["event_type"] = orig_meta.get("event_type", "sub_agent_status")
                                metadata["tool_name"] = orig_meta.get("tool_name")
                                metadata["duration_ms"] = orig_meta.get("duration_ms")
                                metadata["input"] = orig_meta.get("input")  # Tool input args
                                metadata["output"] = orig_meta.get("output")  # Tool output
                                metadata["agent_name"] = orig_meta.get("agent_name")

        # Handle dict-like event
        if isinstance(event_data, dict):
            status = event_data.get("status", {})
            message = status.get("message", {})
            parts = message.get("parts", [])
            for part in parts:
                if isinstance(part, dict):
                    orig_meta = part.get("metadata") or part.get("root", {}).get("metadata")
                    if orig_meta:
                        metadata["event_type"] = orig_meta.get("event_type", "sub_agent_status")
                        metadata["tool_name"] = orig_meta.get("tool_name")
                        metadata["duration_ms"] = orig_meta.get("duration_ms")
                        metadata["input"] = orig_meta.get("input")  # Tool input args
                        metadata["output"] = orig_meta.get("output")  # Tool output
                        metadata["agent_name"] = orig_meta.get("agent_name")

        # Default event_type if not found
        if "event_type" not in metadata:
            metadata["event_type"] = "sub_agent_status"

        return metadata

    def _create_delegation_tool(
        self,
        agent_name: str,
    ) -> BaseTool:
        """Create a delegation tool for a specific agent.

        Args:
            agent_name: Name of the sub-agent.

        Returns:
            A BaseTool that delegates to the sub-agent.
        """
        factory = self  # Capture for closure
        description = self._descriptions.get(agent_name, f"Delegate to {agent_name}")

        # Create tool name (replace hyphens with underscores for valid Python identifier)
        tool_name = f"delegate_to_{agent_name.replace('-', '_')}"

        @tool(tool_name)
        async def delegation_tool(task: str, runtime: Any = None) -> Command:
            """Delegate a task to a sub-agent.

            Args:
                task: The task description to delegate.
                runtime: LangGraph runtime context (provides tool_call_id).

            Returns:
                Command with ToolMessage and optional state updates.
            """
            tool_call_id = getattr(runtime, "tool_call_id", "unknown")

            logger.debug("Delegating to %s: %s", agent_name, task[:100])

            try:
                # Use native SDK to send message
                final_result = await factory._native_factory.send_message_direct(
                    agent_name=agent_name,
                    text=task,
                    context_id=factory.context_id,
                    task_id=factory.task_id,
                )

            except Exception as e:
                logger.error("Delegation to %s failed: %s", agent_name, e)
                final_result = f"Error delegating to {agent_name}: {str(e)}"

            # Build Command update
            update: Dict[str, Any] = {
                "messages": [
                    ToolMessage(
                        content=f"[{agent_name}] {final_result}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }

            # Optionally track delegation history in state
            if factory.track_delegation_history:
                update["delegation_history"] = [
                    {
                        "agent": agent_name,
                        "task": task[:200],
                        "result": final_result[:500],
                    }
                ]

            logger.debug(
                "Delegation to %s completed: result: %s...",
                agent_name,
                final_result[:100],
            )

            return Command(update=update)

        # Set description dynamically
        delegation_tool.description = f"Delegate task to {agent_name}. {description}"
        return delegation_tool

    async def _emit_event_to_queue(
        self,
        event: AgentEvent,
        source_agent: str,
    ) -> None:
        """Emit an AgentEvent to the A2A EventQueue.

        Args:
            event: The AgentEvent to emit.
            source_agent: Name of the source sub-agent.
        """
        if not self.event_queue:
            return

        try:
            # Import here to avoid circular imports
            from uuid import uuid4

            from a2a.types import Message, Part, Role, TaskState, TaskStatus, TaskStatusUpdateEvent

            # Map event types to display text
            event_display_map = {
                "delegation_start": f"📤 Delegating to {source_agent}",
                "delegation_end": f"✅ {source_agent} completed",
                "sub_agent_tool_start": f"🔧 [{source_agent}] {event.name}",
                "sub_agent_tool_end": f"✅ [{source_agent}] {event.name} done",
                "sub_agent_thinking": f"🤔 [{source_agent}] thinking...",
                "sub_agent_error": f"❌ [{source_agent}] error",
            }

            display_text = event_display_map.get(event.type)
            if not display_text:
                # Skip events without display text (e.g., text_delta handled by artifacts)
                return

            # Use context_id and task_id if available, otherwise generate UUIDs
            context_id = self.context_id or str(uuid4())
            task_id = self.task_id or str(uuid4())

            # Emit status update event
            await self.event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    contextId=context_id,
                    taskId=task_id,
                    final=False,
                    status=TaskStatus(
                        state=TaskState.working,
                        message=Message(
                            messageId=str(uuid4()),
                            role=Role.agent,
                            parts=[
                                Part(text=display_text),
                                Part(
                                    data={
                                        "event_type": event.type,
                                        "source_agent": source_agent,
                                        "name": event.name,
                                        **event.data,
                                    }
                                ),
                            ],
                        ),
                    ),
                )
            )

        except Exception as e:
            logger.warning("Failed to emit event to queue: %s", e)

    async def close(self) -> None:
        """Close all client connections."""
        await self._native_factory.close()
        self._descriptions.clear()


async def create_delegation_tools(
    agent_urls: Dict[str, str],
    event_queue: Optional["EventQueue"] = None,
    track_history: bool = True,
) -> List[BaseTool]:
    """Create delegation tools for multiple sub-agents.

    Convenience function to create delegation tools from a dict of agent URLs.

    Args:
        agent_urls: Dict mapping agent names to their URLs.
        event_queue: Optional A2A EventQueue for streaming events.
        track_history: Whether to track delegation history in state.

    Returns:
        List of delegation tools.

    Example:
        tools = await create_delegation_tools({
            "jira-agent": "http://localhost:10001",
            "faq-agent": "http://localhost:10002",
        })
        # Returns: [delegate_to_jira_agent, delegate_to_faq_agent]
    """
    factory = DelegationToolFactory(
        event_queue=event_queue,
        track_delegation_history=track_history,
    )

    for name, url in agent_urls.items():
        await factory.register_agent(url, name=name)

    return factory.get_tools()


# Legacy function for backwards compatibility
def create_delegation_tool_sync(
    agent_name: str,
    client: Any,  # StreamingA2AClient - deprecated
    event_queue: Optional["EventQueue"] = None,
    track_history: bool = True,
) -> BaseTool:
    """Create a single delegation tool (synchronous factory method).

    DEPRECATED: This function uses the old StreamingA2AClient.
    Use DelegationToolFactory.register_agent() instead.

    Args:
        agent_name: Name of the sub-agent.
        client: Connected StreamingA2AClient (deprecated).
        event_queue: Optional A2A EventQueue.
        track_history: Whether to track delegation history.

    Returns:
        Delegation tool for the agent.
    """
    import warnings
    warnings.warn(
        "create_delegation_tool_sync is deprecated. "
        "Use DelegationToolFactory.register_agent() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    factory = DelegationToolFactory(
        event_queue=event_queue,
        track_delegation_history=track_history,
    )
    # This is a compatibility shim - the tool won't work with old clients
    # but allows existing code to not break immediately
    factory._descriptions[agent_name] = f"Delegate tasks to {agent_name}"
    return factory._create_delegation_tool(agent_name)
