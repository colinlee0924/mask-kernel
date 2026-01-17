"""Delegation tool factory for multi-agent orchestration.

This module provides tools for delegating tasks from an orchestrator agent
to sub-agents via A2A protocol, with real-time event streaming.

The delegation tools return LangGraph Command objects to:
1. Inject ToolMessage with sub-agent results
2. Update agent state (e.g., delegation_history)

Following the MASK skill loader pattern with Command support.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command

from mask.a2a.streaming_client import StreamingA2AClient
from mask.core.events import AgentEvent

if TYPE_CHECKING:
    from a2a.server.events import EventQueue

logger = logging.getLogger(__name__)


class DelegationToolFactory:
    """Factory for creating delegation tools for orchestrator agents.

    Creates tools that delegate tasks to sub-agents and stream events
    back to the frontend via A2A EventQueue.

    Example:
        factory = DelegationToolFactory()
        await factory.register_agent("http://localhost:10001", "jira-agent")
        await factory.register_agent("http://localhost:10002", "faq-agent")

        tools = factory.get_tools()
        # Returns: [delegate_to_jira_agent, delegate_to_faq_agent]

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
        self._clients: Dict[str, StreamingA2AClient] = {}
        self._descriptions: Dict[str, str] = {}

        # A2A context for streaming events (set by executor before each invocation)
        self.context_id: Optional[str] = None
        self.task_id: Optional[str] = None

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
        client = StreamingA2AClient(url, agent_name=name)
        await client.connect()

        agent_name = client.agent_name or name or url
        self._clients[agent_name] = client

        # Use provided description or extract from agent card
        if description:
            self._descriptions[agent_name] = description
        elif client.card and client.card.description:
            self._descriptions[agent_name] = client.card.description
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
            self._create_delegation_tool(name, client)
            for name, client in self._clients.items()
        ]

    def get_agent_names(self) -> List[str]:
        """Get names of all registered agents.

        Returns:
            List of agent names.
        """
        return list(self._clients.keys())

    def _create_delegation_tool(
        self,
        agent_name: str,
        client: StreamingA2AClient,
    ) -> BaseTool:
        """Create a delegation tool for a specific agent.

        Args:
            agent_name: Name of the sub-agent.
            client: StreamingA2AClient for the agent.

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

            final_result = ""
            delegation_events: List[Dict[str, Any]] = []
            event_count = 0

            logger.debug("Delegating to %s: %s", agent_name, task[:100])

            try:
                # Stream events from sub-agent
                async for event in client.send_message_streaming(task):
                    event_count += 1

                    # Emit to frontend via EventQueue if available
                    if factory.event_queue:
                        await factory._emit_event_to_queue(event, agent_name)

                    # Collect events for history
                    delegation_events.append({
                        "type": event.type,
                        "name": event.name,
                        "source_agent": event.source_agent,
                    })

                    # Accumulate final result from text deltas
                    if event.type == "sub_agent_text_delta":
                        delta = event.data.get("delta", "")
                        final_result += delta
                    elif event.type == "delegation_end":
                        if not final_result:
                            final_result = event.data.get("result", "Task completed")

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
                        "event_count": event_count,
                    }
                ]

            logger.debug(
                "Delegation to %s completed: %d events, result: %s...",
                agent_name,
                event_count,
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

            # Debug log for sub-agent tool events
            if event.type in ("sub_agent_tool_start", "sub_agent_tool_end"):
                logger.info(
                    "[EMIT-DEBUG] %s: name=%s, source=%s, data_keys=%s",
                    event.type,
                    event.name,
                    source_agent,
                    list(event.data.keys()) if event.data else "empty",
                )

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
        for client in self._clients.values():
            await client.close()
        self._clients.clear()
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


def create_delegation_tool_sync(
    agent_name: str,
    client: StreamingA2AClient,
    event_queue: Optional["EventQueue"] = None,
    track_history: bool = True,
) -> BaseTool:
    """Create a single delegation tool (synchronous factory method).

    For cases where you already have a connected StreamingA2AClient.

    Args:
        agent_name: Name of the sub-agent.
        client: Connected StreamingA2AClient.
        event_queue: Optional A2A EventQueue.
        track_history: Whether to track delegation history.

    Returns:
        Delegation tool for the agent.
    """
    factory = DelegationToolFactory(
        event_queue=event_queue,
        track_delegation_history=track_history,
    )
    factory._clients[agent_name] = client
    return factory._create_delegation_tool(agent_name, client)
