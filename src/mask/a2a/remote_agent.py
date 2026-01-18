"""Remote agent connection using native A2A SDK.

This module provides a wrapper around the official A2A SDK's ClientFactory
and Client classes for connecting to remote A2A agents.

Based on the official A2A Python samples pattern:
https://github.com/google/a2a-python-samples/blob/main/hosts/multiagent/remote_agent_connection.py

The key difference from our StreamingA2AClient is that this uses the native
SDK's Client.send_message() method which handles SSE parsing correctly,
avoiding the event loop issues we encountered with our custom implementation.
"""

import logging
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, Client, ClientConfig, ClientFactory
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
    TransportProtocol,
)

logger = logging.getLogger(__name__)


class NativeRemoteAgentConnection:
    """Connection to a remote A2A agent using native SDK.

    This class wraps the official A2A SDK's Client class for reliable
    communication with remote agents. Unlike our StreamingA2AClient,
    this uses the SDK's built-in SSE parsing which works correctly
    in uvicorn environments.

    Example:
        # Create connection via factory
        factory = NativeRemoteAgentFactory()
        await factory.register_agent("http://localhost:10001", "hr-expert")

        # Send message
        result = await factory.send_message_direct("hr-expert", "Hello")
    """

    def __init__(self, client_factory: ClientFactory, agent_card: AgentCard) -> None:
        """Initialize connection.

        Args:
            client_factory: A2A SDK ClientFactory instance.
            agent_card: Remote agent's AgentCard metadata.
        """
        self.agent_client: Client = client_factory.create(agent_card)
        self.card: AgentCard = agent_card

    @property
    def name(self) -> str:
        """Get agent name."""
        return self.card.name

    @property
    def description(self) -> Optional[str]:
        """Get agent description."""
        return self.card.description

    async def send_message(
        self,
        message: Message,
    ) -> Union[Task, Message, None]:
        """Send message to remote agent using native SDK.

        This method iterates through all events from the SDK's send_message
        generator and returns the final Task or Message. If the stream ends
        with an error after receiving events, we return the last valid Task.

        Args:
            message: A2A Message to send.

        Returns:
            Task or Message response from remote agent.
        """
        last_task: Optional[Task] = None

        try:
            async for event in self.agent_client.send_message(message):
                if isinstance(event, Message):
                    # Direct message response
                    return event
                if isinstance(event, tuple) and len(event) > 0:
                    task = event[0]
                    if isinstance(task, Task):
                        # Keep updating last_task - artifacts accumulate over events
                        last_task = task
                        if self._is_terminal_state(task):
                            return task
        except Exception as e:
            # A2A SDK may throw "streamed Message after first response" error
            # but we may already have complete data in last_task
            if last_task:
                logger.debug("Stream error after receiving task data, using last_task: %s", e)
                return last_task
            logger.error("Error sending message to %s: %s", self.name, e)
            raise

        return last_task

    async def send_text(
        self,
        text: str,
        context_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Union[Task, Message, None]:
        """Send text message to remote agent.

        Convenience method that creates a Message from text.

        Note: task_id is intentionally NOT passed to the sub-agent.
        Each delegation creates a new task on the sub-agent. The
        orchestrator's task_id is for tracking on the orchestrator side.

        Args:
            text: Message text to send.
            context_id: Optional context ID for conversation continuity.
            task_id: Optional task ID (currently ignored - sub-agent creates own task).

        Returns:
            Task or Message response from remote agent.
        """
        # Note: We don't pass task_id to sub-agent - it creates its own task.
        # Passing the orchestrator's task_id would fail because the sub-agent
        # doesn't know about that task.
        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=text))],
            message_id=str(uuid4()),
            context_id=context_id or str(uuid4()),
            # task_id intentionally omitted - let sub-agent create new task
        )

        return await self.send_message(message)

    async def send_message_streaming(
        self,
        message: Message,
        on_event: Optional[Callable[[Any], None]] = None,
    ) -> AsyncGenerator[Tuple[str, Any], None]:
        """Send message and yield streaming events from remote agent.

        Unlike send_message() which only returns the final result, this method
        yields all intermediate events (status updates, artifact updates) for
        propagation to parent event queues.

        Args:
            message: A2A Message to send.
            on_event: Optional callback for each event (for logging/debugging).

        Yields:
            Tuple of (event_type, event_data) for each streaming event:
            - ("status_update", TaskStatusUpdateEvent)
            - ("artifact_update", TaskArtifactUpdateEvent)
            - ("task", Task) - intermediate task state
            - ("message", Message) - direct message response
            - ("final", Task | Message | None) - final result
        """
        last_task: Optional[Task] = None

        try:
            async for event in self.agent_client.send_message(message):
                if on_event:
                    on_event(event)

                # Handle direct Message response
                if isinstance(event, Message):
                    yield ("message", event)
                    yield ("final", event)
                    return

                # Handle tuple events from A2A SDK
                # Format: (Task, Event) or just (Task,)
                if isinstance(event, tuple):
                    if len(event) >= 1:
                        task = event[0]
                        if isinstance(task, Task):
                            last_task = task
                            yield ("task", task)

                            # Check for terminal state
                            if self._is_terminal_state(task):
                                yield ("final", task)
                                return

                    # Check for streaming events in tuple[1]
                    if len(event) >= 2:
                        streaming_event = event[1]

                        # Handle TaskStatusUpdateEvent (thinking, tool calls, etc.)
                        if isinstance(streaming_event, TaskStatusUpdateEvent):
                            yield ("status_update", streaming_event)

                        # Handle TaskArtifactUpdateEvent (content streaming)
                        elif isinstance(streaming_event, TaskArtifactUpdateEvent):
                            yield ("artifact_update", streaming_event)

                        # Handle dict-like events (some SDK versions)
                        elif isinstance(streaming_event, dict):
                            kind = streaming_event.get("kind")
                            if kind == "status-update":
                                yield ("status_update", streaming_event)
                            elif kind == "artifact-update":
                                yield ("artifact_update", streaming_event)

        except Exception as e:
            # A2A SDK may throw errors but we may have valid data
            if last_task:
                logger.debug("Stream error after receiving task data, using last_task: %s", e)
                yield ("final", last_task)
                return
            logger.error("Error in streaming from %s: %s", self.name, e)
            raise

        # Yield final result if we haven't already
        yield ("final", last_task)

    async def send_text_streaming(
        self,
        text: str,
        context_id: Optional[str] = None,
        on_event: Optional[Callable[[Any], None]] = None,
    ) -> AsyncGenerator[Tuple[str, Any], None]:
        """Send text and yield streaming events.

        Convenience method combining send_text with streaming.

        Args:
            text: Message text to send.
            context_id: Optional context ID.
            on_event: Optional callback for each event.

        Yields:
            Streaming events from the remote agent.
        """
        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=text))],
            message_id=str(uuid4()),
            context_id=context_id or str(uuid4()),
        )

        async for event in self.send_message_streaming(message, on_event):
            yield event

    def _is_terminal_state(self, task: Task) -> bool:
        """Check if task is in terminal state.

        Args:
            task: The task to check.

        Returns:
            True if task is complete, canceled, failed, or input_required.
            Note: unknown is NOT terminal - it's an initial/intermediate state.
        """
        terminal_states = {
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
            TaskState.input_required,
            # Note: TaskState.unknown is NOT terminal - it's initial state
        }
        return task.status.state in terminal_states

    def get_skills(self) -> List[Dict[str, Any]]:
        """Get skills advertised by remote agent.

        Returns:
            List of skill information dicts.
        """
        return [
            {
                "id": skill.id,
                "name": skill.name,
                "description": getattr(skill, "description", None),
            }
            for skill in (self.card.skills or [])
        ]


class NativeRemoteAgentFactory:
    """Factory for creating remote agent connections using native A2A SDK.

    This factory manages httpx client lifecycle and creates ClientFactory
    instances for reliable A2A communication.

    Example:
        factory = NativeRemoteAgentFactory()
        await factory.register_agent("http://localhost:10001", "hr-expert")
        await factory.register_agent("http://localhost:10002", "finance-expert")

        # Send message directly (bypasses LLM)
        result = await factory.send_message_direct("hr-expert", "Hello")

        # Get agent names for LLM routing
        names = factory.get_agent_names()
    """

    def __init__(
        self,
        httpx_client: Optional[httpx.AsyncClient] = None,
        timeout: float = 120.0,
    ) -> None:
        """Initialize factory.

        Args:
            httpx_client: Optional httpx client (creates one if not provided).
            timeout: HTTP timeout in seconds.
        """
        self._external_client = httpx_client
        self._timeout = timeout

        # Lazy initialization - will be created on first use in current event loop
        self._httpx_client: Optional[httpx.AsyncClient] = None
        self._client_factory: Optional[ClientFactory] = None
        self._created_in_loop: Optional[Any] = None  # Track which event loop client was created in

        # Store connections and cards (these don't depend on event loop)
        self._connections: Dict[str, NativeRemoteAgentConnection] = {}
        self._cards: Dict[str, AgentCard] = {}
        self._agent_urls: Dict[str, str] = {}  # Store URLs for re-registration

    def _ensure_client_factory(self) -> ClientFactory:
        """Ensure client factory exists in current event loop.

        Creates a fresh httpx client and client factory if needed.
        This handles the case where the factory was created in a different
        event loop (e.g., asyncio.run() vs uvicorn).
        """
        import asyncio

        # Get current event loop
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        # Check if we need to recreate the client
        need_recreate = False
        if self._httpx_client is None:
            need_recreate = True
        elif self._httpx_client.is_closed:
            need_recreate = True
        else:
            # Check if the client was created in a different event loop
            # by checking if any stored loop reference differs
            if hasattr(self, "_created_in_loop") and self._created_in_loop != current_loop:
                logger.info(
                    "Event loop changed (old: %s, new: %s), recreating client",
                    id(self._created_in_loop) if self._created_in_loop else "None",
                    id(current_loop) if current_loop else "None",
                )
                need_recreate = True

        if need_recreate:
            # Create fresh client in current event loop
            if self._external_client:
                self._httpx_client = self._external_client
            else:
                self._httpx_client = httpx.AsyncClient(timeout=self._timeout)

            # Store the loop we created the client in
            self._created_in_loop = current_loop

            # Create fresh client factory
            config = ClientConfig(
                httpx_client=self._httpx_client,
                supported_transports=[
                    TransportProtocol.jsonrpc,
                    TransportProtocol.http_json,
                ],
            )
            self._client_factory = ClientFactory(config)

            # Re-register connections with new factory
            self._connections.clear()
            logger.info("Created fresh httpx client and ClientFactory in loop %s",
                       id(current_loop) if current_loop else "None")

        return self._client_factory

    @property
    def connections(self) -> Dict[str, NativeRemoteAgentConnection]:
        """Get all registered connections."""
        return self._connections

    @property
    def cards(self) -> Dict[str, AgentCard]:
        """Get all registered agent cards."""
        return self._cards

    async def register_agent(
        self,
        url: str,
        name: Optional[str] = None,
    ) -> str:
        """Register a remote agent by URL.

        Args:
            url: Base URL of the A2A agent.
            name: Optional override for agent name.

        Returns:
            The registered agent name.

        Raises:
            httpx.HTTPError: If connection to agent fails.
        """
        # Ensure client factory exists in current event loop
        client_factory = self._ensure_client_factory()

        # Resolve agent card
        card_resolver = A2ACardResolver(self._httpx_client, url)
        card = await card_resolver.get_agent_card()

        # Create connection
        connection = NativeRemoteAgentConnection(client_factory, card)

        # Register with name override if provided
        agent_name = name or card.name
        self._connections[agent_name] = connection
        self._cards[agent_name] = card
        self._agent_urls[agent_name] = url  # Store for re-registration

        logger.info(
            "Registered remote agent: %s at %s (description: %s)",
            agent_name,
            url,
            card.description[:50] if card.description else "N/A",
        )

        return agent_name

    def get_agent_names(self) -> List[str]:
        """Get names of all registered agents.

        Returns:
            List of agent names.
        """
        return list(self._connections.keys())

    def get_agent_info(self) -> List[Dict[str, Any]]:
        """Get info about all registered agents.

        Returns:
            List of agent info dicts with name and description.
        """
        return [
            {
                "name": card.name,
                "description": card.description,
            }
            for card in self._cards.values()
        ]

    def get_connection(self, agent_name: str) -> Optional[NativeRemoteAgentConnection]:
        """Get connection for a specific agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            Connection or None if not found.
        """
        return self._connections.get(agent_name)

    async def send_message_direct(
        self,
        agent_name: str,
        text: str,
        context_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """Send message directly to an agent and return text response.

        This method is for parameter-based routing where we bypass LLM
        and send directly to the target agent.

        Args:
            agent_name: Name of the target agent.
            text: Message text to send.
            context_id: Optional context ID.
            task_id: Optional task ID.

        Returns:
            Text response from the agent.
        """
        # Ensure client factory exists (may need recreation after event loop change)
        self._ensure_client_factory()

        # Check if connection exists, re-register if needed
        if agent_name not in self._connections:
            # Try to re-register from stored URL
            if agent_name in self._agent_urls:
                logger.info("Re-registering agent %s after event loop change", agent_name)
                await self.register_agent(self._agent_urls[agent_name], name=agent_name)
            else:
                return f"Error: Agent '{agent_name}' not found"

        connection = self._connections[agent_name]

        try:
            response = await connection.send_text(
                text=text,
                context_id=context_id,
                task_id=task_id,
            )

            return self._extract_response_text(response)

        except Exception as e:
            logger.error("Error sending to %s: %s", agent_name, e)
            return f"Error delegating to {agent_name}: {str(e)}"

    async def send_message_streaming(
        self,
        agent_name: str,
        text: str,
        context_id: Optional[str] = None,
        on_event: Optional[Callable[[Any], None]] = None,
    ) -> AsyncGenerator[Tuple[str, Any], None]:
        """Send message with streaming events from sub-agent.

        This method propagates all streaming events (status updates, tool calls,
        artifacts) from the sub-agent, allowing the orchestrator to re-emit them.

        Args:
            agent_name: Name of the target agent.
            text: Message text to send.
            context_id: Optional context ID.
            on_event: Optional callback for each raw event.

        Yields:
            Tuple of (event_type, event_data):
            - ("status_update", TaskStatusUpdateEvent | dict)
            - ("artifact_update", TaskArtifactUpdateEvent | dict)
            - ("task", Task)
            - ("final", Task | Message | None)
        """
        # Ensure client factory exists
        self._ensure_client_factory()

        # Check if connection exists, re-register if needed
        if agent_name not in self._connections:
            if agent_name in self._agent_urls:
                logger.info("Re-registering agent %s after event loop change", agent_name)
                await self.register_agent(self._agent_urls[agent_name], name=agent_name)
            else:
                yield ("error", f"Agent '{agent_name}' not found")
                return

        connection = self._connections[agent_name]

        try:
            async for event_type, event_data in connection.send_text_streaming(
                text=text,
                context_id=context_id,
                on_event=on_event,
            ):
                yield (event_type, event_data)

        except Exception as e:
            logger.error("Error streaming from %s: %s", agent_name, e)
            yield ("error", str(e))

    def _extract_response_text(
        self,
        response: Union[Task, Message, None],
    ) -> str:
        """Extract text from Task or Message response.

        Args:
            response: Task or Message from remote agent.

        Returns:
            Extracted text content.
        """
        if response is None:
            return "No response received"

        if isinstance(response, Message):
            return self._extract_parts_text(response.parts)

        if isinstance(response, Task):
            texts = []

            # Extract from artifacts (final response content)
            # Skip tool_call/tool_result artifacts - only get "response" artifacts
            if response.artifacts:
                for artifact in response.artifacts:
                    # Skip tool-related artifacts (tool_call, tool_result)
                    artifact_name = getattr(artifact, "name", "") or ""
                    if artifact_name in ("tool_call", "tool_result"):
                        continue
                    if artifact.parts:
                        artifact_text = self._extract_parts_text(artifact.parts)
                        texts.append(artifact_text)

            # Fallback to status message only if no response artifacts
            if not texts and response.status and response.status.message:
                status_text = self._extract_parts_text(response.status.message.parts)
                if status_text:
                    texts.append(status_text)

            return "\n".join(filter(None, texts)) or "Task completed"

        return str(response)

    def _extract_parts_text(self, parts: Optional[List[Part]]) -> str:
        """Extract text from message parts.

        Handles multiple Part formats:
        - Part(root=TextPart(text="..."))
        - Part with direct text attribute
        - Dict-like parts from JSON

        Args:
            parts: List of message parts.

        Returns:
            Concatenated text content.
        """
        if not parts:
            return ""

        texts = []
        for part in parts:
            text = None

            # Try Part.root.text (standard A2A SDK format)
            if hasattr(part, "root") and part.root is not None:
                root = part.root
                if hasattr(root, "text"):
                    text = root.text
                elif isinstance(root, dict) and "text" in root:
                    text = root["text"]

            # Try direct text attribute
            if text is None and hasattr(part, "text"):
                text = part.text

            # Try dict-like access (for deserialized JSON)
            if text is None and isinstance(part, dict):
                if "text" in part:
                    text = part["text"]
                elif "root" in part and isinstance(part["root"], dict):
                    text = part["root"].get("text")

            if text:
                texts.append(str(text))

        return "".join(texts)

    async def close(self) -> None:
        """Close all connections and cleanup."""
        if self._owns_client and self._httpx_client:
            await self._httpx_client.aclose()
            self._httpx_client = None

        self._connections.clear()
        self._cards.clear()

    def __len__(self) -> int:
        """Return number of registered agents."""
        return len(self._connections)

    def __contains__(self, name: str) -> bool:
        """Check if agent is registered."""
        return name in self._connections
