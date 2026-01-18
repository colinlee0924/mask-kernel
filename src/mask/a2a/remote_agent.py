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
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, Client, ClientConfig, ClientFactory
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    Task,
    TaskState,
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
        generator and returns the final Task or Message.

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
                        if self._is_terminal_state(task):
                            return task
                        last_task = task
        except Exception as e:
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

        Args:
            text: Message text to send.
            context_id: Optional context ID for conversation continuity.
            task_id: Optional task ID for task continuation.

        Returns:
            Task or Message response from remote agent.
        """
        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=text))],
            message_id=str(uuid4()),
            context_id=context_id or str(uuid4()),
            task_id=task_id,
        )

        return await self.send_message(message)

    def _is_terminal_state(self, task: Task) -> bool:
        """Check if task is in terminal state.

        Args:
            task: The task to check.

        Returns:
            True if task is complete, canceled, failed, input_required, or unknown.
        """
        terminal_states = {
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
            TaskState.input_required,
            TaskState.unknown,
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
        self._owns_client = httpx_client is None
        self._httpx_client = httpx_client or httpx.AsyncClient(timeout=timeout)
        self._timeout = timeout

        # Create ClientFactory with supported transports
        config = ClientConfig(
            httpx_client=self._httpx_client,
            supported_transports=[
                TransportProtocol.jsonrpc,
                TransportProtocol.http_json,
            ],
        )
        self._client_factory = ClientFactory(config)

        # Store connections and cards
        self._connections: Dict[str, NativeRemoteAgentConnection] = {}
        self._cards: Dict[str, AgentCard] = {}

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
        # Resolve agent card
        card_resolver = A2ACardResolver(self._httpx_client, url)
        card = await card_resolver.get_agent_card()

        # Create connection
        connection = NativeRemoteAgentConnection(self._client_factory, card)

        # Register with name override if provided
        agent_name = name or card.name
        self._connections[agent_name] = connection
        self._cards[agent_name] = card

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
        if agent_name not in self._connections:
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

            # Extract from status message
            if response.status and response.status.message:
                texts.append(self._extract_parts_text(response.status.message.parts))

            # Extract from artifacts
            if response.artifacts:
                for artifact in response.artifacts:
                    if artifact.parts:
                        texts.append(self._extract_parts_text(artifact.parts))

            return "\n".join(filter(None, texts)) or "Task completed"

        return str(response)

    def _extract_parts_text(self, parts: Optional[List[Part]]) -> str:
        """Extract text from message parts.

        Args:
            parts: List of message parts.

        Returns:
            Concatenated text content.
        """
        if not parts:
            return ""

        texts = []
        for part in parts:
            if hasattr(part, "root"):
                root = part.root
                if hasattr(root, "text"):
                    texts.append(root.text)
                elif hasattr(root, "kind") and root.kind == "text":
                    texts.append(getattr(root, "text", ""))
            elif hasattr(part, "text"):
                texts.append(part.text)

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
