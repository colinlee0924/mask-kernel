"""Remote agent connection for multi-agent scenarios.

This module provides utilities for connecting to remote A2A agents,
enabling multi-agent orchestration patterns.

Following a2a-python-samples hosts/multiagent pattern.
"""

import logging
from typing import Dict, List, Optional, Union
from uuid import uuid4

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    Task,
    TaskState,
    TextPart,
)

logger = logging.getLogger(__name__)


class RemoteAgentConnection:
    """Connection to a remote A2A agent.

    Enables Host Agent to communicate with Remote Agents in
    multi-agent ecosystem.

    Example:
        # In host/routing agent
        conn = await RemoteAgentConnection.from_url("http://localhost:10001")
        result = await conn.send_message("Query all open Jira tickets")
    """

    def __init__(
        self,
        agent_card: AgentCard,
        client: A2AClient,
    ) -> None:
        """Initialize connection.

        Args:
            agent_card: Remote agent's AgentCard metadata.
            client: A2A client for communication.
        """
        self.card = agent_card
        self.client = client

    @classmethod
    async def from_url(
        cls,
        url: str,
        timeout: float = 30.0,
    ) -> "RemoteAgentConnection":
        """Create connection by discovering agent at URL.

        Args:
            url: Remote agent's base URL.
            timeout: HTTP timeout in seconds.

        Returns:
            RemoteAgentConnection instance.
        """
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as http_client:
            # Resolve agent card
            resolver = A2ACardResolver(http_client, url)
            card = await resolver.get_agent_card()

            # Create persistent client
            client = A2AClient(httpx_client=http_client, url=url)

            logger.info("Connected to remote agent: %s at %s", card.name, url)

            return cls(agent_card=card, client=client)

    async def send_message(
        self,
        text: str,
        context_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Union[Task, Message, None]:
        """Send message to remote agent.

        Args:
            text: Message text to send.
            context_id: Optional context ID for conversation continuity.
            task_id: Optional task ID for task continuation.

        Returns:
            Task or Message response from remote agent.
        """
        # Create message
        message = Message(
            message_id=str(uuid4()),
            context_id=context_id or str(uuid4()),
            task_id=task_id,
            role=Role.user,
            parts=[Part(root=TextPart(text=text))],
        )

        # Create request
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(message=message),
        )

        logger.debug(
            "Sending message to %s: %s...",
            self.card.name,
            text[:50],
        )

        # Send and collect response
        async for event in self.client.send_message(request):
            if isinstance(event, Message):
                return event
            if isinstance(event, tuple) and len(event) > 0:
                task = event[0]
                if isinstance(task, Task) and self._is_terminal_state(task):
                    return task

        return None

    def _is_terminal_state(self, task: Task) -> bool:
        """Check if task is in terminal state.

        Args:
            task: The task to check.

        Returns:
            True if task is complete, canceled, or failed.
        """
        terminal_states = {
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
        }
        return task.status.state in terminal_states

    def get_skills(self) -> List[dict]:
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


class RemoteAgentRegistry:
    """Registry of remote agents for multi-agent orchestration.

    Provides discovery and routing to multiple remote agents.

    Example:
        registry = RemoteAgentRegistry()
        await registry.discover([
            "http://localhost:10001",  # Jira agent
            "http://localhost:10002",  # Slack agent
        ])

        # List available agents
        agents = registry.list_agents()

        # Send to specific agent
        result = await registry.send_to("jira-agent", "Get my tickets")
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self.connections: Dict[str, RemoteAgentConnection] = {}
        self.cards: Dict[str, AgentCard] = {}

    async def discover(self, agent_urls: List[str]) -> int:
        """Discover and register agents from URLs.

        Args:
            agent_urls: List of agent base URLs.

        Returns:
            Number of successfully discovered agents.
        """
        count = 0
        for url in agent_urls:
            try:
                conn = await RemoteAgentConnection.from_url(url)
                name = conn.card.name
                self.connections[name] = conn
                self.cards[name] = conn.card
                count += 1
                logger.info("Discovered agent: %s", name)
            except Exception as e:
                logger.warning("Failed to discover agent at %s: %s", url, e)

        return count

    async def add_agent(
        self,
        url: str,
        name: Optional[str] = None,
    ) -> Optional[RemoteAgentConnection]:
        """Add a single agent to the registry.

        Args:
            url: Agent URL.
            name: Optional override for agent name.

        Returns:
            Connection if successful, None otherwise.
        """
        try:
            conn = await RemoteAgentConnection.from_url(url)
            agent_name = name or conn.card.name
            self.connections[agent_name] = conn
            self.cards[agent_name] = conn.card
            return conn
        except Exception as e:
            logger.warning("Failed to add agent at %s: %s", url, e)
            return None

    def remove_agent(self, name: str) -> bool:
        """Remove an agent from the registry.

        Args:
            name: Agent name.

        Returns:
            True if removed, False if not found.
        """
        if name in self.connections:
            del self.connections[name]
            del self.cards[name]
            return True
        return False

    def list_agents(self) -> List[dict]:
        """List available agents with metadata.

        Returns:
            List of agent info dicts.
        """
        return [
            {
                "name": card.name,
                "description": card.description,
                "url": card.url,
                "skills": [s.name for s in (card.skills or [])],
            }
            for card in self.cards.values()
        ]

    def get_agent_names(self) -> List[str]:
        """Get list of registered agent names.

        Returns:
            List of agent names.
        """
        return list(self.connections.keys())

    async def send_to(
        self,
        agent_name: str,
        message: str,
        **kwargs,
    ) -> Union[Task, Message, None]:
        """Send message to named agent.

        Args:
            agent_name: Name of the target agent.
            message: Message text to send.
            **kwargs: Additional arguments for send_message.

        Returns:
            Response from remote agent.

        Raises:
            ValueError: If agent not found.
        """
        if agent_name not in self.connections:
            raise ValueError(f"Agent '{agent_name}' not found in registry")

        return await self.connections[agent_name].send_message(message, **kwargs)

    async def broadcast(
        self,
        message: str,
    ) -> Dict[str, Union[Task, Message, None]]:
        """Send message to all registered agents.

        Args:
            message: Message to broadcast.

        Returns:
            Dict mapping agent names to responses.
        """
        results: Dict[str, Union[Task, Message, None]] = {}
        for name, conn in self.connections.items():
            try:
                results[name] = await conn.send_message(message)
            except Exception as e:
                logger.warning("Failed to send to %s: %s", name, e)
                results[name] = None
        return results

    def get_connection(self, name: str) -> Optional[RemoteAgentConnection]:
        """Get connection for a specific agent.

        Args:
            name: Agent name.

        Returns:
            Connection or None if not found.
        """
        return self.connections.get(name)

    def __len__(self) -> int:
        """Return number of registered agents."""
        return len(self.connections)

    def __contains__(self, name: str) -> bool:
        """Check if agent is registered."""
        return name in self.connections
