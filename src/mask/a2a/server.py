"""A2A Server wrapper for MASK agents.

This module provides MaskA2AServer for exposing MASK agents as
A2A remote services, following patterns from a2a-python-samples.
"""

import logging
from typing import TYPE_CHECKING, List, Optional

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from mask.a2a.executor import MaskAgentExecutor

if TYPE_CHECKING:
    from mask.agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class MaskA2AServer:
    """A2A Server wrapper for exposing MASK Agent as A2A remote service.

    Follows a2a-python-samples pattern:
    - AgentCard for service discovery
    - AgentExecutor for request handling
    - HTTP Server via Starlette

    Example:
        from mask.a2a import MaskA2AServer
        from mask.agent import create_mask_agent

        agent = create_mask_agent()
        server = MaskA2AServer(
            agent=agent,
            name="jira-expertise-agent",
            description="Jira domain expert agent",
            skills=[
                AgentSkill(id="jira_query", name="Query Jira issues"),
                AgentSkill(id="jira_create", name="Create Jira tickets"),
            ],
        )
        server.run(port=10001)
    """

    def __init__(
        self,
        agent: "BaseAgent",
        name: str,
        description: str,
        url: Optional[str] = None,
        version: str = "1.0.0",
        skills: Optional[List[AgentSkill]] = None,
        capabilities: Optional[AgentCapabilities] = None,
        stream: bool = False,
        default_input_modes: Optional[List[str]] = None,
        default_output_modes: Optional[List[str]] = None,
    ) -> None:
        """Initialize A2A Server.

        Args:
            agent: The MASK BaseAgent to expose.
            name: Agent name for AgentCard.
            description: Agent description for AgentCard.
            url: Agent URL (auto-detected if not provided).
            version: Agent version string.
            skills: List of AgentSkill describing agent capabilities.
            capabilities: Agent capabilities configuration.
            stream: Whether to enable streaming responses.
            default_input_modes: Supported input modes (default: ["text"]).
            default_output_modes: Supported output modes (default: ["text"]).
        """
        self.agent = agent
        self.name = name
        self.description = description
        self.url = url
        self.version = version
        self.skills = skills or []
        self.capabilities = capabilities or AgentCapabilities(streaming=stream)
        self.stream = stream
        self.default_input_modes = default_input_modes or ["text"]
        self.default_output_modes = default_output_modes or ["text"]
        self._app: Optional[A2AStarletteApplication] = None

    def create_agent_card(self, host: str, port: int) -> AgentCard:
        """Create AgentCard for service discovery.

        Args:
            host: Server host.
            port: Server port.

        Returns:
            AgentCard instance.
        """
        url = self.url or f"http://{host}:{port}/"
        return AgentCard(
            name=self.name,
            description=self.description,
            url=url,
            version=self.version,
            skills=self.skills,
            capabilities=self.capabilities,
            defaultInputModes=self.default_input_modes,
            defaultOutputModes=self.default_output_modes,
        )

    def create_app(self, host: str, port: int) -> A2AStarletteApplication:
        """Create A2A Starlette application.

        Args:
            host: Server host.
            port: Server port.

        Returns:
            A2AStarletteApplication instance.
        """
        # Create executor
        executor = MaskAgentExecutor(self.agent, stream=self.stream)

        # Create agent card
        agent_card = self.create_agent_card(host, port)

        # Create request handler
        handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )

        # Create application
        self._app = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=handler,
        )

        logger.info(
            "Created A2A application: name=%s, url=%s",
            self.name,
            agent_card.url,
        )

        return self._app

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 10001,
        log_level: str = "info",
    ) -> None:
        """Start A2A HTTP server.

        Args:
            host: Host to bind to.
            port: Port to listen on.
            log_level: Uvicorn log level.
        """
        import uvicorn

        app = self.create_app(host, port)

        logger.info("Starting A2A server on %s:%d", host, port)

        uvicorn.run(
            app.build(),
            host=host,
            port=port,
            log_level=log_level,
        )

    def get_app(self, host: str = "0.0.0.0", port: int = 10001):
        """Get the ASGI application for custom deployment.

        Args:
            host: Host for URL generation.
            port: Port for URL generation.

        Returns:
            Starlette application.
        """
        if self._app is None:
            self.create_app(host, port)
        return self._app.build()
