"""A2A Helper Functions.

This module provides helper functions that return native A2A SDK types.
Developers can use these helpers with the native A2A SDK directly.

Usage:
    from langchain.agents import create_agent
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from mask.a2a import create_a2a_executor

    graph = create_agent(model, tools, system_prompt)
    executor = create_a2a_executor(graph, server_name="my-agent")
    handler = DefaultRequestHandler(agent_executor=executor, task_store=...)
    app = A2AStarletteApplication(agent_card, http_handler=handler)
"""

from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from mask.a2a.executor import MaskAgentExecutor
    from mask.agent.base_agent import BaseAgent


def create_a2a_executor(
    agent: Union["BaseAgent", "CompiledStateGraph"],
    stream: bool = True,
    server_name: Optional[str] = None,
) -> "MaskAgentExecutor":
    """Create an A2A executor from a LangChain CompiledStateGraph or MASK agent.

    This helper function creates a MaskAgentExecutor that bridges your agent
    to the A2A protocol. It supports both:
    - LangChain CompiledStateGraph from create_agent() (recommended)
    - MASK BaseAgent (legacy)

    Features:
    - Real-time streaming via TaskArtifactUpdateEvent (default enabled)
    - Multi-agent handoffs with context isolation
    - OpenTelemetry tracing integration

    Args:
        agent: LangChain CompiledStateGraph or MASK BaseAgent instance.
        stream: Whether to use real-time streaming (default True for Open WebUI).
        server_name: Optional server name for trace display in Phoenix/Langfuse.
            If not provided, falls back to the agent name attribute.

    Returns:
        MaskAgentExecutor instance compatible with A2A SDK.

    Example:
        from langchain.agents import create_agent
        from a2a.server.apps import A2AStarletteApplication
        from a2a.server.request_handlers import DefaultRequestHandler
        from a2a.server.tasks import InMemoryTaskStore
        from a2a.types import AgentCapabilities, AgentCard, AgentSkill
        from mask.a2a import create_a2a_executor

        # Create agent using native LangChain API
        graph = create_agent(model, tools, system_prompt)

        # Create executor
        executor = create_a2a_executor(graph, server_name="my-agent")

        # Build A2A server with native SDK
        agent_card = AgentCard(
            name="my-agent",
            skills=[AgentSkill(id="general", name="General", description="...", tags=["general"])],
            capabilities=AgentCapabilities(streaming=True),
            ...
        )
        handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
        app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)

        import uvicorn
        uvicorn.run(app.build(), host="0.0.0.0", port=10001)
    """
    from mask.a2a.executor import MaskAgentExecutor

    return MaskAgentExecutor(
        agent=agent,
        stream=stream,
        server_name=server_name,
    )
