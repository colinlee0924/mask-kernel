"""A2A Helper Functions.

This module provides helper functions that return native A2A SDK types.
Developers can use these helpers with the native A2A SDK directly.

Usage:
    from a2a import A2AServer
    from mask.a2a.helpers import create_a2a_executor

    agent = create_my_agent()
    executor = create_a2a_executor(agent)
    server = A2AServer(executor)
    server.run(port=10001)
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mask.agent.base_agent import BaseAgent
    from mask.a2a.executor import MaskAgentExecutor


def create_a2a_executor(
    agent: "BaseAgent",
    stream: bool = False,
    server_name: Optional[str] = None,
) -> "MaskAgentExecutor":
    """Create an A2A executor from a MASK agent.

    This helper function creates a MaskAgentExecutor that can be used
    with the native A2A SDK's A2AServer.

    Args:
        agent: The MASK BaseAgent instance to wrap.
        stream: Whether to use streaming responses. Defaults to False.
        server_name: Optional server name for trace display in Phoenix/Langfuse.
            If not provided, falls back to the agent name.

    Returns:
        MaskAgentExecutor instance compatible with A2A SDK.

    Example:
        from a2a import A2AServer
        from mask.a2a.helpers import create_a2a_executor
        from mask.models import LLMFactory, ModelTier

        # Create your agent
        agent = create_my_agent()

        # Create executor and server using native A2A SDK
        executor = create_a2a_executor(agent, server_name="my-agent-server")
        server = A2AServer(executor)
        server.run(port=10001)
    """
    from mask.a2a.executor import MaskAgentExecutor

    return MaskAgentExecutor(
        agent=agent,
        stream=stream,
        server_name=server_name,
    )
