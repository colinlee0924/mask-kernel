"""MASK A2A Protocol integration.

This module provides A2A (Agent-to-Agent) protocol support for MASK agents,
enabling multi-agent ecosystems.

Components:
- create_a2a_executor: Helper function to create A2A executor (recommended)
- MaskA2AServer: Expose MASK agent as A2A remote service (deprecated)
- MaskAgentExecutor: Bridge BaseAgent to A2A AgentExecutor
- RemoteAgentConnection: Connect to remote A2A agents
- RemoteAgentRegistry: Manage multiple remote agent connections
"""

from mask.a2a.executor import MaskAgentExecutor
from mask.a2a.helpers import create_a2a_executor
from mask.a2a.remote_connection import RemoteAgentConnection, RemoteAgentRegistry
from mask.a2a.server import MaskA2AServer

__all__ = [
    # Recommended helper
    "create_a2a_executor",
    # Legacy (deprecated)
    "MaskA2AServer",
    "MaskAgentExecutor",
    "RemoteAgentConnection",
    "RemoteAgentRegistry",
]
