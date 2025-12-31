"""MASK A2A Protocol integration.

This module provides A2A (Agent-to-Agent) protocol support for MASK agents,
enabling multi-agent ecosystems.

Components:
- MaskA2AServer: Expose MASK agent as A2A remote service
- MaskAgentExecutor: Bridge BaseAgent to A2A AgentExecutor
- RemoteAgentConnection: Connect to remote A2A agents
- RemoteAgentRegistry: Manage multiple remote agent connections
"""

from mask.a2a.executor import MaskAgentExecutor
from mask.a2a.remote_connection import RemoteAgentConnection, RemoteAgentRegistry
from mask.a2a.server import MaskA2AServer

__all__ = [
    "MaskA2AServer",
    "MaskAgentExecutor",
    "RemoteAgentConnection",
    "RemoteAgentRegistry",
]
