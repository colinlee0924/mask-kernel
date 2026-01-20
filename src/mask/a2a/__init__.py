"""MASK A2A Protocol integration.

This module provides A2A (Agent-to-Agent) protocol support for MASK agents,
enabling multi-agent ecosystems.

Components:
- create_a2a_executor: Helper function to create A2A executor (recommended)
- create_database_task_store: Create DatabaseTaskStore for A2A task persistence
- create_openai_compat_app: Create OpenAI-compatible API wrapping A2A
- run_openai_compat_server: Run OpenAI-compatible wrapper server
- MaskA2AServer: Expose MASK agent as A2A remote service (deprecated)
- MaskAgentExecutor: Bridge BaseAgent to A2A AgentExecutor

Message Synchronization (Open WebUI integration):
- StateSynchronizer: Sync Open WebUI messages with LangGraph checkpoints
- SyncResult: Result of synchronization analysis (retry/delete detection)

Multi-agent orchestration (Native SDK - Recommended):
- NativeRemoteAgentConnection: Connect to remote A2A agents using native SDK
- NativeRemoteAgentFactory: Factory for managing remote agent connections
- DelegationToolFactory: Create delegation tools for orchestrator agents
- create_delegation_tools: Convenience function to create delegation tools

Legacy (Deprecated - may have issues in uvicorn):
- RemoteAgentConnection: Connect to remote A2A agents (legacy)
- RemoteAgentRegistry: Manage multiple remote agent connections (legacy)
- StreamingA2AClient: Subscribe to sub-agent event streams (deprecated)

For LangGraph checkpoint helpers, use mask.checkpoints module:
    from mask.checkpoints import setup_postgres_tables, create_async_checkpointer
"""

from mask.a2a.delegation import (
    DelegationToolFactory,
    create_delegation_tool_sync,
    create_delegation_tools,
)
from mask.a2a.executor import MaskAgentExecutor
from mask.a2a.helpers import create_a2a_executor, create_database_task_store
from mask.a2a.openai_compat import create_openai_compat_app, run_openai_compat_server
from mask.a2a.remote_agent import NativeRemoteAgentConnection, NativeRemoteAgentFactory
from mask.a2a.remote_connection import RemoteAgentConnection, RemoteAgentRegistry
from mask.a2a.server import MaskA2AServer
from mask.a2a.state_sync import StateSynchronizer, SyncResult
from mask.a2a.streaming_client import StreamingA2AClient, create_streaming_client

__all__ = [
    # Recommended helpers
    "create_a2a_executor",
    "create_database_task_store",
    "create_openai_compat_app",
    "run_openai_compat_server",
    # Message synchronization (Open WebUI integration)
    "StateSynchronizer",
    "SyncResult",
    # Multi-agent orchestration (Native SDK - Recommended)
    "NativeRemoteAgentConnection",
    "NativeRemoteAgentFactory",
    "DelegationToolFactory",
    "create_delegation_tools",
    "create_delegation_tool_sync",
    # Legacy (deprecated)
    "MaskA2AServer",
    "MaskAgentExecutor",
    "RemoteAgentConnection",
    "RemoteAgentRegistry",
    "StreamingA2AClient",
    "create_streaming_client",
]
