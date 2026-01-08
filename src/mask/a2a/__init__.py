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
- RemoteAgentConnection: Connect to remote A2A agents
- RemoteAgentRegistry: Manage multiple remote agent connections

For LangGraph checkpoint helpers, use mask.checkpoints module:
    from mask.checkpoints import setup_postgres_tables, create_async_checkpointer
"""

from mask.a2a.executor import MaskAgentExecutor
from mask.a2a.helpers import create_a2a_executor, create_database_task_store
from mask.a2a.openai_compat import create_openai_compat_app, run_openai_compat_server
from mask.a2a.remote_connection import RemoteAgentConnection, RemoteAgentRegistry
from mask.a2a.server import MaskA2AServer

__all__ = [
    # Recommended helpers
    "create_a2a_executor",
    "create_database_task_store",
    "create_openai_compat_app",
    "run_openai_compat_server",
    # Legacy (deprecated)
    "MaskA2AServer",
    "MaskAgentExecutor",
    "RemoteAgentConnection",
    "RemoteAgentRegistry",
]
