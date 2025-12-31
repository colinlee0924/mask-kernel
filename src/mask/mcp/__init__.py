"""MASK MCP (Model Context Protocol) integration.

This module provides MCP integration using langchain-mcp-adapters,
allowing MASK agents to use tools from MCP servers.

Requires: pip install mask-kernel[mcp]
"""

from mask.mcp.client import MaskMCPClient
from mask.mcp.integration import (
    MCPToolManager,
    load_mcp_tools_for_agent,
    load_mcp_tools_from_config,
)

__all__ = [
    "MaskMCPClient",
    "load_mcp_tools_from_config",
    "load_mcp_tools_for_agent",
    "MCPToolManager",
]
