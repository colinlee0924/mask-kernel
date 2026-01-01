"""MCP integration utilities.

This module provides convenience functions for integrating MCP tools
with MASK agents.
"""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


async def load_mcp_tools_from_config(
    config_path: str | Path,
    server_names: Optional[List[str]] = None,
) -> List[BaseTool]:
    """Load MCP tools from config file.

    Convenience function that handles client lifecycle.

    Args:
        config_path: Path to mcp_servers.json config file.
        server_names: Optional list of server names to load.
            If None, loads all configured servers.

    Returns:
        List of LangChain BaseTool instances.

    Example:
        tools = await load_mcp_tools_from_config("config/mcp_servers.json")
        agent = create_mask_agent(additional_tools=tools)
    """
    from mask.mcp.client import MaskMCPClient

    client = MaskMCPClient.from_config(config_path)

    # Filter servers if specified
    if server_names:
        client.server_configs = {
            k: v for k, v in client.server_configs.items()
            if k in server_names
        }

    if not client.server_configs:
        logger.debug("No MCP servers to load")
        return []

    # Load tools directly (langchain-mcp-adapters 0.1.0+ doesn't need context manager)
    tools = await client.get_tools()
    logger.info("Loaded %d MCP tools", len(tools))
    return tools


async def load_mcp_tools_for_agent(
    config_dir: str | Path = "config",
    server_names: Optional[List[str]] = None,
) -> List[BaseTool]:
    """Load MCP tools from agent's config directory.

    Looks for mcp_servers.json in the config directory.

    Args:
        config_dir: Base config directory.
        server_names: Optional list of servers to load.

    Returns:
        List of MCP tools.
    """
    config_path = Path(config_dir) / "mcp_servers.json"

    if not config_path.exists():
        logger.debug("No MCP config found at %s", config_path)
        return []

    return await load_mcp_tools_from_config(config_path, server_names)


class MCPToolManager:
    """Manager for MCP tools with persistent connections.

    Use this when you need to maintain MCP connections across
    multiple agent invocations.

    Example:
        manager = MCPToolManager("config/mcp_servers.json")
        await manager.connect()

        try:
            tools = manager.get_tools()
            # Use tools in agent...
        finally:
            await manager.disconnect()
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize manager.

        Args:
            config_path: Path to MCP config file.
        """
        from mask.mcp.client import MaskMCPClient

        self.config_path = Path(config_path)
        self._client: Optional[MaskMCPClient] = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to MCP servers."""
        if self._connected:
            return

        from mask.mcp.client import MaskMCPClient

        self._client = MaskMCPClient.from_config(self.config_path)
        await self._client.__aenter__()
        self._connected = True

        logger.info("MCP manager connected")

    async def disconnect(self) -> None:
        """Disconnect from MCP servers."""
        if self._client and self._connected:
            await self._client.__aexit__(None, None, None)
            self._connected = False
            logger.info("MCP manager disconnected")

    def get_tools(self) -> List[BaseTool]:
        """Get MCP tools.

        Must call connect() first.

        Returns:
            List of MCP tools.

        Raises:
            RuntimeError: If not connected.
        """
        if not self._connected or self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")

        # Note: get_tools is sync in langchain-mcp-adapters
        return self._client._client.get_tools() if self._client._client else []

    @property
    def is_connected(self) -> bool:
        """Check if connected to MCP servers."""
        return self._connected

    async def __aenter__(self) -> "MCPToolManager":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()
