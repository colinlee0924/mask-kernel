"""MCP Client wrapper using langchain-mcp-adapters.

This module provides MaskMCPClient for loading MCP tools from
configuration files and integrating them with MASK agents.

Requires: pip install mask-kernel[mcp]
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class MaskMCPClient:
    """MCP Client wrapper using langchain-mcp-adapters.

    Wraps MultiServerMCPClient to load MCP tools from config file.
    Handles environment variable substitution in server configurations.

    Example:
        client = MaskMCPClient.from_config("config/mcp_servers.json")
        async with client:
            tools = await client.get_tools()
            agent = create_mask_agent(additional_tools=tools)
    """

    def __init__(self, server_configs: Dict[str, Dict[str, Any]]) -> None:
        """Initialize with server configurations.

        Args:
            server_configs: Dict mapping server names to their configs.
                Each config should have: command, args, env (for stdio)
                or url, transport (for http).
        """
        self.server_configs = server_configs
        self._client = None
        self._tools: Optional[List[BaseTool]] = None

    @classmethod
    def from_config(cls, config_path: str | Path) -> "MaskMCPClient":
        """Create MaskMCPClient from JSON config file.

        Expected config format (mcp_servers.json):
        ```json
        {
            "mcpServers": {
                "atlassian": {
                    "command": "uvx",
                    "args": ["mcp-atlassian"],
                    "env": {
                        "JIRA_URL": "${JIRA_URL}",
                        "JIRA_EMAIL": "${JIRA_EMAIL}"
                    }
                }
            }
        }
        ```

        Args:
            config_path: Path to mcp_servers.json config file.

        Returns:
            Configured MaskMCPClient instance.
        """
        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning("MCP config not found: %s", config_path)
            return cls(server_configs={})

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        # Transform config to langchain-mcp-adapters format
        servers: Dict[str, Dict[str, Any]] = {}
        for name, server_config in config.get("mcpServers", {}).items():
            servers[name] = cls._transform_server_config(server_config)

        logger.debug("Loaded %d MCP server configs from %s", len(servers), config_path)

        return cls(server_configs=servers)

    @staticmethod
    def _resolve_env_vars(value: str) -> str:
        """Resolve ${VAR} patterns in a string with environment variable values."""
        def replacer(match):
            var_name = match.group(1)
            return os.environ.get(var_name, "")
        return re.sub(r"\$\{(\w+)\}", replacer, value)

    @classmethod
    def _transform_server_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform our config format to langchain-mcp-adapters format.

        Handles environment variable substitution for ${VAR} patterns.
        Supports both stdio and HTTP-based transports.

        Args:
            config: Server configuration dict.

        Returns:
            Transformed configuration.
        """
        # Determine transport type
        transport = config.get("transport") or config.get("type", "stdio")

        # Map 'streamable-http' to 'streamable_http' (underscore)
        if transport == "streamable-http":
            transport = "streamable_http"

        # Handle HTTP-based transports (sse, http, streamable_http, websocket)
        if transport in ("sse", "http", "streamable_http", "websocket"):
            # Resolve environment variables in headers
            resolved_headers: Dict[str, str] = {}
            for key, value in config.get("headers", {}).items():
                if isinstance(value, str):
                    resolved_headers[key] = cls._resolve_env_vars(value)
                else:
                    resolved_headers[key] = str(value)

            result: Dict[str, Any] = {
                "url": config["url"],
                "transport": transport,
            }

            if resolved_headers:
                result["headers"] = resolved_headers

            return result

        # Handle stdio transport (default)
        resolved_env: Dict[str, str] = {}
        for key, value in config.get("env", {}).items():
            if isinstance(value, str):
                resolved_env[key] = cls._resolve_env_vars(value)
            else:
                resolved_env[key] = str(value)

        return {
            "command": config["command"],
            "args": config.get("args", []),
            "env": resolved_env,
            "transport": "stdio",
        }

    async def __aenter__(self) -> "MaskMCPClient":
        """Async context manager entry - initialize MultiServerMCPClient."""
        if not self.server_configs:
            logger.debug("No MCP servers configured")
            return self

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError:
            raise ImportError(
                "MCP support requires the 'langchain-mcp-adapters' package. "
                "Install with: pip install mask-kernel[mcp]"
            )

        self._client = MultiServerMCPClient(self.server_configs)
        # Note: As of langchain-mcp-adapters 0.1.0, MultiServerMCPClient
        # no longer uses context manager pattern. Tools are fetched directly.
        logger.info("Initialized MCP client with %d servers", len(self.server_configs))

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - cleanup."""
        self._client = None
        self._tools = None

    async def get_tools(self) -> List[BaseTool]:
        """Get all tools from configured MCP servers.

        Returns:
            List of LangChain BaseTool instances from all MCP servers.
        """
        if not self.server_configs:
            return []

        # Initialize client if not already done
        if self._client is None:
            try:
                from langchain_mcp_adapters.client import MultiServerMCPClient
            except ImportError:
                raise ImportError(
                    "MCP support requires the 'langchain-mcp-adapters' package. "
                    "Install with: pip install mask-kernel[mcp]"
                )
            self._client = MultiServerMCPClient(self.server_configs)

        if self._tools is None:
            # As of langchain-mcp-adapters 0.1.0, get_tools() is async
            self._tools = await self._client.get_tools()
            logger.debug("Loaded %d tools from MCP servers", len(self._tools))

        return self._tools

    def get_server_names(self) -> List[str]:
        """Get list of configured server names.

        Returns:
            List of server names.
        """
        return list(self.server_configs.keys())

    def has_server(self, name: str) -> bool:
        """Check if a server is configured.

        Args:
            name: Server name.

        Returns:
            True if server is configured.
        """
        return name in self.server_configs
