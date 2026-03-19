"""MCP (Model Context Protocol) adapter for dynamic tool loading."""

from __future__ import annotations

import asyncio
from typing import Any


class MCPAdapter:
    """Adapter to load tools from MCP servers dynamically.

    Uses langchain-mcp-adapters to connect to MCP servers
    and convert their tools to LangChain-compatible tools.
    """

    def __init__(self, server_configs: list[dict[str, Any]] | None = None):
        self._server_configs = server_configs or []
        self._tools: list = []
        self._loaded = False

    async def _load_tools_async(self) -> list:
        """Load tools from configured MCP servers."""
        if self._loaded:
            return self._tools

        tools = []
        for config in self._server_configs:
            server_type = config.get("type", "stdio")
            try:
                if server_type == "stdio":
                    tools.extend(await self._load_stdio_tools(config))
                elif server_type == "sse":
                    tools.extend(await self._load_sse_tools(config))
            except Exception as e:
                print(f"Warning: Failed to load MCP server {config.get('name', 'unknown')}: {e}")

        self._tools = tools
        self._loaded = True
        return tools

    async def _load_stdio_tools(self, config: dict) -> list:
        """Load tools from a stdio MCP server."""
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            client = MultiServerMCPClient(
                {
                    config.get("name", "mcp_server"): {
                        "command": config["command"],
                        "args": config.get("args", []),
                        "env": config.get("env"),
                    }
                }
            )
            async with client:
                return client.get_tools()
        except ImportError:
            print("Warning: langchain-mcp-adapters not installed.")
            return []

    async def _load_sse_tools(self, config: dict) -> list:
        """Load tools from an SSE MCP server."""
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            client = MultiServerMCPClient(
                {
                    config.get("name", "mcp_server"): {
                        "url": config["url"],
                        "transport": "sse",
                    }
                }
            )
            async with client:
                return client.get_tools()
        except ImportError:
            print("Warning: langchain-mcp-adapters not installed.")
            return []

    def load_tools(self) -> list:
        """Synchronous wrapper for loading MCP tools."""
        try:
            loop = asyncio.get_running_loop()
            # If already in an async context, can't run synchronously
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return loop.run_in_executor(pool, lambda: asyncio.run(self._load_tools_async()))
        except RuntimeError:
            return asyncio.run(self._load_tools_async())

    @property
    def tools(self) -> list:
        return self._tools

    @property
    def is_loaded(self) -> bool:
        return self._loaded
