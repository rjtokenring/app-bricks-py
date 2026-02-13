# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
from arduino.app_utils import brick


@brick
class MCPClient:
    """A class to communicate with the MCP server to perform various tasks."""

    def __init__(self, url: str = "http://localhost:8000/mcp", transport: str = "streamable_http", tool_name_prefix: bool = False, **kwargs):
        """Initialize the MCPClient with a MultiServerMCPClient."""
        self._client = MultiServerMCPClient(
            connections={
                "weather": {
                    "transport": transport,
                    "url": url,
                }
            },
            tool_name_prefix=tool_name_prefix,
            **kwargs,
        )

    def list_tools(self) -> str:
        """List the available tools from the MCP server.

        Returns:
            str: A list of available tools.
        """
        return asyncio.run(self._client.get_tools())
