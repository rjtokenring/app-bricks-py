# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Basic mcp client usage example"

from arduino.app_bricks.mcp_client import MCPClient, LocalPythonMCPEndpoint, HTTPEndpoint
from arduino.app_utils import App

external_mcp = HTTPEndpoint(name="filesystem", url="http://localhost:8080/mcp")

print("-------------------------------------------------")

local = LocalPythonMCPEndpoint(name="local_mcp", script_path="math_server.py")

client = MCPClient(clients=[local, external_mcp])

print(client.list_tools())

App.run()
