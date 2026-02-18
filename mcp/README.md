# MCP Prototype

This folder contains the initial MCP server prototype for AssetOpsBench.

## Current Status

- FastMCP server implemented
- Streamable HTTP transport configured
- Inspector connectivity validated
- Basic `hello` tool operational

This milestone validates the MCP transport layer and tool registration workflow.

## How to Run

1. Create virtual environment:

   python3 -m venv .venv
   source .venv/bin/activate

2. Install MCP SDK:

   pip install "mcp[cli]"

3. Run the server:

   python server.py

4. Connect via MCP Inspector to:

   http://localhost:8000/mcp
