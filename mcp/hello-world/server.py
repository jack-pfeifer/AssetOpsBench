from mcp.server.fastmcp import FastMCP

mcp = FastMCP("aob-hello", json_response=True)

@mcp.tool()
def hello(name: str = "world") -> str:
    return f"Hello, {name}! MCP is running."

if __name__ == "__main__":
    mcp.run(transport="streamable-http")

