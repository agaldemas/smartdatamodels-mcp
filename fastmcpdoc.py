import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, Optional, Union

from fastmcp import FastMCP, Client
from pydantic import Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize the FastMCP server
app = FastMCP(
    name="fastmcp-doc",
    instructions="MCP server for FastMCP documentation search."
)

# Tool definitions
@app.tool()
async def search_fast_mcp(
    query: str = Field(..., description="The search query for FastMCP documentation"),
    sessionId: Optional[str] = Field(None, description="The ID of the tool call")
) -> str:
    """Search FastMCP documentation for a given query.

    Returns:
        JSON string with search results from FastMCP.
    """
    try:
        async with Client("https://gofastmcp.com/mcp") as client:
            result = await client.call_tool(
                name="SearchFastMcp",
                arguments={"query": query}
            )
        return json.dumps({
            "success": True,
            "query": query,
            "result": result
        }, indent=2)
    except Exception as e:
        logger.error(f"FastMCP search failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "query": query
        }, indent=2)

def run_http_server(port: int = 8000):
    """Run the MCP server with HTTP/SSE support."""
    logger.info(f"Starting FastMCP Doc MCP Server with HTTP transport on port {port}")
    asyncio.run(app.run_sse_async(port=port))

def main():
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="FastMCP Doc MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default=os.getenv("MCP_TRANSPORT", "sse"), # Default to sse as requested
        help="Transport mode: stdio, sse (default), or http"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MCP_HTTP_PORT", "8000")),
        help="Port for HTTP/SSE transport (default: 8000)"
    )

    args = parser.parse_args()

    transport = args.transport.lower()
    port = args.port

    if transport in ["sse", "http"]:
        run_http_server(port)
    else:  # stdio
        logger.info("Starting FastMCP Doc MCP Server with stdio transport")
        app.run(transport="stdio")

if __name__ == "__main__":
    main()
