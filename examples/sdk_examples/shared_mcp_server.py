"""
Shared MCP Server for All Examples

This MCP server provides tools that all LLM provider examples can use:
- Math operations (add, multiply, calculate)
- Time utilities (get_time, get_timezone)
- Data operations (store_data, retrieve_data)

Run this server first before running any LLM examples that use MCP tools.

Usage:
    python examples/shared_mcp_server.py

The server will run on http://localhost:8000/sse
"""

import json
import os
import sys
from datetime import datetime

import pytz
import logging

# Add SDK to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from neatlogs import flush, init, mcp_tool, shutdown


def _env_default(k: str, v: str) -> None:
    # Do not override caller's env.
    os.environ.setdefault(k, v)


_env_default("NEATLOGS_LOG_SPANS", "true")
_env_default("NEATLOGS_LOG_SPANS_FILE", "spans_agentic_matrix.log")
_env_default("NEATLOGS_LOG_RAW_SPANS", "true")
_env_default("NEATLOGS_LOG_RAW_SPANS_FILE", "spans_raw_agentic_matrix.log")

# Basic server-side logging so tool calls are visible.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neatlogs.mcp_server")

# Initialize Neatlogs
init(
    api_key=os.getenv("NEATLOGS_API_KEY", "test-key"),
    endpoint="http://localhost:3000/api/data/v4/batch",
    instrumentations=["mcp"],
    debug=True,
)

from mcp.server.fastmcp import FastMCP

# Create FastMCP server
mcp = FastMCP("neatlogs-shared-tools")

# In-memory data store
data_store = {}


@mcp.tool()
@mcp_tool(name="add")
def add(a: float, b: float) -> str:
    """Add two numbers together."""
    result = a + b
    return json.dumps({"operation": "add", "a": a, "b": b, "result": result})


@mcp.tool()
@mcp_tool(name="multiply")
def multiply(a: float, b: float) -> str:
    """Multiply two numbers together."""
    result = a * b
    return json.dumps({"operation": "multiply", "a": a, "b": b, "result": result})


@mcp.tool()
@mcp_tool(name="calculate")
def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        # Only allow basic math operations
        allowed_chars = set("0123456789+-*/(). ")
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return json.dumps({"expression": expression, "result": result})
        return json.dumps({"error": "Invalid characters in expression"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
@mcp_tool(name="get_time")
def get_time() -> str:
    """Get the current server time in ISO format."""
    current_time = datetime.now().isoformat()
    return json.dumps({"time": current_time, "timezone": "local"})


@mcp.tool()
@mcp_tool(name="get_timezone")
def get_timezone(timezone: str = "UTC") -> str:
    """Get current time in a specific timezone."""
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz).isoformat()
        return json.dumps({"timezone": timezone, "time": current_time})
    except Exception as e:
        return json.dumps({"error": f"Invalid timezone: {str(e)}"})


@mcp.tool()
@mcp_tool(name="store_data")
def store_data(key: str, value: str) -> str:
    """Store a key-value pair in the server's memory."""
    data_store[key] = value
    return json.dumps({"status": "stored", "key": key, "value": value})


@mcp.tool()
@mcp_tool(name="retrieve_data")
def retrieve_data(key: str) -> str:
    """Retrieve a value from the server's memory by key."""
    if key in data_store:
        return json.dumps({"key": key, "value": data_store[key]})
    return json.dumps({"error": f"Key '{key}' not found"})


@mcp.tool()
@mcp_tool(name="list_data")
def list_data() -> str:
    """List all keys stored in the server's memory."""
    return json.dumps({"keys": list(data_store.keys()), "count": len(data_store)})


@mcp.tool()
@mcp_tool(name="tavily_web_search")
def tavily_web_search(query: str, max_results: int = 3) -> str:
    """
    Search the web using Tavily API.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 3)
    
    Returns:
        JSON string with search results
    """
    try:
        # Use print as well as logging; logging may be configured elsewhere.
        print(f"[MCP] tavily_web_search called: query={query!r} max_results={max_results}", flush=True)
        logger.info("tavily_web_search called: query=%r max_results=%s", query, max_results)
        from langchain_community.tools import TavilySearchResults
        
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            print("[MCP] TAVILY_API_KEY not set on MCP server", flush=True)
            logger.warning("TAVILY_API_KEY not set on MCP server")
            return json.dumps({"error": "TAVILY_API_KEY not set"})
        
        tool = TavilySearchResults(
            api_key=tavily_api_key,
            max_results=max_results,
            search_depth="advanced"
        )
        
        results = tool.invoke({"query": query})
        try:
            n = len(results)  # type: ignore[arg-type]
        except Exception:
            n = -1
        print(f"[MCP] tavily_web_search returning {n} result(s)", flush=True)
        logger.info("tavily_web_search returning %d result(s)", len(results) if hasattr(results, "__len__") else -1)
        return json.dumps({
            "query": query,
            "results": results,
            "count": len(results)
        })
    except Exception as e:
        print(f"[MCP] tavily_web_search error: {e}", flush=True)
        logger.exception("tavily_web_search error")
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    import atexit

    atexit.register(lambda: (flush(), shutdown()))

    print("=" * 60)
    print("🚀 Neatlogs Shared MCP Server")
    print("=" * 60)
    print("\nAvailable Tools:")
    print("  - add(a, b) - Add two numbers")
    print("  - multiply(a, b) - Multiply two numbers")
    print("  - calculate(expression) - Evaluate math expression")
    print("  - get_time() - Get current server time")
    print("  - get_timezone(timezone) - Get time in timezone")
    print("  - store_data(key, value) - Store key-value pair")
    print("  - retrieve_data(key) - Retrieve stored value")
    print("  - list_data() - List all stored keys")
    print("  - tavily_web_search(query, max_results) - Search the web")
    print("\nServer running on: http://localhost:8000/sse")
    print("Press Ctrl+C to stop\n")
    print("=" * 60)

    mcp.run(transport="sse")
