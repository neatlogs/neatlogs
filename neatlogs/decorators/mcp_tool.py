"""
MCP Tool Decorator for Neatlogs SDK

Provides @mcp_tool() decorator to manually instrument MCP tool functions.
"""

import json
import inspect
import functools
import logging
from typing import Any, Callable, TypeVar
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)
F = TypeVar('F', bound=Callable[..., Any])

def mcp_tool(name: str | None = None) -> Callable[[F], F]:
    """
    Decorator to instrument MCP tool functions with tracing.
    
    Args:
        name: Optional custom span name. If not provided, uses function name.
    
    Example:
        ```python
        from neatlogs import init, mcp_tool
        
        init(api_key="...", instrumentations=["mcp"])
        
        from mcp.server.fastmcp import FastMCP
        mcp = FastMCP("my-server")
        
        @mcp.tool()
        @mcp_tool(name="add_numbers")
        def add(a: int, b: int) -> str:
            return f"Result: {a + b}"
        ```
    """
    def decorator(func: F) -> F:
        span_name = name or f"{func.__name__}.tool"
        tool_name = name or func.__name__
        tracer = trace.get_tracer(__name__)
        
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    span_name,
                    kind=trace.SpanKind.INTERNAL,
                ) as span:
                    span.set_attribute("mcp.tool.name", tool_name)
                    span.set_attribute("openinference.span.kind", "MCP_TOOL")
                    span.set_attribute("tool.name", tool_name)
                    
                    try:
                        if args and hasattr(args[0], 'model_dump'):
                            arguments = args[0].model_dump()
                        else:
                            sig = inspect.signature(func)
                            bound = sig.bind_partial(*args, **kwargs)
                            bound.apply_defaults()
                            arguments = dict(bound.arguments)
                        
                        span.set_attribute("mcp.tool.arguments", json.dumps(arguments))
                        span.set_attribute("tool.parameters", json.dumps(arguments))
                        span.set_attribute("input.value", json.dumps(arguments))
                        span.set_attribute("input.mime_type", "application/json")
                    except Exception as e:
                        logger.debug(f"Failed to set MCP tool input attributes: {e}")
                    
                    try:
                        result = await func(*args, **kwargs)
                        try:
                            if isinstance(result, dict):
                                span.set_attribute("mcp.response.value", json.dumps(result))
                                span.set_attribute("output.value", json.dumps(result))
                                span.set_attribute("output.mime_type", "application/json")
                            elif isinstance(result, str):
                                output = {"result": result}
                                span.set_attribute("mcp.response.value", json.dumps(output))
                                span.set_attribute("output.value", json.dumps(output))
                                span.set_attribute("output.mime_type", "application/json")
                        except Exception as e:
                            logger.debug(f"Failed to set MCP tool output attributes: {e}")
                        
                        span.set_status(Status(StatusCode.OK))
                        return result
                        
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
            
            return async_wrapper
        
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    span_name,
                    kind=trace.SpanKind.INTERNAL,
                ) as span:
                    span.set_attribute("mcp.tool.name", tool_name)
                    span.set_attribute("openinference.span.kind", "MCP_TOOL")
                    span.set_attribute("tool.name", tool_name)

                    try:
                        if args and hasattr(args[0], 'model_dump'):
                            arguments = args[0].model_dump()
                        else:
                            sig = inspect.signature(func)
                            bound = sig.bind_partial(*args, **kwargs)
                            bound.apply_defaults()
                            arguments = dict(bound.arguments)
                        
                        span.set_attribute("mcp.tool.arguments", json.dumps(arguments))
                        span.set_attribute("tool.parameters", json.dumps(arguments))
                        span.set_attribute("input.value", json.dumps(arguments))
                        span.set_attribute("input.mime_type", "application/json")
                    except Exception as e:
                        logger.debug(f"Failed to set MCP tool input attributes: {e}")
                    
                    try:
                        result = func(*args, **kwargs)
                        try:
                            if isinstance(result, dict):
                                span.set_attribute("mcp.response.value", json.dumps(result))
                                span.set_attribute("output.value", json.dumps(result))
                                span.set_attribute("output.mime_type", "application/json")
                            elif isinstance(result, str):
                                output = {"result": result}
                                span.set_attribute("mcp.response.value", json.dumps(output))
                                span.set_attribute("output.value", json.dumps(output))
                                span.set_attribute("output.mime_type", "application/json")
                        except Exception as e:
                            logger.debug(f"Failed to set MCP tool output attributes: {e}")
                        
                        span.set_status(Status(StatusCode.OK))
                        return result
                        
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
            
            return sync_wrapper
    
    return decorator
