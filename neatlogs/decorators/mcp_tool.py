"""
MCP Tool Decorator for Neatlogs SDK

Provides @mcp_tool() decorator to manually instrument MCP tool functions.
"""

import json
import inspect
import functools
from typing import Any, Callable, TypeVar
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

F = TypeVar('F', bound=Callable[..., Any])

def mcp_tool(name: str | None = None) -> Callable[[F], F]:
    """
    Decorator to instrument MCP tool functions with tracing.
    
    Similar to Phoenix's @tracer.tool() but for Neatlogs SDK.
    
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
        
        # Handle both sync and async functions
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    span_name,
                    kind=trace.SpanKind.INTERNAL,
                ) as span:
                    # Set MCP-specific attributes
                    span.set_attribute("mcp.tool.name", tool_name)
                    span.set_attribute("openinference.span.kind", "TOOL")
                    # OpenInference tool naming conventions (preferred over Traceloop entity spans).
                    span.set_attribute("tool.name", tool_name)
                    
                    # Capture arguments
                    try:
                        # For FastMCP, first arg is often a Pydantic model
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
                    except Exception:
                        pass
                    
                    try:
                        # Call original function
                        result = await func(*args, **kwargs)
                        
                        # Capture output
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
                        except Exception:
                            pass
                        
                        span.set_status(Status(StatusCode.OK))
                        return result
                        
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
            
            return async_wrapper  # type: ignore
        
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    span_name,
                    kind=trace.SpanKind.INTERNAL,
                ) as span:
                    # Set MCP-specific attributes
                    span.set_attribute("mcp.tool.name", tool_name)
                    span.set_attribute("openinference.span.kind", "TOOL")
                    span.set_attribute("tool.name", tool_name)
                    
                    # Capture arguments
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
                    except Exception:
                        pass
                    
                    try:
                        # Call original function
                        result = func(*args, **kwargs)
                        
                        # Capture output
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
                        except Exception:
                            pass
                        
                        span.set_status(Status(StatusCode.OK))
                        return result
                        
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
            
            return sync_wrapper  # type: ignore
    
    return decorator
