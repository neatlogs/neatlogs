"""
Decorator for automatic prompt variable capture.
"""

import functools
import json
import inspect
from typing import Callable, Optional, Any
from opentelemetry import trace as otel_trace
from opentelemetry.context import attach, detach, set_value, get_current


def observe(name: Optional[str] = None, version: Optional[str] = None, as_type: str = "chain"):
    """
    Decorator that auto-captures function arguments as prompt variables.
    
    Variables are propagated via OpenTelemetry baggage to child LLM spans,
    ensuring they appear on the actual LLM span (openinference.span.kind="LLM")
    for consistent querying in the backend.
    
    Args:
        name: Optional span name (defaults to function name)
        version: Optional version identifier for the prompt template
        as_type: Span kind for the wrapper span ("chain", "tool", "agent")
                 Default "chain" for logical grouping. LLM spans are created by instrumentations.
    
    Example:
        ```python
        @neatlogs.observe()
        def weather_lookup(city: str, date: str):
            prompt = f"Weather in {city} on {date}"
            return openai.create(messages=[{"role": "user", "content": prompt}])
        
        # Creates hierarchy:
        # weather_lookup (CHAIN) - wrapper for logical grouping
        #   └─ openai.chat (LLM) - llm.prompt_template_variables: {"city": "SF", "date": "Jan 21"}
        ```
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = otel_trace.get_tracer(__name__)
            span_name = name or func.__name__
            
            # Capture function arguments
            variables_json = None
            template = None
            try:
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                variables = dict(bound.arguments)
                variables_json = json.dumps(variables, default=str)
                
                # Try to extract template from docstring (optional)
                if func.__doc__:
                    # First line of docstring could be the template
                    template = func.__doc__.strip().split('\n')[0]
            except Exception:
                pass
            
            # Set prompt metadata in context so child LLM spans can access it
            ctx = get_current()
            if variables_json:
                ctx = set_value("neatlogs.prompt_variables", variables_json, context=ctx)
            if template:
                ctx = set_value("neatlogs.prompt_template", template, context=ctx)
            if version:
                ctx = set_value("neatlogs.prompt_version", version, context=ctx)
            
            # Attach the modified context
            token = attach(ctx)
            try:
                # Start wrapper span
                with tracer.start_as_current_span(span_name) as span:
                    # Set span kind for wrapper
                    span_kind_map = {
                        "chain": "CHAIN",
                        "tool": "TOOL",
                        "agent": "AGENT",
                    }
                    span.set_attribute("openinference.span.kind", span_kind_map.get(as_type, "CHAIN"))
                    
                    # Execute function (child LLM spans will inherit these attributes via processor)
                    result = func(*args, **kwargs)
                    
                    return result
            finally:
                detach(token)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = otel_trace.get_tracer(__name__)
            span_name = name or func.__name__
            
            # Capture function arguments
            variables_json = None
            template = None
            try:
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                variables = dict(bound.arguments)
                variables_json = json.dumps(variables, default=str)
                
                # Try to extract template from docstring (optional)
                if func.__doc__:
                    template = func.__doc__.strip().split('\n')[0]
            except Exception:
                pass
            
            # Set prompt metadata in context so child LLM spans can access it
            ctx = get_current()
            if variables_json:
                ctx = set_value("neatlogs.prompt_variables", variables_json, context=ctx)
            if template:
                ctx = set_value("neatlogs.prompt_template", template, context=ctx)
            if version:
                ctx = set_value("neatlogs.prompt_version", version, context=ctx)
            
            # Attach the modified context
            token = attach(ctx)
            try:
                # Start wrapper span
                with tracer.start_as_current_span(span_name) as span:
                    # Set span kind for wrapper
                    span_kind_map = {
                        "chain": "CHAIN",
                        "tool": "TOOL",
                        "agent": "AGENT",
                    }
                    span.set_attribute("openinference.span.kind", span_kind_map.get(as_type, "CHAIN"))
                    
                    # Execute async function (child LLM spans will inherit these attributes via processor)
                    result = await func(*args, **kwargs)
                    
                    return result
            finally:
                detach(token)
        
        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
