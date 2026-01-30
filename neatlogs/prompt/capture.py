"""
Explicit prompt capture helpers.

No AST magic - users explicitly provide template and variables.
"""

import json
from typing import Dict, Any, Optional
from opentelemetry import trace
from opentelemetry.context import attach, detach, set_value, get_current


def capture_prompt(
    template: str,
    variables: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None,
) -> None:
    """
    Low-level function to capture prompt template and variables.
    Attaches to current active span and propagates to child LLM spans.
    
    **Recommended:** Use `with trace()` context manager instead for more observish style:
    ```python
    with neatlogs.trace("query", prompt_template="...", prompt_variables={...}):
        response = openai.create(...)
    ```
    
    Args:
        template: The prompt template string (e.g., "Weather in {city} on {date}")
        variables: Dictionary of variable values (e.g., {"city": "SF", "date": "Jan 21"})
        version: Optional version identifier for the prompt template
    
    Example (low-level usage):
        ```python
        city = "San Francisco"
        date = "Jan 21"
        
        capture_prompt("Weather in {city} on {date}", {"city": city, "date": date})
        response = openai.create(...)  # Span will have prompt metadata
        ```
    """
    current_span = trace.get_current_span()
    
    if current_span and current_span.is_recording():
        # Set OpenInference prompt template attribute on current span
        current_span.set_attribute("llm.prompt_template", template)
        
        # Set variables as JSON (OpenInference format)
        if variables:
            current_span.set_attribute(
                "llm.prompt_template_variables",
                json.dumps(variables)
            )
        
        # Set version if provided
        if version:
            current_span.set_attribute("llm.prompt_template.version", version)
        
        # Also propagate via context so child LLM spans can inherit
        # Note: This modifies the context for subsequent operations in the same execution flow
        ctx = get_current()
        ctx = set_value("neatlogs.prompt_template", template, context=ctx)
        if variables:
            ctx = set_value("neatlogs.prompt_variables", json.dumps(variables, default=str), context=ctx)
        if version:
            ctx = set_value("neatlogs.prompt_version", version, context=ctx)
        
        # Attach the modified context (will be detached when current scope ends)
        attach(ctx)


def capture_vars(**kwargs) -> None:
    """
    Low-level function to capture variables from current scope.
    
    Convenience function when you don't have an explicit template.
    
    **Recommended:** Use `@observe()` decorator or `with trace()` instead:
    ```python
    @neatlogs.observe()
    def query(city: str, date: str):  # Auto-captures variables
        return openai.create(...)
    ```
    
    Args:
        **kwargs: Variable names and values to capture
    
    Example (low-level usage):
        ```python
        city = "SF"
        date = "Jan 21"
        
        capture_vars(city=city, date=date)
        response = openai.create(...)
        ```
    """
    current_span = trace.get_current_span()
    
    if current_span and current_span.is_recording():
        current_span.set_attribute(
            "llm.prompt_template_variables",
            json.dumps(kwargs)
        )
