"""
Context managers for manual span creation.
"""

from contextlib import contextmanager
from typing import Optional, Dict, Any, Union
from opentelemetry import trace as otel_trace


@contextmanager
def trace(
    name: str,
    kind: Optional[str] = None,
    prompt_template: Optional[Union[str, "PromptTemplate"]] = None,
    prompt_variables: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None,
    **attributes
):
    """
    Generic span context manager with optional prompt tracking.

    Creates a span with automatic context propagation.
    Span kind is automatically inferred if not provided.

    **Session-Aware Trace Creation:**
    - If `session_id` is set in init() AND no active parent span exists,
      this creates a NEW root trace (for multi-turn conversations).
    - Otherwise, creates a normal child span within the existing trace.

    Args:
        name: Span name
        kind: OpenInference span kind (LLM, CHAIN, TOOL, etc.)
              If None, will be inferred by span processor
        prompt_template: Optional prompt template - can be:
            - String template (legacy): "Weather in {city}"
            - PromptTemplate object (recommended): PromptTemplate("Weather in {{city}}")
        prompt_variables: Optional dict of prompt variables (for string templates only)
        version: Optional version identifier
        **attributes: Additional attributes to set on the span

    Example (legacy string template):
        ```python
        with neatlogs.trace(
            "weather_query",
            prompt_template="Weather in {city}",
            prompt_variables={"city": "SF"}
        ):
            response = llm.create(...)
        ```

    Example (new PromptTemplate - NO DUPLICATION!):
        ```python
        from neatlogs import trace, PromptTemplate

        template = PromptTemplate("Weather in {{city}}")

        with trace("weather_query", prompt_template=template):
            # Variables specified ONCE in compile() - no duplication!
            prompt_text = template.compile(city="SF")
            response = llm.create(messages=[{"role": "user", "content": prompt_text}])
        ```

    Example (chat session - automatic root trace per turn):
        ```python
        # init() with session_id
        neatlogs.init(api_key="...", auto_session=True)

        # Each trace() at top level creates a NEW root trace
        with trace("turn_1"):  # → New trace (same session_id)
            agent.run(...)

        with trace("turn_2"):  # → New trace (same session_id)
            agent.run(...)
        ```
    """
    import json
    import logging
    from opentelemetry.context import attach, detach, set_value, get_current
    from ..prompt.template import PromptTemplate, PromptContext
    from ..init import get_session_config

    logger = logging.getLogger(__name__)
    tracer = otel_trace.get_tracer(__name__)

    # Auto-detect session-based trace creation
    session_config = get_session_config()
    session_id = session_config.get("session_id")
    user_id = session_config.get("user_id")
    current_span = otel_trace.get_current_span()
    is_in_active_trace = current_span and current_span.is_recording()

    # Determine if we should create a NEW root trace or a child span
    # Logic: If session_id exists AND no active parent → create NEW root trace (for chat turns)
    should_create_root_trace = session_id and not is_in_active_trace

    # Handle PromptTemplate objects vs string templates
    template_string = None
    is_prompt_template_obj = False

    if prompt_template is not None:
        # Check if it's a PromptTemplate object
        if isinstance(prompt_template, PromptTemplate):
            is_prompt_template_obj = True
            template_string = str(prompt_template.template)
            logger.debug(f"[trace] Using PromptTemplate object with variables: {prompt_template.variables}")
        else:
            # Legacy string template
            template_string = prompt_template

    # Set prompt info in context if provided (so child LLM spans can access it)
    ctx = get_current()
    variables_json = json.dumps(prompt_variables, default=str) if prompt_variables else None

    if variables_json:
        ctx = set_value("neatlogs.prompt_variables", variables_json, context=ctx)
        logger.debug(f"[trace] Set neatlogs.prompt_variables in context: {variables_json}")
    if template_string:
        ctx = set_value("neatlogs.prompt_template", template_string, context=ctx)
        logger.debug(f"[trace] Set neatlogs.prompt_template in context: {template_string}")
    if version:
        ctx = set_value("neatlogs.prompt_version", version, context=ctx)
        logger.debug(f"[trace] Set neatlogs.prompt_version in context: {version}")

    # Attach the modified context
    ctx_token = attach(ctx)
    try:
        # Create span with appropriate context
        if should_create_root_trace:
            # Chat mode: Force new root trace by detaching from ambient context
            logger.debug(f"[trace] Creating NEW root trace '{name}' (session_id={session_id})")
            with tracer.start_as_current_span(name, context=None) as span:
                # Note: session.id and user.id are now set globally via Resource attributes in init()
                # No need to set them here - they'll automatically be on ALL spans (including orphans)
                
                _set_span_attributes(span, kind, template_string, prompt_variables, version, attributes)
                yield span
                _finalize_prompt_capture(span, is_prompt_template_obj, logger)
        else:
            # Normal mode: Create child span within existing trace
            # Pass the context explicitly so child spans can read prompt variables
            logger.debug(f"[trace] Creating child span '{name}'")
            with tracer.start_as_current_span(name, context=ctx) as span:
                _set_span_attributes(span, kind, template_string, prompt_variables, version, attributes)
                yield span
                _finalize_prompt_capture(span, is_prompt_template_obj, logger)
    finally:
        # Clear PromptContext to avoid leaking to next trace
        if is_prompt_template_obj:
            PromptContext.clear()
        detach(ctx_token)


def _set_span_attributes(span, kind, template_string, prompt_variables, version, attributes):
    """Helper to set span attributes."""
    import json
    
    # Mark as internal wrapper span (filtered in UI)
    span.set_attribute("neatlogs.internal", True)
    
    # Set span kind (default to CHAIN for user-created spans)
    span_kind = kind if kind else "CHAIN"
    span.set_attribute("openinference.span.kind", span_kind)

    # DON'T set prompt attributes on wrapper span - they'll be propagated to child LLM spans via context
    # The span processor reads from context and applies them to actual LLM spans
    # This avoids duplication and ensures prompt data appears only on the LLM span

    # Set additional attributes
    for key, value in attributes.items():
        span.set_attribute(key, value)


def _finalize_prompt_capture(span, is_prompt_template_obj, logger):
    """Helper to finalize prompt variable capture after yield."""
    import json
    from ..prompt.template import PromptContext
    
    # After yield, check if PromptTemplate.compile() was called and auto-capture variables
    if is_prompt_template_obj:
        captured_vars = PromptContext.get_variables()
        if captured_vars:
            span.set_attribute(
                "llm.prompt_template_variables",
                json.dumps(captured_vars, default=str)
            )
            logger.debug(f"[trace] Auto-captured variables from PromptContext: {list(captured_vars.keys())}")


@contextmanager
def track_prompt(
    template: str,
    variables: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None,
):
    """
    Context manager for prompt tracking.
    
    Automatically captures prompt template and variables for the current span.
    Use this when you want to wrap a block of code that makes LLM calls.
    
    Args:
        template: Prompt template string
        variables: Dictionary of template variables
        version: Optional version identifier
    
    Example:
        ```python
        with neatlogs.track_prompt(
            template="Weather in {city}",
            variables={"city": "SF"}
        ):
            response = openai.create(...)  # Auto-captured
        ```
    """
    from ..prompt.capture import capture_prompt
    
    # Capture the prompt metadata
    capture_prompt(template, variables, version)
    yield
