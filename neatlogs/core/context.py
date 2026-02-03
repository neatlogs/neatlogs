"""
Context managers for manual span creation.
"""

from contextlib import contextmanager
from typing import Any, Dict, Optional, Union

from opentelemetry import trace as otel_trace


@contextmanager
def trace(
    name: str,
    kind: Optional[str] = None,
    prompt_template: Optional[Union[str, "PromptTemplate"]] = None,
    prompt_variables: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None,
    **attributes,
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

    from opentelemetry.context import attach, detach, get_current, set_value

    from ..init import get_session_config
    from ..prompt.template import PromptContext, PromptTemplate

    logger = logging.getLogger(__name__)
    tracer = otel_trace.get_tracer(__name__)

    session_config = get_session_config()
    session_id = session_config.get("session_id")
    user_id = session_config.get("user_id")
    current_span = otel_trace.get_current_span()
    is_in_active_trace = current_span and current_span.is_recording()

    should_create_root_trace = session_id and not is_in_active_trace

    template_string = None
    is_prompt_template_obj = False

    if prompt_template is not None:
        if isinstance(prompt_template, PromptTemplate):
            is_prompt_template_obj = True
            template_string = str(prompt_template.template)
            logger.debug(
                f"[trace] Using PromptTemplate object with variables: {prompt_template.variables}"
            )
        else:
            template_string = prompt_template

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

    ctx_token = attach(ctx)
    try:
        if should_create_root_trace:
            logger.debug(f"[trace] Creating NEW root trace '{name}' (session_id={session_id})")
            with tracer.start_as_current_span(name, context=None) as span:

                _set_span_attributes(
                    span, kind, template_string, prompt_variables, version, attributes
                )
                yield span
                _finalize_prompt_capture(span, is_prompt_template_obj, logger)
        else:
            logger.debug(f"[trace] Creating child span '{name}'")
            with tracer.start_as_current_span(name, context=ctx) as span:
                _set_span_attributes(
                    span, kind, template_string, prompt_variables, version, attributes
                )
                yield span
                _finalize_prompt_capture(span, is_prompt_template_obj, logger)
    finally:
        if is_prompt_template_obj:
            PromptContext.clear()
        detach(ctx_token)


def _set_span_attributes(span, kind, template_string, prompt_variables, version, attributes):
    """Helper to set span attributes."""

    span.set_attribute("neatlogs.internal", True)
    span_kind = kind if kind else "CHAIN"
    span.set_attribute("openinference.span.kind", span_kind)

    for key, value in attributes.items():
        span.set_attribute(key, value)


def _finalize_prompt_capture(span, is_prompt_template_obj, logger):
    """Helper to finalize prompt variable capture after yield."""
    import json

    from ..prompt.template import PromptContext

    if is_prompt_template_obj:
        captured_vars = PromptContext.get_variables()
        if captured_vars:
            span.set_attribute(
                "llm.prompt_template_variables", json.dumps(captured_vars, default=str)
            )
            logger.debug(
                f"[trace] Auto-captured variables from PromptContext: {list(captured_vars.keys())}"
            )


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

    capture_prompt(template, variables, version)
    yield
