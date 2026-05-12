"""
Context managers for manual span creation.
"""

from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Union

from opentelemetry import trace as otel_trace


@contextmanager
def trace(
    name: str,
    kind: Optional[str] = None,
    system_prompt_template: Optional[Union[str, "SystemPromptTemplate"]] = None,
    system_prompt_variables: Optional[Dict[str, Any]] = None,
    user_prompt_template: Optional[Union[str, "UserPromptTemplate"]] = None,
    user_prompt_variables: Optional[Dict[str, Any]] = None,
    prompt_template: Optional[Union[str, "SystemPromptTemplate"]] = None,
    prompt_variables: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None,
    capture_stdout: bool = False,
    mask: Optional[Callable] = None,
    **attributes,
):
    """
    Context manager for prompt tracking and session management.

    IMPORTANT: This is NOT required for basic tracing!
    - Framework code (LangChain, OpenAI, etc.) is auto-instrumented
    - Custom code should use @span(kind="...") decorator

    Use trace() ONLY for:
    1. Prompt template tracking (captures template + variables for versioning)
    2. Multi-turn session management (groups turns within a session)
    3. Grouping multiple top-level operations in main()

    **When NOT to use trace():**
    - Don't wrap a single @span(kind="WORKFLOW") call - it's already traced!
    - Don't use for custom functions - use @span(kind="...") decorator instead
    - Don't use just to "create a span" - auto-instrumentation handles that

    **Session-Aware Trace Creation:**
    - If `session_id` is set in init() AND no active parent span exists,
      this creates a NEW root trace (for multi-turn conversations).
    - Otherwise, creates a normal child span within the existing trace.

    Args:
        name: Span name
        kind: OpenInference span kind (rarely needed - auto-inferred)
        system_prompt_template: SystemPromptTemplate object for system prompt versioning
        system_prompt_variables: Dict of system prompt variables (legacy string templates only)
        user_prompt_template: UserPromptTemplate object for user/human prompt versioning
        user_prompt_variables: Dict of user prompt variables (legacy string templates only)
        prompt_template: Deprecated alias for ``system_prompt_template``. Kept for
            backward compat; new code should use ``system_prompt_template``.
        prompt_variables: Deprecated alias for ``system_prompt_variables``.
        version: Optional version identifier
        **attributes: Additional attributes to set on the span

    Examples:

        Use Case 1: Prompt template tracking (inside your function):

        >>> template = SystemPromptTemplate([{"role": "user", "content": "{{query}}"}])
        >>>
        >>> @span(kind="AGENT")
        >>> def my_agent(query: str):
        ...     with trace(name="prompt", system_prompt_template=template):
        ...         messages = template.compile(query=query)  # Compile ONCE
        ...         response = llm.create(messages=messages)   # No duplication!
        ...     return response

        Use Case 2: Multi-turn sessions:

        >>> neatlogs.init(api_key="...", session_id="user-123")
        >>>
        >>> with trace(name="turn_1"):  # New root trace (same session)
        ...     agent.run("Hello")
        >>>
        >>> with trace(name="turn_2"):  # New root trace (same session)
        ...     agent.run("Tell me more")

        Use Case 3: Grouping multiple operations in main:

        >>> def main():
        ...     with trace(name="pipeline"):  # Groups multiple steps
        ...         step1()  # @span(kind="CHAIN")
        ...         step2()  # @span(kind="CHAIN")
        ...         step3()  # @span(kind="AGENT")

        What NOT to do:

        >>> @span(kind="WORKFLOW")
        >>> def my_workflow():
        ...     pass
        >>>
        >>> # ❌ WRONG: Redundant wrapper!
        >>> with trace(name="main"):
        ...     my_workflow()  # Already traced by @span decorator
        >>>
        >>> # ✅ CORRECT: Just call it
        >>> my_workflow()
    """
    import json
    import logging

    from opentelemetry.context import attach, detach, get_current, set_value

    from ..init import get_session_config
    from ..prompt.template import (
        PromptContext,
        SystemPromptTemplate,
        UserPromptContext,
        UserPromptTemplate,
    )

    logger = logging.getLogger(__name__)
    tracer = otel_trace.get_tracer(__name__)

    # Resolve canonical kwargs. `system_prompt_template` / `system_prompt_variables`
    # are the canonical names; `prompt_template` / `prompt_variables` are kept as
    # deprecated aliases for backward compat. If both are supplied, the canonical
    # (system_*) name wins.
    if system_prompt_template is not None:
        prompt_template = system_prompt_template
    if system_prompt_variables is not None:
        prompt_variables = system_prompt_variables

    session_config = get_session_config()
    session_id = session_config.get("session_id")
    user_id = session_config.get("user_id")
    current_span = otel_trace.get_current_span()
    is_in_active_trace = current_span and current_span.is_recording()

    should_create_root_trace = session_id and not is_in_active_trace

    template_string = None
    is_prompt_template_obj = False

    if prompt_template is not None:
        if isinstance(prompt_template, SystemPromptTemplate):
            is_prompt_template_obj = True
            template_string = str(prompt_template.template)
            logger.debug(
                f"[trace] Using SystemPromptTemplate object with variables: {prompt_template.variables}"
            )
        else:
            template_string = prompt_template

    user_template_string = None
    is_user_prompt_template_obj = False

    if user_prompt_template is not None:
        if isinstance(user_prompt_template, UserPromptTemplate):
            is_user_prompt_template_obj = True
            user_template_string = str(user_prompt_template.template)
            logger.debug(
                f"[trace] Using UserPromptTemplate object with variables: {user_prompt_template.variables}"
            )
        else:
            user_template_string = user_prompt_template

    ctx = get_current()
    variables_json = json.dumps(prompt_variables, default=str) if prompt_variables else None
    user_variables_json = (
        json.dumps(user_prompt_variables, default=str) if user_prompt_variables else None
    )

    if variables_json:
        ctx = set_value("neatlogs.system_prompt_variables", variables_json, context=ctx)
        logger.debug(
            f"[trace] Set neatlogs.system_prompt_variables in context: {variables_json}"
        )
    if template_string:
        ctx = set_value("neatlogs.system_prompt_template", template_string, context=ctx)
        logger.debug(
            f"[trace] Set neatlogs.system_prompt_template in context: {template_string}"
        )
    if user_variables_json:
        ctx = set_value("neatlogs.user_prompt_variables", user_variables_json, context=ctx)
        logger.debug(
            f"[trace] Set neatlogs.user_prompt_variables in context: {user_variables_json}"
        )
    if user_template_string:
        ctx = set_value("neatlogs.user_prompt_template", user_template_string, context=ctx)
        logger.debug(
            f"[trace] Set neatlogs.user_prompt_template in context: {user_template_string}"
        )
    if version:
        ctx = set_value("neatlogs.prompt_version", version, context=ctx)
        logger.debug(f"[trace] Set neatlogs.prompt_version in context: {version}")

    from .log import _CaptureStdoutContext

    ctx_token = attach(ctx)
    stdout_ctx = _CaptureStdoutContext() if capture_stdout else None
    try:
        if should_create_root_trace:
            logger.debug(f"[trace] Creating NEW root trace '{name}' (session_id={session_id})")
            with tracer.start_as_current_span(name, context=None) as span:
                _set_span_attributes(
                    span, kind, template_string, prompt_variables, version, attributes
                )
                if mask is not None:
                    from .mask import register_mask

                    span.set_attribute("neatlogs.mask_id", register_mask(mask))
                if stdout_ctx:
                    stdout_ctx.__enter__()
                try:
                    yield span
                    _finalize_prompt_capture(
                        span, is_prompt_template_obj, is_user_prompt_template_obj, logger
                    )
                finally:
                    if stdout_ctx:
                        stdout_ctx.__exit__(None, None, None)
        else:
            logger.debug(f"[trace] Creating child span '{name}'")
            with tracer.start_as_current_span(name, context=ctx) as span:
                _set_span_attributes(
                    span, kind, template_string, prompt_variables, version, attributes
                )
                if mask is not None:
                    from .mask import register_mask

                    span.set_attribute("neatlogs.mask_id", register_mask(mask))
                if stdout_ctx:
                    stdout_ctx.__enter__()
                try:
                    yield span
                    _finalize_prompt_capture(
                        span, is_prompt_template_obj, is_user_prompt_template_obj, logger
                    )
                finally:
                    if stdout_ctx:
                        stdout_ctx.__exit__(None, None, None)
    finally:
        if is_prompt_template_obj:
            PromptContext.clear()
        if is_user_prompt_template_obj:
            UserPromptContext.clear()
        detach(ctx_token)


def _set_span_attributes(span, kind, template_string, prompt_variables, version, attributes):
    """Helper to set span attributes."""

    span.set_attribute("neatlogs.internal", True)
    span_kind = kind if kind else "CHAIN"
    span.set_attribute("openinference.span.kind", span_kind)

    for key, value in attributes.items():
        span.set_attribute(key, value)


def _finalize_prompt_capture(span, is_prompt_template_obj, is_user_prompt_template_obj, logger):
    """Helper to finalize prompt variable capture after yield."""
    import json

    from ..prompt.template import PromptContext, UserPromptContext

    if is_prompt_template_obj:
        captured_vars = PromptContext.get_variables()
        if captured_vars:
            span.set_attribute(
                "llm.prompt_template_variables", json.dumps(captured_vars, default=str)
            )
            logger.debug(
                f"[trace] Auto-captured variables from PromptContext: {list(captured_vars.keys())}"
            )

    if is_user_prompt_template_obj:
        captured_user_vars = UserPromptContext.get_variables()
        if captured_user_vars:
            span.set_attribute(
                "llm.user_prompt_template_variables", json.dumps(captured_user_vars, default=str)
            )
            logger.debug(
                f"[trace] Auto-captured variables from UserPromptContext: {list(captured_user_vars.keys())}"
            )
