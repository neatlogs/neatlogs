"""
LLM template binding for frameworks that manage LLM calls internally (e.g. CrewAI).

Usage:
    bound_llm = neatlogs.bind_templates(llm, system_tpl, user_tpl, content=content)
    agent = Agent(llm=bound_llm, ...)

When crew.kickoff() calls bound_llm.invoke(...), this wrapper fires first:
  1. Sets neatlogs.system_prompt_template + neatlogs.user_prompt_template in OTel context
  2. Calls the instrumented class-level invoke (creates the LLM span)
  3. span_processor.on_start() reads context → templates land on the LLM span
"""

from __future__ import annotations

from typing import Any, Optional


def bind_templates(
    llm: Any,
    system_tpl: Any,
    user_tpl: Optional[Any] = None,
    **compiled_vars: Any,
) -> Any:
    """
    Return a copy of ``llm`` whose ``invoke()`` injects prompt template context
    before the instrumented LLM span is created.

    Because the instance-level ``invoke`` runs before the class-level
    (OpenInference-instrumented) ``invoke``, span_processor.on_start() sees
    the OTel context values and writes them onto the LLM span.

    Args:
        llm: Any LangChain-compatible chat model (ChatOpenAI, AzureChatOpenAI, …)
        system_tpl: neatlogs.SystemPromptTemplate for the agent backstory / system role
        user_tpl: Optional neatlogs.UserPromptTemplate for the task description /
                  user turn. In CrewAI, prefer register_crewai_task() instead so
                  the template lands on the CREWAI_TASK span rather than LLM spans.
        **compiled_vars: Variable values to pass to user_tpl.compile() so that
                         per-invocation variable captures work correctly.

    Returns:
        A new LLM instance (shallow copy) with template context pre-wired.
        The original ``llm`` is not mutated.
    """
    import logging

    from opentelemetry.context import attach, detach, get_current, set_value

    from ..prompt.template import PromptContext, UserPromptContext

    logger = logging.getLogger(__name__)

    system_str = str(system_tpl.template)
    user_str = str(user_tpl.template) if user_tpl is not None else None

    # Clone the LLM so different agents can each have their own binding.
    # Safe to bind in-place when each agent constructs its own LLM instance.
    import copy as _copy

    llm_copy = None
    for _attempt in (
        lambda: llm.model_copy(),  # Pydantic v2
        lambda: llm.copy(),  # Pydantic v1
        lambda: _copy.copy(llm),  # plain Python classes (e.g. crewai.LLM)
    ):
        try:
            llm_copy = _attempt()
            break
        except Exception:
            continue
    if llm_copy is None:
        llm_copy = llm
        logger.debug(
            "[bind_templates] LLM type %s is not copyable — binding in place.",
            type(llm).__name__,
        )

    # Capture the already-instrumented class-level method BEFORE we set the
    # instance attribute, so our wrapper can delegate to it correctly.
    # Prefer .invoke() (LangChain models); fall back to .call() (crewai.LLM).
    if hasattr(type(llm_copy), "invoke"):
        _method_name = "invoke"
        _class_method = type(llm_copy).invoke
    elif hasattr(type(llm_copy), "call"):
        _method_name = "call"
        _class_method = type(llm_copy).call
    else:
        logger.warning(
            "[bind_templates] LLM type %s has neither invoke() nor call() — "
            "prompt templates will not be captured on spans.",
            type(llm_copy).__name__,
        )
        return llm_copy

    def _wrapped_with_templates(*args: Any, **kwargs: Any) -> Any:
        # Push template strings into OTel context so that when the instrumented
        # method starts the LLM span, on_start() picks them up.
        ctx = get_current()
        ctx = set_value("neatlogs.system_prompt_template", system_str, context=ctx)
        if user_str is not None:
            ctx = set_value("neatlogs.user_prompt_template", user_str, context=ctx)
        token = attach(ctx)
        try:
            system_tpl.compile()
            if user_tpl is not None and compiled_vars:
                user_tpl.compile(**compiled_vars)
            # Call the instrumented class method with our copy as `self`.
            return _class_method(llm_copy, *args, **kwargs)
        finally:
            detach(token)
            PromptContext.clear()
            if user_tpl is not None and compiled_vars:
                UserPromptContext.clear()

    # Instance attribute takes precedence over the class attribute in Python's
    # MRO, so this runs BEFORE the OpenInference class-level patch.
    # Pydantic v2 strict models reject direct attribute assignment — bypass via
    # object.__setattr__ which works for any Python object.
    try:
        setattr(llm_copy, _method_name, _wrapped_with_templates)
    except (ValueError, TypeError):
        object.__setattr__(llm_copy, _method_name, _wrapped_with_templates)
    logger.debug(
        "[bind_templates] Wrapped %s.%s() with template injection.",
        type(llm_copy).__name__,
        _method_name,
    )
    return llm_copy
