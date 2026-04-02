"""
LLM template binding for frameworks that manage LLM calls internally (e.g. CrewAI).

Usage:
    bound_llm = neatlogs.bind_templates(llm, system_tpl, user_tpl, content=content)
    agent = Agent(llm=bound_llm, ...)

When crew.kickoff() calls bound_llm.invoke(...), this wrapper fires first:
  1. Sets neatlogs.prompt_template + neatlogs.user_prompt_template in OTel context
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
        system_tpl: neatlogs.PromptTemplate for the agent backstory / system role
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

    # Clone the LLM so different agents can each have their own binding
    try:
        llm_copy = llm.model_copy()   # Pydantic v2
    except AttributeError:
        try:
            llm_copy = llm.copy()     # Pydantic v1
        except Exception:
            llm_copy = llm            # Fallback: mutate in place (last resort)
            logger.warning(
                "[bind_templates] Could not copy LLM — falling back to in-place binding. "
                "Agents sharing this LLM instance may overwrite each other's templates."
            )

    # Capture the already-instrumented class-level invoke BEFORE we set the
    # instance attribute, so our wrapper can delegate to it correctly.
    # crewai-native LLMs (e.g. AzureCompletion) have no .invoke — skip wrapping.
    if not hasattr(type(llm_copy), "invoke"):
        logger.debug(
            "[bind_templates] LLM type %s has no invoke() — returning as-is.",
            type(llm_copy).__name__,
        )
        return llm_copy
    instrumented_class_invoke = type(llm_copy).invoke

    def _invoke_with_templates(*args: Any, **kwargs: Any) -> Any:
        # Push template strings into OTel context so that when the instrumented
        # class invoke starts the LLM span, on_start() picks them up.
        ctx = get_current()
        ctx = set_value("neatlogs.prompt_template", system_str, context=ctx)
        if user_str is not None:
            ctx = set_value("neatlogs.user_prompt_template", user_str, context=ctx)
        token = attach(ctx)
        try:
            system_tpl.compile()
            if user_tpl is not None and compiled_vars:
                user_tpl.compile(**compiled_vars)
            # Call the instrumented class method with our copy as `self`.
            return instrumented_class_invoke(llm_copy, *args, **kwargs)
        finally:
            detach(token)
            PromptContext.clear()
            if user_tpl is not None and compiled_vars:
                UserPromptContext.clear()

    # Instance attribute takes precedence over the class attribute in Python's
    # MRO, so this runs BEFORE the OpenInference class-level patch.
    llm_copy.invoke = _invoke_with_templates
    return llm_copy
