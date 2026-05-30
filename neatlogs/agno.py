"""
Neatlogs Agno wrapper.

Usage:
    >>> import neatlogs
    >>> from agno.agent import Agent
    >>> agent = neatlogs.wrap(Agent(model=OpenAIChat(id="gpt-4o"), tools=[...]))
    >>> result = agent.run("Hello")
"""

import time
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.trace import StatusCode

from ._wrap_utils import get_tracer, serialize


def wrap_agno(entity: Any) -> Any:
    """
    Wrap an Agno Agent, Team, or Workflow instance.
    Patches run() and arun() methods. Returns the same instance.
    """
    if getattr(entity, "_neatlogs_patched", False):
        return entity

    cls_name = type(entity).__name__
    if cls_name == "Agent" or hasattr(entity, "run"):
        _patch_agent(entity)
    elif cls_name == "Team":
        _patch_agent(entity)
    elif cls_name == "Workflow":
        _patch_workflow(entity)

    return entity


def _get_agent_attributes(agent: Any) -> dict:
    """Extract Agno agent metadata."""
    attrs = {"neatlogs.span.kind": "AGENT"}

    name = getattr(agent, "name", None)
    if name:
        attrs["neatlogs.agent.name"] = name

    role = getattr(agent, "role", None)
    if role:
        attrs["neatlogs.agent.role"] = role

    model = getattr(agent, "model", None)
    if model:
        model_id = getattr(model, "id", None) or getattr(model, "model", None) or str(model)
        attrs["neatlogs.llm.model_name"] = str(model_id)

    agent_id = getattr(agent, "agent_id", None) or getattr(agent, "id", None)
    if agent_id:
        attrs["neatlogs.agent.id"] = str(agent_id)

    tools = getattr(agent, "tools", None)
    if tools:
        for i, tool in enumerate(tools):
            tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
            if tool_name:
                attrs[f"neatlogs.llm.tools.{i}.name"] = str(tool_name)
            tool_desc = getattr(tool, "description", None)
            if tool_desc:
                attrs[f"neatlogs.llm.tools.{i}.description"] = str(tool_desc)[:500]

    return attrs


def _extract_usage(response: Any) -> dict:
    """Extract token usage from Agno RunResponse."""
    attrs = {}

    metrics = getattr(response, "metrics", None)
    if metrics and isinstance(metrics, dict):
        input_tokens = metrics.get("input_tokens") or metrics.get("prompt_tokens")
        output_tokens = metrics.get("output_tokens") or metrics.get("completion_tokens")
        if input_tokens:
            attrs["neatlogs.llm.token_count.prompt"] = sum(input_tokens) if isinstance(input_tokens, list) else input_tokens
        if output_tokens:
            attrs["neatlogs.llm.token_count.completion"] = sum(output_tokens) if isinstance(output_tokens, list) else output_tokens
        total = metrics.get("total_tokens")
        if total:
            attrs["neatlogs.llm.token_count.total"] = sum(total) if isinstance(total, list) else total

    return attrs


def _finalize_agent_span(span: Any, response: Any, duration_ms: float) -> None:
    """Finalize an Agno agent run span."""
    if response is None:
        span.set_status(StatusCode.OK)
        span.end()
        return

    content = getattr(response, "content", None)
    if content:
        span.set_attribute("output.value", str(content)[:10000])

    for attr_name, value in _extract_usage(response).items():
        span.set_attribute(attr_name, value)

    # Tool calls from response
    tools_used = getattr(response, "tools", None)
    if tools_used:
        for i, tool in enumerate(tools_used):
            name = getattr(tool, "name", None) or getattr(tool, "function_name", None)
            if name:
                span.set_attribute(f"neatlogs.llm.tool_calls.{i}.name", name)
            args = getattr(tool, "arguments", None) or getattr(tool, "function_arguments", None)
            if args:
                span.set_attribute(f"neatlogs.llm.tool_calls.{i}.arguments", serialize(args) if not isinstance(args, str) else args)

    model = getattr(response, "model", None)
    if model:
        span.set_attribute("neatlogs.llm.model_name", str(model))

    run_id = getattr(response, "run_id", None)
    if run_id:
        span.set_attribute("neatlogs.agent.run_id", str(run_id))

    session_id = getattr(response, "session_id", None)
    if session_id:
        span.set_attribute("neatlogs.session.id", str(session_id))

    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()


def _patch_agent(agent: Any) -> None:
    """Patch run() and arun() on an Agno Agent or Team."""
    if hasattr(agent, "run"):
        orig_run = agent.run

        def patched_run(*args, **kwargs):
            tracer = get_tracer()
            attrs = _get_agent_attributes(agent)

            message = args[0] if args else kwargs.get("message", kwargs.get("prompt", ""))
            if message:
                attrs["input.value"] = str(message)[:10000]

            span = tracer.start_span(
                name=f"agno.{type(agent).__name__.lower()}.run",
                attributes=attrs,
            )
            ctx = otel_context.set_value("current_span", span)
            token = otel_context.attach(ctx)
            start = time.perf_counter()

            try:
                result = orig_run(*args, **kwargs)
            except Exception as e:
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)
                span.end()
                raise
            finally:
                otel_context.detach(token)

            duration_ms = (time.perf_counter() - start) * 1000
            _finalize_agent_span(span, result, duration_ms)
            return result

        agent.run = patched_run

    if hasattr(agent, "arun"):
        orig_arun = agent.arun

        async def patched_arun(*args, **kwargs):
            tracer = get_tracer()
            attrs = _get_agent_attributes(agent)

            message = args[0] if args else kwargs.get("message", kwargs.get("prompt", ""))
            if message:
                attrs["input.value"] = str(message)[:10000]

            span = tracer.start_span(
                name=f"agno.{type(agent).__name__.lower()}.arun",
                attributes=attrs,
            )
            ctx = otel_context.set_value("current_span", span)
            token = otel_context.attach(ctx)
            start = time.perf_counter()

            try:
                result = await orig_arun(*args, **kwargs)
            except Exception as e:
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)
                span.end()
                raise
            finally:
                otel_context.detach(token)

            duration_ms = (time.perf_counter() - start) * 1000
            _finalize_agent_span(span, result, duration_ms)
            return result

        agent.arun = patched_arun

    agent._neatlogs_patched = True


def _patch_workflow(workflow: Any) -> None:
    """Patch run() on an Agno Workflow."""
    if not hasattr(workflow, "run"):
        return

    orig_run = workflow.run

    def patched_run(*args, **kwargs):
        tracer = get_tracer()
        attrs = {
            "neatlogs.span.kind": "WORKFLOW",
        }
        name = getattr(workflow, "name", None)
        if name:
            attrs["neatlogs.workflow.name"] = name

        if kwargs:
            attrs["input.value"] = serialize(kwargs)

        span = tracer.start_span(
            name="agno.workflow.run",
            attributes=attrs,
        )
        ctx = otel_context.set_value("current_span", span)
        token = otel_context.attach(ctx)
        start = time.perf_counter()

        try:
            result = orig_run(*args, **kwargs)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise
        finally:
            otel_context.detach(token)

        duration_ms = (time.perf_counter() - start) * 1000
        if result is not None:
            span.set_attribute("output.value", str(result)[:10000])
        span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
        span.set_status(StatusCode.OK)
        span.end()
        return result

    workflow.run = patched_run
    workflow._neatlogs_patched = True
