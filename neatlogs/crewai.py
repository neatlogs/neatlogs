"""
Neatlogs CrewAI wrapper.

Usage:
    >>> import neatlogs
    >>> from crewai import Crew, Agent, Task
    >>> crew = neatlogs.wrap(Crew(agents=[...], tasks=[...]))
    >>> result = crew.kickoff()

Creates span hierarchy: WORKFLOW → TASK → AGENT → LLM
"""

import time
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.trace import StatusCode

from ._wrap_utils import get_tracer, serialize


def wrap_crewai(crew: Any) -> Any:
    """
    Wrap a CrewAI Crew instance. Patches kickoff methods and the
    underlying Task/Agent execute methods to produce a full span hierarchy.
    Returns the same crew instance.
    """
    _patch_kickoff(crew)
    _patch_kickoff_async(crew)
    _patch_kickoff_for_each(crew)
    _patch_kickoff_for_each_async(crew)

    # Patch tasks and agents for child spans
    for task in getattr(crew, "tasks", []) or []:
        _patch_task_execute(task)
    for agent in getattr(crew, "agents", []) or []:
        _patch_agent_execute(agent)

    return crew


# ---------------------------------------------------------------------------
# Crew-level (WORKFLOW spans)
# ---------------------------------------------------------------------------


def _get_crew_attributes(crew: Any) -> dict:
    """Extract crew metadata as span attributes."""
    attrs = {"neatlogs.span.kind": "WORKFLOW"}

    name = getattr(crew, "name", None) or getattr(crew, "_name", None)
    if name:
        attrs["neatlogs.workflow.name"] = name

    crew_id = getattr(crew, "id", None)
    if crew_id:
        attrs["neatlogs.crewai.crew_id"] = str(crew_id)

    crew_key = getattr(crew, "key", None)
    if crew_key:
        attrs["neatlogs.crewai.crew_key"] = str(crew_key)

    process = getattr(crew, "process", None)
    if process:
        attrs["neatlogs.crewai.process"] = str(process.value) if hasattr(process, "value") else str(process)

    agents = getattr(crew, "agents", None)
    if agents:
        attrs["neatlogs.crewai.crew_number_of_agents"] = len(agents)

    tasks = getattr(crew, "tasks", None)
    if tasks:
        attrs["neatlogs.crewai.crew_number_of_tasks"] = len(tasks)

    try:
        import crewai
        attrs["neatlogs.crewai.version"] = getattr(crewai, "__version__", "")
    except (ImportError, AttributeError):
        pass

    return attrs


def _extract_token_usage(result: Any) -> dict:
    """Extract token usage from CrewOutput."""
    attrs = {}
    token_usage = getattr(result, "token_usage", None)
    if not token_usage:
        return attrs

    if isinstance(token_usage, dict):
        usage = token_usage
    else:
        usage = token_usage.__dict__ if hasattr(token_usage, "__dict__") else {}

    if usage.get("prompt_tokens"):
        attrs["neatlogs.llm.token_count.prompt"] = usage["prompt_tokens"]
    if usage.get("completion_tokens"):
        attrs["neatlogs.llm.token_count.completion"] = usage["completion_tokens"]
    if usage.get("total_tokens"):
        attrs["neatlogs.llm.token_count.total"] = usage["total_tokens"]
    if usage.get("cached_tokens"):
        attrs["neatlogs.llm.token_count.cache_read"] = usage["cached_tokens"]

    return attrs


def _finalize_crew_span(span: Any, result: Any, duration_ms: float) -> None:
    """Finalize a crew execution span with result data."""
    if result is None:
        span.set_status(StatusCode.OK)
        span.end()
        return

    raw = getattr(result, "raw", None)
    if raw:
        span.set_attribute("output.value", str(raw)[:10000])

    for attr_name, value in _extract_token_usage(result).items():
        span.set_attribute(attr_name, value)

    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()


def _patch_kickoff(crew: Any) -> None:
    if getattr(crew, "_neatlogs_kickoff_patched", False):
        return

    orig_kickoff = crew.kickoff

    def patched_kickoff(*args, **kwargs):
        tracer = get_tracer()
        attrs = _get_crew_attributes(crew)
        if kwargs.get("inputs"):
            attrs["input.value"] = serialize(kwargs["inputs"])

        span = tracer.start_span(name="crewai.crew.kickoff", attributes=attrs)
        ctx = otel_context.set_value("current_span", span)
        token = otel_context.attach(ctx)
        start = time.perf_counter()

        try:
            result = orig_kickoff(*args, **kwargs)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise
        finally:
            otel_context.detach(token)

        duration_ms = (time.perf_counter() - start) * 1000
        _finalize_crew_span(span, result, duration_ms)
        return result

    crew.kickoff = patched_kickoff
    crew._neatlogs_kickoff_patched = True


def _patch_kickoff_async(crew: Any) -> None:
    if not hasattr(crew, "kickoff_async"):
        return
    if getattr(crew, "_neatlogs_kickoff_async_patched", False):
        return

    orig_kickoff_async = crew.kickoff_async

    async def patched_kickoff_async(*args, **kwargs):
        tracer = get_tracer()
        attrs = _get_crew_attributes(crew)
        if kwargs.get("inputs"):
            attrs["input.value"] = serialize(kwargs["inputs"])

        span = tracer.start_span(name="crewai.crew.kickoff_async", attributes=attrs)
        ctx = otel_context.set_value("current_span", span)
        token = otel_context.attach(ctx)
        start = time.perf_counter()

        try:
            result = await orig_kickoff_async(*args, **kwargs)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise
        finally:
            otel_context.detach(token)

        duration_ms = (time.perf_counter() - start) * 1000
        _finalize_crew_span(span, result, duration_ms)
        return result

    crew.kickoff_async = patched_kickoff_async
    crew._neatlogs_kickoff_async_patched = True


def _patch_kickoff_for_each(crew: Any) -> None:
    if not hasattr(crew, "kickoff_for_each"):
        return
    if getattr(crew, "_neatlogs_kickoff_for_each_patched", False):
        return

    orig = crew.kickoff_for_each

    def patched_kickoff_for_each(*args, **kwargs):
        tracer = get_tracer()
        inputs = kwargs.get("inputs") or (args[0] if args else None)
        attrs = _get_crew_attributes(crew)
        if inputs and hasattr(inputs, "__len__"):
            attrs["neatlogs.workflow.batch_size"] = len(inputs)

        span = tracer.start_span(name="crewai.crew.kickoff_for_each", attributes=attrs)
        ctx = otel_context.set_value("current_span", span)
        token = otel_context.attach(ctx)
        start = time.perf_counter()

        try:
            results = orig(*args, **kwargs)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise
        finally:
            otel_context.detach(token)

        duration_ms = (time.perf_counter() - start) * 1000
        span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
        span.set_status(StatusCode.OK)
        span.end()
        return results

    crew.kickoff_for_each = patched_kickoff_for_each
    crew._neatlogs_kickoff_for_each_patched = True


def _patch_kickoff_for_each_async(crew: Any) -> None:
    if not hasattr(crew, "kickoff_for_each_async"):
        return
    if getattr(crew, "_neatlogs_kickoff_for_each_async_patched", False):
        return

    orig = crew.kickoff_for_each_async

    async def patched_kickoff_for_each_async(*args, **kwargs):
        tracer = get_tracer()
        inputs = kwargs.get("inputs") or (args[0] if args else None)
        attrs = _get_crew_attributes(crew)
        if inputs and hasattr(inputs, "__len__"):
            attrs["neatlogs.workflow.batch_size"] = len(inputs)

        span = tracer.start_span(name="crewai.crew.kickoff_for_each_async", attributes=attrs)
        ctx = otel_context.set_value("current_span", span)
        token = otel_context.attach(ctx)
        start = time.perf_counter()

        try:
            results = await orig(*args, **kwargs)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise
        finally:
            otel_context.detach(token)

        duration_ms = (time.perf_counter() - start) * 1000
        span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
        span.set_status(StatusCode.OK)
        span.end()
        return results

    crew.kickoff_for_each_async = patched_kickoff_for_each_async
    crew._neatlogs_kickoff_for_each_async_patched = True


# ---------------------------------------------------------------------------
# Task-level (TASK spans)
# ---------------------------------------------------------------------------


def _patch_task_execute(task: Any) -> None:
    """Patch Task._execute_core or execute_sync to create TASK child spans."""
    if getattr(task, "_neatlogs_task_patched", False):
        return

    # CrewAI >=0.80 uses _execute_core; older versions use execute_sync
    method_name = "_execute_core" if hasattr(task, "_execute_core") else "execute_sync"
    if not hasattr(task, method_name):
        method_name = "execute"
    if not hasattr(task, method_name):
        return

    orig = getattr(task, method_name)

    def patched_execute(*args, **kwargs):
        tracer = get_tracer()
        attrs = {"neatlogs.span.kind": "TASK"}

        task_id = getattr(task, "id", None)
        if task_id:
            attrs["neatlogs.task.id"] = str(task_id)

        task_key = getattr(task, "key", None)
        if task_key:
            attrs["neatlogs.task.key"] = str(task_key)

        description = getattr(task, "description", "")
        if description:
            attrs["input.value"] = str(description)[:10000]

        agent = getattr(task, "agent", None)
        if agent:
            role = getattr(agent, "role", "")
            if role:
                attrs["neatlogs.agent.role"] = role

        span_name = f"crewai.task"
        span = tracer.start_span(name=span_name, attributes=attrs)
        ctx = otel_context.set_value("current_span", span)
        token = otel_context.attach(ctx)
        start = time.perf_counter()

        try:
            result = orig(*args, **kwargs)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise
        finally:
            otel_context.detach(token)

        duration_ms = (time.perf_counter() - start) * 1000

        if result is not None:
            raw = getattr(result, "raw", None) if hasattr(result, "raw") else str(result)
            if raw:
                span.set_attribute("output.value", str(raw)[:10000])

        span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
        span.set_status(StatusCode.OK)
        span.end()
        return result

    setattr(task, method_name, patched_execute)
    task._neatlogs_task_patched = True


# ---------------------------------------------------------------------------
# Agent-level (AGENT spans)
# ---------------------------------------------------------------------------


def _patch_agent_execute(agent: Any) -> None:
    """Patch Agent.execute_task to create AGENT child spans."""
    if getattr(agent, "_neatlogs_agent_patched", False):
        return
    if not hasattr(agent, "execute_task"):
        return

    orig = agent.execute_task

    def patched_execute_task(*args, **kwargs):
        tracer = get_tracer()
        attrs = {"neatlogs.span.kind": "AGENT"}

        role = getattr(agent, "role", "")
        if role:
            attrs["neatlogs.agent.role"] = role

        agent_name = getattr(agent, "name", None)
        if agent_name:
            attrs["neatlogs.agent.name"] = agent_name

        # Capture tools available to this agent
        tools = getattr(agent, "tools", None)
        if tools:
            for i, tool in enumerate(tools):
                tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", "")
                if tool_name:
                    attrs[f"neatlogs.llm.tools.{i}.name"] = str(tool_name)
                tool_desc = getattr(tool, "description", None)
                if tool_desc:
                    attrs[f"neatlogs.llm.tools.{i}.description"] = str(tool_desc)[:500]

        span_name = f"crewai.agent.{role}" if role else "crewai.agent"
        span = tracer.start_span(name=span_name, attributes=attrs)
        ctx = otel_context.set_value("current_span", span)
        token = otel_context.attach(ctx)
        start = time.perf_counter()

        try:
            result = orig(*args, **kwargs)
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

    agent.execute_task = patched_execute_task
    agent._neatlogs_agent_patched = True
