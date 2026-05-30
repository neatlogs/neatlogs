"""
Neatlogs Google ADK wrapper.

Usage:
    >>> import neatlogs
    >>> from google.adk.runners import Runner
    >>> runner = neatlogs.wrap(Runner(agent=my_agent, app_name="my_app"))
    >>> # runner.run() / runner.run_async() are now traced
"""

import time
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.trace import StatusCode

from ._wrap_utils import get_tracer, serialize


def wrap_google_adk(runner: Any) -> Any:
    """
    Wrap a Google ADK Runner instance. Patches run() and run_async().
    Returns the same runner instance.
    """
    if getattr(runner, "_neatlogs_patched", False):
        return runner

    _patch_run(runner)
    _patch_run_async(runner)
    return runner


def _get_runner_attributes(runner: Any) -> dict:
    """Extract runner metadata as span attributes."""
    attrs = {"neatlogs.span.kind": "WORKFLOW"}

    app_name = getattr(runner, "app_name", None)
    if app_name:
        attrs["neatlogs.workflow.name"] = app_name

    agent = getattr(runner, "agent", None)
    if agent:
        agent_name = getattr(agent, "name", None)
        if agent_name:
            attrs["neatlogs.agent.name"] = agent_name
        model = getattr(agent, "model", None)
        if model:
            attrs["neatlogs.llm.model_name"] = str(model)

    return attrs


def _collect_events(events) -> tuple:
    """Consume a generator/iterator of events, collecting them and extracting attributes."""
    collected = []
    total_input_tokens = 0
    total_output_tokens = 0
    last_content = None
    tool_calls = []
    author = None

    for event in events:
        collected.append(event)

        content = getattr(event, "content", None)
        if content:
            last_content = content

        event_author = getattr(event, "author", None)
        if event_author:
            author = event_author

        # Token usage from event actions
        actions = getattr(event, "actions", None)
        if actions:
            for action in (actions if isinstance(actions, list) else [actions]):
                usage = getattr(action, "usage_metadata", None) or getattr(action, "usage", None)
                if usage:
                    input_t = getattr(usage, "prompt_token_count", None) or getattr(usage, "input_tokens", 0)
                    output_t = getattr(usage, "candidates_token_count", None) or getattr(usage, "output_tokens", 0)
                    if input_t:
                        total_input_tokens += input_t
                    if output_t:
                        total_output_tokens += output_t

        # Tool use from function calls
        parts = None
        if content and hasattr(content, "parts"):
            parts = content.parts
        elif hasattr(event, "parts"):
            parts = event.parts

        if parts:
            for part in parts:
                fn_call = getattr(part, "function_call", None)
                if fn_call:
                    tool_calls.append({
                        "name": getattr(fn_call, "name", ""),
                        "arguments": serialize(getattr(fn_call, "args", {})),
                    })

    attrs = {}
    if total_input_tokens:
        attrs["neatlogs.llm.token_count.prompt"] = total_input_tokens
    if total_output_tokens:
        attrs["neatlogs.llm.token_count.completion"] = total_output_tokens
    if total_input_tokens and total_output_tokens:
        attrs["neatlogs.llm.token_count.total"] = total_input_tokens + total_output_tokens

    if last_content:
        text_parts = []
        if hasattr(last_content, "parts"):
            for part in last_content.parts:
                text = getattr(part, "text", None)
                if text:
                    text_parts.append(text)
        if text_parts:
            attrs["output.value"] = "\n".join(text_parts)[:10000]

    if author:
        attrs["neatlogs.agent.name"] = str(author)

    for i, tc in enumerate(tool_calls):
        attrs[f"neatlogs.llm.tool_calls.{i}.name"] = tc["name"]
        attrs[f"neatlogs.llm.tool_calls.{i}.arguments"] = tc["arguments"]

    return collected, attrs


async def _collect_events_async(events) -> tuple:
    """Async version of _collect_events."""
    collected = []
    total_input_tokens = 0
    total_output_tokens = 0
    last_content = None
    tool_calls = []
    author = None

    async for event in events:
        collected.append(event)

        content = getattr(event, "content", None)
        if content:
            last_content = content

        event_author = getattr(event, "author", None)
        if event_author:
            author = event_author

        actions = getattr(event, "actions", None)
        if actions:
            for action in (actions if isinstance(actions, list) else [actions]):
                usage = getattr(action, "usage_metadata", None) or getattr(action, "usage", None)
                if usage:
                    input_t = getattr(usage, "prompt_token_count", None) or getattr(usage, "input_tokens", 0)
                    output_t = getattr(usage, "candidates_token_count", None) or getattr(usage, "output_tokens", 0)
                    if input_t:
                        total_input_tokens += input_t
                    if output_t:
                        total_output_tokens += output_t

        parts = None
        if content and hasattr(content, "parts"):
            parts = content.parts
        elif hasattr(event, "parts"):
            parts = event.parts

        if parts:
            for part in parts:
                fn_call = getattr(part, "function_call", None)
                if fn_call:
                    tool_calls.append({
                        "name": getattr(fn_call, "name", ""),
                        "arguments": serialize(getattr(fn_call, "args", {})),
                    })

    attrs = {}
    if total_input_tokens:
        attrs["neatlogs.llm.token_count.prompt"] = total_input_tokens
    if total_output_tokens:
        attrs["neatlogs.llm.token_count.completion"] = total_output_tokens
    if total_input_tokens and total_output_tokens:
        attrs["neatlogs.llm.token_count.total"] = total_input_tokens + total_output_tokens

    if last_content:
        text_parts = []
        if hasattr(last_content, "parts"):
            for part in last_content.parts:
                text = getattr(part, "text", None)
                if text:
                    text_parts.append(text)
        if text_parts:
            attrs["output.value"] = "\n".join(text_parts)[:10000]

    if author:
        attrs["neatlogs.agent.name"] = str(author)

    for i, tc in enumerate(tool_calls):
        attrs[f"neatlogs.llm.tool_calls.{i}.name"] = tc["name"]
        attrs[f"neatlogs.llm.tool_calls.{i}.arguments"] = tc["arguments"]

    return collected, attrs


def _patch_run(runner: Any) -> None:
    """Patch Runner.run() (synchronous generator)."""
    if not hasattr(runner, "run"):
        return

    orig_run = runner.run

    def patched_run(*args, **kwargs):
        tracer = get_tracer()
        attrs = _get_runner_attributes(runner)

        user_id = kwargs.get("user_id")
        if user_id:
            attrs["neatlogs.user.id"] = str(user_id)
        session_id = kwargs.get("session_id")
        if session_id:
            attrs["neatlogs.session.id"] = str(session_id)

        new_message = kwargs.get("new_message")
        if new_message:
            if hasattr(new_message, "parts"):
                text_parts = [getattr(p, "text", "") for p in new_message.parts if getattr(p, "text", None)]
                if text_parts:
                    attrs["input.value"] = "\n".join(text_parts)[:10000]
            else:
                attrs["input.value"] = str(new_message)[:10000]

        span = tracer.start_span(name="google_adk.runner.run", attributes=attrs)
        start = time.perf_counter()

        try:
            events = orig_run(*args, **kwargs)
            collected, event_attrs = _collect_events(events)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        for attr_name, value in event_attrs.items():
            span.set_attribute(attr_name, value)
        span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
        span.set_status(StatusCode.OK)
        span.end()

        return iter(collected)

    runner.run = patched_run
    runner._neatlogs_patched = True


def _patch_run_async(runner: Any) -> None:
    """Patch Runner.run_async() (async generator)."""
    if not hasattr(runner, "run_async"):
        return

    orig_run_async = runner.run_async

    async def patched_run_async(*args, **kwargs):
        tracer = get_tracer()
        attrs = _get_runner_attributes(runner)

        user_id = kwargs.get("user_id")
        if user_id:
            attrs["neatlogs.user.id"] = str(user_id)
        session_id = kwargs.get("session_id")
        if session_id:
            attrs["neatlogs.session.id"] = str(session_id)

        new_message = kwargs.get("new_message")
        if new_message:
            if hasattr(new_message, "parts"):
                text_parts = [getattr(p, "text", "") for p in new_message.parts if getattr(p, "text", None)]
                if text_parts:
                    attrs["input.value"] = "\n".join(text_parts)[:10000]
            else:
                attrs["input.value"] = str(new_message)[:10000]

        span = tracer.start_span(name="google_adk.runner.run_async", attributes=attrs)
        start = time.perf_counter()

        try:
            events = orig_run_async(*args, **kwargs)
            collected, event_attrs = await _collect_events_async(events)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        for attr_name, value in event_attrs.items():
            span.set_attribute(attr_name, value)
        span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
        span.set_status(StatusCode.OK)
        span.end()

        async def _yield_collected():
            for event in collected:
                yield event

        return _yield_collected()

    runner.run_async = patched_run_async
