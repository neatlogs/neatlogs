"""
Neatlogs Pydantic AI wrapper.

Usage:
    >>> import neatlogs
    >>> from pydantic_ai import Agent
    >>> agent = neatlogs.wrap(Agent("openai:gpt-4o", system_prompt="..."))
    >>> result = agent.run_sync("Hello")
"""

import time
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.trace import StatusCode

from ._wrap_utils import get_tracer, serialize


def wrap_pydantic_ai(agent: Any) -> Any:
    """
    Wrap a Pydantic AI Agent instance. Patches run(), run_sync(), and run_stream().
    Returns the same agent instance.
    """
    _patch_run(agent)
    _patch_run_sync(agent)
    _patch_run_stream(agent)
    return agent


def _get_agent_attributes(agent: Any) -> dict:
    """Extract agent metadata as span attributes."""
    attrs = {"neatlogs.span.kind": "AGENT"}

    model = getattr(agent, "model", None)
    if model:
        model_name = getattr(model, "model_name", None) or getattr(model, "name", None) or str(model)
        attrs["neatlogs.llm.model_name"] = model_name

    name = getattr(agent, "name", None)
    if name:
        attrs["neatlogs.agent.name"] = name

    system_prompt = getattr(agent, "system_prompt", None) or getattr(agent, "_system_prompt", None)
    if system_prompt and isinstance(system_prompt, str):
        attrs["neatlogs.llm.input_messages.0.role"] = "system"
        attrs["neatlogs.llm.input_messages.0.content"] = system_prompt[:10000]

    return attrs


def _extract_usage(result: Any) -> dict:
    """Extract token usage from RunResult."""
    attrs = {}

    usage_obj = getattr(result, "usage", None) or getattr(result, "_usage", None)
    if not usage_obj:
        cost = getattr(result, "cost", None)
        if cost:
            usage_obj = cost

    if not usage_obj:
        return attrs

    if hasattr(usage_obj, "request_tokens"):
        attrs["neatlogs.llm.token_count.prompt"] = usage_obj.request_tokens
    elif hasattr(usage_obj, "prompt_tokens"):
        attrs["neatlogs.llm.token_count.prompt"] = usage_obj.prompt_tokens

    if hasattr(usage_obj, "response_tokens"):
        attrs["neatlogs.llm.token_count.completion"] = usage_obj.response_tokens
    elif hasattr(usage_obj, "completion_tokens"):
        attrs["neatlogs.llm.token_count.completion"] = usage_obj.completion_tokens

    if hasattr(usage_obj, "total_tokens"):
        attrs["neatlogs.llm.token_count.total"] = usage_obj.total_tokens

    return attrs


def _finalize_run_span(span: Any, result: Any, duration_ms: float) -> None:
    """Finalize a Pydantic AI run span with result data."""
    if result is None:
        span.set_status(StatusCode.OK)
        span.end()
        return

    data = getattr(result, "data", None)
    if data is not None:
        output = str(data) if not isinstance(data, str) else data
        span.set_attribute("output.value", output[:10000])

    for attr_name, value in _extract_usage(result).items():
        span.set_attribute(attr_name, value)

    all_messages = getattr(result, "all_messages", None) or getattr(result, "messages", None)
    if all_messages:
        tool_call_idx = 0
        for msg in all_messages:
            kind = getattr(msg, "kind", None) or type(msg).__name__
            if kind == "tool-return" or "ToolReturn" in str(type(msg)):
                tool_name = getattr(msg, "tool_name", None)
                if tool_name:
                    span.set_attribute(f"neatlogs.llm.tool_calls.{tool_call_idx}.name", tool_name)
                    tool_call_idx += 1

    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()


def _patch_run(agent: Any) -> None:
    if not hasattr(agent, "run"):
        return
    if getattr(agent, "_neatlogs_run_patched", False):
        return

    orig_run = agent.run

    async def patched_run(*args, **kwargs):
        tracer = get_tracer()
        attrs = _get_agent_attributes(agent)

        user_prompt = args[0] if args else kwargs.get("user_prompt", kwargs.get("prompt", ""))
        if user_prompt:
            attrs["input.value"] = str(user_prompt)[:10000]

        span = tracer.start_span(name="pydantic_ai.agent.run", attributes=attrs)
        ctx = otel_context.set_value("current_span", span)
        token = otel_context.attach(ctx)
        start = time.perf_counter()

        try:
            result = await orig_run(*args, **kwargs)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise
        finally:
            otel_context.detach(token)

        duration_ms = (time.perf_counter() - start) * 1000
        _finalize_run_span(span, result, duration_ms)
        return result

    agent.run = patched_run
    agent._neatlogs_run_patched = True


def _patch_run_sync(agent: Any) -> None:
    if not hasattr(agent, "run_sync"):
        return
    if getattr(agent, "_neatlogs_run_sync_patched", False):
        return

    orig_run_sync = agent.run_sync

    def patched_run_sync(*args, **kwargs):
        tracer = get_tracer()
        attrs = _get_agent_attributes(agent)

        user_prompt = args[0] if args else kwargs.get("user_prompt", kwargs.get("prompt", ""))
        if user_prompt:
            attrs["input.value"] = str(user_prompt)[:10000]

        span = tracer.start_span(name="pydantic_ai.agent.run_sync", attributes=attrs)
        ctx = otel_context.set_value("current_span", span)
        token = otel_context.attach(ctx)
        start = time.perf_counter()

        try:
            result = orig_run_sync(*args, **kwargs)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise
        finally:
            otel_context.detach(token)

        duration_ms = (time.perf_counter() - start) * 1000
        _finalize_run_span(span, result, duration_ms)
        return result

    agent.run_sync = patched_run_sync
    agent._neatlogs_run_sync_patched = True


def _patch_run_stream(agent: Any) -> None:
    if not hasattr(agent, "run_stream"):
        return
    if getattr(agent, "_neatlogs_run_stream_patched", False):
        return

    orig_run_stream = agent.run_stream

    async def patched_run_stream(*args, **kwargs):
        tracer = get_tracer()
        attrs = _get_agent_attributes(agent)

        user_prompt = args[0] if args else kwargs.get("user_prompt", kwargs.get("prompt", ""))
        if user_prompt:
            attrs["input.value"] = str(user_prompt)[:10000]

        span = tracer.start_span(name="pydantic_ai.agent.run_stream", attributes=attrs)
        start = time.perf_counter()

        try:
            stream_result = await orig_run_stream(*args, **kwargs)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise

        return _StreamResultWrapper(stream_result, span, start)

    agent.run_stream = patched_run_stream
    agent._neatlogs_run_stream_patched = True


class _StreamResultWrapper:
    """Wraps a Pydantic AI StreamedRunResult to capture final data on exit."""

    def __init__(self, stream_result: Any, span: Any, start_time: float):
        self._stream_result = stream_result
        self._span = span
        self._start_time = start_time

    def __getattr__(self, name):
        return getattr(self._stream_result, name)

    async def __aenter__(self):
        await self._stream_result.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self._stream_result.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            duration_ms = (time.perf_counter() - self._start_time) * 1000
            if exc_type:
                self._span.set_status(StatusCode.ERROR, str(exc_val))
                if exc_val:
                    self._span.record_exception(exc_val)
            else:
                data = getattr(self._stream_result, "data", None)
                if data is not None:
                    self._span.set_attribute("output.value", str(data)[:10000])

                usage = getattr(self._stream_result, "usage", None)
                if usage:
                    if hasattr(usage, "request_tokens"):
                        self._span.set_attribute("neatlogs.llm.token_count.prompt", usage.request_tokens)
                    if hasattr(usage, "response_tokens"):
                        self._span.set_attribute("neatlogs.llm.token_count.completion", usage.response_tokens)

                self._span.set_status(StatusCode.OK)

            self._span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
            self._span.end()

    async def stream(self):
        async for chunk in self._stream_result.stream():
            yield chunk

    async def stream_text(self, delta=True):
        async for text in self._stream_result.stream_text(delta=delta):
            yield text
