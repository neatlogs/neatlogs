"""
Neatlogs Strands Agents hook registration.

Usage:
    >>> import neatlogs
    >>> from strands import Agent
    >>> agent = Agent(model=model)
    >>> neatlogs.strands_hooks(agent)
    >>> response = agent("Hello")

Creates span hierarchy:
    AGENT → TOOL (per tool call)
         → LLM (per model invocation)
"""

import time
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.trace import StatusCode

from ._wrap_utils import get_tracer, serialize


def strands_hooks(agent: Any) -> Any:
    """
    Register Neatlogs tracing hooks on a Strands Agent.
    Uses Strands' native hook system. Returns the same agent.
    """
    if getattr(agent, "_neatlogs_hooked", False):
        return agent

    handler = _NeatlogsStrandsHandler()

    hooks = getattr(agent, "hooks", None)
    if hooks is not None:
        if hasattr(hooks, "register"):
            hooks.register(handler)
        elif hasattr(hooks, "add"):
            hooks.add(handler)
        else:
            _patch_agent_call(agent, handler)
    else:
        _patch_agent_call(agent, handler)

    agent._neatlogs_hooked = True
    return agent


class _NeatlogsStrandsHandler:
    """Handler that creates OTel spans from Strands agent lifecycle hooks."""

    def __init__(self):
        self._agent_spans = {}
        self._agent_tokens = {}
        self._agent_start_times = {}
        self._tool_spans = {}
        self._tool_start_times = {}
        self._llm_spans = {}
        self._llm_start_times = {}

    # ------ AGENT lifecycle (AGENT span) ------

    def on_agent_start(self, agent: Any, **kwargs):
        tracer = get_tracer()
        attrs = {"neatlogs.span.kind": "AGENT"}

        name = getattr(agent, "name", None)
        if name:
            attrs["neatlogs.agent.name"] = name

        model = getattr(agent, "model", None)
        if model:
            model_name = getattr(model, "model_id", None) or getattr(model, "model_name", None) or str(model)
            attrs["neatlogs.llm.model_name"] = model_name

        prompt = kwargs.get("prompt") or kwargs.get("message") or kwargs.get("input")
        if prompt:
            attrs["input.value"] = str(prompt)[:10000]

        span = tracer.start_span(name="strands.agent.run", attributes=attrs)
        ctx = otel_context.set_value("current_span", span)
        token = otel_context.attach(ctx)

        agent_id = id(agent)
        self._agent_spans[agent_id] = span
        self._agent_tokens[agent_id] = token
        self._agent_start_times[agent_id] = time.perf_counter()

    def on_agent_end(self, agent: Any, response: Any = None, **kwargs):
        agent_id = id(agent)
        span = self._agent_spans.pop(agent_id, None)
        token = self._agent_tokens.pop(agent_id, None)
        start_time = self._agent_start_times.pop(agent_id, None)
        if not span:
            return

        if token:
            otel_context.detach(token)

        if response is not None:
            content = getattr(response, "content", None) or getattr(response, "text", None)
            if content:
                span.set_attribute("output.value", str(content)[:10000])

            usage = getattr(response, "usage", None) or kwargs.get("usage")
            if usage:
                _set_usage_attrs(span, usage)

            stop_reason = (
                getattr(response, "stop_reason", None)
                or kwargs.get("stop_reason")
                or getattr(response, "finish_reason", None)
                or kwargs.get("finish_reason")
            )
            if stop_reason:
                span.set_attribute("neatlogs.llm.finish_reason", str(stop_reason))

        if start_time:
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))

        span.set_status(StatusCode.OK)
        span.end()

    def on_agent_error(self, agent: Any, error: Any = None, **kwargs):
        agent_id = id(agent)
        span = self._agent_spans.pop(agent_id, None)
        token = self._agent_tokens.pop(agent_id, None)
        start_time = self._agent_start_times.pop(agent_id, None)
        if not span:
            return

        if token:
            otel_context.detach(token)

        if start_time:
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))

        error_msg = str(error) if error else "Unknown error"
        span.set_status(StatusCode.ERROR, error_msg)
        if isinstance(error, BaseException):
            span.record_exception(error)
        span.end()

    # ------ TOOL lifecycle (TOOL spans) ------

    def on_tool_start(self, agent: Any, tool_name: str = "", tool_input: Any = None, **kwargs):
        tracer = get_tracer()
        attrs = {
            "neatlogs.span.kind": "TOOL",
            "neatlogs.tool.name": tool_name,
        }
        if tool_input is not None:
            attrs["input.value"] = serialize(tool_input) if not isinstance(tool_input, str) else tool_input

        span = tracer.start_span(name=f"strands.tool.{tool_name}", attributes=attrs)
        ctx = otel_context.set_value("current_span", span)
        token = otel_context.attach(ctx)

        tool_key = (id(agent), tool_name)
        self._tool_spans[tool_key] = (span, token)
        self._tool_start_times[tool_key] = time.perf_counter()

    def on_tool_end(self, agent: Any, tool_name: str = "", tool_output: Any = None, **kwargs):
        tool_key = (id(agent), tool_name)
        entry = self._tool_spans.pop(tool_key, None)
        start_time = self._tool_start_times.pop(tool_key, None)

        if not entry:
            return

        span, token = entry
        if token:
            otel_context.detach(token)

        if tool_output is not None:
            span.set_attribute("output.value", str(tool_output)[:10000])

        if start_time:
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))

        span.set_status(StatusCode.OK)
        span.end()

    def on_tool_error(self, agent: Any, tool_name: str = "", error: Any = None, **kwargs):
        tool_key = (id(agent), tool_name)
        entry = self._tool_spans.pop(tool_key, None)
        start_time = self._tool_start_times.pop(tool_key, None)

        if not entry:
            return

        span, token = entry
        if token:
            otel_context.detach(token)

        if start_time:
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))

        error_msg = str(error) if error else "Unknown error"
        span.set_status(StatusCode.ERROR, error_msg)
        if isinstance(error, BaseException):
            span.record_exception(error)
        span.end()

    # ------ LLM lifecycle (LLM spans for model invocations) ------

    def on_model_start(self, agent: Any, model_name: str = "", messages: Any = None, **kwargs):
        tracer = get_tracer()
        attrs = {
            "neatlogs.span.kind": "LLM",
            "neatlogs.llm.provider": "strands",
        }
        if model_name:
            attrs["neatlogs.llm.model_name"] = model_name

        if messages:
            for i, msg in enumerate(messages if isinstance(messages, list) else [messages]):
                role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
                content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                if role:
                    attrs[f"neatlogs.llm.input_messages.{i}.role"] = role
                if content:
                    content_str = content if isinstance(content, str) else serialize(content)
                    attrs[f"neatlogs.llm.input_messages.{i}.content"] = content_str[:10000]

        span = tracer.start_span(name="strands.model.invoke", attributes=attrs)
        ctx = otel_context.set_value("current_span", span)
        token = otel_context.attach(ctx)

        llm_key = id(agent)
        self._llm_spans[llm_key] = (span, token)
        self._llm_start_times[llm_key] = time.perf_counter()

    def on_model_end(self, agent: Any, response: Any = None, **kwargs):
        llm_key = id(agent)
        entry = self._llm_spans.pop(llm_key, None)
        start_time = self._llm_start_times.pop(llm_key, None)

        if not entry:
            return

        span, token = entry
        if token:
            otel_context.detach(token)

        if response is not None:
            content = getattr(response, "content", None) or getattr(response, "text", None)
            if content:
                span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
                span.set_attribute("neatlogs.llm.output_messages.0.content", str(content)[:10000])

            usage = getattr(response, "usage", None) or kwargs.get("usage")
            if usage:
                _set_usage_attrs(span, usage)

            stop_reason = (
                getattr(response, "stop_reason", None)
                or kwargs.get("stop_reason")
                or getattr(response, "finish_reason", None)
            )
            if stop_reason:
                span.set_attribute("neatlogs.llm.finish_reason", str(stop_reason))

            # Tool calls in the response
            tool_calls = getattr(response, "tool_calls", None) or getattr(response, "tool_use", None)
            if tool_calls and isinstance(tool_calls, list):
                for j, tc in enumerate(tool_calls):
                    tc_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    tc_args = tc.get("input", tc.get("arguments", "")) if isinstance(tc, dict) else getattr(tc, "input", "")
                    tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                    if tc_name:
                        span.set_attribute(f"neatlogs.llm.tool_calls.{j}.name", tc_name)
                    if tc_args:
                        span.set_attribute(f"neatlogs.llm.tool_calls.{j}.arguments", serialize(tc_args) if not isinstance(tc_args, str) else tc_args)
                    if tc_id:
                        span.set_attribute(f"neatlogs.llm.tool_calls.{j}.id", str(tc_id))

        if start_time:
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))

        span.set_status(StatusCode.OK)
        span.end()

    def on_model_error(self, agent: Any, error: Any = None, **kwargs):
        llm_key = id(agent)
        entry = self._llm_spans.pop(llm_key, None)
        start_time = self._llm_start_times.pop(llm_key, None)

        if not entry:
            return

        span, token = entry
        if token:
            otel_context.detach(token)

        if start_time:
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))

        error_msg = str(error) if error else "Unknown error"
        span.set_status(StatusCode.ERROR, error_msg)
        if isinstance(error, BaseException):
            span.record_exception(error)
        span.end()


def _set_usage_attrs(span: Any, usage: Any) -> None:
    """Set token usage attributes from various usage formats."""
    if isinstance(usage, dict):
        input_tokens = usage.get("inputTokens") or usage.get("input_tokens") or usage.get("prompt_tokens")
        output_tokens = usage.get("outputTokens") or usage.get("output_tokens") or usage.get("completion_tokens")
        cache_read = usage.get("cacheReadInputTokens") or usage.get("cache_read_input_tokens")
        cache_write = usage.get("cacheCreationInputTokens") or usage.get("cache_creation_input_tokens")
    else:
        input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "inputTokens", None)
        output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "outputTokens", None)
        cache_read = getattr(usage, "cacheReadInputTokens", None) or getattr(usage, "cache_read_input_tokens", None)
        cache_write = getattr(usage, "cacheCreationInputTokens", None) or getattr(usage, "cache_creation_input_tokens", None)

    if input_tokens:
        span.set_attribute("neatlogs.llm.token_count.prompt", input_tokens)
    if output_tokens:
        span.set_attribute("neatlogs.llm.token_count.completion", output_tokens)
    if cache_read:
        span.set_attribute("neatlogs.llm.token_count.cache_read", cache_read)
    if cache_write:
        span.set_attribute("neatlogs.llm.token_count.cache_write", cache_write)


def _patch_agent_call(agent: Any, handler: _NeatlogsStrandsHandler) -> None:
    """Fallback: patch __call__ if hooks system not available."""
    if not hasattr(agent, "__call__"):
        return

    orig_call = agent.__call__

    def patched_call(*args, **kwargs):
        handler.on_agent_start(agent, prompt=args[0] if args else kwargs.get("prompt"))
        try:
            result = orig_call(*args, **kwargs)
        except Exception as e:
            handler.on_agent_error(agent, error=e)
            raise
        handler.on_agent_end(agent, response=result)
        return result

    agent.__call__ = patched_call
