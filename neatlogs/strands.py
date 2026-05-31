"""
Neatlogs Strands Agents hook registration.

Usage:
    >>> import neatlogs
    >>> from strands import Agent
    >>> agent = Agent(model=model)
    >>> neatlogs.strands_hooks(agent)
    >>> response = agent("Hello")

Uses the native Strands hook system (HookProvider / HookRegistry). Registers
callbacks for the real Strands event classes:

    BeforeInvocationEvent / AfterInvocationEvent   → AGENT span
    BeforeToolCallEvent / AfterToolCallEvent       → TOOL span (per tool call)
    BeforeModelCallEvent / AfterModelCallEvent     → LLM span (per model call)

Span hierarchy:
    AGENT → LLM (per model invocation)
          → TOOL (per tool call)
"""

import time
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.trace import StatusCode

from ._wrap_utils import attach_as_current, detach, get_tracer, serialize


def strands_hooks(agent: Any) -> Any:
    """
    Register Neatlogs tracing hooks on a Strands Agent.
    Uses Strands' native hook system. Returns the same agent.
    """
    if getattr(agent, "_neatlogs_hooked", False):
        return agent

    handler = _NeatlogsStrandsHooks()

    registry = getattr(agent, "hooks", None)
    registered = False

    # Preferred: agent.hooks is a HookRegistry — register via add_hook (HookProvider).
    if registry is not None and hasattr(registry, "add_hook"):
        try:
            registry.add_hook(handler)
            registered = True
        except Exception:
            registered = False

    # Some versions accept individual callbacks via add_callback.
    if not registered and registry is not None and hasattr(registry, "add_callback"):
        try:
            handler.register_hooks(registry)
            registered = True
        except Exception:
            registered = False

    # Fallback: wrap the agent's __call__ / invoke_async directly.
    if not registered:
        _patch_agent_call(agent, handler)

    agent._neatlogs_hooked = True
    return agent


class _NeatlogsStrandsHooks:
    """
    HookProvider that emits OTel spans from Strands agent lifecycle events.

    Implements `register_hooks(registry, **kwargs)` so it can be passed to
    `agent.hooks.add_hook(self)`. Keyed by id(agent) with stacks so nested /
    concurrent tool and model calls on the same agent don't clobber each other.
    """

    def __init__(self):
        self._agent_spans = {}        # id(agent) -> (span, token, start)
        self._tool_spans = {}         # tool_use_id -> (span, token, start)
        self._model_spans = {}        # id(agent) -> list[(span, token, start)]

    # -- HookProvider protocol -------------------------------------------------

    def register_hooks(self, registry: Any, **kwargs: Any) -> None:
        # Import lazily so importing neatlogs never hard-depends on strands.
        try:
            from strands.hooks import (
                AfterInvocationEvent,
                AfterModelCallEvent,
                AfterToolCallEvent,
                BeforeInvocationEvent,
                BeforeModelCallEvent,
                BeforeToolCallEvent,
            )
        except Exception:
            return

        registry.add_callback(BeforeInvocationEvent, self._on_before_invocation)
        registry.add_callback(AfterInvocationEvent, self._on_after_invocation)
        registry.add_callback(BeforeToolCallEvent, self._on_before_tool)
        registry.add_callback(AfterToolCallEvent, self._on_after_tool)
        registry.add_callback(BeforeModelCallEvent, self._on_before_model)
        registry.add_callback(AfterModelCallEvent, self._on_after_model)

    # -- AGENT lifecycle -------------------------------------------------------

    def _on_before_invocation(self, event: Any) -> None:
        agent = getattr(event, "agent", None)
        tracer = get_tracer()
        attrs = {"neatlogs.span.kind": "AGENT"}

        name = getattr(agent, "name", None)
        if name:
            attrs["neatlogs.agent.name"] = name

        model = getattr(agent, "model", None)
        if model is not None:
            model_name = (
                getattr(model, "model_id", None)
                or getattr(model, "model_name", None)
                or _config_get(model, "model_id")
            )
            if model_name:
                attrs["neatlogs.llm.model_name"] = str(model_name)

        messages = getattr(event, "messages", None)
        prompt = _last_user_text(messages)
        if prompt:
            attrs["input.value"] = prompt[:10000]

        span = tracer.start_span(name="strands.agent.run", attributes=attrs)
        token = attach_as_current(span)
        self._agent_spans[id(agent)] = (span, token, time.perf_counter())

    def _on_after_invocation(self, event: Any) -> None:
        agent = getattr(event, "agent", None)
        entry = self._agent_spans.pop(id(agent), None)
        if not entry:
            return
        span, token, start = entry
        if token:
            detach(token)

        result = getattr(event, "result", None)
        if result is not None:
            text = _agent_result_text(result)
            if text:
                span.set_attribute("output.value", text[:10000])

            stop_reason = getattr(result, "stop_reason", None)
            if stop_reason:
                span.set_attribute("neatlogs.llm.finish_reason", str(stop_reason))

            metrics = getattr(result, "metrics", None)
            usage = getattr(metrics, "accumulated_usage", None) if metrics else None
            if usage:
                _set_usage_attrs(span, usage)

        _set_duration(span, start)
        span.set_status(StatusCode.OK)
        span.end()

    # -- TOOL lifecycle --------------------------------------------------------

    def _on_before_tool(self, event: Any) -> None:
        tool_use = getattr(event, "tool_use", None) or {}
        tool_name = tool_use.get("name", "") if isinstance(tool_use, dict) else getattr(tool_use, "name", "")
        tool_id = tool_use.get("toolUseId", "") if isinstance(tool_use, dict) else getattr(tool_use, "toolUseId", "")
        tool_input = tool_use.get("input") if isinstance(tool_use, dict) else getattr(tool_use, "input", None)

        tracer = get_tracer()
        attrs = {"neatlogs.span.kind": "TOOL", "neatlogs.tool.name": tool_name}
        if tool_id:
            attrs["neatlogs.tool.call_id"] = str(tool_id)
        if tool_input is not None:
            attrs["input.value"] = tool_input if isinstance(tool_input, str) else serialize(tool_input)

        span = tracer.start_span(name=f"strands.tool.{tool_name}", attributes=attrs)
        token = attach_as_current(span)
        key = str(tool_id) or f"{id(event)}"
        self._tool_spans[key] = (span, token, time.perf_counter())

    def _on_after_tool(self, event: Any) -> None:
        tool_use = getattr(event, "tool_use", None) or {}
        tool_id = tool_use.get("toolUseId", "") if isinstance(tool_use, dict) else getattr(tool_use, "toolUseId", "")
        key = str(tool_id) or f"{id(event)}"
        entry = self._tool_spans.pop(key, None)
        if not entry:
            return
        span, token, start = entry
        if token:
            detach(token)

        result = getattr(event, "result", None)
        if result is not None:
            out = _tool_result_text(result)
            if out:
                span.set_attribute("output.value", out[:10000])
            status = result.get("status") if isinstance(result, dict) else getattr(result, "status", None)
            if status:
                span.set_attribute("neatlogs.tool.status", str(status))

        exception = getattr(event, "exception", None)
        _set_duration(span, start)
        if exception is not None:
            span.set_status(StatusCode.ERROR, str(exception))
            if isinstance(exception, BaseException):
                span.record_exception(exception)
        else:
            span.set_status(StatusCode.OK)
        span.end()

    # -- LLM / model lifecycle -------------------------------------------------

    def _on_before_model(self, event: Any) -> None:
        agent = getattr(event, "agent", None)
        tracer = get_tracer()
        attrs = {"neatlogs.span.kind": "LLM", "neatlogs.llm.provider": "strands"}

        model = getattr(agent, "model", None)
        if model is not None:
            model_name = (
                getattr(model, "model_id", None)
                or getattr(model, "model_name", None)
                or _config_get(model, "model_id")
            )
            if model_name:
                attrs["neatlogs.llm.model_name"] = str(model_name)

        messages = getattr(agent, "messages", None)
        if messages:
            _set_input_messages(attrs, messages)

        projected = getattr(event, "projected_input_tokens", None)
        if projected:
            attrs["neatlogs.llm.projected_input_tokens"] = projected

        span = tracer.start_span(name="strands.model.invoke", attributes=attrs)
        token = attach_as_current(span)
        self._model_spans.setdefault(id(agent), []).append((span, token, time.perf_counter()))

    def _on_after_model(self, event: Any) -> None:
        agent = getattr(event, "agent", None)
        stack = self._model_spans.get(id(agent))
        if not stack:
            return
        span, token, start = stack.pop()
        if token:
            detach(token)

        stop_response = getattr(event, "stop_response", None)
        if stop_response is not None:
            message = getattr(stop_response, "message", None)
            text = _message_text(message)
            if text:
                span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
                span.set_attribute("neatlogs.llm.output_messages.0.content", text[:10000])
            _set_tool_calls_from_message(span, message)
            stop_reason = getattr(stop_response, "stop_reason", None)
            if stop_reason:
                span.set_attribute("neatlogs.llm.finish_reason", str(stop_reason))

        exception = getattr(event, "exception", None)
        _set_duration(span, start)
        if exception is not None:
            span.set_status(StatusCode.ERROR, str(exception))
            if isinstance(exception, BaseException):
                span.record_exception(exception)
        else:
            span.set_status(StatusCode.OK)
        span.end()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config_get(model: Any, key: str) -> Any:
    """Strands model config lives in model.config (a dict) on some providers."""
    cfg = getattr(model, "config", None)
    if isinstance(cfg, dict):
        return cfg.get(key)
    return None


def _last_user_text(messages: Any) -> str:
    if not messages:
        return ""
    for msg in reversed(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "user":
            return _message_text(msg)
    return _message_text(messages[-1]) if messages else ""


def _message_text(message: Any) -> str:
    """Extract concatenated text from a Strands Message (content is a list of blocks)."""
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content if isinstance(content, list) else [content]:
        if isinstance(block, dict):
            if "text" in block:
                parts.append(block["text"])
        else:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
    return "".join(parts)


def _set_input_messages(attrs: dict, messages: Any) -> None:
    for i, msg in enumerate(messages if isinstance(messages, list) else [messages]):
        role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
        content = _message_text(msg)
        if role:
            attrs[f"neatlogs.llm.input_messages.{i}.role"] = role
        if content:
            attrs[f"neatlogs.llm.input_messages.{i}.content"] = content[:10000]


def _set_tool_calls_from_message(span: Any, message: Any) -> None:
    """Pull toolUse blocks out of an assistant Message into tool_call attributes."""
    if message is None:
        return
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if not isinstance(content, list):
        return
    j = 0
    for block in content:
        tool_use = block.get("toolUse") if isinstance(block, dict) else getattr(block, "toolUse", None)
        if not tool_use:
            continue
        name = tool_use.get("name", "") if isinstance(tool_use, dict) else getattr(tool_use, "name", "")
        args = tool_use.get("input") if isinstance(tool_use, dict) else getattr(tool_use, "input", None)
        tid = tool_use.get("toolUseId", "") if isinstance(tool_use, dict) else getattr(tool_use, "toolUseId", "")
        if name:
            span.set_attribute(f"neatlogs.llm.tool_calls.{j}.name", name)
        if args is not None:
            span.set_attribute(f"neatlogs.llm.tool_calls.{j}.arguments", args if isinstance(args, str) else serialize(args))
        if tid:
            span.set_attribute(f"neatlogs.llm.tool_calls.{j}.id", str(tid))
        j += 1


def _agent_result_text(result: Any) -> str:
    message = getattr(result, "message", None)
    text = _message_text(message)
    if text:
        return text
    return str(result) if result is not None else ""


def _tool_result_text(result: Any) -> str:
    content = result.get("content") if isinstance(result, dict) else getattr(result, "content", None)
    if not content:
        return str(result)
    parts = []
    for block in content if isinstance(content, list) else [content]:
        if isinstance(block, dict):
            if "text" in block:
                parts.append(block["text"])
            elif "json" in block:
                parts.append(serialize(block["json"]))
            else:
                parts.append(serialize(block))
        else:
            parts.append(str(block))
    return "".join(parts)


def _set_usage_attrs(span: Any, usage: Any) -> None:
    """Set token usage attributes from Strands usage (dict or object)."""
    if isinstance(usage, dict):
        input_tokens = usage.get("inputTokens") or usage.get("input_tokens") or usage.get("prompt_tokens")
        output_tokens = usage.get("outputTokens") or usage.get("output_tokens") or usage.get("completion_tokens")
        total_tokens = usage.get("totalTokens") or usage.get("total_tokens")
        cache_read = usage.get("cacheReadInputTokens") or usage.get("cache_read_input_tokens")
        cache_write = usage.get("cacheCreationInputTokens") or usage.get("cache_creation_input_tokens")
    else:
        input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "inputTokens", None)
        output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "outputTokens", None)
        total_tokens = getattr(usage, "total_tokens", None) or getattr(usage, "totalTokens", None)
        cache_read = getattr(usage, "cacheReadInputTokens", None) or getattr(usage, "cache_read_input_tokens", None)
        cache_write = getattr(usage, "cacheCreationInputTokens", None) or getattr(usage, "cache_creation_input_tokens", None)

    if input_tokens:
        span.set_attribute("neatlogs.llm.token_count.prompt", input_tokens)
    if output_tokens:
        span.set_attribute("neatlogs.llm.token_count.completion", output_tokens)
    if total_tokens:
        span.set_attribute("neatlogs.llm.token_count.total", total_tokens)
    if cache_read:
        span.set_attribute("neatlogs.llm.token_count.cache_read", cache_read)
    if cache_write:
        span.set_attribute("neatlogs.llm.token_count.cache_write", cache_write)


def _set_duration(span: Any, start: float) -> None:
    span.set_attribute("neatlogs.llm.metrics.duration_ms", round((time.perf_counter() - start) * 1000, 3))


def _patch_agent_call(agent: Any, handler: "_NeatlogsStrandsHooks") -> None:
    """Fallback: wrap the agent's __call__ to at least emit the AGENT span."""

    orig_call = getattr(type(agent), "__call__", None) or getattr(agent, "__call__", None)
    if orig_call is None:
        return

    class _Evt:
        def __init__(self, agent, messages=None, result=None):
            self.agent = agent
            self.messages = messages
            self.result = result

    def patched_call(self, *args, **kwargs):
        prompt = args[0] if args else kwargs.get("prompt")
        msgs = [{"role": "user", "content": [{"text": str(prompt)}]}] if prompt else None
        handler._on_before_invocation(_Evt(self, messages=msgs))
        try:
            result = orig_call(self, *args, **kwargs)
        except Exception as e:
            entry = handler._agent_spans.pop(id(self), None)
            if entry:
                span, token, start = entry
                if token:
                    detach(token)
                _set_duration(span, start)
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)
                span.end()
            raise
        handler._on_after_invocation(_Evt(self, result=result))
        return result

    try:
        type(agent).__call__ = patched_call
    except Exception:
        agent.__call__ = patched_call.__get__(agent, type(agent))
