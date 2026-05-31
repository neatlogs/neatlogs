"""
Neatlogs Pydantic AI wrapper.

Usage:
    >>> import neatlogs
    >>> from pydantic_ai import Agent
    >>> agent = neatlogs.wrap(Agent("openai:gpt-4o", system_prompt="..."))
    >>> result = agent.run_sync("Hello")

Span hierarchy:
    AGENT (run / run_sync / run_stream / iter)
      ↳ LLM   (Model.request / request_stream — one per model call)
      ↳ TOOL  (FunctionToolset.call_tool — one per tool invocation)

The AGENT span is per-agent-instance; the LLM and TOOL spans are installed once
at the class level so every model request and tool call nests under the active
agent run (and under user @span / trace() blocks too).
"""

import contextvars
import time
from typing import Any

from opentelemetry.trace import StatusCode

from ._wrap_utils import attach_as_current, detach, get_tracer, serialize

_CLASS_HOOKS_INSTALLED = False

# Pydantic AI's run_sync() calls run(), which calls iter(). Without a guard we
# would emit three nested AGENT spans for one logical run. This flag marks that
# an AGENT span is already open on the current context so inner entry points
# skip creating their own.
_agent_span_active = contextvars.ContextVar("neatlogs_pai_agent_active", default=False)


def wrap_pydantic_ai(agent: Any) -> Any:
    """
    Wrap a Pydantic AI Agent. Patches run(), run_sync(), run_stream(), iter()
    and installs class-level Model (LLM) + toolset (TOOL) hooks.
    Returns the same agent instance.
    """
    _install_class_hooks()
    _patch_agent_model(agent)
    _patch_run(agent)
    _patch_run_sync(agent)
    _patch_run_stream(agent)
    _patch_iter(agent)
    return agent


def _get_agent_attributes(agent: Any) -> dict:
    attrs = {"neatlogs.span.kind": "AGENT"}

    model = getattr(agent, "model", None)
    if model is not None:
        model_name = getattr(model, "model_name", None) or getattr(model, "name", None) or str(model)
        attrs["neatlogs.llm.model_name"] = str(model_name)

    name = getattr(agent, "name", None)
    if name:
        attrs["neatlogs.agent.name"] = name

    system_prompt = getattr(agent, "system_prompt", None) or getattr(agent, "_system_prompts", None)
    if system_prompt:
        if isinstance(system_prompt, (list, tuple)):
            system_prompt = "\n".join(str(s) for s in system_prompt)
        if isinstance(system_prompt, str) and system_prompt:
            attrs["neatlogs.llm.input_messages.0.role"] = "system"
            attrs["neatlogs.llm.input_messages.0.content"] = system_prompt[:10000]

    return attrs


def _extract_usage(result: Any) -> dict:
    """Extract token usage from a RunResult / StreamedRunResult."""
    attrs = {}
    usage_obj = None
    usage_attr = getattr(result, "usage", None)
    if callable(usage_attr):
        try:
            usage_obj = usage_attr()
        except Exception:
            usage_obj = None
    else:
        usage_obj = usage_attr
    if usage_obj is None:
        usage_obj = getattr(result, "_usage", None)
    if usage_obj is None:
        return attrs

    prompt = getattr(usage_obj, "input_tokens", None)
    if prompt is None:
        prompt = getattr(usage_obj, "request_tokens", None) or getattr(usage_obj, "prompt_tokens", None)
    completion = getattr(usage_obj, "output_tokens", None)
    if completion is None:
        completion = getattr(usage_obj, "response_tokens", None) or getattr(usage_obj, "completion_tokens", None)
    total = getattr(usage_obj, "total_tokens", None)
    cache_read = getattr(usage_obj, "cache_read_tokens", None)
    cache_write = getattr(usage_obj, "cache_write_tokens", None)

    if prompt:
        attrs["neatlogs.llm.token_count.prompt"] = prompt
    if completion:
        attrs["neatlogs.llm.token_count.completion"] = completion
    if total:
        attrs["neatlogs.llm.token_count.total"] = total
    if cache_read:
        attrs["neatlogs.llm.token_count.cache_read"] = cache_read
    if cache_write:
        attrs["neatlogs.llm.token_count.cache_write"] = cache_write
    return attrs


def _result_output(result: Any) -> Any:
    for attr in ("output", "data"):
        val = getattr(result, attr, None)
        if val is not None:
            return val
    return None


def _finalize_run_span(span: Any, result: Any, duration_ms: float) -> None:
    if result is None:
        span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
        span.set_status(StatusCode.OK)
        span.end()
        return

    output = _result_output(result)
    if output is not None:
        span.set_attribute("output.value", (output if isinstance(output, str) else serialize(output))[:10000])

    for attr_name, value in _extract_usage(result).items():
        span.set_attribute(attr_name, value)

    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()


def _user_prompt(args, kwargs) -> str:
    up = args[0] if args else kwargs.get("user_prompt", kwargs.get("prompt"))
    if up is None:
        return ""
    return up if isinstance(up, str) else serialize(up)


# ---------------------------------------------------------------------------
# Agent.run (async)
# ---------------------------------------------------------------------------


def _patch_run(agent: Any) -> None:
    if not hasattr(agent, "run") or getattr(agent, "_neatlogs_run_patched", False):
        return
    orig_run = agent.run

    async def patched_run(*args, **kwargs):
        if _agent_span_active.get():
            return await orig_run(*args, **kwargs)
        tracer = get_tracer()
        attrs = _get_agent_attributes(agent)
        up = _user_prompt(args, kwargs)
        if up:
            attrs["input.value"] = up[:10000]

        span = tracer.start_span(name="pydantic_ai.agent.run", attributes=attrs)
        token = attach_as_current(span)
        guard = _agent_span_active.set(True)
        start = time.perf_counter()
        try:
            result = await orig_run(*args, **kwargs)
        except Exception as e:
            _err(span, e); raise
        finally:
            _agent_span_active.reset(guard)
            detach(token)
        _finalize_run_span(span, result, (time.perf_counter() - start) * 1000)
        return result

    agent.run = patched_run
    agent._neatlogs_run_patched = True


# ---------------------------------------------------------------------------
# Agent.run_sync
# ---------------------------------------------------------------------------


def _patch_run_sync(agent: Any) -> None:
    if not hasattr(agent, "run_sync") or getattr(agent, "_neatlogs_run_sync_patched", False):
        return
    orig_run_sync = agent.run_sync

    def patched_run_sync(*args, **kwargs):
        if _agent_span_active.get():
            return orig_run_sync(*args, **kwargs)
        tracer = get_tracer()
        attrs = _get_agent_attributes(agent)
        up = _user_prompt(args, kwargs)
        if up:
            attrs["input.value"] = up[:10000]

        span = tracer.start_span(name="pydantic_ai.agent.run_sync", attributes=attrs)
        token = attach_as_current(span)
        guard = _agent_span_active.set(True)
        start = time.perf_counter()
        try:
            result = orig_run_sync(*args, **kwargs)
        except Exception as e:
            _err(span, e); raise
        finally:
            _agent_span_active.reset(guard)
            detach(token)
        _finalize_run_span(span, result, (time.perf_counter() - start) * 1000)
        return result

    agent.run_sync = patched_run_sync
    agent._neatlogs_run_sync_patched = True


# ---------------------------------------------------------------------------
# Agent.run_stream (async context manager)
# ---------------------------------------------------------------------------


def _patch_run_stream(agent: Any) -> None:
    if not hasattr(agent, "run_stream") or getattr(agent, "_neatlogs_run_stream_patched", False):
        return
    orig_run_stream = agent.run_stream

    def patched_run_stream(*args, **kwargs):
        # run_stream() returns an async context manager (not a coroutine).
        if _agent_span_active.get():
            return orig_run_stream(*args, **kwargs)
        attrs = _get_agent_attributes(agent)
        up = _user_prompt(args, kwargs)
        if up:
            attrs["input.value"] = up[:10000]
        cm = orig_run_stream(*args, **kwargs)
        return _StreamCtxWrapper(cm, attrs)

    agent.run_stream = patched_run_stream
    agent._neatlogs_run_stream_patched = True


class _StreamCtxWrapper:
    """Wraps the async context manager returned by Agent.run_stream()."""

    def __init__(self, cm: Any, attrs: dict):
        self._cm = cm
        self._attrs = attrs
        self._span = None
        self._token = None
        self._guard = None
        self._start = None
        self._result = None

    async def __aenter__(self):
        tracer = get_tracer()
        self._span = tracer.start_span(name="pydantic_ai.agent.run_stream", attributes=self._attrs)
        self._span.set_attribute("neatlogs.llm.is_streaming", True)
        self._token = attach_as_current(self._span)
        self._guard = _agent_span_active.set(True)
        self._start = time.perf_counter()
        self._result = await self._cm.__aenter__()
        return self._result

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            return await self._cm.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            if self._guard is not None:
                _agent_span_active.reset(self._guard)
            if self._token:
                detach(self._token)
            duration_ms = (time.perf_counter() - self._start) * 1000
            if exc_type:
                self._span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
                self._span.set_status(StatusCode.ERROR, str(exc_val))
                if exc_val:
                    self._span.record_exception(exc_val)
                self._span.end()
            else:
                _finalize_run_span(self._span, self._result, duration_ms)

    def __getattr__(self, name):
        return getattr(self._cm, name)


# ---------------------------------------------------------------------------
# Agent.iter (graph iteration → async context manager yielding an AgentRun)
# ---------------------------------------------------------------------------


def _patch_iter(agent: Any) -> None:
    if not hasattr(agent, "iter") or getattr(agent, "_neatlogs_iter_patched", False):
        return
    orig_iter = agent.iter

    def patched_iter(*args, **kwargs):
        if _agent_span_active.get():
            return orig_iter(*args, **kwargs)
        attrs = _get_agent_attributes(agent)
        up = _user_prompt(args, kwargs)
        if up:
            attrs["input.value"] = up[:10000]
        cm = orig_iter(*args, **kwargs)
        return _IterCtxWrapper(cm, attrs)

    agent.iter = patched_iter
    agent._neatlogs_iter_patched = True


class _IterCtxWrapper:
    """Wraps the async context manager returned by Agent.iter()."""

    def __init__(self, cm: Any, attrs: dict):
        self._cm = cm
        self._attrs = attrs
        self._span = None
        self._token = None
        self._guard = None
        self._start = None
        self._run = None

    async def __aenter__(self):
        tracer = get_tracer()
        self._span = tracer.start_span(name="pydantic_ai.agent.iter", attributes=self._attrs)
        self._token = attach_as_current(self._span)
        self._guard = _agent_span_active.set(True)
        self._start = time.perf_counter()
        self._run = await self._cm.__aenter__()
        return self._run

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            return await self._cm.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            if self._guard is not None:
                _agent_span_active.reset(self._guard)
            if self._token:
                detach(self._token)
            duration_ms = (time.perf_counter() - self._start) * 1000
            if exc_type:
                self._span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
                self._span.set_status(StatusCode.ERROR, str(exc_val))
                if exc_val:
                    self._span.record_exception(exc_val)
                self._span.end()
            else:
                result = getattr(self._run, "result", None)
                _finalize_run_span(self._span, result, duration_ms)

    def __getattr__(self, name):
        return getattr(self._cm, name)


# ---------------------------------------------------------------------------
# Class-level hooks: LLM (Model.request) + TOOL (FunctionToolset.call_tool)
# ---------------------------------------------------------------------------


def _install_class_hooks() -> None:
    global _CLASS_HOOKS_INSTALLED
    if _CLASS_HOOKS_INSTALLED:
        return
    _CLASS_HOOKS_INSTALLED = True
    _patch_model_class()
    _patch_toolset_class()


def _patch_agent_model(agent: Any) -> None:
    """
    Patch the concrete model class of this agent's model instance.

    Concrete models (TestModel, OpenAIModel, ...) override ``request`` /
    ``request_stream`` on their own class, which shadows the base ``Model``
    methods. Patch the actual class so per-request LLM spans are emitted.
    """
    model = getattr(agent, "model", None)
    if model is None:
        return
    _patch_model_class(type(model))


def _patch_model_class(model_cls=None) -> None:
    try:
        from pydantic_ai.models import Model
    except Exception:
        return
    target = model_cls or Model
    # Use __dict__ (not getattr) so a subclass isn't considered "patched" just
    # because the base Model class carries the flag via inheritance.
    if target.__dict__.get("_neatlogs_patched", False):
        return
    # Only patch a subclass if it actually defines its own request method;
    # otherwise it inherits an already-patched base.
    if model_cls is not None and "request" not in target.__dict__ and "request_stream" not in target.__dict__:
        return

    def _set_request_inputs(span, messages):
        if not messages:
            return
        idx = 0
        for msg in messages:
            parts = getattr(msg, "parts", None) or []
            for part in parts:
                kind = getattr(part, "part_kind", "")
                if kind in ("system-prompt", "user-prompt"):
                    role = "system" if kind == "system-prompt" else "user"
                    content = getattr(part, "content", "")
                    span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", role)
                    span.set_attribute(
                        f"neatlogs.llm.input_messages.{idx}.content",
                        (content if isinstance(content, str) else serialize(content))[:10000],
                    )
                    idx += 1

    def _finalize_request(span, response):
        if response is not None:
            parts = getattr(response, "parts", None) or []
            text_parts = []
            j = 0
            for part in parts:
                kind = getattr(part, "part_kind", "")
                if kind == "text":
                    text_parts.append(getattr(part, "content", ""))
                elif kind == "tool-call":
                    name = getattr(part, "tool_name", "")
                    args = getattr(part, "args", None)
                    tc_id = getattr(part, "tool_call_id", "")
                    if name:
                        span.set_attribute(f"neatlogs.llm.tool_calls.{j}.name", name)
                    if args is not None:
                        span.set_attribute(f"neatlogs.llm.tool_calls.{j}.arguments", args if isinstance(args, str) else serialize(args))
                    if tc_id:
                        span.set_attribute(f"neatlogs.llm.tool_calls.{j}.id", str(tc_id))
                    j += 1
                elif kind == "thinking":
                    thinking = getattr(part, "content", "")
                    if thinking:
                        span.set_attribute("neatlogs.llm.output_messages.0.thinking", thinking)
            if text_parts:
                span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
                span.set_attribute("neatlogs.llm.output_messages.0.content", "".join(text_parts)[:10000])

            model_name = getattr(response, "model_name", None)
            if model_name:
                span.set_attribute("neatlogs.llm.model_name", str(model_name))

            usage = getattr(response, "usage", None)
            if usage:
                if getattr(usage, "input_tokens", None):
                    span.set_attribute("neatlogs.llm.token_count.prompt", usage.input_tokens)
                if getattr(usage, "output_tokens", None):
                    span.set_attribute("neatlogs.llm.token_count.completion", usage.output_tokens)
                if getattr(usage, "total_tokens", None):
                    span.set_attribute("neatlogs.llm.token_count.total", usage.total_tokens)
        span.set_status(StatusCode.OK)
        span.end()

    def _model_name(self):
        return getattr(self, "model_name", None) or str(self)

    if hasattr(target, "request") and "request" in target.__dict__ or (model_cls is None and hasattr(target, "request")):
        orig_request = target.request

        async def patched_request(self, messages, *a, **k):
            tracer = get_tracer()
            attrs = {"neatlogs.span.kind": "LLM", "neatlogs.llm.model_name": str(_model_name(self))}
            span = tracer.start_span(name="pydantic_ai.model.request", attributes=attrs)
            _set_request_inputs(span, messages)
            token = attach_as_current(span)
            try:
                response = await orig_request(self, messages, *a, **k)
            except Exception as e:
                _err(span, e); raise
            finally:
                detach(token)
            _finalize_request(span, response)
            return response

        target.request = patched_request

    if (hasattr(target, "request_stream") and "request_stream" in target.__dict__) or (model_cls is None and hasattr(target, "request_stream")):
        orig_request_stream = target.request_stream

        def patched_request_stream(self, messages, *a, **k):
            # request_stream returns an async context manager yielding a streamed response.
            tracer = get_tracer()
            attrs = {
                "neatlogs.span.kind": "LLM",
                "neatlogs.llm.model_name": str(_model_name(self)),
                "neatlogs.llm.is_streaming": True,
            }
            cm = orig_request_stream(self, messages, *a, **k)
            return _ModelStreamCtx(cm, attrs, messages, _set_request_inputs)

        target.request_stream = patched_request_stream

    target._neatlogs_patched = True


class _ModelStreamCtx:
    def __init__(self, cm, attrs, messages, set_inputs):
        self._cm = cm
        self._attrs = attrs
        self._messages = messages
        self._set_inputs = set_inputs
        self._span = None
        self._token = None
        self._streamed = None

    async def __aenter__(self):
        self._span = get_tracer().start_span(name="pydantic_ai.model.request_stream", attributes=self._attrs)
        self._set_inputs(self._span, self._messages)
        self._token = attach_as_current(self._span)
        self._streamed = await self._cm.__aenter__()
        return self._streamed

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            return await self._cm.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            if self._token:
                detach(self._token)
            if exc_type:
                self._span.set_status(StatusCode.ERROR, str(exc_val))
                if exc_val:
                    self._span.record_exception(exc_val)
            else:
                # try to capture final usage/text from the streamed response
                try:
                    text = self._streamed.get() if hasattr(self._streamed, "get") else None
                    if text:
                        self._span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
                        self._span.set_attribute("neatlogs.llm.output_messages.0.content", str(text)[:10000])
                except Exception:
                    pass
                usage = getattr(self._streamed, "usage", None)
                if callable(usage):
                    try:
                        usage = usage()
                    except Exception:
                        usage = None
                if usage:
                    if getattr(usage, "input_tokens", None):
                        self._span.set_attribute("neatlogs.llm.token_count.prompt", usage.input_tokens)
                    if getattr(usage, "output_tokens", None):
                        self._span.set_attribute("neatlogs.llm.token_count.completion", usage.output_tokens)
                self._span.set_status(StatusCode.OK)
            self._span.end()

    def __getattr__(self, name):
        return getattr(self._cm, name)


def _patch_toolset_class() -> None:
    """Patch FunctionToolset.call_tool (the standard user toolset) for TOOL spans."""
    try:
        from pydantic_ai.toolsets.function import FunctionToolset
    except Exception:
        return
    if getattr(FunctionToolset, "_neatlogs_patched", False):
        return
    if not hasattr(FunctionToolset, "call_tool"):
        return

    orig_call_tool = FunctionToolset.call_tool

    async def patched_call_tool(self, name, tool_args, ctx, tool, *a, **k):
        tracer = get_tracer()
        attrs = {"neatlogs.span.kind": "TOOL", "neatlogs.tool.name": str(name)}
        if tool_args is not None:
            attrs["input.value"] = tool_args if isinstance(tool_args, str) else serialize(tool_args)
        span = tracer.start_span(name=f"pydantic_ai.tool.{name}", attributes=attrs)
        token = attach_as_current(span)
        try:
            result = await orig_call_tool(self, name, tool_args, ctx, tool, *a, **k)
        except Exception as e:
            _err(span, e); raise
        finally:
            detach(token)
        if result is not None:
            span.set_attribute("output.value", str(result)[:10000])
        span.set_status(StatusCode.OK)
        span.end()
        return result

    FunctionToolset.call_tool = patched_call_tool
    FunctionToolset._neatlogs_patched = True


def _err(span: Any, e: Exception) -> None:
    span.set_status(StatusCode.ERROR, str(e))
    span.record_exception(e)
    span.end()
