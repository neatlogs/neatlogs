"""
Neatlogs Agno wrapper.

Usage:
    >>> import neatlogs
    >>> from agno.agent import Agent
    >>> agent = neatlogs.wrap(Agent(model=OpenAIChat(id="gpt-4o"), tools=[...]))
    >>> result = agent.run("Hello")            # sync
    >>> result = await agent.arun("Hello")     # async
    >>> for ev in agent.run("Hi", stream=True): ...   # streaming

Span hierarchy:
    AGENT / TEAM / WORKFLOW (run/arun, incl. streaming)
      ↳ LLM   (model.invoke / ainvoke / invoke_stream / ainvoke_stream)
      ↳ TOOL  (FunctionCall.execute / aexecute)

Tool and model spans are installed once at the class level (idempotent), so they
nest under whichever agent/team/workflow run is active — including tools and
models added after wrap().
"""

import time
from typing import Any

from opentelemetry.trace import StatusCode

from ._wrap_utils import attach_as_current, detach, get_tracer, serialize

_CLASS_HOOKS_INSTALLED = False


def wrap_agno(entity: Any) -> Any:
    """
    Wrap an Agno Agent, Team, or Workflow instance.
    Patches run()/arun() and installs class-level TOOL + LLM hooks.
    Returns the same instance.
    """
    # Class-level hooks (tools + model) are global; install once.
    _install_class_hooks()

    if getattr(entity, "_neatlogs_patched", False):
        return entity

    cls_name = type(entity).__name__
    if cls_name == "Workflow" or (hasattr(entity, "run") and _looks_like_workflow(entity)):
        _patch_workflow(entity)
    elif cls_name in ("Agent", "Team") or hasattr(entity, "run") or hasattr(entity, "arun"):
        _patch_agent(entity)

    entity._neatlogs_patched = True
    return entity


def _looks_like_workflow(entity: Any) -> bool:
    return type(entity).__name__ == "Workflow" or (
        hasattr(entity, "steps") and not hasattr(entity, "model")
    )


# ---------------------------------------------------------------------------
# Agent / Team (AGENT span)
# ---------------------------------------------------------------------------


def _get_agent_attributes(agent: Any) -> dict:
    kind = "TEAM" if type(agent).__name__ == "Team" else "AGENT"
    attrs = {"neatlogs.span.kind": kind if kind != "TEAM" else "AGENT"}
    if kind == "TEAM":
        attrs["neatlogs.agent.type"] = "team"

    name = getattr(agent, "name", None)
    if name:
        attrs["neatlogs.agent.name"] = name

    role = getattr(agent, "role", None)
    if role:
        attrs["neatlogs.agent.role"] = role

    model = getattr(agent, "model", None)
    if model is not None:
        model_id = getattr(model, "id", None) or getattr(model, "model", None) or str(model)
        attrs["neatlogs.llm.model_name"] = str(model_id)
        provider = getattr(model, "provider", None)
        if provider:
            attrs["neatlogs.llm.provider"] = str(provider)

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
    """Extract token usage from an Agno RunOutput (metrics is a Metrics dataclass)."""
    attrs = {}
    metrics = getattr(response, "metrics", None)
    if metrics is None:
        return attrs

    def _val(name):
        v = getattr(metrics, name, None)
        if v is None and isinstance(metrics, dict):
            v = metrics.get(name)
        if isinstance(v, list):
            v = sum(x for x in v if isinstance(x, (int, float)))
        return v

    prompt = _val("input_tokens") or _val("prompt_tokens")
    completion = _val("output_tokens") or _val("completion_tokens")
    total = _val("total_tokens")
    cache_read = _val("cache_read_tokens")
    cache_write = _val("cache_write_tokens")
    reasoning = _val("reasoning_tokens")

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
    if reasoning:
        attrs["neatlogs.llm.token_count.reasoning"] = reasoning
    return attrs


def _finalize_agent_span(span: Any, response: Any, duration_ms: float) -> None:
    if response is None:
        span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
        span.set_status(StatusCode.OK)
        span.end()
        return

    content = getattr(response, "content", None)
    if content:
        span.set_attribute("output.value", str(content)[:10000])

    for attr_name, value in _extract_usage(response).items():
        span.set_attribute(attr_name, value)

    tools_used = getattr(response, "tools", None)
    if tools_used:
        for i, tool in enumerate(tools_used):
            name = (
                getattr(tool, "tool_name", None)
                or getattr(tool, "name", None)
                or getattr(tool, "function_name", None)
                or (tool.get("tool_name") or tool.get("name") if isinstance(tool, dict) else None)
            )
            if name:
                span.set_attribute(f"neatlogs.llm.tool_calls.{i}.name", str(name))
            args = (
                getattr(tool, "tool_args", None)
                or getattr(tool, "arguments", None)
                or (tool.get("tool_args") or tool.get("arguments") if isinstance(tool, dict) else None)
            )
            if args:
                span.set_attribute(f"neatlogs.llm.tool_calls.{i}.arguments", args if isinstance(args, str) else serialize(args))

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


def _input_value(args, kwargs) -> str:
    message = None
    if args:
        message = args[0]
    else:
        message = kwargs.get("input", kwargs.get("message", kwargs.get("prompt")))
    if message is None:
        return ""
    return message if isinstance(message, str) else serialize(message)


def _patch_agent(agent: Any) -> None:
    span_kind_name = type(agent).__name__.lower()

    if hasattr(agent, "run"):
        orig_run = agent.run

        def patched_run(*args, **kwargs):
            tracer = get_tracer()
            attrs = _get_agent_attributes(agent)
            iv = _input_value(args, kwargs)
            if iv:
                attrs["input.value"] = iv[:10000]
            is_stream = bool(kwargs.get("stream"))
            if is_stream:
                attrs["neatlogs.llm.is_streaming"] = True

            span = tracer.start_span(name=f"agno.{span_kind_name}.run", attributes=attrs)
            token = attach_as_current(span)
            start = time.perf_counter()

            if is_stream:
                # run(stream=True) returns an iterator of events; keep the span
                # active across iteration so LLM/TOOL children nest, finalize at end.
                try:
                    iterator = orig_run(*args, **kwargs)
                except Exception as e:
                    _err(span, e)
                    detach(token)
                    raise
                return _AgnoStreamIter(iterator, span, token, start, sync=True)

            try:
                result = orig_run(*args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            finally:
                detach(token)
            _finalize_agent_span(span, result, (time.perf_counter() - start) * 1000)
            return result

        agent.run = patched_run

    if hasattr(agent, "arun"):
        orig_arun = agent.arun

        async def patched_arun(*args, **kwargs):
            tracer = get_tracer()
            attrs = _get_agent_attributes(agent)
            iv = _input_value(args, kwargs)
            if iv:
                attrs["input.value"] = iv[:10000]
            is_stream = bool(kwargs.get("stream"))
            if is_stream:
                attrs["neatlogs.llm.is_streaming"] = True

            span = tracer.start_span(name=f"agno.{span_kind_name}.arun", attributes=attrs)
            token = attach_as_current(span)
            start = time.perf_counter()

            if is_stream:
                try:
                    aiter = orig_arun(*args, **kwargs)
                except Exception as e:
                    _err(span, e)
                    detach(token)
                    raise
                return _AgnoAsyncStreamIter(aiter, span, token, start)

            try:
                result = await orig_arun(*args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            finally:
                detach(token)
            _finalize_agent_span(span, result, (time.perf_counter() - start) * 1000)
            return result

        agent.arun = patched_arun


# ---------------------------------------------------------------------------
# Workflow (WORKFLOW span)
# ---------------------------------------------------------------------------


def _patch_workflow(workflow: Any) -> None:
    def _attrs():
        attrs = {"neatlogs.span.kind": "WORKFLOW"}
        name = getattr(workflow, "name", None)
        if name:
            attrs["neatlogs.workflow.name"] = name
        return attrs

    if hasattr(workflow, "run"):
        orig_run = workflow.run

        def patched_run(*args, **kwargs):
            tracer = get_tracer()
            attrs = _attrs()
            iv = _input_value(args, kwargs)
            if iv:
                attrs["input.value"] = iv[:10000]
            is_stream = bool(kwargs.get("stream"))
            span = tracer.start_span(name="agno.workflow.run", attributes=attrs)
            token = attach_as_current(span)
            start = time.perf_counter()
            if is_stream:
                try:
                    iterator = orig_run(*args, **kwargs)
                except Exception as e:
                    _err(span, e); detach(token); raise
                return _AgnoStreamIter(iterator, span, token, start, sync=True, workflow=True)
            try:
                result = orig_run(*args, **kwargs)
            except Exception as e:
                _err(span, e); raise
            finally:
                detach(token)
            _finalize_workflow_span(span, result, (time.perf_counter() - start) * 1000)
            return result

        workflow.run = patched_run

    if hasattr(workflow, "arun"):
        orig_arun = workflow.arun

        async def patched_arun(*args, **kwargs):
            tracer = get_tracer()
            attrs = _attrs()
            iv = _input_value(args, kwargs)
            if iv:
                attrs["input.value"] = iv[:10000]
            is_stream = bool(kwargs.get("stream"))
            span = tracer.start_span(name="agno.workflow.arun", attributes=attrs)
            token = attach_as_current(span)
            start = time.perf_counter()
            if is_stream:
                try:
                    aiter = orig_arun(*args, **kwargs)
                except Exception as e:
                    _err(span, e); detach(token); raise
                return _AgnoAsyncStreamIter(aiter, span, token, start, workflow=True)
            try:
                result = await orig_arun(*args, **kwargs)
            except Exception as e:
                _err(span, e); raise
            finally:
                detach(token)
            _finalize_workflow_span(span, result, (time.perf_counter() - start) * 1000)
            return result

        workflow.arun = patched_arun


def _finalize_workflow_span(span: Any, result: Any, duration_ms: float) -> None:
    if result is not None:
        content = getattr(result, "content", None)
        span.set_attribute("output.value", str(content if content is not None else result)[:10000])
        for attr_name, value in _extract_usage(result).items():
            span.set_attribute(attr_name, value)
    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()


# ---------------------------------------------------------------------------
# Streaming iterators (keep parent span active across iteration)
# ---------------------------------------------------------------------------


class _AgnoStreamIter:
    def __init__(self, iterator, span, token, start, sync=True, workflow=False):
        self._it = iter(iterator)
        self._span = span
        self._token = token
        self._start = start
        self._workflow = workflow
        self._last = None
        self._done = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            ev = next(self._it)
        except StopIteration:
            self._finalize()
            raise
        except Exception as e:
            self._finalize(error=e)
            raise
        self._last = ev
        return ev

    def _finalize(self, error=None):
        if self._done:
            return
        self._done = True
        detach(self._token)
        duration_ms = (time.perf_counter() - self._start) * 1000
        if error is not None:
            self._span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
            self._span.set_status(StatusCode.ERROR, str(error))
            self._span.record_exception(error)
            self._span.end()
            return
        if self._workflow:
            _finalize_workflow_span(self._span, self._last, duration_ms)
        else:
            _finalize_agent_span(self._span, self._last, duration_ms)

    def __getattr__(self, name):
        return getattr(self._it, name)


class _AgnoAsyncStreamIter:
    def __init__(self, aiterable, span, token, start, workflow=False):
        self._aiterable = aiterable
        self._aiter = None
        self._span = span
        self._token = token
        self._start = start
        self._workflow = workflow
        self._last = None
        self._done = False

    def __aiter__(self):
        self._aiter = self._aiterable.__aiter__()
        return self

    async def __anext__(self):
        if self._aiter is None:
            self._aiter = self._aiterable.__aiter__()
        try:
            ev = await self._aiter.__anext__()
        except StopAsyncIteration:
            self._finalize()
            raise
        except Exception as e:
            self._finalize(error=e)
            raise
        self._last = ev
        return ev

    def _finalize(self, error=None):
        if self._done:
            return
        self._done = True
        detach(self._token)
        duration_ms = (time.perf_counter() - self._start) * 1000
        if error is not None:
            self._span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
            self._span.set_status(StatusCode.ERROR, str(error))
            self._span.record_exception(error)
            self._span.end()
            return
        if self._workflow:
            _finalize_workflow_span(self._span, self._last, duration_ms)
        else:
            _finalize_agent_span(self._span, self._last, duration_ms)

    def __getattr__(self, name):
        return getattr(self._aiterable, name)


# ---------------------------------------------------------------------------
# Class-level hooks: TOOL (FunctionCall) + LLM (Model)
# ---------------------------------------------------------------------------


def _install_class_hooks() -> None:
    global _CLASS_HOOKS_INSTALLED
    if _CLASS_HOOKS_INSTALLED:
        return
    _CLASS_HOOKS_INSTALLED = True
    _patch_function_call_class()
    _patch_model_class()


def _patch_function_call_class() -> None:
    try:
        from agno.tools.function import FunctionCall
    except Exception:
        return
    if getattr(FunctionCall, "_neatlogs_patched", False):
        return

    def _tool_attrs(self):
        attrs = {"neatlogs.span.kind": "TOOL"}
        fn = getattr(self, "function", None)
        name = getattr(fn, "name", None) if fn else None
        if name:
            attrs["neatlogs.tool.name"] = name
        args = getattr(self, "arguments", None)
        if args:
            attrs["input.value"] = args if isinstance(args, str) else serialize(args)
        call_id = getattr(self, "call_id", None)
        if call_id:
            attrs["neatlogs.tool.call_id"] = str(call_id)
        return attrs, (name or "tool")

    if hasattr(FunctionCall, "execute"):
        orig_execute = FunctionCall.execute

        def patched_execute(self, *a, **k):
            tracer = get_tracer()
            attrs, name = _tool_attrs(self)
            span = tracer.start_span(name=f"agno.tool.{name}", attributes=attrs)
            token = attach_as_current(span)
            try:
                result = orig_execute(self, *a, **k)
            except Exception as e:
                _err(span, e); raise
            finally:
                detach(token)
            _finalize_tool(span, self, result)
            return result

        FunctionCall.execute = patched_execute

    if hasattr(FunctionCall, "aexecute"):
        orig_aexecute = FunctionCall.aexecute

        async def patched_aexecute(self, *a, **k):
            tracer = get_tracer()
            attrs, name = _tool_attrs(self)
            span = tracer.start_span(name=f"agno.tool.{name}", attributes=attrs)
            token = attach_as_current(span)
            try:
                result = await orig_aexecute(self, *a, **k)
            except Exception as e:
                _err(span, e); raise
            finally:
                detach(token)
            _finalize_tool(span, self, result)
            return result

        FunctionCall.aexecute = patched_aexecute

    FunctionCall._neatlogs_patched = True


def _finalize_tool(span: Any, fcall: Any, result: Any) -> None:
    # result is a FunctionExecutionResult; fcall.result is also populated
    out = getattr(result, "result", None) if result is not None else None
    if out is None:
        out = getattr(fcall, "result", None)
    if out is not None:
        span.set_attribute("output.value", str(out)[:10000])
    error = getattr(result, "error", None) if result is not None else None
    if error:
        span.set_status(StatusCode.ERROR, str(error))
    else:
        span.set_status(StatusCode.OK)
    span.end()


def _patch_model_class() -> None:
    try:
        from agno.models.base import Model
    except Exception:
        return
    if getattr(Model, "_neatlogs_patched", False):
        return

    def _model_attrs(self):
        attrs = {"neatlogs.span.kind": "LLM"}
        model_id = getattr(self, "id", None) or getattr(self, "model", None)
        if model_id:
            attrs["neatlogs.llm.model_name"] = str(model_id)
        provider = getattr(self, "provider", None)
        if provider:
            attrs["neatlogs.llm.provider"] = str(provider)
        return attrs

    for method in ("invoke", "ainvoke", "invoke_stream", "ainvoke_stream"):
        if not hasattr(Model, method):
            continue
        orig = getattr(Model, method)
        is_async = method.startswith("a")
        is_stream = "stream" in method

        if is_async and is_stream:
            def make(orig=orig):
                async def wrapper(self, *a, **k):
                    tracer = get_tracer()
                    span = tracer.start_span(name="agno.model.invoke", attributes=_model_attrs(self))
                    token = attach_as_current(span)
                    try:
                        agen = orig(self, *a, **k)
                        async for chunk in agen:
                            yield chunk
                    except Exception as e:
                        _err(span, e); raise
                    finally:
                        detach(token)
                        if span.is_recording():
                            span.set_status(StatusCode.OK); span.end()
                return wrapper
            setattr(Model, method, make())
        elif is_async:
            def make(orig=orig):
                async def wrapper(self, *a, **k):
                    tracer = get_tracer()
                    span = tracer.start_span(name="agno.model.invoke", attributes=_model_attrs(self))
                    token = attach_as_current(span)
                    try:
                        result = await orig(self, *a, **k)
                    except Exception as e:
                        _err(span, e); raise
                    finally:
                        detach(token)
                    _finalize_model(span, result)
                    return result
                return wrapper
            setattr(Model, method, make())
        elif is_stream:
            def make(orig=orig):
                def wrapper(self, *a, **k):
                    tracer = get_tracer()
                    span = tracer.start_span(name="agno.model.invoke", attributes=_model_attrs(self))
                    token = attach_as_current(span)
                    try:
                        gen = orig(self, *a, **k)
                        for chunk in gen:
                            yield chunk
                    except Exception as e:
                        _err(span, e); raise
                    finally:
                        detach(token)
                        if span.is_recording():
                            span.set_status(StatusCode.OK); span.end()
                return wrapper
            setattr(Model, method, make())
        else:
            def make(orig=orig):
                def wrapper(self, *a, **k):
                    tracer = get_tracer()
                    span = tracer.start_span(name="agno.model.invoke", attributes=_model_attrs(self))
                    token = attach_as_current(span)
                    try:
                        result = orig(self, *a, **k)
                    except Exception as e:
                        _err(span, e); raise
                    finally:
                        detach(token)
                    _finalize_model(span, result)
                    return result
                return wrapper
            setattr(Model, method, make())

    Model._neatlogs_patched = True


def _finalize_model(span: Any, result: Any) -> None:
    """result is a ModelResponse (non-streaming)."""
    if result is not None:
        content = getattr(result, "content", None)
        if content:
            span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
            span.set_attribute("neatlogs.llm.output_messages.0.content", str(content)[:10000])

        for src, dst in (
            ("input_tokens", "prompt"),
            ("output_tokens", "completion"),
            ("total_tokens", "total"),
            ("cache_read_tokens", "cache_read"),
            ("cache_write_tokens", "cache_write"),
            ("reasoning_tokens", "reasoning"),
        ):
            v = getattr(result, src, None)
            if v:
                span.set_attribute(f"neatlogs.llm.token_count.{dst}", v)

        tool_calls = getattr(result, "tool_calls", None)
        if tool_calls:
            for j, tc in enumerate(tool_calls):
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name") if fn else getattr(tc, "name", None)
                args = fn.get("arguments") if fn else getattr(tc, "arguments", None)
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if name:
                    span.set_attribute(f"neatlogs.llm.tool_calls.{j}.name", name)
                if args:
                    span.set_attribute(f"neatlogs.llm.tool_calls.{j}.arguments", args if isinstance(args, str) else serialize(args))
                if tc_id:
                    span.set_attribute(f"neatlogs.llm.tool_calls.{j}.id", str(tc_id))
    span.set_status(StatusCode.OK)
    span.end()


def _err(span: Any, e: Exception) -> None:
    span.set_status(StatusCode.ERROR, str(e))
    span.record_exception(e)
    span.end()
