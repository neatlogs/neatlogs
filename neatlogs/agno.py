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
    is_team = type(agent).__name__ == "Team"
    attrs = {"neatlogs.span.kind": "agent"}  # Team is rendered as an AGENT span
    if is_team:
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
        span.set_attribute("output.value", str(content))

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

    # Concrete model classes (OpenAIChat, Claude, …) override invoke/ainvoke, so the
    # base-Model patch is shadowed. Patch the agent's actual model class too.
    model = getattr(agent, "model", None)
    if model is not None:
        try:
            _patch_model_class(type(model))
        except Exception:
            pass

    if hasattr(agent, "run"):
        orig_run = agent.run

        def patched_run(*args, **kwargs):
            tracer = get_tracer()
            attrs = _get_agent_attributes(agent)
            iv = _input_value(args, kwargs)
            if iv:
                attrs["input.value"] = iv
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
                attrs["input.value"] = iv
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
        attrs = {"neatlogs.span.kind": "workflow"}
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
                attrs["input.value"] = iv
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
                attrs["input.value"] = iv
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
        span.set_attribute("output.value", str(content if content is not None else result))
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
        attrs = {"neatlogs.span.kind": "tool"}
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
        span.set_attribute("output.value", str(out))
    error = getattr(result, "error", None) if result is not None else None
    if error:
        span.set_status(StatusCode.ERROR, str(error))
    else:
        span.set_status(StatusCode.OK)
    span.end()


def _patch_model_class(target=None) -> None:
    """
    Install LLM-span hooks on an Agno model class.

    Concrete model classes (e.g. OpenAIChat, Claude) OVERRIDE invoke/ainvoke, so
    patching only the base `Model` is shadowed by the subclass method. We therefore
    patch the methods on whichever class actually DEFINES them — by default the base
    `Model`, and additionally the concrete class passed as `target` (the agent's model
    class), patching only methods present in that class's own __dict__.
    """
    try:
        from agno.models.base import Model
    except Exception:
        return

    cls = target if target is not None else Model
    if getattr(cls, "_neatlogs_patched", False):
        return

    def _model_attrs(self, call_kwargs=None):
        attrs = {"neatlogs.span.kind": "llm"}
        model_id = getattr(self, "id", None) or getattr(self, "model", None)
        if model_id:
            attrs["neatlogs.llm.model_name"] = str(model_id)
        provider = getattr(self, "provider", None)
        if provider:
            attrs["neatlogs.llm.provider"] = str(provider)
        # Input messages: agno passes them as the `messages` kwarg (list[Message]).
        msgs = (call_kwargs or {}).get("messages")
        if isinstance(msgs, list) and msgs:
            collected = []
            for i, m in enumerate(msgs):
                role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None) or ""
                content = getattr(m, "content", None)
                if content is None and isinstance(m, dict):
                    content = m.get("content")
                content_str = content if isinstance(content, str) else serialize(content)
                if role:
                    attrs[f"neatlogs.llm.input_messages.{i}.role"] = role
                if content:
                    attrs[f"neatlogs.llm.input_messages.{i}.content"] = content_str
                collected.append({"role": role or "user", "content": content_str})
            if collected:
                # Flat input blob the UI renders. Backend enforces the 1 MB cap.
                attrs["neatlogs.llm.input"] = serialize({"messages": collected})
        return attrs

    # agno's high-level response()/aresponse() call the *_with_retry wrappers, which
    # are defined on the base Model and internally call invoke/ainvoke. Concrete model
    # classes (OpenAIChat, …) override the bare invoke/ainvoke but NOT the retry
    # wrappers, so patching the retry methods on base Model reliably captures every
    # model call regardless of provider. Fall back to bare invoke if retry absent.
    candidates = [
        "_invoke_with_retry",
        "_ainvoke_with_retry",
        "_invoke_stream_with_retry",
        "_ainvoke_stream_with_retry",
    ]
    if not any(m in cls.__dict__ for m in candidates):
        candidates = ["invoke", "ainvoke", "invoke_stream", "ainvoke_stream"]

    for method in candidates:
        # Only patch methods this class DEFINES itself (so we hit the real override,
        # and don't double-wrap an inherited-then-already-patched base method).
        if method not in cls.__dict__:
            continue
        orig = getattr(cls, method)
        is_async = method.startswith("a") or method.startswith("_a")
        is_stream = "stream" in method

        if is_async and is_stream:
            def make(orig=orig):
                async def wrapper(self, *a, **k):
                    tracer = get_tracer()
                    span = tracer.start_span(name="agno.model.invoke", attributes=_model_attrs(self, k))
                    token = attach_as_current(span)
                    chunks = []
                    try:
                        agen = orig(self, *a, **k)
                        async for chunk in agen:
                            chunks.append(chunk)
                            yield chunk
                    except Exception as e:
                        _err(span, e); raise
                    finally:
                        detach(token)
                        if span.is_recording():
                            _finalize_model_stream(span, chunks)
                return wrapper
            setattr(cls, method, make())
        elif is_async:
            def make(orig=orig):
                async def wrapper(self, *a, **k):
                    tracer = get_tracer()
                    span = tracer.start_span(name="agno.model.invoke", attributes=_model_attrs(self, k))
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
            setattr(cls, method, make())
        elif is_stream:
            def make(orig=orig):
                def wrapper(self, *a, **k):
                    tracer = get_tracer()
                    span = tracer.start_span(name="agno.model.invoke", attributes=_model_attrs(self, k))
                    token = attach_as_current(span)
                    chunks = []
                    try:
                        gen = orig(self, *a, **k)
                        for chunk in gen:
                            chunks.append(chunk)
                            yield chunk
                    except Exception as e:
                        _err(span, e); raise
                    finally:
                        detach(token)
                        if span.is_recording():
                            _finalize_model_stream(span, chunks)
                return wrapper
            setattr(cls, method, make())
        else:
            def make(orig=orig):
                def wrapper(self, *a, **k):
                    tracer = get_tracer()
                    span = tracer.start_span(name="agno.model.invoke", attributes=_model_attrs(self, k))
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
            setattr(cls, method, make())

    cls._neatlogs_patched = True


def _finalize_model_stream(span: Any, chunks: list) -> None:
    """
    Finalize a streamed model call. Agno yields ModelResponse deltas: accumulate
    text content across chunks and take token usage from whichever chunk carries
    `response_usage` (usually the last). Mirrors _finalize_model's output so
    streaming LLM spans get the same I/O + tokens as non-streaming.
    """
    text_parts = []
    tool_calls = []
    usage = None
    for ch in chunks or []:
        c = getattr(ch, "content", None)
        if c:
            text_parts.append(c if isinstance(c, str) else str(c))
        ru = getattr(ch, "response_usage", None)
        if ru is not None:
            usage = ru
        tcs = getattr(ch, "tool_calls", None)
        if tcs:
            for tc in tcs:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name") if fn else getattr(tc, "name", None)
                args = fn.get("arguments") if fn else getattr(tc, "arguments", None)
                if name and not any(x["name"] == name for x in tool_calls):
                    tool_calls.append({"name": name, "arguments": args})

    out_text = "".join(text_parts)
    if not out_text and tool_calls:
        out_text = "\n".join(
            f"{c['name']}({c['arguments'] if isinstance(c['arguments'], str) else serialize(c['arguments'])})"
            for c in tool_calls
        )
    if out_text or tool_calls:
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        span.set_attribute("neatlogs.llm.output_messages.0.content", out_text)
        out_blob = {"role": "assistant", "content": out_text}
        if tool_calls:
            out_blob["tool_calls"] = tool_calls
        span.set_attribute("neatlogs.llm.output", serialize(out_blob))
    if usage is not None:
        for src, dst in (
            ("input_tokens", "prompt"),
            ("output_tokens", "completion"),
            ("total_tokens", "total"),
            ("cache_read_tokens", "cache_read"),
            ("reasoning_tokens", "reasoning"),
        ):
            v = getattr(usage, src, None)
            if v is None and isinstance(usage, dict):
                v = usage.get(src)
            if v:
                span.set_attribute(f"neatlogs.llm.token_count.{dst}", v)
    span.set_status(StatusCode.OK)
    span.end()


def _finalize_model(span: Any, result: Any) -> None:
    """result is a ModelResponse (non-streaming)."""
    if result is not None:
        content = getattr(result, "content", None)

        # Collect tool calls first so we can include them in the output blob.
        tool_calls = getattr(result, "tool_calls", None)
        collected_calls = []
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
                collected_calls.append({"name": name, "arguments": args})

        # Output message + flat blob the UI renders. When the model returns a
        # tool-call (no text content), summarise the call into content so the
        # span output isn't blank.
        if content:
            out_text = str(content)
        elif collected_calls:
            out_text = "\n".join(
                f"{c['name']}({c['arguments'] if isinstance(c['arguments'], str) else serialize(c['arguments'])})"
                for c in collected_calls
            )
        else:
            out_text = ""
        if out_text or collected_calls:
            span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
            span.set_attribute("neatlogs.llm.output_messages.0.content", out_text)
            out_blob = {"role": "assistant", "content": out_text}
            if collected_calls:
                out_blob["tool_calls"] = collected_calls
            span.set_attribute("neatlogs.llm.output", serialize(out_blob))

        # Token usage. Agno's ModelResponse exposes the live usage on
        # `response_usage` (a MessageMetrics); the top-level input_tokens/
        # output_tokens attributes are typically None. Read response_usage first.
        usage = getattr(result, "response_usage", None) or result
        for src, dst in (
            ("input_tokens", "prompt"),
            ("output_tokens", "completion"),
            ("total_tokens", "total"),
            ("cache_read_tokens", "cache_read"),
            ("cache_write_tokens", "cache_write"),
            ("reasoning_tokens", "reasoning"),
        ):
            v = getattr(usage, src, None)
            if v is None and isinstance(usage, dict):
                v = usage.get(src)
            if v:
                span.set_attribute(f"neatlogs.llm.token_count.{dst}", v)
    span.set_status(StatusCode.OK)
    span.end()


def _err(span: Any, e: Exception) -> None:
    span.set_status(StatusCode.ERROR, str(e))
    span.record_exception(e)
    span.end()
