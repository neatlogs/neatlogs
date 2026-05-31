"""
Neatlogs Anthropic wrapper.

Two usage patterns:

  1. Explicit wrap (primary):
     >>> import neatlogs, anthropic
     >>> client = neatlogs.wrap(anthropic.Anthropic())
     >>> client.messages.create(...)

  2. Import replacement (Langfuse-style):
     >>> from neatlogs.anthropic import anthropic
     >>> client = anthropic.Anthropic()
     >>> client.messages.create(...)
"""

import time
from typing import Any, List, Optional

from opentelemetry.trace import StatusCode

from ._wrap_utils import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
    get_tracer,
    is_suppressed,
    serialize,
)

_PATCHED = False
_ORIG_INIT = None
_ORIG_ASYNC_INIT = None


class AnthropicInstrumentor:
    """Instrumentor class for InstrumentationManager integration."""

    def instrument(self, tracer_provider=None):
        _patch_anthropic_module()

    def uninstrument(self):
        _unpatch_anthropic_module()


def wrap_anthropic_client(client: Any) -> Any:
    """
    Wrap an Anthropic client instance. Patches messages (create/stream/parse/
    count_tokens), legacy completions, and beta.messages.
    """
    _patch_messages(client.messages)
    _extra_message_methods(client.messages, is_async=False)
    _patch_legacy_completions(getattr(client, "completions", None), is_async=False)
    beta = getattr(client, "beta", None)
    if beta is not None and getattr(beta, "messages", None) is not None:
        _patch_messages(beta.messages)
        _extra_message_methods(beta.messages, is_async=False)
    return client


def wrap_async_anthropic_client(client: Any) -> Any:
    """Wrap an AsyncAnthropic client instance — full coverage."""
    _patch_async_messages(client.messages)
    _extra_message_methods(client.messages, is_async=True)
    _patch_legacy_completions(getattr(client, "completions", None), is_async=True)
    beta = getattr(client, "beta", None)
    if beta is not None and getattr(beta, "messages", None) is not None:
        _patch_async_messages(beta.messages)
        _extra_message_methods(beta.messages, is_async=True)
    return client


def _patch_messages(messages: Any) -> None:
    if getattr(messages, "_neatlogs_patched", False):
        return

    orig_create = messages.create
    orig_stream = getattr(messages, "stream", None)

    def patched_create(*args, **kwargs):
        if is_suppressed():
            return orig_create(*args, **kwargs)

        model = kwargs.get("model", "")
        input_messages = kwargs.get("messages", [])
        system = kwargs.get("system")
        is_stream = kwargs.get("stream", False)

        tracer = get_tracer()
        span = tracer.start_span(
            name="anthropic.messages.create",
            attributes={
                "neatlogs.span.kind": "LLM",
                "neatlogs.llm.provider": "anthropic",
                "neatlogs.llm.system": "anthropic",
                "neatlogs.llm.model_name": model,
                "neatlogs.llm.is_streaming": is_stream,
            },
        )

        _set_input_attributes(span, input_messages, system, kwargs)

        start = time.perf_counter()

        try:
            response = orig_create(*args, **kwargs)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise

        if is_stream:
            return SyncStreamWrapper(response, span, _finalize_stream)

        duration_ms = (time.perf_counter() - start) * 1000
        _finalize_response(span, response, duration_ms)
        return response

    messages.create = patched_create

    if orig_stream:
        def patched_stream(*args, **kwargs):
            if is_suppressed():
                return orig_stream(*args, **kwargs)

            model = kwargs.get("model", "")
            input_messages = kwargs.get("messages", [])
            system = kwargs.get("system")

            tracer = get_tracer()
            span = tracer.start_span(
                name="anthropic.messages.create",
                attributes={
                    "neatlogs.span.kind": "LLM",
                    "neatlogs.llm.provider": "anthropic",
                    "neatlogs.llm.system": "anthropic",
                    "neatlogs.llm.model_name": model,
                    "neatlogs.llm.is_streaming": True,
                },
            )

            _set_input_attributes(span, input_messages, system, kwargs)

            stream_mgr = orig_stream(*args, **kwargs)
            return _SyncStreamManagerWrapper(stream_mgr, span)

        messages.stream = patched_stream

    messages._neatlogs_patched = True


def _patch_async_messages(messages: Any) -> None:
    if getattr(messages, "_neatlogs_patched", False):
        return

    orig_create = messages.create
    orig_stream = getattr(messages, "stream", None)

    async def patched_create(*args, **kwargs):
        if is_suppressed():
            return await orig_create(*args, **kwargs)

        model = kwargs.get("model", "")
        input_messages = kwargs.get("messages", [])
        system = kwargs.get("system")
        is_stream = kwargs.get("stream", False)

        tracer = get_tracer()
        span = tracer.start_span(
            name="anthropic.messages.create",
            attributes={
                "neatlogs.span.kind": "LLM",
                "neatlogs.llm.provider": "anthropic",
                "neatlogs.llm.system": "anthropic",
                "neatlogs.llm.model_name": model,
                "neatlogs.llm.is_streaming": is_stream,
            },
        )

        _set_input_attributes(span, input_messages, system, kwargs)

        start = time.perf_counter()

        try:
            response = await orig_create(*args, **kwargs)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise

        if is_stream:
            return AsyncStreamWrapper(response, span, _finalize_stream)

        duration_ms = (time.perf_counter() - start) * 1000
        _finalize_response(span, response, duration_ms)
        return response

    messages.create = patched_create

    if orig_stream:
        def patched_stream(*args, **kwargs):
            if is_suppressed():
                return orig_stream(*args, **kwargs)

            model = kwargs.get("model", "")
            input_messages = kwargs.get("messages", [])
            system = kwargs.get("system")

            tracer = get_tracer()
            span = tracer.start_span(
                name="anthropic.messages.create",
                attributes={
                    "neatlogs.span.kind": "LLM",
                    "neatlogs.llm.provider": "anthropic",
                    "neatlogs.llm.system": "anthropic",
                    "neatlogs.llm.model_name": model,
                    "neatlogs.llm.is_streaming": True,
                },
            )

            _set_input_attributes(span, input_messages, system, kwargs)

            stream_mgr = orig_stream(*args, **kwargs)
            return _AsyncStreamManagerWrapper(stream_mgr, span)

        messages.stream = patched_stream

    messages._neatlogs_patched = True


def _set_input_attributes(span: Any, messages: list, system: Any, kwargs: dict) -> None:
    """Set input message and tool attributes on the span."""
    idx = 0

    if system:
        span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", "system")
        if isinstance(system, str):
            span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", system)
        else:
            span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", serialize(system))
        idx += 1

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", role)
        if isinstance(content, str):
            span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", content)
        else:
            span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", serialize(content))
        if msg.get("tool_call_id"):
            span.set_attribute(f"neatlogs.llm.input_messages.{idx}.tool_call_id", msg["tool_call_id"])
        idx += 1

    tools = kwargs.get("tools")
    if tools:
        for i, tool in enumerate(tools):
            span.set_attribute(f"neatlogs.llm.tools.{i}.name", tool.get("name", ""))
            if tool.get("description"):
                span.set_attribute(f"neatlogs.llm.tools.{i}.description", tool["description"])
            if tool.get("input_schema"):
                span.set_attribute(f"neatlogs.llm.tools.{i}.input_schema", serialize(tool["input_schema"]))

    for param in ("temperature", "top_p", "top_k", "max_tokens"):
        if param in kwargs and kwargs[param] is not None:
            span.set_attribute(f"neatlogs.llm.{param}", kwargs[param])


def _finalize_response(span: Any, response: Any, duration_ms: float) -> None:
    """Extract attributes from a non-streaming Anthropic Message response."""
    content = getattr(response, "content", []) or []
    text_parts: List[str] = []
    tool_call_idx = 0

    for block in content:
        block_type = getattr(block, "type", "")
        if block_type == "text":
            text_parts.append(getattr(block, "text", ""))
        elif block_type == "tool_use":
            span.set_attribute(f"neatlogs.llm.tool_calls.{tool_call_idx}.id", getattr(block, "id", ""))
            span.set_attribute(f"neatlogs.llm.tool_calls.{tool_call_idx}.name", getattr(block, "name", ""))
            span.set_attribute(f"neatlogs.llm.tool_calls.{tool_call_idx}.arguments", serialize(getattr(block, "input", {})))
            tool_call_idx += 1
        elif block_type == "thinking":
            thinking_text = getattr(block, "thinking", "")
            if thinking_text:
                span.set_attribute("neatlogs.llm.output_messages.0.thinking", thinking_text)

    if text_parts:
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        span.set_attribute("neatlogs.llm.output_messages.0.content", "".join(text_parts))

    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason:
        span.set_attribute("neatlogs.llm.finish_reason", stop_reason)

    model = getattr(response, "model", None)
    if model:
        span.set_attribute("neatlogs.llm.model_name", model)

    usage = getattr(response, "usage", None)
    if usage:
        _set_usage_attributes(span, usage)

    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()


def _set_usage_attributes(span: Any, usage: Any) -> None:
    """Set token usage attributes from Anthropic usage object."""
    input_tokens = getattr(usage, "input_tokens", None)
    if input_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.prompt", input_tokens)

    output_tokens = getattr(usage, "output_tokens", None)
    if output_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.completion", output_tokens)

    if input_tokens is not None and output_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.total", input_tokens + output_tokens)

    cache_read = getattr(usage, "cache_read_input_tokens", None)
    if cache_read is not None:
        span.set_attribute("neatlogs.llm.token_count.cache_read", cache_read)

    cache_write = getattr(usage, "cache_creation_input_tokens", None)
    if cache_write is not None:
        span.set_attribute("neatlogs.llm.token_count.cache_write", cache_write)


def _finalize_stream(span: Any, chunks: List[Any], duration_ms: float, ttft_ms: Optional[float]) -> None:
    """Finalize a streaming response span from accumulated Anthropic stream events."""
    text_parts: List[str] = []
    tool_calls_acc: dict = {}
    thinking_parts: List[str] = []
    stop_reason = None
    model = None
    input_tokens = None
    output_tokens = None
    cache_read = None
    cache_write = None

    for event in chunks:
        event_type = getattr(event, "type", "")

        if event_type == "message_start":
            message = getattr(event, "message", None)
            if message:
                model = getattr(message, "model", None)
                msg_usage = getattr(message, "usage", None)
                if msg_usage:
                    input_tokens = getattr(msg_usage, "input_tokens", None)
                    cache_read = getattr(msg_usage, "cache_read_input_tokens", None)
                    cache_write = getattr(msg_usage, "cache_creation_input_tokens", None)

        elif event_type == "content_block_start":
            block = getattr(event, "content_block", None)
            if block and getattr(block, "type", "") == "tool_use":
                idx = getattr(event, "index", len(tool_calls_acc))
                tool_calls_acc[idx] = {
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "arguments": "",
                }

        elif event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if not delta:
                continue
            delta_type = getattr(delta, "type", "")
            if delta_type == "text_delta":
                text_parts.append(getattr(delta, "text", ""))
            elif delta_type == "input_json_delta":
                idx = getattr(event, "index", 0)
                if idx in tool_calls_acc:
                    tool_calls_acc[idx]["arguments"] += getattr(delta, "partial_json", "")
            elif delta_type == "thinking_delta":
                thinking_parts.append(getattr(delta, "thinking", ""))

        elif event_type == "message_delta":
            delta = getattr(event, "delta", None)
            if delta:
                stop_reason = getattr(delta, "stop_reason", None)
            msg_usage = getattr(event, "usage", None)
            if msg_usage:
                output_tokens = getattr(msg_usage, "output_tokens", None)

    # Output messages
    full_text = "".join(text_parts)
    if full_text:
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        span.set_attribute("neatlogs.llm.output_messages.0.content", full_text)

    # Thinking
    full_thinking = "".join(thinking_parts)
    if full_thinking:
        span.set_attribute("neatlogs.llm.output_messages.0.thinking", full_thinking)

    # Tool calls
    for j, tc in enumerate(tool_calls_acc.values()):
        span.set_attribute(f"neatlogs.llm.tool_calls.{j}.id", tc["id"])
        span.set_attribute(f"neatlogs.llm.tool_calls.{j}.name", tc["name"])
        span.set_attribute(f"neatlogs.llm.tool_calls.{j}.arguments", tc["arguments"])

    if model:
        span.set_attribute("neatlogs.llm.model_name", model)
    if stop_reason:
        span.set_attribute("neatlogs.llm.finish_reason", stop_reason)

    # Usage
    if input_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.prompt", input_tokens)
    if output_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.completion", output_tokens)
    if input_tokens is not None and output_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.total", input_tokens + output_tokens)
    if cache_read is not None:
        span.set_attribute("neatlogs.llm.token_count.cache_read", cache_read)
    if cache_write is not None:
        span.set_attribute("neatlogs.llm.token_count.cache_write", cache_write)

    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    if ttft_ms is not None:
        span.set_attribute("neatlogs.llm.metrics.ttft_ms", round(ttft_ms, 3))
        if duration_ms > ttft_ms:
            span.set_attribute(
                "neatlogs.llm.metrics.streaming_time_to_generate_ms",
                round(duration_ms - ttft_ms, 3),
            )

    span.set_status(StatusCode.OK)
    span.end()


class _SyncStreamManagerWrapper:
    """Wraps Anthropic's MessageStreamManager context manager for messages.stream()."""

    def __init__(self, stream_mgr: Any, span: Any):
        self._stream_mgr = stream_mgr
        self._span = span
        self._start_time = time.perf_counter()
        self._first_chunk_time: Optional[float] = None
        self._chunks: List[Any] = []
        self._finalized = False

    def __enter__(self):
        self._stream = self._stream_mgr.__enter__()
        return _SyncStreamIterator(self)

    def __exit__(self, *args):
        try:
            self._stream_mgr.__exit__(*args)
        finally:
            self._finalize()

    def _finalize(self):
        if self._finalized:
            return
        self._finalized = True
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        ttft_ms = None
        if self._first_chunk_time is not None:
            ttft_ms = (self._first_chunk_time - self._start_time) * 1000
        _finalize_stream(self._span, self._chunks, elapsed_ms, ttft_ms)

    def __getattr__(self, name):
        return getattr(self._stream_mgr, name)


class _SyncStreamIterator:
    """Iterates events from an Anthropic MessageStream, collecting chunks."""

    def __init__(self, wrapper: _SyncStreamManagerWrapper):
        self._wrapper = wrapper

    def __iter__(self):
        for event in self._wrapper._stream:
            if self._wrapper._first_chunk_time is None:
                self._wrapper._first_chunk_time = time.perf_counter()
            self._wrapper._chunks.append(event)
            yield event

    def __getattr__(self, name):
        return getattr(self._wrapper._stream, name)


class _AsyncStreamManagerWrapper:
    """Wraps Anthropic's AsyncMessageStreamManager for async messages.stream()."""

    def __init__(self, stream_mgr: Any, span: Any):
        self._stream_mgr = stream_mgr
        self._span = span
        self._start_time = time.perf_counter()
        self._first_chunk_time: Optional[float] = None
        self._chunks: List[Any] = []
        self._finalized = False

    async def __aenter__(self):
        self._stream = await self._stream_mgr.__aenter__()
        return _AsyncStreamIterator(self)

    async def __aexit__(self, *args):
        try:
            await self._stream_mgr.__aexit__(*args)
        finally:
            self._finalize()

    def _finalize(self):
        if self._finalized:
            return
        self._finalized = True
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        ttft_ms = None
        if self._first_chunk_time is not None:
            ttft_ms = (self._first_chunk_time - self._start_time) * 1000
        _finalize_stream(self._span, self._chunks, elapsed_ms, ttft_ms)

    def __getattr__(self, name):
        return getattr(self._stream_mgr, name)


class _AsyncStreamIterator:
    """Iterates events from an Anthropic AsyncMessageStream, collecting chunks."""

    def __init__(self, wrapper: _AsyncStreamManagerWrapper):
        self._wrapper = wrapper

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            event = await self._wrapper._stream.__anext__()
        except StopAsyncIteration:
            raise

        if self._wrapper._first_chunk_time is None:
            self._wrapper._first_chunk_time = time.perf_counter()
        self._wrapper._chunks.append(event)
        return event

    def __getattr__(self, name):
        return getattr(self._wrapper._stream, name)


# ---------------------------------------------------------------------------
# Additional message methods: parse (structured) + count_tokens; legacy completions
# ---------------------------------------------------------------------------


def _extra_message_methods(messages: Any, is_async: bool) -> None:
    """Patch messages.parse (structured output) and messages.count_tokens."""
    # parse → same shape as a Message response
    if hasattr(messages, "parse") and not getattr(messages, "_neatlogs_parse_patched", False):
        orig_parse = messages.parse
        if is_async:
            async def patched_parse(*args, **kwargs):
                if is_suppressed():
                    return await orig_parse(*args, **kwargs)
                span = _start_message_span(kwargs, "anthropic.messages.parse", structured=True)
                start = time.perf_counter()
                try:
                    resp = await orig_parse(*args, **kwargs)
                except Exception as e:
                    _err(span, e); raise
                _finalize_response(span, resp, (time.perf_counter() - start) * 1000)
                return resp
        else:
            def patched_parse(*args, **kwargs):
                if is_suppressed():
                    return orig_parse(*args, **kwargs)
                span = _start_message_span(kwargs, "anthropic.messages.parse", structured=True)
                start = time.perf_counter()
                try:
                    resp = orig_parse(*args, **kwargs)
                except Exception as e:
                    _err(span, e); raise
                _finalize_response(span, resp, (time.perf_counter() - start) * 1000)
                return resp
        messages.parse = patched_parse
        messages._neatlogs_parse_patched = True

    # count_tokens → small utility call
    if hasattr(messages, "count_tokens") and not getattr(messages, "_neatlogs_count_patched", False):
        orig_count = messages.count_tokens
        if is_async:
            async def patched_count(*args, **kwargs):
                if is_suppressed():
                    return await orig_count(*args, **kwargs)
                span = get_tracer().start_span(
                    name="anthropic.messages.count_tokens",
                    attributes={"neatlogs.span.kind": "LLM", "neatlogs.llm.provider": "anthropic",
                                "neatlogs.llm.task": "count_tokens", "neatlogs.llm.model_name": kwargs.get("model", "")},
                )
                try:
                    resp = await orig_count(*args, **kwargs)
                except Exception as e:
                    _err(span, e); raise
                _finalize_count_tokens(span, resp)
                return resp
        else:
            def patched_count(*args, **kwargs):
                if is_suppressed():
                    return orig_count(*args, **kwargs)
                span = get_tracer().start_span(
                    name="anthropic.messages.count_tokens",
                    attributes={"neatlogs.span.kind": "LLM", "neatlogs.llm.provider": "anthropic",
                                "neatlogs.llm.task": "count_tokens", "neatlogs.llm.model_name": kwargs.get("model", "")},
                )
                try:
                    resp = orig_count(*args, **kwargs)
                except Exception as e:
                    _err(span, e); raise
                _finalize_count_tokens(span, resp)
                return resp
        messages.count_tokens = patched_count
        messages._neatlogs_count_patched = True


def _start_message_span(kwargs: dict, name: str, structured: bool = False) -> Any:
    span = get_tracer().start_span(
        name=name,
        attributes={
            "neatlogs.span.kind": "LLM",
            "neatlogs.llm.provider": "anthropic",
            "neatlogs.llm.system": "anthropic",
            "neatlogs.llm.model_name": kwargs.get("model", ""),
        },
    )
    if structured:
        span.set_attribute("neatlogs.llm.structured_output", True)
    _set_input_attributes(span, kwargs.get("messages", []), kwargs.get("system"), kwargs)
    return span


def _finalize_count_tokens(span: Any, resp: Any) -> None:
    tokens = getattr(resp, "input_tokens", None)
    if tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.prompt", tokens)
    span.set_status(StatusCode.OK)
    span.end()


def _patch_legacy_completions(completions: Any, is_async: bool) -> None:
    """Legacy text completions API (client.completions.create)."""
    if completions is None or not hasattr(completions, "create"):
        return
    if getattr(completions, "_neatlogs_patched", False):
        return
    orig = completions.create

    def _attrs(kwargs):
        prompt = kwargs.get("prompt", "")
        return {
            "neatlogs.span.kind": "LLM",
            "neatlogs.llm.provider": "anthropic",
            "neatlogs.llm.system": "anthropic",
            "neatlogs.llm.model_name": kwargs.get("model", ""),
            "neatlogs.llm.input_messages.0.role": "user",
            "neatlogs.llm.input_messages.0.content": str(prompt)[:10000],
        }

    def _finalize(span, resp, duration_ms):
        completion = getattr(resp, "completion", None)
        if completion:
            span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
            span.set_attribute("neatlogs.llm.output_messages.0.content", str(completion)[:10000])
        stop = getattr(resp, "stop_reason", None)
        if stop:
            span.set_attribute("neatlogs.llm.finish_reason", str(stop))
        span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
        span.set_status(StatusCode.OK)
        span.end()

    if is_async:
        async def patched(*args, **kwargs):
            if is_suppressed():
                return await orig(*args, **kwargs)
            span = get_tracer().start_span(name="anthropic.completions.create", attributes=_attrs(kwargs))
            start = time.perf_counter()
            try:
                resp = await orig(*args, **kwargs)
            except Exception as e:
                _err(span, e); raise
            _finalize(span, resp, (time.perf_counter() - start) * 1000)
            return resp
    else:
        def patched(*args, **kwargs):
            if is_suppressed():
                return orig(*args, **kwargs)
            span = get_tracer().start_span(name="anthropic.completions.create", attributes=_attrs(kwargs))
            start = time.perf_counter()
            try:
                resp = orig(*args, **kwargs)
            except Exception as e:
                _err(span, e); raise
            _finalize(span, resp, (time.perf_counter() - start) * 1000)
            return resp

    completions.create = patched
    completions._neatlogs_patched = True


def _err(span: Any, e: Exception) -> None:
    span.set_status(StatusCode.ERROR, str(e))
    span.record_exception(e)
    span.end()


# ---------------------------------------------------------------------------
# Import-replacement: `from neatlogs.anthropic import anthropic`
# Patches Anthropic/AsyncAnthropic.__init__ so every client is auto-wrapped.
# ---------------------------------------------------------------------------

def _patch_anthropic_module() -> None:
    global _PATCHED, _ORIG_INIT, _ORIG_ASYNC_INIT
    if _PATCHED:
        return
    _PATCHED = True

    import anthropic as _anthropic

    _ORIG_INIT = _anthropic.Anthropic.__init__
    _ORIG_ASYNC_INIT = _anthropic.AsyncAnthropic.__init__

    def _patched_init(self, *args, **kwargs):
        _ORIG_INIT(self, *args, **kwargs)
        wrap_anthropic_client(self)

    _anthropic.Anthropic.__init__ = _patched_init

    def _patched_async_init(self, *args, **kwargs):
        _ORIG_ASYNC_INIT(self, *args, **kwargs)
        wrap_async_anthropic_client(self)

    _anthropic.AsyncAnthropic.__init__ = _patched_async_init


def _unpatch_anthropic_module() -> None:
    global _PATCHED, _ORIG_INIT, _ORIG_ASYNC_INIT
    if not _PATCHED:
        return

    import anthropic as _anthropic

    if _ORIG_INIT is not None:
        _anthropic.Anthropic.__init__ = _ORIG_INIT
    if _ORIG_ASYNC_INIT is not None:
        _anthropic.AsyncAnthropic.__init__ = _ORIG_ASYNC_INIT

    _PATCHED = False
    _ORIG_INIT = None
    _ORIG_ASYNC_INIT = None


_patch_anthropic_module()

import anthropic  # noqa: E402
