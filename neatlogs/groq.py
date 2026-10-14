"""
Neatlogs Groq wrapper.

Groq's official Python SDK (``groq``) is OpenAI-compatible and exposes a
``chat.completions`` resource on both ``groq.Groq`` (sync) and
``groq.AsyncGroq`` (async). This module traces chat completions — sync,
async, and streaming — on a wrapped client instance.

Usage (the client is mutated in place and also returned):

  >>> import neatlogs
  >>> from groq import Groq
  >>> client = neatlogs.wrap(Groq(api_key=os.environ["GROQ_API_KEY"]))
  >>> client.chat.completions.create(
  ...     model="llama-3.3-70b-versatile",
  ...     messages=[{"role": "user", "content": "hi"}],
  ... )

``neatlogs.llm.provider`` is always ``groq``; ``neatlogs.llm.system`` is also
``groq`` because every model the SDK serves is hosted by Groq's own inference
infrastructure (model slug is preserved as ``neatlogs.llm.model_name``).
"""

import time
from typing import Any, List, Optional

from opentelemetry.trace import StatusCode

from ._wrap_utils import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
    _safe_finalize,
    _telemetry_fallback,
    get_provider_tracer,
    is_suppressed,
    serialize,
)

try:  # Groq SDK Omit sentinel — distinguish "user did not set" from "user set None".
    from groq import Omit as _GroqOmit  # noqa: F401
except Exception:  # pragma: no cover - groq not installed
    _GroqOmit = type("_NoOmit", (), {})


def _is_set(val: Any) -> bool:
    """True if the user actually passed a value (filtering out Groq's ``Omit``)."""
    if val is None:
        return False
    if isinstance(val, _GroqOmit):
        return False
    return True


_PROVIDER = "groq"


class GroqInstrumentor:
    """
    Instrumentor class for InstrumentationManager integration.

    The Groq SDK has no import-time public class we patch globally; clients
    are wrapped explicitly via ``neatlogs.wrap(Groq(...))``. This shim lets
    ``init(instrumentations=["groq"])`` patch the constructor so every client
    built afterward is auto-wrapped.
    """

    def instrument(self, tracer_provider=None):
        _patch_groq_module()

    def uninstrument(self):
        _unpatch_groq_module()


# ---------------------------------------------------------------------------
# Public wrap entrypoint
# ---------------------------------------------------------------------------


def wrap_groq_client(client: Any) -> Any:
    """
    Wrap a ``groq.Groq`` or ``groq.AsyncGroq`` client instance. Patches
    ``client.chat.completions.create`` to auto-trace; returns the same client.
    Idempotent.
    """
    if getattr(client, "_neatlogs_groq_wrapped", False):
        return client

    chat = getattr(client, "chat", None)
    if chat is not None:
        completions = getattr(chat, "completions", None)
        if completions is not None:
            _safe(_patch_completions, completions)

    client._neatlogs_groq_wrapped = True
    return client


def _safe(fn, resource) -> None:
    if resource is None:
        return
    try:
        fn(resource)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Shared attribute helpers
# ---------------------------------------------------------------------------

_PARAM_KEYS = (
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "stop",
    "reasoning_effort",
    "service_tier",
)


def _set_invocation_params(span: Any, kwargs: dict) -> None:
    """Capture sampling/generation params + write the canonical
    ``neatlogs.llm.invocation_parameters`` blob the backend reads as
    ``model_settings`` in the UI."""
    params = {}
    for key in _PARAM_KEYS:
        val = kwargs.get(key)
        if not _is_set(val):
            continue
        params[key] = val
        span.set_attribute(
            f"neatlogs.llm.{key}",
            val if not isinstance(val, (list, dict)) else serialize(val),
        )
    if params:
        span.set_attribute("neatlogs.llm.invocation_parameters", serialize(params))


def _set_chat_input(span: Any, kwargs: dict) -> None:
    messages = kwargs.get("messages", []) or []
    for i, msg in enumerate(messages):
        role = _get(msg, "role", "")
        content = _get(msg, "content", "")
        span.set_attribute(f"neatlogs.llm.input_messages.{i}.role", role or "")
        span.set_attribute(
            f"neatlogs.llm.input_messages.{i}.content",
            content if isinstance(content, str) else serialize(content),
        )
        tool_call_id = _get(msg, "tool_call_id", None)
        if tool_call_id:
            span.set_attribute(f"neatlogs.llm.input_messages.{i}.tool_call_id", tool_call_id)
    if messages:
        span.set_attribute("input.value", serialize(_plain(messages)))

    tools = kwargs.get("tools")
    if tools:
        for i, tool in enumerate(tools):
            fn = _get(tool, "function", {}) or {}
            name = _get(fn, "name", None) or _get(tool, "name", None)
            if name:
                span.set_attribute(f"neatlogs.llm.tools.{i}.name", name)
            desc = _get(fn, "description", None) or _get(tool, "description", None)
            if desc:
                span.set_attribute(f"neatlogs.llm.tools.{i}.description", desc)
            schema = _get(fn, "parameters", None)
            if schema:
                span.set_attribute(
                    f"neatlogs.llm.tools.{i}.input_schema", serialize(_plain(schema))
                )


def _get(obj: Any, key: str, default: Any) -> Any:
    """Read ``key`` from a dict or a pydantic/attr object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _plain(obj: Any) -> Any:
    """Best-effort convert pydantic models to plain dicts for serialization."""
    if isinstance(obj, list):
        return [_plain(o) for o in obj]
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump(exclude_none=True)
        except Exception:
            pass
    return obj


# ---------------------------------------------------------------------------
# Chat Completions: client.chat.completions.create (sync + async)
# ---------------------------------------------------------------------------


def _patch_completions(completions: Any) -> None:
    if getattr(completions, "_neatlogs_groq_patched", False):
        return

    orig_create = getattr(completions, "create", None)
    if not callable(orig_create):
        return

    is_async_cls = _is_async_completions(completions)

    if is_async_cls:

        async def patched_create_async(*args, **kwargs):
            if is_suppressed():
                return await orig_create(*args, **kwargs)
            is_stream = bool(kwargs.get("stream", False))
            try:
                span = _start(kwargs, is_stream)
                start = time.perf_counter()
            except Exception:
                return await _telemetry_fallback(orig_create, *args, **kwargs)
            try:
                response = await orig_create(*args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            if is_stream:
                return AsyncStreamWrapper(response, span, _finalize_chat_stream)
            _safe_finalize(span, _finalize_chat, response, (time.perf_counter() - start) * 1000)
            return response

        completions.create = patched_create_async
    else:

        def patched_create(*args, **kwargs):
            if is_suppressed():
                return orig_create(*args, **kwargs)
            is_stream = bool(kwargs.get("stream", False))
            try:
                span = _start(kwargs, is_stream)
                start = time.perf_counter()
            except Exception:
                return _telemetry_fallback(orig_create, *args, **kwargs)
            try:
                response = orig_create(*args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            if is_stream:
                return SyncStreamWrapper(response, span, _finalize_chat_stream)
            _safe_finalize(span, _finalize_chat, response, (time.perf_counter() - start) * 1000)
            return response

        completions.create = patched_create

    completions._neatlogs_groq_patched = True


def _is_async_completions(completions: Any) -> bool:
    """Groq SDK uses the same method name on sync/AsyncGroq; infer from class name."""
    cls = type(completions)
    return "Async" in cls.__name__


def _start(kwargs: dict, is_stream: bool) -> Any:
    model = kwargs.get("model", "")
    span = get_provider_tracer().start_span(
        name="groq.chat.completions.create",
        attributes={
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.provider": _PROVIDER,
            "neatlogs.llm.system": _PROVIDER,
            "neatlogs.llm.model_name": model,
            "neatlogs.llm.is_streaming": is_stream,
        },
    )
    _set_chat_input(span, kwargs)
    _set_invocation_params(span, kwargs)
    return span


def _finalize_chat(span: Any, response: Any, duration_ms: float) -> None:
    """Extract attributes from a non-streaming ChatCompletion."""
    choices = _get(response, "choices", []) or []
    for i, choice in enumerate(choices):
        message = _get(choice, "message", None)
        if message is None:
            continue
        span.set_attribute(f"neatlogs.llm.output_messages.{i}.role", "assistant")
        content = _get(message, "content", None)
        if content:
            text = content if isinstance(content, str) else serialize(_plain(content))
            span.set_attribute(f"neatlogs.llm.output_messages.{i}.content", text)
            if i == 0:
                span.set_attribute("output.value", text)
        reasoning = _get(message, "reasoning", None)
        if reasoning:
            rtext = reasoning if isinstance(reasoning, str) else serialize(_plain(reasoning))
            span.set_attribute(f"neatlogs.llm.output_messages.{i}.reasoning", rtext)
        tool_calls = _get(message, "tool_calls", None)
        if tool_calls:
            for j, tc in enumerate(tool_calls):
                fn = _get(tc, "function", None)
                span.set_attribute(f"neatlogs.llm.tool_calls.{j}.id", _get(tc, "id", "") or "")
                span.set_attribute(f"neatlogs.llm.tool_calls.{j}.name", _get(fn, "name", "") or "")
                args = _get(fn, "arguments", "")
                span.set_attribute(
                    f"neatlogs.llm.tool_calls.{j}.arguments",
                    args if isinstance(args, str) else serialize(_plain(args)),
                )
        finish_reason = _get(choice, "finish_reason", None)
        if finish_reason:
            span.set_attribute("neatlogs.llm.finish_reason", str(finish_reason))

    _set_chat_usage(span, _get(response, "usage", None))

    model = _get(response, "model", None)
    if model:
        span.set_attribute("neatlogs.llm.model_name", str(model))
    response_id = _get(response, "id", None)
    if response_id:
        span.set_attribute("neatlogs.llm.response_id", str(response_id))

    _ok(span, duration_ms)


def _finalize_chat_stream(
    span: Any, chunks: List[Any], duration_ms: float, ttft_ms: Optional[float]
) -> None:
    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls_acc: dict = {}
    finish_reason = None
    model = None
    usage = None

    for chunk in chunks:
        if _get(chunk, "model", None):
            model = _get(chunk, "model", None)
        if _get(chunk, "usage", None):
            usage = _get(chunk, "usage", None)
        choices = _get(chunk, "choices", None) or []
        if not choices:
            continue
        choice = choices[0]
        delta = _get(choice, "delta", None)
        if delta is not None:
            content = _get(delta, "content", None)
            if content:
                text_parts.append(content)
            reasoning = _get(delta, "reasoning", None)
            if reasoning:
                reasoning_parts.append(reasoning)
            for tc in _get(delta, "tool_calls", None) or []:
                idx = _get(tc, "index", 0) or 0
                acc = tool_calls_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if _get(tc, "id", None):
                    acc["id"] = _get(tc, "id", "")
                fn = _get(tc, "function", None)
                if fn is not None:
                    if _get(fn, "name", None):
                        acc["name"] = _get(fn, "name", "")
                    if _get(fn, "arguments", None):
                        acc["arguments"] += _get(fn, "arguments", "")
        fr = _get(choice, "finish_reason", None)
        if fr:
            finish_reason = fr

    full_text = "".join(text_parts)
    if full_text:
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        span.set_attribute("neatlogs.llm.output_messages.0.content", full_text)
        span.set_attribute("output.value", full_text)
    full_reasoning = "".join(reasoning_parts)
    if full_reasoning:
        span.set_attribute("neatlogs.llm.output_messages.0.reasoning", full_reasoning)
    for j, tc in enumerate(tool_calls_acc.values()):
        span.set_attribute(f"neatlogs.llm.tool_calls.{j}.id", tc["id"])
        span.set_attribute(f"neatlogs.llm.tool_calls.{j}.name", tc["name"])
        span.set_attribute(f"neatlogs.llm.tool_calls.{j}.arguments", tc["arguments"])
    if model:
        span.set_attribute("neatlogs.llm.model_name", str(model))
    if finish_reason:
        span.set_attribute("neatlogs.llm.finish_reason", str(finish_reason))
    _set_chat_usage(span, usage)

    _ok(span, duration_ms, ttft_ms)


def _set_chat_usage(span: Any, usage: Any) -> None:
    if usage is None:
        return
    prompt = _get(usage, "prompt_tokens", None)
    completion = _get(usage, "completion_tokens", None)
    total = _get(usage, "total_tokens", None)
    if prompt is not None:
        span.set_attribute("neatlogs.llm.token_count.prompt", prompt)
    if completion is not None:
        span.set_attribute("neatlogs.llm.token_count.completion", completion)
    if total is not None:
        span.set_attribute("neatlogs.llm.token_count.total", total)
    elif prompt is not None and completion is not None:
        span.set_attribute("neatlogs.llm.token_count.total", prompt + completion)
    details = _get(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = _get(details, "cached_tokens", None)
        if cached is not None:
            span.set_attribute("neatlogs.llm.token_count.cache_read", cached)
    cdetails = _get(usage, "completion_tokens_details", None)
    if cdetails is not None:
        reasoning = _get(cdetails, "reasoning_tokens", None)
        if reasoning is not None:
            span.set_attribute("neatlogs.llm.token_count.reasoning", reasoning)


# ---------------------------------------------------------------------------
# Span finalization helpers
# ---------------------------------------------------------------------------


def _ok(span: Any, duration_ms: float, ttft_ms: Optional[float] = None) -> None:
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


def _err(span: Any, e: Exception) -> None:
    span.set_status(StatusCode.ERROR, str(e))
    span.record_exception(e)
    span.end()


# ---------------------------------------------------------------------------
# Import-replacement: `from neatlogs.groq import Groq`
# Patches groq.Groq.__init__ so every client constructed is auto-wrapped.
# ---------------------------------------------------------------------------

_PATCHED = False
_ORIG_INIT = None
_ORIG_ASYNC_INIT = None


def _patch_groq_module() -> None:
    global _PATCHED, _ORIG_INIT, _ORIG_ASYNC_INIT
    if _PATCHED:
        return
    try:
        from groq import AsyncGroq as _AsyncGroq
        from groq import Groq as _Groq
    except Exception:
        return

    _PATCHED = True
    _ORIG_INIT = _Groq.__init__
    _ORIG_ASYNC_INIT = _AsyncGroq.__init__

    def _patched_init(self, *args, **kwargs):
        _ORIG_INIT(self, *args, **kwargs)
        wrap_groq_client(self)

    def _patched_async_init(self, *args, **kwargs):
        _ORIG_ASYNC_INIT(self, *args, **kwargs)
        wrap_groq_client(self)

    _Groq.__init__ = _patched_init
    _AsyncGroq.__init__ = _patched_async_init


def _unpatch_groq_module() -> None:
    global _PATCHED, _ORIG_INIT, _ORIG_ASYNC_INIT
    if not _PATCHED:
        return
    try:
        from groq import AsyncGroq as _AsyncGroq
        from groq import Groq as _Groq
    except Exception:
        return
    if _ORIG_INIT is not None:
        _Groq.__init__ = _ORIG_INIT
    if _ORIG_ASYNC_INIT is not None:
        _AsyncGroq.__init__ = _ORIG_ASYNC_INIT
    _PATCHED = False
    _ORIG_INIT = None
    _ORIG_ASYNC_INIT = None


try:  # noqa: E402 - re-export for `from neatlogs.groq import Groq`
    from groq import AsyncGroq  # noqa: F401
    from groq import Groq  # noqa: F401
except Exception:  # pragma: no cover - groq not installed
    pass
