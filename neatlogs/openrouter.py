"""
Neatlogs OpenRouter wrapper.

OpenRouter's official Python SDK (the ``openrouter`` package) is a Speakeasy
client exposing OpenAI-compatible surfaces. This module traces the LLM-relevant
ones on a wrapped ``OpenRouter`` client instance:

  - ``client.chat.send`` / ``send_async``            -> Chat Completions (LLM)
  - ``client.beta.responses.send`` / ``send_async``  -> Responses API (LLM)
  - ``client.embeddings.generate``                   -> Embeddings (EMBEDDING)
  - ``client.rerank.rerank``                         -> Rerank (RERANKER)

Usage (the client is mutated in place and also returned):

  >>> import neatlogs
  >>> from openrouter import OpenRouter
  >>> client = neatlogs.wrap(OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"]))
  >>> client.chat.send(model="openai/gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

``neatlogs.llm.provider`` is always ``openrouter``; ``neatlogs.llm.system`` is the
underlying model vendor (openai / anthropic / google / ...), inferred from the
``vendor/model`` slug OpenRouter uses for model ids.
"""

import time
from typing import Any, List, Optional

from opentelemetry.trace import StatusCode

from ._wrap_utils import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
    get_provider_tracer,
    is_suppressed,
    serialize,
)

_PROVIDER = "openrouter"


class OpenRouterInstrumentor:
    """
    Instrumentor class for InstrumentationManager integration.

    The OpenRouter SDK has no import-time public class we patch globally; clients
    are wrapped explicitly via ``neatlogs.wrap(OpenRouter(...))``. This shim lets
    ``init(instrumentations=["openrouter"])`` patch the constructor so every
    client built afterward is auto-wrapped.
    """

    def instrument(self, tracer_provider=None):
        _patch_openrouter_module()

    def uninstrument(self):
        _unpatch_openrouter_module()


# ---------------------------------------------------------------------------
# Vendor helper
# ---------------------------------------------------------------------------


def _vendor_from_model(model: Any) -> str:
    """Infer the model vendor from an OpenRouter ``vendor/model`` slug."""
    mid = str(model or "")
    if "/" in mid:
        return mid.split("/", 1)[0]
    return mid or _PROVIDER


# ---------------------------------------------------------------------------
# Public wrap entrypoint
# ---------------------------------------------------------------------------


def wrap_openrouter_client(client: Any) -> Any:
    """
    Wrap an ``openrouter.OpenRouter`` client instance. Patches the LLM-relevant
    sub-SDK methods to auto-trace; returns the same client. Idempotent.
    """
    if getattr(client, "_neatlogs_openrouter_wrapped", False):
        return client

    _safe(_patch_chat, getattr(client, "chat", None))
    # Responses API lives under client.beta.responses.
    beta = getattr(client, "beta", None)
    if beta is not None:
        _safe(_patch_responses, getattr(beta, "responses", None))
    _safe(_patch_embeddings, getattr(client, "embeddings", None))
    _safe(_patch_rerank, getattr(client, "rerank", None))

    client._neatlogs_openrouter_wrapped = True
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
    "top_k",
    "max_tokens",
    "max_output_tokens",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "stop",
)


def _set_invocation_params(span: Any, kwargs: dict) -> None:
    """
    Capture sampling/generation params.

    The backend surfaces model settings ONLY from the JSON blob
    ``neatlogs.llm.invocation_parameters`` (parsed into ``model_settings`` and
    shown in the UI). We set that blob AND the individual ``neatlogs.llm.*``
    attributes (used by enrichment / other consumers).
    """
    params = {}
    for key in _PARAM_KEYS:
        val = kwargs.get(key)
        if val is not None:
            params[key] = val
            # max_output_tokens normalizes to the max_tokens individual attr.
            attr = "max_tokens" if key == "max_output_tokens" else key
            span.set_attribute(f"neatlogs.llm.{attr}", val if not isinstance(val, (list, dict)) else serialize(val))
    if params:
        span.set_attribute("neatlogs.llm.invocation_parameters", serialize(params))


def _set_chat_input(span: Any, kwargs: dict) -> None:
    messages = kwargs.get("messages", []) or []
    for i, msg in enumerate(messages):
        # messages may be dicts or pydantic models.
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
                span.set_attribute(f"neatlogs.llm.tools.{i}.input_schema", serialize(_plain(schema)))


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
# Chat Completions: client.chat.send / send_async
# ---------------------------------------------------------------------------


def _patch_chat(chat: Any) -> None:
    if getattr(chat, "_neatlogs_openrouter_patched", False):
        return

    orig_send = getattr(chat, "send", None)
    orig_send_async = getattr(chat, "send_async", None)

    def _start(kwargs: dict, is_stream: bool) -> Any:
        model = kwargs.get("model", "")
        span = get_provider_tracer().start_span(
            name="openrouter.chat.send",
            attributes={
                "neatlogs.span.kind": "llm",
                "neatlogs.llm.provider": _PROVIDER,
                "neatlogs.llm.system": _vendor_from_model(model),
                "neatlogs.llm.model_name": model,
                "neatlogs.llm.is_streaming": is_stream,
            },
        )
        _set_chat_input(span, kwargs)
        _set_invocation_params(span, kwargs)
        return span

    if callable(orig_send):
        def patched_send(*args, **kwargs):
            if is_suppressed():
                return orig_send(*args, **kwargs)
            is_stream = bool(kwargs.get("stream", False))
            span = _start(kwargs, is_stream)
            start = time.perf_counter()
            try:
                response = orig_send(*args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            if is_stream:
                return SyncStreamWrapper(response, span, _finalize_chat_stream)
            _finalize_chat(span, response, (time.perf_counter() - start) * 1000)
            return response

        chat.send = patched_send

    if callable(orig_send_async):
        async def patched_send_async(*args, **kwargs):
            if is_suppressed():
                return await orig_send_async(*args, **kwargs)
            is_stream = bool(kwargs.get("stream", False))
            span = _start(kwargs, is_stream)
            start = time.perf_counter()
            try:
                response = await orig_send_async(*args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            if is_stream:
                return AsyncStreamWrapper(response, span, _finalize_chat_stream)
            _finalize_chat(span, response, (time.perf_counter() - start) * 1000)
            return response

        chat.send_async = patched_send_async

    chat._neatlogs_openrouter_patched = True


def _finalize_chat(span: Any, response: Any, duration_ms: float) -> None:
    """Extract attributes from a non-streaming ChatResult."""
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


def _finalize_chat_stream(span: Any, chunks: List[Any], duration_ms: float, ttft_ms: Optional[float]) -> None:
    text_parts: List[str] = []
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
# Responses API: client.beta.responses.send / send_async
# ---------------------------------------------------------------------------


def _patch_responses(responses: Any) -> None:
    if getattr(responses, "_neatlogs_openrouter_patched", False):
        return

    orig_send = getattr(responses, "send", None)
    orig_send_async = getattr(responses, "send_async", None)

    def _start(kwargs: dict, is_stream: bool) -> Any:
        model = kwargs.get("model", "")
        span = get_provider_tracer().start_span(
            name="openrouter.responses.send",
            attributes={
                "neatlogs.span.kind": "llm",
                "neatlogs.llm.provider": _PROVIDER,
                "neatlogs.llm.system": _vendor_from_model(model),
                "neatlogs.llm.model_name": model,
                "neatlogs.llm.is_streaming": is_stream,
            },
        )
        inp = kwargs.get("input")
        if inp is not None:
            span.set_attribute("neatlogs.llm.input_messages.0.role", "user")
            span.set_attribute(
                "neatlogs.llm.input_messages.0.content",
                inp if isinstance(inp, str) else serialize(_plain(inp)),
            )
            span.set_attribute("input.value", inp if isinstance(inp, str) else serialize(_plain(inp)))
        instructions = kwargs.get("instructions")
        if isinstance(instructions, str) and instructions:
            span.set_attribute("neatlogs.llm.system_prompt", instructions)
        _set_invocation_params(span, kwargs)
        return span

    if callable(orig_send):
        def patched_send(*args, **kwargs):
            if is_suppressed():
                return orig_send(*args, **kwargs)
            is_stream = bool(kwargs.get("stream", False))
            span = _start(kwargs, is_stream)
            start = time.perf_counter()
            try:
                response = orig_send(*args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            if is_stream:
                return SyncStreamWrapper(response, span, _finalize_responses_stream)
            _finalize_responses(span, response, (time.perf_counter() - start) * 1000)
            return response

        responses.send = patched_send

    if callable(orig_send_async):
        async def patched_send_async(*args, **kwargs):
            if is_suppressed():
                return await orig_send_async(*args, **kwargs)
            is_stream = bool(kwargs.get("stream", False))
            span = _start(kwargs, is_stream)
            start = time.perf_counter()
            try:
                response = await orig_send_async(*args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            if is_stream:
                return AsyncStreamWrapper(response, span, _finalize_responses_stream)
            _finalize_responses(span, response, (time.perf_counter() - start) * 1000)
            return response

        responses.send_async = patched_send_async

    responses._neatlogs_openrouter_patched = True


def _extract_responses_text(response: Any) -> Optional[str]:
    output_text = _get(response, "output_text", None)
    if output_text:
        return output_text
    output = _get(response, "output", None)
    if not isinstance(output, list):
        return None
    parts: List[str] = []
    for item in output:
        if _get(item, "type", None) == "message":
            for c in _get(item, "content", None) or []:
                if _get(c, "type", None) in ("output_text", "text"):
                    text = _get(c, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
    return "".join(parts) if parts else None


def _finalize_responses(span: Any, response: Any, duration_ms: float) -> None:
    text = _extract_responses_text(response)
    if text:
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        span.set_attribute("neatlogs.llm.output_messages.0.content", text)
        span.set_attribute("output.value", text)
    model = _get(response, "model", None)
    if model:
        span.set_attribute("neatlogs.llm.model_name", str(model))
    response_id = _get(response, "id", None)
    if response_id:
        span.set_attribute("neatlogs.llm.response_id", str(response_id))
    status = _get(response, "status", None)
    if status:
        span.set_attribute("neatlogs.llm.finish_reason", str(status))
    _set_responses_usage(span, _get(response, "usage", None))
    _ok(span, duration_ms)


def _finalize_responses_stream(span: Any, chunks: List[Any], duration_ms: float, ttft_ms: Optional[float]) -> None:
    text_parts: List[str] = []
    model = None
    usage = None
    for ev in chunks:
        ev_type = _get(ev, "type", "")
        if ev_type and "output_text.delta" in str(ev_type):
            d = _get(ev, "delta", None)
            if d:
                text_parts.append(d)
        resp = _get(ev, "response", None)
        if resp is not None:
            if _get(resp, "model", None):
                model = _get(resp, "model", None)
            if _get(resp, "usage", None):
                usage = _get(resp, "usage", None)
    if text_parts:
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        span.set_attribute("neatlogs.llm.output_messages.0.content", "".join(text_parts))
        span.set_attribute("output.value", "".join(text_parts))
    if model:
        span.set_attribute("neatlogs.llm.model_name", str(model))
    _set_responses_usage(span, usage)
    _ok(span, duration_ms, ttft_ms)


def _set_responses_usage(span: Any, usage: Any) -> None:
    if usage is None:
        return
    input_tokens = _get(usage, "input_tokens", None)
    output_tokens = _get(usage, "output_tokens", None)
    total = _get(usage, "total_tokens", None)
    if input_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.prompt", input_tokens)
    if output_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.completion", output_tokens)
    if total is not None:
        span.set_attribute("neatlogs.llm.token_count.total", total)
    elif input_tokens is not None and output_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.total", input_tokens + output_tokens)


# ---------------------------------------------------------------------------
# Embeddings: client.embeddings.generate
# ---------------------------------------------------------------------------


def _patch_embeddings(embeddings: Any) -> None:
    if getattr(embeddings, "_neatlogs_openrouter_patched", False):
        return
    orig = getattr(embeddings, "generate", None)
    if not callable(orig):
        return

    def patched(*args, **kwargs):
        if is_suppressed():
            return orig(*args, **kwargs)
        inp = kwargs.get("input", "")
        span = get_provider_tracer().start_span(
            name="openrouter.embeddings.generate",
            attributes={
                "neatlogs.span.kind": "embedding",
                "neatlogs.llm.provider": _PROVIDER,
                "neatlogs.embedding.model_name": kwargs.get("model", ""),
                "neatlogs.embedding.text": (inp if isinstance(inp, str) else serialize(_plain(inp)))[:10000],
            },
        )
        start = time.perf_counter()
        try:
            response = orig(*args, **kwargs)
        except Exception as e:
            _err(span, e)
            raise
        try:
            _finalize_embeddings(span, response, (time.perf_counter() - start) * 1000)
        except Exception:
            _ok(span, (time.perf_counter() - start) * 1000)
        return response

    embeddings.generate = patched
    embeddings._neatlogs_openrouter_patched = True


def _finalize_embeddings(span: Any, response: Any, duration_ms: float) -> None:
    # response is a CreateEmbeddingsResponse wrapper; the body may be under
    # common attribute names — probe a few.
    body = _first_present(response, ("data", "object_", "create_embeddings_response_body", "result"))
    data = _get(response, "data", None)
    if data is None and body is not None and body is not response:
        data = _get(body, "data", None)
    if data is not None:
        try:
            span.set_attribute("neatlogs.embedding.count", len(data))
            if data and _get(data[0], "embedding", None) is not None:
                span.set_attribute("neatlogs.embedding.dimensions", len(data[0].embedding))
        except (TypeError, AttributeError):
            pass
    usage = _get(response, "usage", None)
    if usage is None and body is not None:
        usage = _get(body, "usage", None)
    if usage is not None:
        prompt = _get(usage, "prompt_tokens", None)
        total = _get(usage, "total_tokens", None)
        if prompt is not None:
            span.set_attribute("neatlogs.llm.token_count.prompt", prompt)
        if total is not None:
            span.set_attribute("neatlogs.embedding.token_count", total)
    _ok(span, duration_ms)


def _first_present(obj: Any, names) -> Any:
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# Rerank: client.rerank.rerank
# ---------------------------------------------------------------------------


def _patch_rerank(rerank: Any) -> None:
    if getattr(rerank, "_neatlogs_openrouter_patched", False):
        return
    orig = getattr(rerank, "rerank", None)
    if not callable(orig):
        return

    def patched(*args, **kwargs):
        if is_suppressed():
            return orig(*args, **kwargs)
        documents = kwargs.get("documents", []) or []
        span = get_provider_tracer().start_span(
            name="openrouter.rerank.rerank",
            attributes={
                "neatlogs.span.kind": "reranker",
                "neatlogs.llm.provider": _PROVIDER,
                "neatlogs.reranker.model_name": kwargs.get("model", ""),
                "neatlogs.reranker.query": str(kwargs.get("query", "")),
            },
        )
        for i, doc in enumerate(documents):
            span.set_attribute(
                f"neatlogs.reranker.input_documents.{i}",
                doc if isinstance(doc, str) else serialize(_plain(doc)),
            )
        start = time.perf_counter()
        try:
            response = orig(*args, **kwargs)
        except Exception as e:
            _err(span, e)
            raise
        try:
            _finalize_rerank(span, response, (time.perf_counter() - start) * 1000)
        except Exception:
            _ok(span, (time.perf_counter() - start) * 1000)
        return response

    rerank.rerank = patched
    rerank._neatlogs_openrouter_patched = True


def _finalize_rerank(span: Any, response: Any, duration_ms: float) -> None:
    results = _get(response, "results", None)
    if results is None:
        body = _first_present(response, ("create_rerank_response_body", "result", "object_"))
        if body is not None:
            results = _get(body, "results", None)
    if results is not None:
        try:
            span.set_attribute("neatlogs.reranker.top_k", len(results))
            for i, r in enumerate(results):
                doc = _get(r, "document", None)
                text = _get(doc, "text", None) if doc is not None else None
                if text is not None:
                    span.set_attribute(f"neatlogs.reranker.output_documents.{i}", str(text))
        except (TypeError, AttributeError):
            pass
    _ok(span, duration_ms)


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
# Import-replacement: `from neatlogs.openrouter import OpenRouter`
# Patches OpenRouter.__init__ so every client constructed is auto-wrapped.
# ---------------------------------------------------------------------------

_PATCHED = False
_ORIG_INIT = None


def _patch_openrouter_module() -> None:
    global _PATCHED, _ORIG_INIT
    if _PATCHED:
        return
    try:
        from openrouter import OpenRouter as _OpenRouter
    except Exception:
        return

    _PATCHED = True
    _ORIG_INIT = _OpenRouter.__init__

    def _patched_init(self, *args, **kwargs):
        _ORIG_INIT(self, *args, **kwargs)
        wrap_openrouter_client(self)

    _OpenRouter.__init__ = _patched_init


def _unpatch_openrouter_module() -> None:
    global _PATCHED, _ORIG_INIT
    if not _PATCHED:
        return
    try:
        from openrouter import OpenRouter as _OpenRouter
    except Exception:
        return
    if _ORIG_INIT is not None:
        _OpenRouter.__init__ = _ORIG_INIT
    _PATCHED = False
    _ORIG_INIT = None


try:  # noqa: E402 - re-export for `from neatlogs.openrouter import OpenRouter`
    from openrouter import OpenRouter  # noqa: F401
except Exception:  # pragma: no cover - openrouter not installed
    pass
