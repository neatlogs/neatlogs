"""
Neatlogs Azure OpenAI wrapper.

Azure OpenAI is accessed through the ``openai`` SDK's ``AzureOpenAI`` /
``AsyncAzureOpenAI`` clients (subclasses of ``OpenAI`` / ``AsyncOpenAI``). This
module traces those clients with provider ``azure`` so Azure traffic is
distinguished from direct OpenAI traffic.

Two usage patterns:

  1. Explicit wrap (primary):
     >>> import neatlogs
     >>> from openai import AzureOpenAI
     >>> client = neatlogs.wrap(AzureOpenAI(azure_endpoint=..., api_version=...))
     >>> client.chat.completions.create(...)

  2. Import replacement:
     >>> from neatlogs.azure_openai import AzureOpenAI
     >>> client = AzureOpenAI(azure_endpoint=..., api_version=...)
     >>> client.chat.completions.create(...)

This module is intentionally self-contained (it does NOT import
``neatlogs.openai``) so importing it never patches the base ``OpenAI`` client.
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

_PROVIDER = "azure"
_SYSTEM = "azure"

_PATCHED = False
_ORIG_INIT = None
_ORIG_ASYNC_INIT = None


class AzureOpenAIInstrumentor:
    """Instrumentor class for InstrumentationManager integration."""

    def instrument(self, tracer_provider=None):
        _patch_azure_openai_module()

    def uninstrument(self):
        _unpatch_azure_openai_module()


def wrap_azure_openai_client(client: Any) -> Any:
    """
    Wrap an AzureOpenAI client instance (sync). Returns the same client with all
    LLM-relevant resources patched to auto-trace (parity with neatlogs.openai).
    """
    _safe(_patch_completions, _resource(client, "chat", "completions"))
    _safe(_patch_chat_parse, _resource(client, "chat", "completions"), sync=True)
    _safe(_patch_legacy_completions, getattr(client, "completions", None), sync=True)
    _safe(_patch_responses, getattr(client, "responses", None))
    _safe(_patch_responses_parse, getattr(client, "responses", None), sync=True)
    _safe(_patch_embeddings, getattr(client, "embeddings", None), sync=True)
    _safe(_patch_images, getattr(client, "images", None), sync=True)
    _safe(_patch_audio, getattr(client, "audio", None), sync=True)
    _safe(_patch_moderations, getattr(client, "moderations", None), sync=True)
    _safe(_patch_batches, getattr(client, "batches", None), sync=True)
    beta = getattr(client, "beta", None)
    if beta is not None:
        _safe(_patch_chat_parse, _resource(beta, "chat", "completions"), sync=True)
    return client


def wrap_async_azure_openai_client(client: Any) -> Any:
    """Wrap an AsyncAzureOpenAI client instance — full resource coverage."""
    _safe(_patch_async_completions, _resource(client, "chat", "completions"))
    _safe(_patch_chat_parse, _resource(client, "chat", "completions"), sync=False)
    _safe(_patch_legacy_completions, getattr(client, "completions", None), sync=False)
    _safe(_patch_async_responses, getattr(client, "responses", None))
    _safe(_patch_responses_parse, getattr(client, "responses", None), sync=False)
    _safe(_patch_embeddings, getattr(client, "embeddings", None), sync=False)
    _safe(_patch_images, getattr(client, "images", None), sync=False)
    _safe(_patch_audio, getattr(client, "audio", None), sync=False)
    _safe(_patch_moderations, getattr(client, "moderations", None), sync=False)
    _safe(_patch_batches, getattr(client, "batches", None), sync=False)
    beta = getattr(client, "beta", None)
    if beta is not None:
        _safe(_patch_chat_parse, _resource(beta, "chat", "completions"), sync=False)
    return client


def _resource(client: Any, *path):
    obj = client
    for p in path:
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj


def _safe(fn, resource, **kw):
    if resource is None:
        return
    try:
        fn(resource, **kw) if kw else fn(resource)
    except Exception:
        pass


def _chat_start_span(model: Any, is_stream: bool) -> Any:
    return get_provider_tracer().start_span(
        name="azure_openai.chat.completions.create",
        attributes={
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.provider": _PROVIDER,
            "neatlogs.llm.system": _SYSTEM,
            "neatlogs.llm.model_name": model,
            "neatlogs.llm.is_streaming": is_stream,
        },
    )


def _set_chat_input(span: Any, kwargs: dict) -> None:
    messages = kwargs.get("messages", [])
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        span.set_attribute(f"neatlogs.llm.input_messages.{i}.role", role)
        if isinstance(content, str):
            span.set_attribute(f"neatlogs.llm.input_messages.{i}.content", content)
        else:
            span.set_attribute(f"neatlogs.llm.input_messages.{i}.content", serialize(content))
        if msg.get("tool_call_id"):
            span.set_attribute(f"neatlogs.llm.input_messages.{i}.tool_call_id", msg["tool_call_id"])

    tools = kwargs.get("tools")
    if tools:
        for i, tool in enumerate(tools):
            fn = tool.get("function", {})
            span.set_attribute(f"neatlogs.llm.tools.{i}.name", fn.get("name", ""))
            if fn.get("description"):
                span.set_attribute(f"neatlogs.llm.tools.{i}.description", fn["description"])
            if fn.get("parameters"):
                span.set_attribute(f"neatlogs.llm.tools.{i}.input_schema", serialize(fn["parameters"]))

    for param in ("temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty"):
        if param in kwargs and kwargs[param] is not None:
            span.set_attribute(f"neatlogs.llm.{param}", kwargs[param])


def _patch_completions(completions: Any) -> None:
    if getattr(completions, "_neatlogs_azure_patched", False):
        return

    orig_create = completions.create

    def patched_create(*args, **kwargs):
        if is_suppressed():
            return orig_create(*args, **kwargs)

        model = kwargs.get("model", "")
        is_stream = kwargs.get("stream", False)

        if is_stream:
            opts = kwargs.get("stream_options") or {}
            if not opts.get("include_usage"):
                opts["include_usage"] = True
                kwargs["stream_options"] = opts

        span = _chat_start_span(model, is_stream)
        _set_chat_input(span, kwargs)

        start = time.perf_counter()
        try:
            response = orig_create(*args, **kwargs)
        except Exception as e:
            _err(span, e)
            raise

        if is_stream:
            return SyncStreamWrapper(response, span, _finalize_stream)

        _finalize_response(span, response, (time.perf_counter() - start) * 1000)
        return response

    completions.create = patched_create
    completions._neatlogs_azure_patched = True


def _patch_async_completions(completions: Any) -> None:
    if getattr(completions, "_neatlogs_azure_patched", False):
        return

    orig_create = completions.create

    async def patched_create(*args, **kwargs):
        if is_suppressed():
            return await orig_create(*args, **kwargs)

        model = kwargs.get("model", "")
        is_stream = kwargs.get("stream", False)

        if is_stream:
            opts = kwargs.get("stream_options") or {}
            if not opts.get("include_usage"):
                opts["include_usage"] = True
                kwargs["stream_options"] = opts

        span = _chat_start_span(model, is_stream)
        _set_chat_input(span, kwargs)

        start = time.perf_counter()
        try:
            response = await orig_create(*args, **kwargs)
        except Exception as e:
            _err(span, e)
            raise

        if is_stream:
            return AsyncStreamWrapper(response, span, _finalize_stream)

        _finalize_response(span, response, (time.perf_counter() - start) * 1000)
        return response

    completions.create = patched_create
    completions._neatlogs_azure_patched = True


def _patch_responses(responses: Any) -> None:
    if getattr(responses, "_neatlogs_azure_patched", False):
        return
    orig_create = responses.create

    def patched_create(*args, **kwargs):
        if is_suppressed():
            return orig_create(*args, **kwargs)
        span = _responses_start_span(kwargs)
        start = time.perf_counter()
        try:
            response = orig_create(*args, **kwargs)
        except Exception as e:
            _err(span, e)
            raise
        if kwargs.get("stream", False):
            return SyncStreamWrapper(response, span, _finalize_responses_stream)
        _finalize_responses_response(span, response, (time.perf_counter() - start) * 1000)
        return response

    responses.create = patched_create
    responses._neatlogs_azure_patched = True


def _patch_async_responses(responses: Any) -> None:
    if getattr(responses, "_neatlogs_azure_patched", False):
        return
    orig_create = responses.create

    async def patched_create(*args, **kwargs):
        if is_suppressed():
            return await orig_create(*args, **kwargs)
        span = _responses_start_span(kwargs)
        start = time.perf_counter()
        try:
            response = await orig_create(*args, **kwargs)
        except Exception as e:
            _err(span, e)
            raise
        if kwargs.get("stream", False):
            return AsyncStreamWrapper(response, span, _finalize_responses_stream)
        _finalize_responses_response(span, response, (time.perf_counter() - start) * 1000)
        return response

    responses.create = patched_create
    responses._neatlogs_azure_patched = True


def _responses_start_span(kwargs: dict) -> Any:
    return get_provider_tracer().start_span(
        name="azure_openai.responses.create",
        attributes={
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.provider": _PROVIDER,
            "neatlogs.llm.system": _SYSTEM,
            "neatlogs.llm.model_name": kwargs.get("model", ""),
            "neatlogs.llm.is_streaming": bool(kwargs.get("stream", False)),
            "neatlogs.llm.input_messages.0.role": "user",
            "neatlogs.llm.input_messages.0.content": serialize(kwargs.get("input", "")),
        },
    )


def _patch_embeddings(embeddings: Any, sync: bool = True) -> None:
    if getattr(embeddings, "_neatlogs_azure_patched", False):
        return
    orig = embeddings.create

    def _start(kwargs):
        inp = kwargs.get("input", "")
        attrs = {
            "neatlogs.span.kind": "embedding",
            "neatlogs.embedding.model_name": kwargs.get("model", ""),
            "neatlogs.llm.provider": _PROVIDER,
        }
        if isinstance(inp, str):
            attrs["neatlogs.embedding.text"] = inp[:10000]
        elif isinstance(inp, list):
            attrs["neatlogs.embedding.text"] = serialize(inp[:20])[:10000]
        return attrs

    def _finalize(span, response, duration_ms):
        usage = getattr(response, "usage", None)
        if usage is not None:
            if getattr(usage, "prompt_tokens", None) is not None:
                span.set_attribute("neatlogs.llm.token_count.prompt", usage.prompt_tokens)
            if getattr(usage, "total_tokens", None) is not None:
                span.set_attribute("neatlogs.embedding.token_count", usage.total_tokens)
        data = getattr(response, "data", None)
        if data is not None:
            try:
                span.set_attribute("neatlogs.embedding.count", len(data))
                if data and getattr(data[0], "embedding", None) is not None:
                    span.set_attribute("neatlogs.embedding.dimensions", len(data[0].embedding))
            except (TypeError, AttributeError):
                pass
        _ok(span, duration_ms)

    if sync:
        def patched(*args, **kwargs):
            if is_suppressed():
                return orig(*args, **kwargs)
            span = get_provider_tracer().start_span(name="azure_openai.embeddings.create", attributes=_start(kwargs))
            start = time.perf_counter()
            try:
                response = orig(*args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            _finalize(span, response, (time.perf_counter() - start) * 1000)
            return response
    else:
        async def patched(*args, **kwargs):
            if is_suppressed():
                return await orig(*args, **kwargs)
            span = get_provider_tracer().start_span(name="azure_openai.embeddings.create", attributes=_start(kwargs))
            start = time.perf_counter()
            try:
                response = await orig(*args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            _finalize(span, response, (time.perf_counter() - start) * 1000)
            return response

    embeddings.create = patched
    embeddings._neatlogs_azure_patched = True


def _finalize_response(span: Any, response: Any, duration_ms: float) -> None:
    """Extract attributes from a non-streaming ChatCompletion response."""
    choices = getattr(response, "choices", []) or []
    for i, choice in enumerate(choices):
        message = getattr(choice, "message", None)
        if not message:
            continue
        span.set_attribute(f"neatlogs.llm.output_messages.{i}.role", "assistant")
        if getattr(message, "content", None):
            span.set_attribute(f"neatlogs.llm.output_messages.{i}.content", message.content)
        if getattr(message, "tool_calls", None):
            for j, tc in enumerate(message.tool_calls):
                span.set_attribute(f"neatlogs.llm.tool_calls.{j}.id", tc.id)
                span.set_attribute(f"neatlogs.llm.tool_calls.{j}.name", tc.function.name)
                span.set_attribute(f"neatlogs.llm.tool_calls.{j}.arguments", tc.function.arguments)
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason:
            span.set_attribute("neatlogs.llm.finish_reason", finish_reason)

    usage = getattr(response, "usage", None)
    if usage:
        if getattr(usage, "prompt_tokens", None) is not None:
            span.set_attribute("neatlogs.llm.token_count.prompt", usage.prompt_tokens)
        if getattr(usage, "completion_tokens", None) is not None:
            span.set_attribute("neatlogs.llm.token_count.completion", usage.completion_tokens)
        total = getattr(usage, "total_tokens", None)
        if total is not None:
            span.set_attribute("neatlogs.llm.token_count.total", total)
        if getattr(usage, "prompt_tokens_details", None):
            cached = getattr(usage.prompt_tokens_details, "cached_tokens", None)
            if cached is not None:
                span.set_attribute("neatlogs.llm.token_count.cache_read", cached)
        if getattr(usage, "completion_tokens_details", None):
            reasoning = getattr(usage.completion_tokens_details, "reasoning_tokens", None)
            if reasoning is not None:
                span.set_attribute("neatlogs.llm.token_count.reasoning", reasoning)

    model = getattr(response, "model", None)
    if model:
        span.set_attribute("neatlogs.llm.model_name", model)

    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()


def _finalize_stream(span: Any, chunks: List[Any], duration_ms: float, ttft_ms: Optional[float]) -> None:
    text_parts: List[str] = []
    tool_calls_acc: dict = {}
    finish_reason = None
    model = None
    usage = None

    for chunk in chunks:
        # Usage may ride with a choices-bearing chunk (some OpenAI-compatible
        # gateways) rather than a dedicated choices-empty one — read it always.
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if not getattr(chunk, "choices", None):
            continue
        choice = chunk.choices[0]
        delta = getattr(choice, "delta", None)
        if not delta:
            continue
        if getattr(delta, "content", None):
            text_parts.append(delta.content)
        for tc in getattr(delta, "tool_calls", None) or []:
            idx = tc.index
            if idx not in tool_calls_acc:
                tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
            if tc.id:
                tool_calls_acc[idx]["id"] = tc.id
            if tc.function and tc.function.name:
                tool_calls_acc[idx]["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                tool_calls_acc[idx]["arguments"] += tc.function.arguments
        if getattr(choice, "finish_reason", None):
            finish_reason = choice.finish_reason
        if getattr(chunk, "model", None):
            model = chunk.model

    full_text = "".join(text_parts)
    if full_text:
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        span.set_attribute("neatlogs.llm.output_messages.0.content", full_text)

    for j, tc in enumerate(tool_calls_acc.values()):
        span.set_attribute(f"neatlogs.llm.tool_calls.{j}.id", tc["id"])
        span.set_attribute(f"neatlogs.llm.tool_calls.{j}.name", tc["name"])
        span.set_attribute(f"neatlogs.llm.tool_calls.{j}.arguments", tc["arguments"])

    if model:
        span.set_attribute("neatlogs.llm.model_name", model)
    if finish_reason:
        span.set_attribute("neatlogs.llm.finish_reason", finish_reason)

    if usage:
        if getattr(usage, "prompt_tokens", None) is not None:
            span.set_attribute("neatlogs.llm.token_count.prompt", usage.prompt_tokens)
        if getattr(usage, "completion_tokens", None) is not None:
            span.set_attribute("neatlogs.llm.token_count.completion", usage.completion_tokens)
        total = getattr(usage, "total_tokens", None)
        if total is not None:
            span.set_attribute("neatlogs.llm.token_count.total", total)

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


def _finalize_responses_response(span: Any, response: Any, duration_ms: float) -> None:
    output_text = getattr(response, "output_text", None)
    if output_text:
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        span.set_attribute("neatlogs.llm.output_messages.0.content", output_text)
    model = getattr(response, "model", None)
    if model:
        span.set_attribute("neatlogs.llm.model_name", model)
    usage = getattr(response, "usage", None)
    if usage:
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        if input_tokens is not None:
            span.set_attribute("neatlogs.llm.token_count.prompt", input_tokens)
        if output_tokens is not None:
            span.set_attribute("neatlogs.llm.token_count.completion", output_tokens)
    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()


def _finalize_responses_stream(span: Any, chunks: List[Any], duration_ms: float, ttft_ms: Optional[float]) -> None:
    text_parts: List[str] = []
    model = None
    usage = None
    for ev in chunks:
        ev_type = getattr(ev, "type", "")
        if ev_type == "response.output_text.delta":
            d = getattr(ev, "delta", None)
            if d:
                text_parts.append(d)
        resp = getattr(ev, "response", None)
        if resp is not None:
            if getattr(resp, "model", None):
                model = resp.model
            if getattr(resp, "usage", None):
                usage = resp.usage
    if text_parts:
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        span.set_attribute("neatlogs.llm.output_messages.0.content", "".join(text_parts)[:10000])
    if model:
        span.set_attribute("neatlogs.llm.model_name", model)
    if usage:
        it = getattr(usage, "input_tokens", None)
        ot_ = getattr(usage, "output_tokens", None)
        if it is not None:
            span.set_attribute("neatlogs.llm.token_count.prompt", it)
        if ot_ is not None:
            span.set_attribute("neatlogs.llm.token_count.completion", ot_)
    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    if ttft_ms is not None:
        span.set_attribute("neatlogs.llm.metrics.ttft_ms", round(ttft_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()


def _ok(span, duration_ms):
    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()


def _err(span: Any, e: Exception) -> None:
    span.set_status(StatusCode.ERROR, str(e))
    span.record_exception(e)
    span.end()


# ---------------------------------------------------------------------------
# Generic method-patching helper (used by the additional resources below)
# ---------------------------------------------------------------------------


def _patch_method(resource: Any, method_name: str, flag: str, start_attrs, finalize, is_async: bool) -> None:
    """
    Wrap resource.<method_name> with a span. start_attrs(kwargs)->dict builds the
    initial attributes; finalize(span, response, duration_ms) records the result.
    Idempotent per-resource via `flag`.
    """
    if resource is None or getattr(resource, flag, False) or not hasattr(resource, method_name):
        return
    orig = getattr(resource, method_name)

    if is_async:
        async def patched(*args, **kwargs):
            if is_suppressed():
                return await orig(*args, **kwargs)
            span = get_provider_tracer().start_span(name=start_attrs.__name__, attributes=start_attrs(kwargs))
            start = time.perf_counter()
            try:
                response = await orig(*args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            try:
                finalize(span, response, (time.perf_counter() - start) * 1000)
            except Exception:
                span.set_status(StatusCode.OK); span.end()
            return response
    else:
        def patched(*args, **kwargs):
            if is_suppressed():
                return orig(*args, **kwargs)
            span = get_provider_tracer().start_span(name=start_attrs.__name__, attributes=start_attrs(kwargs))
            start = time.perf_counter()
            try:
                response = orig(*args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            try:
                finalize(span, response, (time.perf_counter() - start) * 1000)
            except Exception:
                span.set_status(StatusCode.OK); span.end()
            return response

    setattr(resource, method_name, patched)
    setattr(resource, flag, True)


# ---------------------------------------------------------------------------
# chat.completions.parse / beta.chat.completions.parse (structured outputs)
# ---------------------------------------------------------------------------


def _patch_chat_parse(completions: Any, sync: bool = True) -> None:
    def start_attrs(kwargs):
        return {
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.provider": _PROVIDER,
            "neatlogs.llm.system": _SYSTEM,
            "neatlogs.llm.model_name": kwargs.get("model", ""),
            "neatlogs.llm.structured_output": True,
        }
    start_attrs.__name__ = "azure_openai.chat.completions.parse"

    def finalize(span, response, duration_ms):
        _finalize_response(span, response, duration_ms)

    _patch_method(completions, "parse", "_neatlogs_azure_parse_patched", start_attrs, finalize, is_async=not sync)


def _patch_responses_parse(responses: Any, sync: bool = True) -> None:
    def start_attrs(kwargs):
        return {
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.provider": _PROVIDER,
            "neatlogs.llm.system": _SYSTEM,
            "neatlogs.llm.model_name": kwargs.get("model", ""),
            "neatlogs.llm.structured_output": True,
            "neatlogs.llm.input_messages.0.role": "user",
            "neatlogs.llm.input_messages.0.content": serialize(kwargs.get("input", "")),
        }
    start_attrs.__name__ = "azure_openai.responses.parse"

    def finalize(span, response, duration_ms):
        _finalize_responses_response(span, response, duration_ms)

    _patch_method(responses, "parse", "_neatlogs_azure_parse_patched", start_attrs, finalize, is_async=not sync)


# ---------------------------------------------------------------------------
# Legacy completions (text)
# ---------------------------------------------------------------------------


def _patch_legacy_completions(completions: Any, sync: bool = True) -> None:
    def start_attrs(kwargs):
        prompt = kwargs.get("prompt", "")
        return {
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.provider": _PROVIDER,
            "neatlogs.llm.system": _SYSTEM,
            "neatlogs.llm.model_name": kwargs.get("model", ""),
            "neatlogs.llm.input_messages.0.role": "user",
            "neatlogs.llm.input_messages.0.content": (prompt if isinstance(prompt, str) else serialize(prompt))[:10000],
        }
    start_attrs.__name__ = "azure_openai.completions.create"

    def finalize(span, response, duration_ms):
        choices = getattr(response, "choices", []) or []
        if choices:
            text = getattr(choices[0], "text", None)
            if text:
                span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
                span.set_attribute("neatlogs.llm.output_messages.0.content", text[:10000])
            fr = getattr(choices[0], "finish_reason", None)
            if fr:
                span.set_attribute("neatlogs.llm.finish_reason", fr)
        usage = getattr(response, "usage", None)
        if usage:
            if getattr(usage, "prompt_tokens", None) is not None:
                span.set_attribute("neatlogs.llm.token_count.prompt", usage.prompt_tokens)
            if getattr(usage, "completion_tokens", None) is not None:
                span.set_attribute("neatlogs.llm.token_count.completion", usage.completion_tokens)
            if getattr(usage, "total_tokens", None) is not None:
                span.set_attribute("neatlogs.llm.token_count.total", usage.total_tokens)
        _ok(span, duration_ms)

    _patch_method(completions, "create", "_neatlogs_azure_legacy_patched", start_attrs, finalize, is_async=not sync)


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def _patch_images(images: Any, sync: bool = True) -> None:
    for method, span_name in (("generate", "azure_openai.images.generate"),
                              ("edit", "azure_openai.images.edit"),
                              ("create_variation", "azure_openai.images.create_variation")):
        def make(method=method, span_name=span_name):
            def start_attrs(kwargs):
                attrs = {"neatlogs.span.kind": "llm", "neatlogs.llm.provider": _PROVIDER, "neatlogs.llm.task": "image"}
                if kwargs.get("model"):
                    attrs["neatlogs.llm.model_name"] = kwargs["model"]
                if kwargs.get("prompt"):
                    attrs["input.value"] = str(kwargs["prompt"])[:10000]
                if kwargs.get("size"):
                    attrs["neatlogs.image.size"] = str(kwargs["size"])
                return attrs
            start_attrs.__name__ = span_name

            def finalize(span, response, duration_ms):
                data = getattr(response, "data", None)
                if data is not None:
                    try:
                        span.set_attribute("neatlogs.image.count", len(data))
                    except TypeError:
                        pass
                _ok(span, duration_ms)
            return start_attrs, finalize

        sa, fin = make()
        _patch_method(images, method, f"_neatlogs_azure_{method}_patched", sa, fin, is_async=not sync)


# ---------------------------------------------------------------------------
# Audio (speech / transcriptions / translations)
# ---------------------------------------------------------------------------


def _patch_audio(audio: Any, sync: bool = True) -> None:
    speech = getattr(audio, "speech", None)
    if speech is not None:
        def start_attrs(kwargs):
            attrs = {"neatlogs.span.kind": "llm", "neatlogs.llm.provider": _PROVIDER, "neatlogs.llm.task": "tts"}
            if kwargs.get("model"):
                attrs["neatlogs.llm.model_name"] = kwargs["model"]
            if kwargs.get("input"):
                attrs["input.value"] = str(kwargs["input"])[:10000]
            if kwargs.get("voice"):
                attrs["neatlogs.audio.voice"] = str(kwargs["voice"])
            return attrs
        start_attrs.__name__ = "azure_openai.audio.speech.create"
        _patch_method(speech, "create", "_neatlogs_azure_patched", start_attrs, lambda s, r, d: _ok(s, d), is_async=not sync)

    for sub, task in (("transcriptions", "stt"), ("translations", "translation")):
        res = getattr(audio, sub, None)
        if res is None:
            continue
        def make(task=task, sub=sub):
            def start_attrs(kwargs):
                attrs = {"neatlogs.span.kind": "llm", "neatlogs.llm.provider": _PROVIDER, "neatlogs.llm.task": task}
                if kwargs.get("model"):
                    attrs["neatlogs.llm.model_name"] = kwargs["model"]
                return attrs
            start_attrs.__name__ = f"azure_openai.audio.{sub}.create"

            def finalize(span, response, duration_ms):
                text = getattr(response, "text", None)
                if text:
                    span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
                    span.set_attribute("neatlogs.llm.output_messages.0.content", str(text)[:10000])
                _ok(span, duration_ms)
            return start_attrs, finalize
        sa, fin = make()
        _patch_method(res, "create", "_neatlogs_azure_patched", sa, fin, is_async=not sync)


# ---------------------------------------------------------------------------
# Moderations
# ---------------------------------------------------------------------------


def _patch_moderations(moderations: Any, sync: bool = True) -> None:
    def start_attrs(kwargs):
        attrs = {"neatlogs.span.kind": "llm", "neatlogs.llm.provider": _PROVIDER, "neatlogs.llm.task": "moderation"}
        if kwargs.get("model"):
            attrs["neatlogs.llm.model_name"] = kwargs["model"]
        inp = kwargs.get("input")
        if inp:
            attrs["input.value"] = (inp if isinstance(inp, str) else serialize(inp))[:10000]
        return attrs
    start_attrs.__name__ = "azure_openai.moderations.create"

    def finalize(span, response, duration_ms):
        results = getattr(response, "results", None)
        if results:
            try:
                span.set_attribute("neatlogs.moderation.flagged", bool(getattr(results[0], "flagged", False)))
            except (TypeError, AttributeError):
                pass
        _ok(span, duration_ms)

    _patch_method(moderations, "create", "_neatlogs_azure_patched", start_attrs, finalize, is_async=not sync)


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------


def _patch_batches(batches: Any, sync: bool = True) -> None:
    def start_attrs(kwargs):
        return {"neatlogs.span.kind": "task", "neatlogs.batch.endpoint": kwargs.get("endpoint", "")}
    start_attrs.__name__ = "azure_openai.batches.create"

    def finalize(span, response, duration_ms):
        bid = getattr(response, "id", None)
        if bid:
            span.set_attribute("neatlogs.batch.id", str(bid))
        status = getattr(response, "status", None)
        if status:
            span.set_attribute("neatlogs.batch.status", str(status))
        _ok(span, duration_ms)

    _patch_method(batches, "create", "_neatlogs_azure_patched", start_attrs, finalize, is_async=not sync)


# ---------------------------------------------------------------------------
# Import-replacement: `from neatlogs.azure_openai import AzureOpenAI`
# Patches AzureOpenAI/AsyncAzureOpenAI.__init__ so every client is auto-wrapped.
# ---------------------------------------------------------------------------

def _patch_azure_openai_module() -> None:
    global _PATCHED, _ORIG_INIT, _ORIG_ASYNC_INIT
    if _PATCHED:
        return

    try:
        from openai import AsyncAzureOpenAI as _AsyncAzureOpenAI
        from openai import AzureOpenAI as _AzureOpenAI
    except Exception:
        return

    _PATCHED = True
    _ORIG_INIT = _AzureOpenAI.__init__
    _ORIG_ASYNC_INIT = _AsyncAzureOpenAI.__init__

    def _patched_init(self, *args, **kwargs):
        _ORIG_INIT(self, *args, **kwargs)
        wrap_azure_openai_client(self)

    _AzureOpenAI.__init__ = _patched_init

    def _patched_async_init(self, *args, **kwargs):
        _ORIG_ASYNC_INIT(self, *args, **kwargs)
        wrap_async_azure_openai_client(self)

    _AsyncAzureOpenAI.__init__ = _patched_async_init


def _unpatch_azure_openai_module() -> None:
    global _PATCHED, _ORIG_INIT, _ORIG_ASYNC_INIT
    if not _PATCHED:
        return

    try:
        from openai import AsyncAzureOpenAI as _AsyncAzureOpenAI
        from openai import AzureOpenAI as _AzureOpenAI
    except Exception:
        return

    if _ORIG_INIT is not None:
        _AzureOpenAI.__init__ = _ORIG_INIT
    if _ORIG_ASYNC_INIT is not None:
        _AsyncAzureOpenAI.__init__ = _ORIG_ASYNC_INIT

    _PATCHED = False
    _ORIG_INIT = None
    _ORIG_ASYNC_INIT = None


_patch_azure_openai_module()

try:  # noqa: E402
    from openai import AsyncAzureOpenAI, AzureOpenAI  # noqa: F401
except Exception:  # pragma: no cover - openai not installed
    pass
