"""
Neatlogs Vertex AI wrapper.

Vertex AI is accessed through the same ``google-genai`` SDK as Gemini, but in
Vertex mode (``genai.Client(vertexai=True, project=..., location=...)``). This
module traces those clients with provider ``vertex_ai`` so Vertex traffic is
distinguished from Google AI Studio (Gemini) traffic.

Two usage patterns:

  1. Explicit wrap (primary):
     >>> import neatlogs
     >>> from google import genai
     >>> client = neatlogs.wrap(genai.Client(vertexai=True, project="p", location="us-central1"))
     >>> client.models.generate_content(model="gemini-2.0-flash", contents="Hello")

  2. Import replacement:
     >>> from neatlogs.vertex_ai import genai
     >>> client = genai.Client(vertexai=True, project="p", location="us-central1")
     >>> client.models.generate_content(model="gemini-2.0-flash", contents="Hello")

This module is intentionally self-contained (it does NOT import
``neatlogs.google_genai``) so importing it never patches ``genai.Client`` for
the non-Vertex Gemini path.
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

_PROVIDER = "vertex_ai"
_SYSTEM = "vertexai"

_PATCHED = False
_ORIG_INIT = None


class VertexAIInstrumentor:
    """Instrumentor class for InstrumentationManager integration."""

    def instrument(self, tracer_provider=None):
        _patch_vertex_ai_module()

    def uninstrument(self):
        _unpatch_vertex_ai_module()


def _is_vertex_client(client: Any) -> bool:
    """True if a google.genai Client is configured for Vertex AI."""
    if getattr(client, "vertexai", None):
        return True
    api_client = getattr(client, "_api_client", None)
    return bool(getattr(api_client, "vertexai", False))


def wrap_vertex_ai_client(client: Any) -> Any:
    """
    Wrap a google.genai.Client running in Vertex mode. Patches models
    (generate_content + generate_content_stream) on sync and async, and chat
    sessions (Chat/AsyncChat send_message + send_message_stream).
    """
    _patch_models(client.models)
    _patch_models_extra(client.models, is_async=False)
    if hasattr(client, "aio") and hasattr(client.aio, "models"):
        _patch_async_models(client.aio.models)
        _patch_models_extra(client.aio.models, is_async=True)
    _patch_chat_classes()
    return client


def _start_span(model: Any, stream: bool, name: str = "vertex_ai.models.generate_content") -> Any:
    return get_tracer().start_span(
        name=name,
        attributes={
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.provider": _PROVIDER,
            "neatlogs.llm.system": _SYSTEM,
            "neatlogs.llm.model_name": str(model),
            "neatlogs.llm.is_streaming": bool(stream),
        },
    )


def _patch_models(models: Any) -> None:
    if getattr(models, "_neatlogs_vertex_patched", False):
        return

    orig_generate = models.generate_content
    orig_stream = getattr(models, "generate_content_stream", None)

    def patched_generate_content(*args, **kwargs):
        if is_suppressed():
            return orig_generate(*args, **kwargs)

        model = kwargs.get("model", args[0] if args else "")
        contents = kwargs.get("contents", args[1] if len(args) > 1 else "")

        span = _start_span(model, stream=False)
        _set_input_attributes(span, contents, kwargs)

        start = time.perf_counter()
        try:
            response = orig_generate(*args, **kwargs)
        except Exception as e:
            _err(span, e)
            raise

        _finalize_response(span, response, (time.perf_counter() - start) * 1000)
        return response

    models.generate_content = patched_generate_content

    if orig_stream:
        def patched_generate_content_stream(*args, **kwargs):
            if is_suppressed():
                return orig_stream(*args, **kwargs)

            model = kwargs.get("model", args[0] if args else "")
            contents = kwargs.get("contents", args[1] if len(args) > 1 else "")

            span = _start_span(model, stream=True)
            _set_input_attributes(span, contents, kwargs)

            stream = orig_stream(*args, **kwargs)
            return SyncStreamWrapper(stream, span, _finalize_stream)

        models.generate_content_stream = patched_generate_content_stream

    models._neatlogs_vertex_patched = True


def _patch_async_models(models: Any) -> None:
    if getattr(models, "_neatlogs_vertex_patched", False):
        return

    orig_generate = models.generate_content
    orig_stream = getattr(models, "generate_content_stream", None)

    async def patched_generate_content(*args, **kwargs):
        if is_suppressed():
            return await orig_generate(*args, **kwargs)

        model = kwargs.get("model", args[0] if args else "")
        contents = kwargs.get("contents", args[1] if len(args) > 1 else "")

        span = _start_span(model, stream=False)
        _set_input_attributes(span, contents, kwargs)

        start = time.perf_counter()
        try:
            response = await orig_generate(*args, **kwargs)
        except Exception as e:
            _err(span, e)
            raise

        _finalize_response(span, response, (time.perf_counter() - start) * 1000)
        return response

    models.generate_content = patched_generate_content

    if orig_stream:
        # The async google-genai generate_content_stream is a COROUTINE that
        # returns an async iterator. Preserve that contract: an async def that
        # RETURNS an AsyncStreamWrapper (not an async generator).
        async def patched_generate_content_stream(*args, **kwargs):
            if is_suppressed():
                return await orig_stream(*args, **kwargs)

            model = kwargs.get("model", args[0] if args else "")
            contents = kwargs.get("contents", args[1] if len(args) > 1 else "")

            span = _start_span(model, stream=True)
            _set_input_attributes(span, contents, kwargs)

            try:
                stream = await orig_stream(*args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise

            return AsyncStreamWrapper(stream, span, _finalize_stream)

        models.generate_content_stream = patched_generate_content_stream

    models._neatlogs_vertex_patched = True


def _set_input_attributes(span: Any, contents: Any, kwargs: dict) -> None:
    """Set input attributes from contents and config."""
    config = kwargs.get("config")

    idx = 0
    if config:
        system_instruction = (
            getattr(config, "system_instruction", None)
            if not isinstance(config, dict)
            else config.get("system_instruction")
        )
        if system_instruction:
            span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", "system")
            if isinstance(system_instruction, str):
                span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", system_instruction)
            else:
                span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", serialize(system_instruction))
            idx += 1

    if isinstance(contents, str):
        span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", "user")
        span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", contents)
    elif isinstance(contents, list):
        for item in contents:
            if isinstance(item, str):
                span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", "user")
                span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", item)
                idx += 1
            elif isinstance(item, dict):
                role = item.get("role", "user")
                parts = item.get("parts", [])
                span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", role)
                text_parts = []
                for part in parts:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and part.get("text"):
                        text_parts.append(part["text"])
                if text_parts:
                    span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", "\n".join(text_parts))
                else:
                    span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", serialize(parts))
                idx += 1
            else:
                span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", "user")
                span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", serialize(item))
                idx += 1
    else:
        span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", "user")
        span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", serialize(contents))

    tools = None
    if config:
        tools = getattr(config, "tools", None) if not isinstance(config, dict) else config.get("tools")
    if tools:
        for i, tool in enumerate(tools):
            if isinstance(tool, dict):
                fn_decls = tool.get("function_declarations", [])
                for j, fn in enumerate(fn_decls):
                    span.set_attribute(f"neatlogs.llm.tools.{i + j}.name", fn.get("name", ""))
                    if fn.get("description"):
                        span.set_attribute(f"neatlogs.llm.tools.{i + j}.description", fn["description"])
                    if fn.get("parameters"):
                        span.set_attribute(f"neatlogs.llm.tools.{i + j}.input_schema", serialize(fn["parameters"]))

    if config:
        cfg = config if isinstance(config, dict) else config.__dict__ if hasattr(config, "__dict__") else {}
        if isinstance(cfg, dict):
            # Set individual attrs AND the invocation_parameters blob: the backend
            # surfaces model settings to the UI ONLY from the JSON blob.
            params = {}
            for param in ("temperature", "top_p", "top_k", "max_output_tokens",
                          "frequency_penalty", "presence_penalty"):
                val = cfg.get(param)
                if val is not None:
                    attr_name = "max_tokens" if param == "max_output_tokens" else param
                    span.set_attribute(f"neatlogs.llm.{attr_name}", val)
                    params[param] = val
            if params:
                span.set_attribute("neatlogs.llm.invocation_parameters", serialize(params))


def _finalize_response(span: Any, response: Any, duration_ms: float) -> None:
    """Extract attributes from a non-streaming GenerateContentResponse."""
    candidates = getattr(response, "candidates", None) or []
    text_parts: List[str] = []
    tool_call_idx = 0

    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", None) or []
        for part in parts:
            if getattr(part, "text", None) and not getattr(part, "thought", False):
                text_parts.append(part.text)
            elif getattr(part, "thought", False) and getattr(part, "text", None):
                span.set_attribute("neatlogs.llm.output_messages.0.thinking", part.text)
            elif getattr(part, "function_call", None):
                fc = part.function_call
                span.set_attribute(f"neatlogs.llm.tool_calls.{tool_call_idx}.name", getattr(fc, "name", ""))
                args = getattr(fc, "args", None)
                span.set_attribute(f"neatlogs.llm.tool_calls.{tool_call_idx}.arguments", serialize(args) if args else "{}")
                tool_call_idx += 1

        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason:
            span.set_attribute("neatlogs.llm.finish_reason", str(finish_reason))

    if text_parts:
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        span.set_attribute("neatlogs.llm.output_messages.0.content", "".join(text_parts))

    usage = getattr(response, "usage_metadata", None)
    if usage:
        _set_usage_attributes(span, usage)

    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()


def _set_usage_attributes(span: Any, usage: Any) -> None:
    """Set token usage attributes from Vertex/GenAI usage_metadata."""
    prompt_tokens = getattr(usage, "prompt_token_count", None)
    if prompt_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.prompt", prompt_tokens)

    completion_tokens = getattr(usage, "candidates_token_count", None)
    if completion_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.completion", completion_tokens)

    total_tokens = getattr(usage, "total_token_count", None)
    if total_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.total", total_tokens)

    cached_tokens = getattr(usage, "cached_content_token_count", None)
    if cached_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.cache_read", cached_tokens)

    reasoning_tokens = getattr(usage, "thoughts_token_count", None)
    if reasoning_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.reasoning", reasoning_tokens)


def _finalize_stream(span: Any, chunks: List[Any], duration_ms: float, ttft_ms: Optional[float]) -> None:
    """Finalize a streaming response span from accumulated chunks."""
    text_parts: List[str] = []
    thinking_parts: List[str] = []
    tool_calls_acc: List[dict] = []
    finish_reason = None
    usage = None

    for chunk in chunks:
        candidates = getattr(chunk, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                if getattr(part, "text", None) and not getattr(part, "thought", False):
                    text_parts.append(part.text)
                elif getattr(part, "thought", False) and getattr(part, "text", None):
                    thinking_parts.append(part.text)
                elif getattr(part, "function_call", None):
                    fc = part.function_call
                    tool_calls_acc.append({
                        "name": getattr(fc, "name", ""),
                        "arguments": serialize(getattr(fc, "args", None) or {}),
                    })

            fr = getattr(candidate, "finish_reason", None)
            if fr:
                finish_reason = str(fr)

        chunk_usage = getattr(chunk, "usage_metadata", None)
        if chunk_usage:
            usage = chunk_usage

    full_text = "".join(text_parts)
    if full_text:
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        span.set_attribute("neatlogs.llm.output_messages.0.content", full_text)

    full_thinking = "".join(thinking_parts)
    if full_thinking:
        span.set_attribute("neatlogs.llm.output_messages.0.thinking", full_thinking)

    for j, tc in enumerate(tool_calls_acc):
        span.set_attribute(f"neatlogs.llm.tool_calls.{j}.name", tc["name"])
        span.set_attribute(f"neatlogs.llm.tool_calls.{j}.arguments", tc["arguments"])

    if finish_reason:
        span.set_attribute("neatlogs.llm.finish_reason", finish_reason)

    if usage:
        _set_usage_attributes(span, usage)

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


# ---------------------------------------------------------------------------
# embed_content + count_tokens (sync + async)
# ---------------------------------------------------------------------------


def _patch_models_extra(models: Any, is_async: bool) -> None:
    # embed_content
    if hasattr(models, "embed_content") and not getattr(models, "_neatlogs_vertex_embed_patched", False):
        orig = models.embed_content

        def _embed_attrs(kwargs):
            attrs = {
                "neatlogs.span.kind": "embedding",
                "neatlogs.embedding.model_name": str(kwargs.get("model", "")),
                "neatlogs.llm.provider": _PROVIDER,
            }
            contents = kwargs.get("contents")
            if contents is not None:
                attrs["neatlogs.embedding.text"] = (contents if isinstance(contents, str) else serialize(contents))[:10000]
            return attrs

        def _embed_finalize(span, resp):
            embeddings = getattr(resp, "embeddings", None)
            if embeddings is not None:
                try:
                    span.set_attribute("neatlogs.embedding.count", len(embeddings))
                    vals = getattr(embeddings[0], "values", None)
                    if vals is not None:
                        span.set_attribute("neatlogs.embedding.dimensions", len(vals))
                except (TypeError, AttributeError, IndexError):
                    pass
            span.set_status(StatusCode.OK)
            span.end()

        if is_async:
            async def patched_embed(*args, **kwargs):
                if is_suppressed():
                    return await orig(*args, **kwargs)
                span = get_tracer().start_span(name="vertex_ai.models.embed_content", attributes=_embed_attrs(kwargs))
                try:
                    resp = await orig(*args, **kwargs)
                except Exception as e:
                    _err(span, e); raise
                _embed_finalize(span, resp); return resp
        else:
            def patched_embed(*args, **kwargs):
                if is_suppressed():
                    return orig(*args, **kwargs)
                span = get_tracer().start_span(name="vertex_ai.models.embed_content", attributes=_embed_attrs(kwargs))
                try:
                    resp = orig(*args, **kwargs)
                except Exception as e:
                    _err(span, e); raise
                _embed_finalize(span, resp); return resp
        models.embed_content = patched_embed
        models._neatlogs_vertex_embed_patched = True

    # count_tokens
    if hasattr(models, "count_tokens") and not getattr(models, "_neatlogs_vertex_count_patched", False):
        orig_ct = models.count_tokens

        def _ct_attrs(kwargs):
            return {"neatlogs.span.kind": "llm", "neatlogs.llm.provider": _PROVIDER,
                    "neatlogs.llm.task": "count_tokens", "neatlogs.llm.model_name": str(kwargs.get("model", ""))}

        def _ct_finalize(span, resp):
            total = getattr(resp, "total_tokens", None)
            if total is not None:
                span.set_attribute("neatlogs.llm.token_count.prompt", total)
            span.set_status(StatusCode.OK)
            span.end()

        if is_async:
            async def patched_ct(*args, **kwargs):
                if is_suppressed():
                    return await orig_ct(*args, **kwargs)
                span = get_tracer().start_span(name="vertex_ai.models.count_tokens", attributes=_ct_attrs(kwargs))
                try:
                    resp = await orig_ct(*args, **kwargs)
                except Exception as e:
                    _err(span, e); raise
                _ct_finalize(span, resp); return resp
        else:
            def patched_ct(*args, **kwargs):
                if is_suppressed():
                    return orig_ct(*args, **kwargs)
                span = get_tracer().start_span(name="vertex_ai.models.count_tokens", attributes=_ct_attrs(kwargs))
                try:
                    resp = orig_ct(*args, **kwargs)
                except Exception as e:
                    _err(span, e); raise
                _ct_finalize(span, resp); return resp
        models.count_tokens = patched_ct
        models._neatlogs_vertex_count_patched = True


# ---------------------------------------------------------------------------
# Chat sessions (Chat / AsyncChat send_message + send_message_stream)
# ---------------------------------------------------------------------------


def _patch_chat_classes() -> None:
    try:
        from google.genai.chats import AsyncChat, Chat
    except Exception:
        return

    if (
        hasattr(Chat, "send_message")
        and "send_message" in Chat.__dict__
        and not Chat.__dict__.get("_neatlogs_vertex_patched", False)
    ):
        orig_send = Chat.send_message

        def patched_send(self, message, *args, **kwargs):
            if is_suppressed():
                return orig_send(self, message, *args, **kwargs)
            span = _start_chat_span(self, message, stream=False)
            start = time.perf_counter()
            try:
                resp = orig_send(self, message, *args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            _finalize_response(span, resp, (time.perf_counter() - start) * 1000)
            return resp

        Chat.send_message = patched_send

        if hasattr(Chat, "send_message_stream"):
            orig_send_stream = Chat.send_message_stream

            def patched_send_stream(self, message, *args, **kwargs):
                if is_suppressed():
                    return orig_send_stream(self, message, *args, **kwargs)
                span = _start_chat_span(self, message, stream=True)
                stream = orig_send_stream(self, message, *args, **kwargs)
                return SyncStreamWrapper(stream, span, _finalize_stream)

            Chat.send_message_stream = patched_send_stream

        Chat._neatlogs_vertex_patched = True

    if (
        hasattr(AsyncChat, "send_message")
        and "send_message" in AsyncChat.__dict__
        and not AsyncChat.__dict__.get("_neatlogs_vertex_patched", False)
    ):
        orig_asend = AsyncChat.send_message

        async def patched_asend(self, message, *args, **kwargs):
            if is_suppressed():
                return await orig_asend(self, message, *args, **kwargs)
            span = _start_chat_span(self, message, stream=False)
            start = time.perf_counter()
            try:
                resp = await orig_asend(self, message, *args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            _finalize_response(span, resp, (time.perf_counter() - start) * 1000)
            return resp

        AsyncChat.send_message = patched_asend

        if hasattr(AsyncChat, "send_message_stream"):
            orig_asend_stream = AsyncChat.send_message_stream

            async def patched_asend_stream(self, message, *args, **kwargs):
                if is_suppressed():
                    return await orig_asend_stream(self, message, *args, **kwargs)
                span = _start_chat_span(self, message, stream=True)
                try:
                    stream = await orig_asend_stream(self, message, *args, **kwargs)
                except Exception as e:
                    _err(span, e)
                    raise
                return AsyncStreamWrapper(stream, span, _finalize_stream)

            AsyncChat.send_message_stream = patched_asend_stream

        AsyncChat._neatlogs_vertex_patched = True


def _start_chat_span(chat: Any, message: Any, stream: bool) -> Any:
    model = getattr(chat, "_model", None) or getattr(chat, "model", None) or ""
    span = _start_span(model, stream=stream, name="vertex_ai.chat.send_message")
    if message is not None:
        span.set_attribute("neatlogs.llm.input_messages.0.role", "user")
        span.set_attribute(
            "neatlogs.llm.input_messages.0.content",
            (message if isinstance(message, str) else serialize(message))[:10000],
        )
    return span


def _err(span: Any, e: Exception) -> None:
    span.set_status(StatusCode.ERROR, str(e))
    span.record_exception(e)
    span.end()


# ---------------------------------------------------------------------------
# Import-replacement: `from neatlogs.vertex_ai import genai`
# Patches Client.__init__ so every Vertex-mode client is auto-wrapped.
# ---------------------------------------------------------------------------

def _patch_vertex_ai_module() -> None:
    global _PATCHED, _ORIG_INIT
    if _PATCHED:
        return
    _PATCHED = True

    from google import genai as _genai

    _ORIG_INIT = _genai.Client.__init__

    def _patched_init(self, *args, **kwargs):
        _ORIG_INIT(self, *args, **kwargs)
        # Only wrap when this client targets Vertex AI; leave Gemini (AI Studio)
        # clients untouched so they can be traced via neatlogs.google_genai.
        if _is_vertex_client(self):
            wrap_vertex_ai_client(self)

    _genai.Client.__init__ = _patched_init


def _unpatch_vertex_ai_module() -> None:
    global _PATCHED, _ORIG_INIT
    if not _PATCHED:
        return

    from google import genai as _genai

    if _ORIG_INIT is not None:
        _genai.Client.__init__ = _ORIG_INIT

    _PATCHED = False
    _ORIG_INIT = None


_patch_vertex_ai_module()

from google import genai  # noqa: E402
