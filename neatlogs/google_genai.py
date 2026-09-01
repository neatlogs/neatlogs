"""
Neatlogs Google GenAI wrapper.

Two usage patterns:

  1. Explicit wrap (primary):
     >>> import neatlogs
     >>> from google import genai
     >>> client = neatlogs.wrap(genai.Client())
     >>> client.models.generate_content(model="gemini-2.0-flash", contents="Hello")

  2. Import replacement (Langfuse-style):
     >>> from neatlogs.google_genai import genai
     >>> client = genai.Client()
     >>> client.models.generate_content(model="gemini-2.0-flash", contents="Hello")
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
from .core.choice_accumulator import ChoiceAccumulator, GoogleStreamFinalizer
from .core.media import set_media_attributes

_PATCHED = False
_ORIG_INIT = None


class GoogleGenAIInstrumentor:
    """Instrumentor class for InstrumentationManager integration."""

    def instrument(self, tracer_provider=None):
        _patch_google_genai_module()

    def uninstrument(self):
        _unpatch_google_genai_module()


def wrap_google_genai_client(client: Any) -> Any:
    """
    Wrap a google.genai.Client instance. Patches models (generate_content,
    generate_content_stream, embed_content, count_tokens) on sync + async, and
    chat sessions (Chat/AsyncChat send_message + send_message_stream).
    """
    _patch_models(client.models)
    _patch_models_extra(client.models, is_async=False)
    if hasattr(client, "aio") and hasattr(client.aio, "models"):
        _patch_async_models(client.aio.models)
        _patch_models_extra(client.aio.models, is_async=True)
    # Chat sessions are created lazily; patch the classes once so every session
    # (sync + async) is traced.
    _patch_chat_classes()
    return client


def _patch_models(models: Any) -> None:
    if getattr(models, "_neatlogs_patched", False):
        return

    orig_generate = models.generate_content
    orig_stream = getattr(models, "generate_content_stream", None)

    def patched_generate_content(*args, **kwargs):
        if is_suppressed():
            return orig_generate(*args, **kwargs)

        model = kwargs.get("model", args[0] if args else "")
        contents = kwargs.get("contents", args[1] if len(args) > 1 else "")

        tracer = get_provider_tracer()
        span = tracer.start_span(
            name="google_genai.models.generate_content",
            attributes={
                "neatlogs.span.kind": "llm",
                "neatlogs.llm.provider": "google_genai",
                "neatlogs.llm.system": "google_genai",
                "neatlogs.llm.model_name": str(model),
                "neatlogs.llm.is_streaming": False,
            },
        )

        _set_input_attributes(span, contents, kwargs)

        start = time.perf_counter()

        try:
            response = orig_generate(*args, **kwargs)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        _finalize_response(span, response, duration_ms)
        return response

    models.generate_content = patched_generate_content

    if orig_stream:

        def patched_generate_content_stream(*args, **kwargs):
            if is_suppressed():
                return orig_stream(*args, **kwargs)

            model = kwargs.get("model", args[0] if args else "")
            contents = kwargs.get("contents", args[1] if len(args) > 1 else "")

            tracer = get_provider_tracer()
            span = tracer.start_span(
                name="google_genai.models.generate_content",
                attributes={
                    "neatlogs.span.kind": "llm",
                    "neatlogs.llm.provider": "google_genai",
                    "neatlogs.llm.system": "google_genai",
                    "neatlogs.llm.model_name": str(model),
                    "neatlogs.llm.is_streaming": True,
                },
            )

            _set_input_attributes(span, contents, kwargs)

            stream = orig_stream(*args, **kwargs)
            return SyncStreamWrapper(stream, span, GoogleStreamFinalizer())

        models.generate_content_stream = patched_generate_content_stream

    models._neatlogs_patched = True


def _patch_async_models(models: Any) -> None:
    if getattr(models, "_neatlogs_patched", False):
        return

    orig_generate = models.generate_content
    orig_stream = getattr(models, "generate_content_stream", None)

    async def patched_generate_content(*args, **kwargs):
        if is_suppressed():
            return await orig_generate(*args, **kwargs)

        model = kwargs.get("model", args[0] if args else "")
        contents = kwargs.get("contents", args[1] if len(args) > 1 else "")

        tracer = get_provider_tracer()
        span = tracer.start_span(
            name="google_genai.models.generate_content",
            attributes={
                "neatlogs.span.kind": "llm",
                "neatlogs.llm.provider": "google_genai",
                "neatlogs.llm.system": "google_genai",
                "neatlogs.llm.model_name": str(model),
                "neatlogs.llm.is_streaming": False,
            },
        )

        _set_input_attributes(span, contents, kwargs)

        start = time.perf_counter()

        try:
            response = await orig_generate(*args, **kwargs)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        _finalize_response(span, response, duration_ms)
        return response

    models.generate_content = patched_generate_content

    if orig_stream:
        # NOTE: the async google-genai `generate_content_stream` is a COROUTINE that
        # returns an async iterator — callers do `stream = await models.generate_content_stream(...)`
        # then `async for chunk in stream`. The patch MUST preserve that contract: it has to be
        # an `async def` that RETURNS an async-iterable (AsyncStreamWrapper), NOT an async
        # generator (`async def` + top-level `yield`). An async generator is not awaitable, so
        # `await ...generate_content_stream(...)` would raise
        # "object async_generator can't be used in 'await' expression".
        async def patched_generate_content_stream(*args, **kwargs):
            if is_suppressed():
                return await orig_stream(*args, **kwargs)

            model = kwargs.get("model", args[0] if args else "")
            contents = kwargs.get("contents", args[1] if len(args) > 1 else "")

            tracer = get_provider_tracer()
            span = tracer.start_span(
                name="google_genai.models.generate_content",
                attributes={
                    "neatlogs.span.kind": "llm",
                    "neatlogs.llm.provider": "google_genai",
                    "neatlogs.llm.system": "google_genai",
                    "neatlogs.llm.model_name": str(model),
                    "neatlogs.llm.is_streaming": True,
                },
            )

            _set_input_attributes(span, contents, kwargs)

            try:
                stream = await orig_stream(*args, **kwargs)
            except Exception as e:
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)
                span.end()
                raise

            return AsyncStreamWrapper(stream, span, GoogleStreamFinalizer())

        models.generate_content_stream = patched_generate_content_stream

    models._neatlogs_patched = True


def _text_from_parts(parts: Any) -> str:
    """Join the human-readable text of a list of parts (str / dict / typed Part).

    Typed genai Parts carry `.text`, or a `.function_call` / `.function_response`
    for tool turns — surface those as compact JSON so a tool turn isn't blank.
    """
    text_parts: List[str] = []
    for part in parts or []:
        if isinstance(part, str):
            text_parts.append(part)
        elif isinstance(part, dict):
            if part.get("text"):
                text_parts.append(part["text"])
            elif part.get("function_call"):
                text_parts.append(serialize(part["function_call"]))
            elif part.get("function_response"):
                text_parts.append(serialize(part["function_response"]))
        else:
            # Typed Part object.
            text = getattr(part, "text", None)
            fc = getattr(part, "function_call", None)
            fr = getattr(part, "function_response", None)
            if text:
                text_parts.append(text)
            elif fc is not None:
                name = getattr(fc, "name", "") or ""
                args = getattr(fc, "args", None)
                text_parts.append(serialize({"function_call": {"name": name, "args": args}}))
            elif fr is not None:
                name = getattr(fr, "name", "") or ""
                resp = getattr(fr, "response", None)
                text_parts.append(
                    serialize({"function_response": {"name": name, "response": resp}})
                )
    return "\n".join(text_parts)


def _normalize_content_item(item: Any) -> tuple:
    """Normalize one `contents` entry to (role, text).

    Handles a plain string, a dict ({role, parts}), and — crucially for real
    multi-turn conversations — a typed ``types.Content`` object, whose ``.role``
    ("user"/"model") and ``.parts`` were previously dropped (every turn was
    mislabeled "user" and the content became a Python repr).
    """
    if isinstance(item, str):
        return "user", item
    if isinstance(item, dict):
        role = item.get("role") or "user"
        parts = item.get("parts", [])
        text = _text_from_parts(parts) if parts else ""
        return role, (text if text else serialize(parts))
    # Typed Content object.
    role = getattr(item, "role", None)
    parts = getattr(item, "parts", None)
    if role is not None or parts is not None:
        text = _text_from_parts(parts)
        return (role or "user"), (text if text else serialize(item))
    return "user", serialize(item)


def _set_input_attributes(span: Any, contents: Any, kwargs: dict) -> None:
    """Set input attributes from contents and config."""
    config = kwargs.get("config")

    # System instruction
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
                span.set_attribute(
                    f"neatlogs.llm.input_messages.{idx}.content", serialize(system_instruction)
                )
                set_media_attributes(
                    span, f"neatlogs.llm.input_messages.{idx}", system_instruction, "input"
                )
            idx += 1

    # Contents — a single string, or a list of turns (str / dict / typed Content).
    if isinstance(contents, str):
        span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", "user")
        span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", contents)
    elif isinstance(contents, list):
        for item in contents:
            role, text = _normalize_content_item(item)
            span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", role)
            span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", text)
            set_media_attributes(span, f"neatlogs.llm.input_messages.{idx}", item, "input")
            idx += 1
    else:
        # A single typed Content (or anything else) — normalize it too.
        role, text = _normalize_content_item(contents)
        span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", role)
        span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", text)
        set_media_attributes(span, f"neatlogs.llm.input_messages.{idx}", contents, "input")

    # Tools
    tools = None
    if config:
        tools = (
            getattr(config, "tools", None) if not isinstance(config, dict) else config.get("tools")
        )
    if tools:
        for i, tool in enumerate(tools):
            if isinstance(tool, dict):
                fn_decls = tool.get("function_declarations", [])
                for j, fn in enumerate(fn_decls):
                    span.set_attribute(f"neatlogs.llm.tools.{i + j}.name", fn.get("name", ""))
                    if fn.get("description"):
                        span.set_attribute(
                            f"neatlogs.llm.tools.{i + j}.description", fn["description"]
                        )
                    if fn.get("parameters"):
                        span.set_attribute(
                            f"neatlogs.llm.tools.{i + j}.input_schema", serialize(fn["parameters"])
                        )

    # Invocation parameters from config
    if config:
        cfg = (
            config
            if isinstance(config, dict)
            else config.__dict__ if hasattr(config, "__dict__") else {}
        )
        if isinstance(cfg, dict):
            for param in ("temperature", "top_p", "top_k", "max_output_tokens"):
                val = cfg.get(param)
                if val is not None:
                    attr_name = "max_tokens" if param == "max_output_tokens" else param
                    span.set_attribute(f"neatlogs.llm.{attr_name}", val)


def _finalize_response(span: Any, response: Any, duration_ms: float) -> None:
    """Extract attributes from a non-streaming GenerateContentResponse."""
    accumulator = ChoiceAccumulator()
    accumulator.add_google_response(response)
    accumulator.apply(span)

    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()


def _set_usage_attributes(span: Any, usage: Any) -> None:
    """Set token usage attributes from Google GenAI usage_metadata."""
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


def _finalize_stream(
    span: Any,
    chunks: List[Any],
    duration_ms: float,
    ttft_ms: Optional[float],
    *,
    interrupted: bool = False,
) -> None:
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
                    tool_calls_acc.append(
                        {
                            "name": getattr(fc, "name", ""),
                            "arguments": serialize(getattr(fc, "args", None) or {}),
                        }
                    )

            fr = getattr(candidate, "finish_reason", None)
            if fr:
                finish_reason = str(fr)

        chunk_usage = getattr(chunk, "usage_metadata", None)
        if chunk_usage:
            usage = chunk_usage

    # Output messages
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

    span.set_status(StatusCode.UNSET if interrupted else StatusCode.OK)
    span.end()


# ---------------------------------------------------------------------------
# embed_content + count_tokens (sync + async)
# ---------------------------------------------------------------------------


def _patch_models_extra(models: Any, is_async: bool) -> None:
    # embed_content
    if hasattr(models, "embed_content") and not getattr(models, "_neatlogs_embed_patched", False):
        orig = models.embed_content

        def _embed_attrs(kwargs):
            attrs = {
                "neatlogs.span.kind": "embedding",
                "neatlogs.embedding.model_name": str(kwargs.get("model", "")),
            }
            contents = kwargs.get("contents")
            if contents is not None:
                attrs["neatlogs.embedding.text"] = (
                    contents if isinstance(contents, str) else serialize(contents)
                )[:10000]
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
                span = get_provider_tracer().start_span(
                    name="google_genai.models.embed_content", attributes=_embed_attrs(kwargs)
                )
                try:
                    resp = await orig(*args, **kwargs)
                except Exception as e:
                    _err(span, e)
                    raise
                _embed_finalize(span, resp)
                return resp

        else:

            def patched_embed(*args, **kwargs):
                if is_suppressed():
                    return orig(*args, **kwargs)
                span = get_provider_tracer().start_span(
                    name="google_genai.models.embed_content", attributes=_embed_attrs(kwargs)
                )
                try:
                    resp = orig(*args, **kwargs)
                except Exception as e:
                    _err(span, e)
                    raise
                _embed_finalize(span, resp)
                return resp

        models.embed_content = patched_embed
        models._neatlogs_embed_patched = True

    # count_tokens
    if hasattr(models, "count_tokens") and not getattr(models, "_neatlogs_count_patched", False):
        orig_ct = models.count_tokens

        def _ct_attrs(kwargs):
            return {
                "neatlogs.span.kind": "llm",
                "neatlogs.llm.provider": "google_genai",
                "neatlogs.llm.task": "count_tokens",
                "neatlogs.llm.model_name": str(kwargs.get("model", "")),
            }

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
                span = get_provider_tracer().start_span(
                    name="google_genai.models.count_tokens", attributes=_ct_attrs(kwargs)
                )
                try:
                    resp = await orig_ct(*args, **kwargs)
                except Exception as e:
                    _err(span, e)
                    raise
                _ct_finalize(span, resp)
                return resp

        else:

            def patched_ct(*args, **kwargs):
                if is_suppressed():
                    return orig_ct(*args, **kwargs)
                span = get_provider_tracer().start_span(
                    name="google_genai.models.count_tokens", attributes=_ct_attrs(kwargs)
                )
                try:
                    resp = orig_ct(*args, **kwargs)
                except Exception as e:
                    _err(span, e)
                    raise
                _ct_finalize(span, resp)
                return resp

        models.count_tokens = patched_ct
        models._neatlogs_count_patched = True


# ---------------------------------------------------------------------------
# Chat sessions (Chat / AsyncChat send_message + send_message_stream)
# ---------------------------------------------------------------------------


def _patch_chat_classes() -> None:
    try:
        from google.genai.chats import AsyncChat, Chat
    except Exception:
        return

    # Sync Chat.send_message
    if (
        hasattr(Chat, "send_message")
        and "send_message" in Chat.__dict__
        and not Chat.__dict__.get("_neatlogs_patched", False)
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
                return SyncStreamWrapper(stream, span, GoogleStreamFinalizer())

            Chat.send_message_stream = patched_send_stream

        Chat._neatlogs_patched = True

    # Async Chat
    if (
        hasattr(AsyncChat, "send_message")
        and "send_message" in AsyncChat.__dict__
        and not AsyncChat.__dict__.get("_neatlogs_patched", False)
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

            # Like generate_content_stream, the async send_message_stream is a COROUTINE
            # returning an async iterator (callers `await` it then iterate). Patch as an
            # `async def` that RETURNS an AsyncStreamWrapper — NOT an async generator,
            # which would break `await chat.send_message_stream(...)`.
            async def patched_asend_stream(self, message, *args, **kwargs):
                if is_suppressed():
                    return await orig_asend_stream(self, message, *args, **kwargs)
                span = _start_chat_span(self, message, stream=True)
                try:
                    stream = await orig_asend_stream(self, message, *args, **kwargs)
                except Exception as e:
                    _err(span, e)
                    raise
                return AsyncStreamWrapper(stream, span, GoogleStreamFinalizer())

            AsyncChat.send_message_stream = patched_asend_stream

        AsyncChat._neatlogs_patched = True


def _start_chat_span(chat: Any, message: Any, stream: bool) -> Any:
    model = getattr(chat, "_model", None) or getattr(chat, "model", None) or ""
    span = get_provider_tracer().start_span(
        name="google_genai.chat.send_message",
        attributes={
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.provider": "google_genai",
            "neatlogs.llm.system": "google_genai",
            "neatlogs.llm.model_name": str(model),
            "neatlogs.llm.is_streaming": bool(stream),
        },
    )
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
# Import-replacement: `from neatlogs.google_genai import genai`
# Patches Client.__init__ so every client is auto-wrapped.
# ---------------------------------------------------------------------------


def _patch_google_genai_module() -> None:
    global _PATCHED, _ORIG_INIT
    if _PATCHED:
        return
    _PATCHED = True

    from google import genai as _genai

    _ORIG_INIT = _genai.Client.__init__

    def _patched_init(self, *args, **kwargs):
        _ORIG_INIT(self, *args, **kwargs)
        wrap_google_genai_client(self)

    _genai.Client.__init__ = _patched_init


def _unpatch_google_genai_module() -> None:
    global _PATCHED, _ORIG_INIT
    if not _PATCHED:
        return

    from google import genai as _genai

    if _ORIG_INIT is not None:
        _genai.Client.__init__ = _ORIG_INIT

    _PATCHED = False
    _ORIG_INIT = None


try:  # noqa: E402 - optional import-replacement surface
    from google import genai  # type: ignore  # noqa: E402
except ImportError:  # pragma: no cover - exercised by clean-collection CI
    genai = None
else:
    _patch_google_genai_module()
