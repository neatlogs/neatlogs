"""
Neatlogs OpenAI wrapper.

Two usage patterns:

  1. Explicit wrap (primary):
     >>> import neatlogs, openai
     >>> client = neatlogs.wrap(openai.OpenAI())
     >>> client.chat.completions.create(...)

  2. Import replacement (Langfuse-style):
     >>> from neatlogs.openai import openai
     >>> client = openai.OpenAI()
     >>> client.chat.completions.create(...)
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


class OpenAIInstrumentor:
    """Instrumentor class for InstrumentationManager integration."""

    def instrument(self, tracer_provider=None):
        _patch_openai_module()

    def uninstrument(self):
        _unpatch_openai_module()


def wrap_openai_client(client: Any) -> Any:
    """
    Wrap an OpenAI client instance. Returns the same client with
    chat.completions.create patched to auto-trace.
    """
    _patch_completions(client.chat.completions)
    if hasattr(client, "responses"):
        _patch_responses(client.responses)
    return client


def wrap_async_openai_client(client: Any) -> Any:
    """Wrap an AsyncOpenAI client instance."""
    _patch_async_completions(client.chat.completions)
    return client


def _patch_completions(completions: Any) -> None:
    if getattr(completions, "_neatlogs_patched", False):
        return

    orig_create = completions.create

    def patched_create(*args, **kwargs):
        if is_suppressed():
            return orig_create(*args, **kwargs)

        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        is_stream = kwargs.get("stream", False)

        if is_stream:
            opts = kwargs.get("stream_options") or {}
            if not opts.get("include_usage"):
                opts["include_usage"] = True
                kwargs["stream_options"] = opts

        tracer = get_tracer()
        span = tracer.start_span(
            name="openai.chat.completions.create",
            attributes={
                "neatlogs.span.kind": "LLM",
                "neatlogs.llm.provider": "openai",
                "neatlogs.llm.system": "openai",
                "neatlogs.llm.model_name": model,
                "neatlogs.llm.is_streaming": is_stream,
            },
        )

        # Input messages
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

        # Tools
        tools = kwargs.get("tools")
        if tools:
            for i, tool in enumerate(tools):
                fn = tool.get("function", {})
                span.set_attribute(f"neatlogs.llm.tools.{i}.name", fn.get("name", ""))
                if fn.get("description"):
                    span.set_attribute(f"neatlogs.llm.tools.{i}.description", fn["description"])
                if fn.get("parameters"):
                    span.set_attribute(f"neatlogs.llm.tools.{i}.input_schema", serialize(fn["parameters"]))

        # Invocation parameters
        for param in ("temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty"):
            if param in kwargs and kwargs[param] is not None:
                span.set_attribute(f"neatlogs.llm.{param}", kwargs[param])

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

    completions.create = patched_create
    completions._neatlogs_patched = True


def _patch_async_completions(completions: Any) -> None:
    if getattr(completions, "_neatlogs_patched", False):
        return

    orig_create = completions.create

    async def patched_create(*args, **kwargs):
        if is_suppressed():
            return await orig_create(*args, **kwargs)

        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        is_stream = kwargs.get("stream", False)

        if is_stream:
            opts = kwargs.get("stream_options") or {}
            if not opts.get("include_usage"):
                opts["include_usage"] = True
                kwargs["stream_options"] = opts

        tracer = get_tracer()
        span = tracer.start_span(
            name="openai.chat.completions.create",
            attributes={
                "neatlogs.span.kind": "LLM",
                "neatlogs.llm.provider": "openai",
                "neatlogs.llm.system": "openai",
                "neatlogs.llm.model_name": model,
                "neatlogs.llm.is_streaming": is_stream,
            },
        )

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            span.set_attribute(f"neatlogs.llm.input_messages.{i}.role", role)
            if isinstance(content, str):
                span.set_attribute(f"neatlogs.llm.input_messages.{i}.content", content)
            else:
                span.set_attribute(f"neatlogs.llm.input_messages.{i}.content", serialize(content))

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

    completions.create = patched_create
    completions._neatlogs_patched = True


def _patch_responses(responses: Any) -> None:
    if getattr(responses, "_neatlogs_patched", False):
        return

    orig_create = responses.create

    def patched_create(*args, **kwargs):
        if is_suppressed():
            return orig_create(*args, **kwargs)

        model = kwargs.get("model", "")
        tracer = get_tracer()
        span = tracer.start_span(
            name="openai.responses.create",
            attributes={
                "neatlogs.span.kind": "LLM",
                "neatlogs.llm.provider": "openai",
                "neatlogs.llm.system": "openai",
                "neatlogs.llm.model_name": model,
                "neatlogs.llm.input_messages.0.role": "user",
                "neatlogs.llm.input_messages.0.content": serialize(kwargs.get("input", "")),
            },
        )

        start = time.perf_counter()
        try:
            response = orig_create(*args, **kwargs)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        _finalize_responses_response(span, response, duration_ms)
        return response

    responses.create = patched_create
    responses._neatlogs_patched = True


def _finalize_response(span: Any, response: Any, duration_ms: float) -> None:
    """Extract neatlogs attributes from a non-streaming ChatCompletion response."""
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
    """Finalize a streaming response span from accumulated chunks."""
    text_parts: List[str] = []
    tool_calls_acc: dict = {}
    finish_reason = None
    model = None
    usage = None

    for chunk in chunks:
        if not getattr(chunk, "choices", None):
            if getattr(chunk, "usage", None):
                usage = chunk.usage
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

    # Output messages
    full_text = "".join(text_parts)
    if full_text:
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        span.set_attribute("neatlogs.llm.output_messages.0.content", full_text)

    # Tool calls
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
    """Extract attributes from a Responses API response."""
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


# ---------------------------------------------------------------------------
# Import-replacement: `from neatlogs.openai import openai`
# Patches OpenAI/AsyncOpenAI.__init__ so every client is auto-wrapped.
# ---------------------------------------------------------------------------

def _patch_openai_module() -> None:
    global _PATCHED, _ORIG_INIT, _ORIG_ASYNC_INIT
    if _PATCHED:
        return
    _PATCHED = True

    import openai as _openai

    _ORIG_INIT = _openai.OpenAI.__init__
    _ORIG_ASYNC_INIT = _openai.AsyncOpenAI.__init__

    def _patched_init(self, *args, **kwargs):
        _ORIG_INIT(self, *args, **kwargs)
        wrap_openai_client(self)

    _openai.OpenAI.__init__ = _patched_init

    def _patched_async_init(self, *args, **kwargs):
        _ORIG_ASYNC_INIT(self, *args, **kwargs)
        wrap_async_openai_client(self)

    _openai.AsyncOpenAI.__init__ = _patched_async_init


def _unpatch_openai_module() -> None:
    global _PATCHED, _ORIG_INIT, _ORIG_ASYNC_INIT
    if not _PATCHED:
        return

    import openai as _openai

    if _ORIG_INIT is not None:
        _openai.OpenAI.__init__ = _ORIG_INIT
    if _ORIG_ASYNC_INIT is not None:
        _openai.AsyncOpenAI.__init__ = _ORIG_ASYNC_INIT

    _PATCHED = False
    _ORIG_INIT = None
    _ORIG_ASYNC_INIT = None


_patch_openai_module()

import openai  # noqa: E402
