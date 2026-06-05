"""
Neatlogs AWS Bedrock wrapper.

Bedrock is accessed through a boto3 ``bedrock-runtime`` client. This module
traces both the Converse API (``converse`` / ``converse_stream``) and the legacy
InvokeModel API (``invoke_model`` / ``invoke_model_with_response_stream``).
Token extraction handles Claude, Titan, and Llama response formats.

Usage (the boto3 client is mutated in place):

  >>> import boto3, neatlogs
  >>> client = boto3.client("bedrock-runtime", region_name="us-east-1")
  >>> neatlogs.wrap(client)
  >>> client.converse(modelId="anthropic.claude-3-5-sonnet-20240620-v1:0", messages=[...])

`neatlogs.llm.provider` is always ``bedrock``; `neatlogs.llm.system` is the
underlying model vendor (anthropic / amazon / meta / ...), inferred from modelId.
"""

import json
import time
from typing import Any, List, Optional

from opentelemetry.trace import StatusCode

from ._wrap_utils import get_tracer, is_suppressed, serialize

_PROVIDER = "bedrock"


class BedrockInstrumentor:
    """
    Instrumentor class for InstrumentationManager integration.

    Unlike OpenAI/GenAI, boto3 clients are constructed via a factory
    (``boto3.client(...)``) rather than a public class we can patch at import
    time, so there is no module-level auto-patch. Enabling ``bedrock`` through
    ``init(instrumentations=["bedrock"])`` patches the botocore client creator so
    every ``bedrock-runtime`` client is wrapped automatically.
    """

    def instrument(self, tracer_provider=None):
        _patch_botocore_factory()

    def uninstrument(self):
        _unpatch_botocore_factory()


# ---------------------------------------------------------------------------
# Vendor / model helpers
# ---------------------------------------------------------------------------


def _vendor_from_model(model_id: Any) -> str:
    """Infer the model vendor from a Bedrock modelId (or inference profile ARN)."""
    mid = str(model_id or "")
    # Strip cross-region inference profile prefixes like "us." / "eu." / ARNs.
    tail = mid.split("/")[-1]
    for prefix in ("us.", "eu.", "apac.", "us-gov."):
        if tail.startswith(prefix):
            tail = tail[len(prefix):]
    vendor = tail.split(".")[0] if "." in tail else ""
    return vendor or "bedrock"


def _start_span(name: str, model_id: Any, is_stream: bool) -> Any:
    return get_tracer().start_span(
        name=name,
        attributes={
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.provider": _PROVIDER,
            "neatlogs.llm.system": _vendor_from_model(model_id),
            "neatlogs.llm.model_name": str(model_id or ""),
            "neatlogs.llm.is_streaming": bool(is_stream),
        },
    )


def _err(span: Any, e: Exception) -> None:
    span.set_status(StatusCode.ERROR, str(e))
    span.record_exception(e)
    span.end()


def _ok(span: Any, duration_ms: float) -> None:
    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()


# ---------------------------------------------------------------------------
# Public wrap
# ---------------------------------------------------------------------------


def wrap_bedrock_client(client: Any) -> Any:
    """
    Patch a boto3 ``bedrock-runtime`` client in place to trace converse /
    converse_stream / invoke_model / invoke_model_with_response_stream. Returns
    the same client. No-op for non-Bedrock clients.
    """
    if getattr(client, "_neatlogs_bedrock_patched", False):
        return client

    service = getattr(getattr(client, "meta", None), "service_model", None)
    service_name = getattr(service, "service_name", None)
    # Be permissive: if we cannot determine the service, still patch the known
    # method names if present.
    if service_name not in (None, "bedrock-runtime"):
        return client

    if hasattr(client, "converse"):
        _patch_converse(client)
    if hasattr(client, "converse_stream"):
        _patch_converse_stream(client)
    if hasattr(client, "invoke_model"):
        _patch_invoke_model(client)
    if hasattr(client, "invoke_model_with_response_stream"):
        _patch_invoke_model_stream(client)

    client._neatlogs_bedrock_patched = True
    return client


# ---------------------------------------------------------------------------
# Converse API
# ---------------------------------------------------------------------------


def _set_converse_input(span: Any, kwargs: dict) -> None:
    idx = 0
    system = kwargs.get("system")
    if system:
        text = " ".join(
            blk.get("text", "") for blk in system if isinstance(blk, dict)
        ).strip() if isinstance(system, list) else str(system)
        if text:
            span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", "system")
            span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", text)
            idx += 1

    for msg in kwargs.get("messages", []) or []:
        role = msg.get("role", "user")
        content = msg.get("content", [])
        span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", role)
        span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", _converse_blocks_to_text(content))
        idx += 1

    cfg = kwargs.get("inferenceConfig") or {}
    if isinstance(cfg, dict):
        if cfg.get("temperature") is not None:
            span.set_attribute("neatlogs.llm.temperature", cfg["temperature"])
        if cfg.get("topP") is not None:
            span.set_attribute("neatlogs.llm.top_p", cfg["topP"])
        if cfg.get("maxTokens") is not None:
            span.set_attribute("neatlogs.llm.max_tokens", cfg["maxTokens"])

    tool_config = kwargs.get("toolConfig") or {}
    tools = tool_config.get("tools", []) if isinstance(tool_config, dict) else []
    for i, tool in enumerate(tools):
        spec = tool.get("toolSpec", {}) if isinstance(tool, dict) else {}
        if spec.get("name"):
            span.set_attribute(f"neatlogs.llm.tools.{i}.name", spec["name"])
        if spec.get("description"):
            span.set_attribute(f"neatlogs.llm.tools.{i}.description", spec["description"])
        schema = spec.get("inputSchema")
        if schema:
            span.set_attribute(f"neatlogs.llm.tools.{i}.input_schema", serialize(schema))


def _converse_blocks_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return serialize(content)
    parts: List[str] = []
    for block in content:
        if isinstance(block, dict):
            if "text" in block:
                parts.append(str(block["text"]))
            elif "toolResult" in block:
                parts.append(serialize(block["toolResult"]))
            elif "toolUse" in block:
                parts.append(serialize(block["toolUse"]))
            else:
                parts.append(serialize(block))
        else:
            parts.append(str(block))
    return "\n".join(parts)


def _finalize_converse(span: Any, response: dict, duration_ms: float) -> None:
    output = (response or {}).get("output", {})
    message = output.get("message", {}) if isinstance(output, dict) else {}
    content = message.get("content", []) if isinstance(message, dict) else []

    text_parts: List[str] = []
    tool_idx = 0
    for block in content or []:
        if not isinstance(block, dict):
            continue
        if "text" in block:
            text_parts.append(str(block["text"]))
        elif "toolUse" in block:
            tu = block["toolUse"]
            span.set_attribute(f"neatlogs.llm.tool_calls.{tool_idx}.id", str(tu.get("toolUseId", "")))
            span.set_attribute(f"neatlogs.llm.tool_calls.{tool_idx}.name", str(tu.get("name", "")))
            span.set_attribute(f"neatlogs.llm.tool_calls.{tool_idx}.arguments", serialize(tu.get("input", {})))
            tool_idx += 1

    if text_parts:
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        span.set_attribute("neatlogs.llm.output_messages.0.content", "".join(text_parts))

    if response.get("stopReason"):
        span.set_attribute("neatlogs.llm.finish_reason", str(response["stopReason"]))

    _set_converse_usage(span, response.get("usage"))
    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()


def _set_converse_usage(span: Any, usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    if usage.get("inputTokens") is not None:
        span.set_attribute("neatlogs.llm.token_count.prompt", usage["inputTokens"])
    if usage.get("outputTokens") is not None:
        span.set_attribute("neatlogs.llm.token_count.completion", usage["outputTokens"])
    if usage.get("totalTokens") is not None:
        span.set_attribute("neatlogs.llm.token_count.total", usage["totalTokens"])
    if usage.get("cacheReadInputTokens") is not None:
        span.set_attribute("neatlogs.llm.token_count.cache_read", usage["cacheReadInputTokens"])
    if usage.get("cacheWriteInputTokens") is not None:
        span.set_attribute("neatlogs.llm.token_count.cache_write", usage["cacheWriteInputTokens"])


def _patch_converse(client: Any) -> None:
    orig = client.converse

    def patched(*args, **kwargs):
        if is_suppressed():
            return orig(*args, **kwargs)
        span = _start_span("bedrock.converse", kwargs.get("modelId"), is_stream=False)
        _set_converse_input(span, kwargs)
        start = time.perf_counter()
        try:
            response = orig(*args, **kwargs)
        except Exception as e:
            _err(span, e)
            raise
        _finalize_converse(span, response, (time.perf_counter() - start) * 1000)
        return response

    client.converse = patched


def _patch_converse_stream(client: Any) -> None:
    orig = client.converse_stream

    def patched(*args, **kwargs):
        if is_suppressed():
            return orig(*args, **kwargs)
        span = _start_span("bedrock.converse_stream", kwargs.get("modelId"), is_stream=True)
        _set_converse_input(span, kwargs)
        start = time.perf_counter()
        try:
            response = orig(*args, **kwargs)
        except Exception as e:
            _err(span, e)
            raise
        stream = response.get("stream") if isinstance(response, dict) else None
        if stream is None:
            _ok(span, (time.perf_counter() - start) * 1000)
            return response
        response["stream"] = _wrap_converse_stream(stream, span, start)
        return response

    client.converse_stream = patched


def _wrap_converse_stream(stream: Any, span: Any, start: float):
    """Generator that passes through Converse stream events while accumulating."""
    text_parts: List[str] = []
    tool_calls: dict = {}
    finish_reason = None
    usage = None
    errored = False
    try:
        for event in stream:
            if isinstance(event, dict):
                delta = event.get("contentBlockDelta", {}).get("delta", {})
                if delta.get("text"):
                    text_parts.append(delta["text"])
                if delta.get("toolUse", {}).get("input"):
                    blk = event["contentBlockDelta"].get("contentBlockIndex", 0)
                    tool_calls.setdefault(blk, {"name": "", "arguments": ""})
                    tool_calls[blk]["arguments"] += delta["toolUse"]["input"]
                start_blk = event.get("contentBlockStart", {}).get("start", {}).get("toolUse")
                if start_blk:
                    blk = event["contentBlockStart"].get("contentBlockIndex", 0)
                    tc = tool_calls.setdefault(blk, {"name": "", "arguments": ""})
                    tc["name"] = start_blk.get("name", "")
                    tc["id"] = start_blk.get("toolUseId", "")
                if event.get("messageStop", {}).get("stopReason"):
                    finish_reason = event["messageStop"]["stopReason"]
                if event.get("metadata", {}).get("usage"):
                    usage = event["metadata"]["usage"]
            yield event
    except Exception as e:
        errored = True
        _err(span, e)
        raise
    finally:
        if not errored:
            full = "".join(text_parts)
            if full:
                span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
                span.set_attribute("neatlogs.llm.output_messages.0.content", full)
            for j, tc in enumerate(tool_calls.values()):
                if tc.get("id"):
                    span.set_attribute(f"neatlogs.llm.tool_calls.{j}.id", tc["id"])
                span.set_attribute(f"neatlogs.llm.tool_calls.{j}.name", tc.get("name", ""))
                span.set_attribute(f"neatlogs.llm.tool_calls.{j}.arguments", tc.get("arguments", ""))
            if finish_reason:
                span.set_attribute("neatlogs.llm.finish_reason", str(finish_reason))
            _set_converse_usage(span, usage)
            _ok(span, (time.perf_counter() - start) * 1000)


# ---------------------------------------------------------------------------
# InvokeModel API (vendor-specific body formats)
# ---------------------------------------------------------------------------


def _decode_body(body: Any) -> dict:
    try:
        if isinstance(body, (bytes, bytearray)):
            return json.loads(body)
        if isinstance(body, str):
            return json.loads(body)
        if isinstance(body, dict):
            return body
    except (ValueError, TypeError):
        pass
    return {}


def _set_invoke_input(span: Any, vendor: str, body: dict) -> None:
    idx = 0
    # Claude messages format
    if body.get("system"):
        span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", "system")
        span.set_attribute(
            f"neatlogs.llm.input_messages.{idx}.content",
            body["system"] if isinstance(body["system"], str) else serialize(body["system"]),
        )
        idx += 1
    if isinstance(body.get("messages"), list):
        for msg in body["messages"]:
            span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", msg.get("role", "user"))
            content = msg.get("content", "")
            span.set_attribute(
                f"neatlogs.llm.input_messages.{idx}.content",
                content if isinstance(content, str) else serialize(content),
            )
            idx += 1
    elif body.get("prompt"):  # Claude legacy / Llama
        span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", "user")
        span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", str(body["prompt"])[:10000])
    elif body.get("inputText"):  # Titan
        span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", "user")
        span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", str(body["inputText"])[:10000])

    # Invocation params (best-effort across vendors)
    for key, attr in (("temperature", "temperature"), ("top_p", "top_p"), ("max_tokens", "max_tokens"),
                      ("maxTokens", "max_tokens"), ("max_tokens_to_sample", "max_tokens")):
        if body.get(key) is not None:
            span.set_attribute(f"neatlogs.llm.{attr}", body[key])
    cfg = body.get("textGenerationConfig")  # Titan
    if isinstance(cfg, dict):
        if cfg.get("temperature") is not None:
            span.set_attribute("neatlogs.llm.temperature", cfg["temperature"])
        if cfg.get("topP") is not None:
            span.set_attribute("neatlogs.llm.top_p", cfg["topP"])
        if cfg.get("maxTokenCount") is not None:
            span.set_attribute("neatlogs.llm.max_tokens", cfg["maxTokenCount"])


def _finalize_invoke(span: Any, vendor: str, body: dict, duration_ms: float) -> None:
    text = None
    prompt_tokens = completion_tokens = None
    finish_reason = None

    if vendor == "anthropic":
        content = body.get("content")
        if isinstance(content, list):  # messages API
            text = "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
            tool_idx = 0
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    span.set_attribute(f"neatlogs.llm.tool_calls.{tool_idx}.id", str(b.get("id", "")))
                    span.set_attribute(f"neatlogs.llm.tool_calls.{tool_idx}.name", str(b.get("name", "")))
                    span.set_attribute(f"neatlogs.llm.tool_calls.{tool_idx}.arguments", serialize(b.get("input", {})))
                    tool_idx += 1
        elif body.get("completion") is not None:  # legacy text completion
            text = body["completion"]
        usage = body.get("usage", {})
        prompt_tokens = usage.get("input_tokens")
        completion_tokens = usage.get("output_tokens")
        finish_reason = body.get("stop_reason")
    elif vendor == "amazon":  # Titan
        results = body.get("results")
        if isinstance(results, list) and results:
            text = results[0].get("outputText")
            completion_tokens = results[0].get("tokenCount")
            finish_reason = results[0].get("completionReason")
        prompt_tokens = body.get("inputTextTokenCount")
    elif vendor == "meta":  # Llama
        text = body.get("generation")
        prompt_tokens = body.get("prompt_token_count")
        completion_tokens = body.get("generation_token_count")
        finish_reason = body.get("stop_reason")
    elif vendor == "cohere":
        generations = body.get("generations")
        if isinstance(generations, list) and generations:
            text = generations[0].get("text")
            finish_reason = generations[0].get("finish_reason")
    else:
        # Unknown vendor: best-effort common keys
        text = body.get("generation") or body.get("completion") or body.get("outputText")

    if text:
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        span.set_attribute("neatlogs.llm.output_messages.0.content", str(text))
    if prompt_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.prompt", prompt_tokens)
    if completion_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.completion", completion_tokens)
    if prompt_tokens is not None and completion_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.total", prompt_tokens + completion_tokens)
    if finish_reason:
        span.set_attribute("neatlogs.llm.finish_reason", str(finish_reason))

    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()


def _reusable_body(raw: bytes) -> Any:
    """Return a botocore StreamingBody backed by ``raw`` so the caller can read it."""
    import io

    try:
        from botocore.response import StreamingBody

        return StreamingBody(io.BytesIO(raw), len(raw))
    except Exception:
        return io.BytesIO(raw)


def _is_embedding_model(model_id: Any) -> bool:
    return "embed" in str(model_id or "").lower()


def _patch_invoke_model(client: Any) -> None:
    orig = client.invoke_model

    def patched(*args, **kwargs):
        if is_suppressed():
            return orig(*args, **kwargs)
        model_id = kwargs.get("modelId")
        vendor = _vendor_from_model(model_id)
        is_embedding = _is_embedding_model(model_id)
        body_in = _decode_body(kwargs.get("body"))

        if is_embedding:
            span = get_tracer().start_span(
                name="bedrock.invoke_model",
                attributes={
                    "neatlogs.span.kind": "embedding",
                    "neatlogs.llm.provider": _PROVIDER,
                    "neatlogs.embedding.model_name": str(model_id or ""),
                },
            )
            text = body_in.get("inputText") or body_in.get("texts") or body_in.get("input_text")
            if text:
                span.set_attribute(
                    "neatlogs.embedding.text",
                    (text if isinstance(text, str) else serialize(text))[:10000],
                )
        else:
            span = _start_span("bedrock.invoke_model", model_id, is_stream=False)
            _set_invoke_input(span, vendor, body_in)

        start = time.perf_counter()
        try:
            response = orig(*args, **kwargs)
        except Exception as e:
            _err(span, e)
            raise
        # Reading the StreamingBody consumes it; read once, parse, then replace
        # with a fresh body so the caller still gets the bytes.
        try:
            raw = response["body"].read()
            response["body"] = _reusable_body(raw)
            data = _decode_body(raw)
            if is_embedding:
                _finalize_invoke_embedding(span, data, (time.perf_counter() - start) * 1000)
            else:
                _finalize_invoke(span, vendor, data, (time.perf_counter() - start) * 1000)
        except Exception:
            _ok(span, (time.perf_counter() - start) * 1000)
        return response

    client.invoke_model = patched


def _finalize_invoke_embedding(span: Any, body: dict, duration_ms: float) -> None:
    # Titan: {"embedding": [...], "inputTextTokenCount": N}
    # Cohere: {"embeddings": [[...], ...]}
    emb = body.get("embedding")
    embs = body.get("embeddings")
    if isinstance(emb, list):
        span.set_attribute("neatlogs.embedding.count", 1)
        span.set_attribute("neatlogs.embedding.dimensions", len(emb))
    elif isinstance(embs, list) and embs:
        span.set_attribute("neatlogs.embedding.count", len(embs))
        if isinstance(embs[0], list):
            span.set_attribute("neatlogs.embedding.dimensions", len(embs[0]))
    if body.get("inputTextTokenCount") is not None:
        span.set_attribute("neatlogs.llm.token_count.prompt", body["inputTextTokenCount"])
        span.set_attribute("neatlogs.embedding.token_count", body["inputTextTokenCount"])
    _ok(span, duration_ms)


def _patch_invoke_model_stream(client: Any) -> None:
    orig = client.invoke_model_with_response_stream

    def patched(*args, **kwargs):
        if is_suppressed():
            return orig(*args, **kwargs)
        model_id = kwargs.get("modelId")
        vendor = _vendor_from_model(model_id)
        span = _start_span("bedrock.invoke_model_with_response_stream", model_id, is_stream=True)
        _set_invoke_input(span, vendor, _decode_body(kwargs.get("body")))
        start = time.perf_counter()
        try:
            response = orig(*args, **kwargs)
        except Exception as e:
            _err(span, e)
            raise
        body = response.get("body") if isinstance(response, dict) else None
        if body is None:
            _ok(span, (time.perf_counter() - start) * 1000)
            return response
        response["body"] = _wrap_invoke_stream(body, span, vendor, start)
        return response

    client.invoke_model_with_response_stream = patched


def _wrap_invoke_stream(body: Any, span: Any, vendor: str, start: float):
    """Generator passing through InvokeModel stream chunks while accumulating text/usage."""
    text_parts: List[str] = []
    prompt_tokens = completion_tokens = None
    finish_reason = None
    errored = False
    try:
        for event in body:
            chunk = event.get("chunk", {}) if isinstance(event, dict) else {}
            data = _decode_body(chunk.get("bytes")) if chunk.get("bytes") is not None else {}
            if data:
                # Claude streaming
                if data.get("type") == "content_block_delta":
                    text_parts.append(data.get("delta", {}).get("text", ""))
                elif data.get("type") == "message_delta":
                    if data.get("delta", {}).get("stop_reason"):
                        finish_reason = data["delta"]["stop_reason"]
                    usage = data.get("usage", {})
                    if usage.get("output_tokens") is not None:
                        completion_tokens = usage["output_tokens"]
                # Titan / Llama streaming (single-field chunks)
                if data.get("outputText"):
                    text_parts.append(data["outputText"])
                if data.get("generation"):
                    text_parts.append(data["generation"])
                metrics = data.get("amazon-bedrock-invocationMetrics")
                if metrics:
                    prompt_tokens = metrics.get("inputTokenCount", prompt_tokens)
                    completion_tokens = metrics.get("outputTokenCount", completion_tokens)
            yield event
    except Exception as e:
        errored = True
        _err(span, e)
        raise
    finally:
        if not errored:
            full = "".join(text_parts)
            if full:
                span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
                span.set_attribute("neatlogs.llm.output_messages.0.content", full)
            if prompt_tokens is not None:
                span.set_attribute("neatlogs.llm.token_count.prompt", prompt_tokens)
            if completion_tokens is not None:
                span.set_attribute("neatlogs.llm.token_count.completion", completion_tokens)
            if finish_reason:
                span.set_attribute("neatlogs.llm.finish_reason", str(finish_reason))
            _ok(span, (time.perf_counter() - start) * 1000)


# ---------------------------------------------------------------------------
# botocore factory patch (for init(instrumentations=["bedrock"]))
# ---------------------------------------------------------------------------

_FACTORY_PATCHED = False
_ORIG_CREATE_CLIENT = None


def _patch_botocore_factory() -> None:
    global _FACTORY_PATCHED, _ORIG_CREATE_CLIENT
    if _FACTORY_PATCHED:
        return

    try:
        from botocore.client import ClientCreator
    except Exception:
        return

    _FACTORY_PATCHED = True
    _ORIG_CREATE_CLIENT = ClientCreator.create_client

    def _patched_create_client(self, service_name, *args, **kwargs):
        client = _ORIG_CREATE_CLIENT(self, service_name, *args, **kwargs)
        if service_name == "bedrock-runtime":
            try:
                wrap_bedrock_client(client)
            except Exception:
                pass
        return client

    ClientCreator.create_client = _patched_create_client


def _unpatch_botocore_factory() -> None:
    global _FACTORY_PATCHED, _ORIG_CREATE_CLIENT
    if not _FACTORY_PATCHED:
        return
    try:
        from botocore.client import ClientCreator

        if _ORIG_CREATE_CLIENT is not None:
            ClientCreator.create_client = _ORIG_CREATE_CLIENT
    except Exception:
        pass
    _FACTORY_PATCHED = False
    _ORIG_CREATE_CLIENT = None
