"""Lightweight client wrappers for OpenAI, Anthropic, and Google GenAI.

Intercepts LLM calls, captures full span attributes (tokens, cache, reasoning,
thinking blocks, messages, tool calls, streaming metrics), and writes LLM spans
into the shared SpanBuffer. Reads parent context from contextvars so spans nest
correctly under framework callbacks (LangChain/CrewAI).

Usage::

    nl = NeatlogsCallback(mcp_url="...")
    oai = nl.wrap(openai.OpenAI())
    resp = oai.chat.completions.create(model="gpt-4o", messages=[...])
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from . import context as ctx
from .buffer import SpanBuffer, SpanRecord


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(obj: Any, max_len: int = 50_000) -> str:
    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(obj)
    return s[:max_len] if len(s) > max_len else s


def _detect_client_type(client: Any) -> Optional[str]:
    cls_name = type(client).__name__
    module = type(client).__module__ or ""

    if "openai" in module or cls_name in ("OpenAI", "AsyncOpenAI"):
        return "openai"
    if "anthropic" in module or cls_name in ("Anthropic", "AsyncAnthropic"):
        return "anthropic"
    if "google" in module and ("genai" in module or "generativeai" in module):
        return "google_genai"
    if cls_name == "Client" and "google" in module:
        return "google_genai"
    return None


def wrap_client(client: Any, buffer: SpanBuffer, workflow_name: str) -> Any:
    """Wrap an LLM provider client to auto-capture LLM spans.

    Detects the provider from the client type and patches the relevant methods.
    Returns a proxy that behaves identically to the original client.
    """
    provider = _detect_client_type(client)
    if provider == "openai":
        return _OpenAIProxy(client, buffer, workflow_name)
    elif provider == "anthropic":
        return _AnthropicProxy(client, buffer, workflow_name)
    elif provider == "google_genai":
        return _GoogleGenAIProxy(client, buffer, workflow_name)
    else:
        raise ValueError(
            f"Unsupported client type: {type(client).__name__}. "
            "Supported: openai.OpenAI, anthropic.Anthropic, google.genai.Client"
        )


class _BaseProxy:
    """Base proxy that delegates attribute access to the wrapped client."""

    def __init__(self, client: Any, buffer: SpanBuffer, workflow_name: str, provider: str):
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_buffer", buffer)
        object.__setattr__(self, "_workflow_name", workflow_name)
        object.__setattr__(self, "_provider", provider)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_client"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_client"), name, value)

    def _get_parent(self) -> Optional[str]:
        return ctx.get_parent_span_id()

    def _get_trace_id(self) -> str:
        tid = ctx.get_trace_id()
        if not tid:
            tid = ctx.generate_trace_id()
            ctx.set_trace_id(tid)
        return tid

    def _base_attrs(self, model: str) -> Dict[str, Any]:
        provider = object.__getattribute__(self, "_provider")
        attrs: Dict[str, Any] = {
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.model_name": model,
            "neatlogs.llm.provider": provider,
            "neatlogs.llm.system": provider,
            "neatlogs.llm.request_type": "chat",
        }
        pt = ctx.get_prompt_template()
        if pt:
            attrs["neatlogs.llm.prompt_template"] = pt
            pv = ctx.get_prompt_variables()
            if pv:
                attrs["neatlogs.llm.prompt_template_variables"] = _safe_json(pv)
        upt = ctx.get_user_prompt_template()
        if upt:
            attrs["neatlogs.llm.user_prompt_template"] = upt
            upv = ctx.get_user_prompt_variables()
            if upv:
                attrs["neatlogs.llm.user_prompt_template_variables"] = _safe_json(upv)
        return attrs

    def _record_span(
        self,
        name: str,
        kind: str,
        start_iso: str,
        end_iso: str,
        attrs: Dict[str, Any],
        status_code: str = "OK",
        status_message: str = "",
    ) -> None:
        buffer = object.__getattribute__(self, "_buffer")
        workflow_name = object.__getattribute__(self, "_workflow_name")
        trace_id = self._get_trace_id()
        buffer.get_or_create_trace(trace_id, workflow_name)

        span = SpanRecord(
            span_id=ctx.generate_span_id(),
            parent_span_id=self._get_parent(),
            name=name,
            kind=kind,
            start_time=start_iso,
            end_time=end_iso,
            status_code=status_code,
            status_message=status_message,
            attributes=attrs,
        )
        buffer.add_span(trace_id, span)

    def _capture_params(self, attrs: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        param_map = {
            "temperature": "neatlogs.llm.temperature",
            "max_tokens": "neatlogs.llm.max_tokens",
            "max_output_tokens": "neatlogs.llm.max_tokens",
            "top_p": "neatlogs.llm.top_p",
            "top_k": "neatlogs.llm.top_k",
            "frequency_penalty": "neatlogs.llm.frequency_penalty",
            "presence_penalty": "neatlogs.llm.presence_penalty",
            "stop": "neatlogs.llm.stop_sequences",
            "stop_sequences": "neatlogs.llm.stop_sequences",
        }
        for src, tgt in param_map.items():
            if src in kwargs and kwargs[src] is not None:
                v = kwargs[src]
                attrs[tgt] = _safe_json(v) if isinstance(v, (list, dict)) else v

        if kwargs.get("stream"):
            attrs["neatlogs.llm.is_streaming"] = True
        if kwargs.get("response_format"):
            attrs["neatlogs.llm.request.structured_output_schema"] = _safe_json(kwargs["response_format"])
        if kwargs.get("reasoning_effort"):
            attrs["neatlogs.llm.reasoning_effort"] = str(kwargs["reasoning_effort"])

        safe = {k: v for k, v in kwargs.items() if k not in ("messages", "system", "contents", "api_key") and not k.startswith("_")}
        attrs["neatlogs.llm.invocation_parameters"] = _safe_json(safe)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class _OpenAIChatCompletionsProxy:
    def __init__(self, completions: Any, proxy: _BaseProxy):
        object.__setattr__(self, "_completions", completions)
        object.__setattr__(self, "_proxy", proxy)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_completions"), name)

    def create(self, **kwargs: Any) -> Any:
        proxy = object.__getattribute__(self, "_proxy")
        completions = object.__getattribute__(self, "_completions")
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])

        attrs = proxy._base_attrs(model)

        # Input messages
        for i, msg in enumerate(messages):
            if isinstance(msg, dict):
                attrs[f"neatlogs.llm.input_messages.{i}.role"] = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in content
                    )
                attrs[f"neatlogs.llm.input_messages.{i}.content"] = str(content)[:50_000]

        proxy._capture_params(attrs, kwargs)

        start_iso = _now_iso()
        t0 = time.monotonic()
        try:
            response = completions.create(**kwargs)
        except Exception as e:
            proxy._record_span(model, "LLM", start_iso, _now_iso(), attrs, "ERROR", str(e))
            raise

        end_iso = _now_iso()

        # Output
        choices = getattr(response, "choices", None) or []
        for i, choice in enumerate(choices):
            msg = getattr(choice, "message", None)
            if msg:
                attrs[f"neatlogs.llm.output_messages.{i}.role"] = getattr(msg, "role", "assistant")
                attrs[f"neatlogs.llm.output_messages.{i}.content"] = str(getattr(msg, "content", "") or "")[:50_000]

                fr = getattr(choice, "finish_reason", None)
                if fr:
                    attrs[f"neatlogs.llm.output_messages.{i}.message.finish_reason"] = str(fr)
                    if i == 0:
                        attrs["neatlogs.llm.finish_reason"] = str(fr)

                tool_calls = getattr(msg, "tool_calls", None) or []
                for ti, tc in enumerate(tool_calls):
                    attrs[f"neatlogs.llm.tool_calls.{ti}"] = _safe_json({
                        "id": getattr(tc, "id", ""),
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": getattr(getattr(tc, "function", None), "name", ""),
                            "arguments": getattr(getattr(tc, "function", None), "arguments", ""),
                        },
                    })

        # Token usage
        usage = getattr(response, "usage", None)
        if usage:
            attrs["neatlogs.llm.token_count.prompt"] = getattr(usage, "prompt_tokens", 0) or 0
            attrs["neatlogs.llm.token_count.completion"] = getattr(usage, "completion_tokens", 0) or 0
            attrs["neatlogs.llm.token_count.total"] = getattr(usage, "total_tokens", 0) or 0

            cd = getattr(usage, "completion_tokens_details", None)
            if cd and getattr(cd, "reasoning_tokens", None):
                attrs["neatlogs.llm.token_count.reasoning"] = cd.reasoning_tokens

            pd = getattr(usage, "prompt_tokens_details", None)
            if pd and getattr(pd, "cached_tokens", None):
                attrs["neatlogs.llm.token_count.cache_read"] = pd.cached_tokens

        proxy._record_span(model, "LLM", start_iso, end_iso, attrs)
        return response


class _OpenAIChatProxy:
    def __init__(self, chat: Any, proxy: _BaseProxy):
        object.__setattr__(self, "_chat", chat)
        object.__setattr__(self, "_proxy", proxy)
        object.__setattr__(self, "completions", _OpenAIChatCompletionsProxy(chat.completions, proxy))

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_chat"), name)


class _OpenAIEmbeddingsProxy:
    def __init__(self, embeddings: Any, proxy: _BaseProxy):
        object.__setattr__(self, "_embeddings", embeddings)
        object.__setattr__(self, "_proxy", proxy)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_embeddings"), name)

    def create(self, **kwargs: Any) -> Any:
        proxy = object.__getattribute__(self, "_proxy")
        embeddings = object.__getattribute__(self, "_embeddings")
        model = kwargs.get("model", "")

        attrs: Dict[str, Any] = {
            "neatlogs.span.kind": "embedding",
            "neatlogs.embedding.model_name": model,
            "input.value": _safe_json(kwargs.get("input", "")),
        }

        start_iso = _now_iso()
        try:
            response = embeddings.create(**kwargs)
        except Exception as e:
            proxy._record_span(model, "EMBEDDING", start_iso, _now_iso(), attrs, "ERROR", str(e))
            raise

        end_iso = _now_iso()
        data = getattr(response, "data", None) or []
        if data:
            first_emb = getattr(data[0], "embedding", None) or []
            attrs["neatlogs.embedding.vector_size"] = len(first_emb)

        usage = getattr(response, "usage", None)
        if usage:
            attrs["neatlogs.llm.token_count.prompt"] = getattr(usage, "prompt_tokens", 0) or 0
            attrs["neatlogs.llm.token_count.total"] = getattr(usage, "total_tokens", 0) or 0

        proxy._record_span(model, "EMBEDDING", start_iso, end_iso, attrs)
        return response


class _OpenAIProxy(_BaseProxy):
    def __init__(self, client: Any, buffer: SpanBuffer, workflow_name: str):
        super().__init__(client, buffer, workflow_name, "openai")
        object.__setattr__(self, "chat", _OpenAIChatProxy(client.chat, self))
        object.__setattr__(self, "embeddings", _OpenAIEmbeddingsProxy(client.embeddings, self))


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class _AnthropicMessagesProxy:
    def __init__(self, messages: Any, proxy: _BaseProxy):
        object.__setattr__(self, "_messages", messages)
        object.__setattr__(self, "_proxy", proxy)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_messages"), name)

    def create(self, **kwargs: Any) -> Any:
        proxy = object.__getattribute__(self, "_proxy")
        messages_api = object.__getattribute__(self, "_messages")
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        system = kwargs.get("system", "")

        attrs = proxy._base_attrs(model)

        # Input messages — system first, then messages
        idx = 0
        if system:
            if isinstance(system, str):
                attrs[f"neatlogs.llm.input_messages.{idx}.role"] = "system"
                attrs[f"neatlogs.llm.input_messages.{idx}.content"] = system[:50_000]
                idx += 1
            elif isinstance(system, list):
                for block in system:
                    if isinstance(block, dict) and block.get("text"):
                        attrs[f"neatlogs.llm.input_messages.{idx}.role"] = "system"
                        attrs[f"neatlogs.llm.input_messages.{idx}.content"] = str(block["text"])[:50_000]
                        idx += 1

        for msg in messages:
            if isinstance(msg, dict):
                attrs[f"neatlogs.llm.input_messages.{idx}.role"] = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", str(b)) if isinstance(b, dict) else str(b) for b in content
                    )
                attrs[f"neatlogs.llm.input_messages.{idx}.content"] = str(content)[:50_000]
                idx += 1

        proxy._capture_params(attrs, kwargs)

        start_iso = _now_iso()
        try:
            response = messages_api.create(**kwargs)
        except Exception as e:
            proxy._record_span(model, "LLM", start_iso, _now_iso(), attrs, "ERROR", str(e))
            raise

        end_iso = _now_iso()

        # Output messages + thinking blocks
        content_blocks = getattr(response, "content", None) or []
        out_idx = 0
        for block in content_blocks:
            btype = getattr(block, "type", "")
            if btype == "thinking":
                attrs[f"neatlogs.llm.output_messages.{out_idx}.role"] = "thinking"
                attrs[f"neatlogs.llm.output_messages.{out_idx}.content"] = str(getattr(block, "thinking", ""))[:50_000]
                out_idx += 1
            elif btype == "text":
                attrs[f"neatlogs.llm.output_messages.{out_idx}.role"] = "assistant"
                attrs[f"neatlogs.llm.output_messages.{out_idx}.content"] = str(getattr(block, "text", ""))[:50_000]
                out_idx += 1
            elif btype == "tool_use":
                attrs[f"neatlogs.llm.tool_calls.{out_idx}"] = _safe_json({
                    "id": getattr(block, "id", ""),
                    "type": "function",
                    "function": {
                        "name": getattr(block, "name", ""),
                        "arguments": _safe_json(getattr(block, "input", {})),
                    },
                })

        # Finish reason
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason:
            attrs["neatlogs.llm.finish_reason"] = str(stop_reason)
            attrs["neatlogs.llm.stop_reason"] = str(stop_reason)

        # Token usage
        usage = getattr(response, "usage", None)
        if usage:
            attrs["neatlogs.llm.token_count.prompt"] = getattr(usage, "input_tokens", 0) or 0
            attrs["neatlogs.llm.token_count.completion"] = getattr(usage, "output_tokens", 0) or 0
            total = (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0)
            attrs["neatlogs.llm.token_count.total"] = total

            if getattr(usage, "cache_read_input_tokens", None):
                attrs["neatlogs.llm.token_count.cache_read"] = usage.cache_read_input_tokens
            if getattr(usage, "cache_creation_input_tokens", None):
                attrs["neatlogs.llm.token_count.cache_write"] = usage.cache_creation_input_tokens

        proxy._record_span(model, "LLM", start_iso, end_iso, attrs)
        return response


class _AnthropicProxy(_BaseProxy):
    def __init__(self, client: Any, buffer: SpanBuffer, workflow_name: str):
        super().__init__(client, buffer, workflow_name, "anthropic")
        object.__setattr__(self, "messages", _AnthropicMessagesProxy(client.messages, self))


# ---------------------------------------------------------------------------
# Google GenAI
# ---------------------------------------------------------------------------

class _GoogleGenAIModelsProxy:
    def __init__(self, models: Any, proxy: _BaseProxy):
        object.__setattr__(self, "_models", models)
        object.__setattr__(self, "_proxy", proxy)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_models"), name)

    def generate_content(self, **kwargs: Any) -> Any:
        proxy = object.__getattribute__(self, "_proxy")
        models = object.__getattribute__(self, "_models")
        model = kwargs.get("model", "")
        contents = kwargs.get("contents", [])

        attrs = proxy._base_attrs(model)

        # Input messages from contents
        idx = 0
        config = kwargs.get("config", {}) or {}
        system_instruction = config.get("system_instruction") if isinstance(config, dict) else getattr(config, "system_instruction", None)
        if system_instruction:
            attrs[f"neatlogs.llm.input_messages.{idx}.role"] = "system"
            attrs[f"neatlogs.llm.input_messages.{idx}.content"] = str(system_instruction)[:50_000]
            idx += 1

        if isinstance(contents, str):
            attrs[f"neatlogs.llm.input_messages.{idx}.role"] = "user"
            attrs[f"neatlogs.llm.input_messages.{idx}.content"] = contents[:50_000]
        elif isinstance(contents, list):
            for item in contents:
                if isinstance(item, str):
                    attrs[f"neatlogs.llm.input_messages.{idx}.role"] = "user"
                    attrs[f"neatlogs.llm.input_messages.{idx}.content"] = item[:50_000]
                    idx += 1
                elif isinstance(item, dict):
                    role = item.get("role", "user")
                    parts = item.get("parts", [])
                    text = " ".join(
                        p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in parts
                    ) if isinstance(parts, list) else str(parts)
                    attrs[f"neatlogs.llm.input_messages.{idx}.role"] = role
                    attrs[f"neatlogs.llm.input_messages.{idx}.content"] = text[:50_000]
                    idx += 1
                elif hasattr(item, "role") and hasattr(item, "parts"):
                    role = getattr(item, "role", "user")
                    parts = getattr(item, "parts", [])
                    text = " ".join(
                        getattr(p, "text", str(p)) for p in parts
                    )
                    attrs[f"neatlogs.llm.input_messages.{idx}.role"] = role
                    attrs[f"neatlogs.llm.input_messages.{idx}.content"] = text[:50_000]
                    idx += 1

        proxy._capture_params(attrs, kwargs)

        start_iso = _now_iso()
        try:
            response = models.generate_content(**kwargs)
        except Exception as e:
            proxy._record_span(model, "LLM", start_iso, _now_iso(), attrs, "ERROR", str(e))
            raise

        end_iso = _now_iso()

        # Output
        candidates = getattr(response, "candidates", None) or []
        out_idx = 0
        for ci, candidate in enumerate(candidates):
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                thought = getattr(part, "thought", False)
                text = getattr(part, "text", None)
                if thought and text:
                    attrs[f"neatlogs.llm.output_messages.{out_idx}.role"] = "thinking"
                    attrs[f"neatlogs.llm.output_messages.{out_idx}.content"] = text[:50_000]
                    out_idx += 1
                elif text:
                    attrs[f"neatlogs.llm.output_messages.{out_idx}.role"] = "model"
                    attrs[f"neatlogs.llm.output_messages.{out_idx}.content"] = text[:50_000]
                    out_idx += 1

            fr = getattr(candidate, "finish_reason", None)
            if fr:
                fr_str = fr.name if hasattr(fr, "name") else str(fr)
                attrs["neatlogs.llm.finish_reason"] = fr_str

            safety = getattr(candidate, "safety_ratings", None)
            if safety:
                try:
                    attrs[f"neatlogs.llm.output_messages.{ci}.message.safety_ratings"] = _safe_json([
                        {
                            "category": r.category.name if hasattr(r.category, "name") else str(r.category),
                            "probability": r.probability.name if hasattr(r.probability, "name") else str(r.probability),
                        }
                        for r in safety
                    ])
                except Exception:
                    pass

        # Token usage
        usage = getattr(response, "usage_metadata", None)
        if usage:
            attrs["neatlogs.llm.token_count.prompt"] = getattr(usage, "prompt_token_count", 0) or 0
            attrs["neatlogs.llm.token_count.completion"] = getattr(usage, "candidates_token_count", 0) or 0
            attrs["neatlogs.llm.token_count.total"] = getattr(usage, "total_token_count", 0) or 0

            if getattr(usage, "thinking_token_count", None):
                attrs["neatlogs.llm.token_count.reasoning"] = usage.thinking_token_count
            if getattr(usage, "cached_content_token_count", None):
                attrs["neatlogs.llm.token_count.cache_read"] = usage.cached_content_token_count

        proxy._record_span(model, "LLM", start_iso, end_iso, attrs)
        return response


class _GoogleGenAIProxy(_BaseProxy):
    def __init__(self, client: Any, buffer: SpanBuffer, workflow_name: str):
        super().__init__(client, buffer, workflow_name, "google_genai")
        object.__setattr__(self, "models", _GoogleGenAIModelsProxy(client.models, self))
