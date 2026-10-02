"""Reconstruct an LLM invocation from a captured span.

Reads the ``neatlogs.llm.*`` attribute family written by ``neatlogs/openai.py``
(and OpenAI-compatible wrappers). Both the wrapper form
(``neatlogs.llm.temperature``) and the normalized form
(``neatlogs.llm.invocation_parameters.temperature``) are accepted.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional

from .types import LLMRequest, ReplayError, UnsupportedSpanError

SUPPORTED_PROVIDERS = frozenset({"openai", "azure_openai", "openrouter", "openai-compatible"})


def _attributes(span: Any) -> Mapping[str, Any]:
    attrs = getattr(span, "attributes", None) or {}
    return dict(attrs)


def span_kind(span: Any) -> str:
    """Uppercase canonical kind (LLM/TOOL/...) from neatlogs or OI attributes."""
    attrs = _attributes(span)
    kind = str(attrs.get("neatlogs.span.kind") or attrs.get("openinference.span.kind") or "")
    if "." in kind:
        kind = kind.rsplit(".", 1)[-1]
    return kind.upper()


def span_provider(span: Any) -> str:
    """Raw provider string from `neatlogs.llm.provider` (empty if unset)."""
    attrs = _attributes(span)
    return str(attrs.get("neatlogs.llm.provider") or attrs.get("neatlogs.llm.system") or "")


def _indexed(attrs: Mapping[str, Any], prefix: str) -> Dict[int, Dict[str, Any]]:
    result: Dict[int, Dict[str, Any]] = {}
    needle = f"{prefix}."
    for key, value in attrs.items():
        if not key.startswith(needle):
            continue
        rest = key[len(needle) :]
        index_text, sep, field = rest.partition(".")
        if not sep or not index_text.isdigit() or not field:
            continue
        result.setdefault(int(index_text), {})[field] = value
    return result


def _decode(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[:1] not in {"[", "{"}:
        return value
    try:
        return json.loads(stripped)
    except (TypeError, ValueError):
        return value


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _param(attrs: Mapping[str, Any], name: str) -> Any:
    if f"neatlogs.llm.{name}" in attrs:
        return attrs[f"neatlogs.llm.{name}"]
    return attrs.get(f"neatlogs.llm.invocation_parameters.{name}")


def extract_messages(span: Any) -> List[Dict[str, Any]]:
    """Rebuild the chat message list ({role, content, ...}) from span attributes."""
    attrs = _attributes(span)
    records = _indexed(attrs, "neatlogs.llm.input_messages")
    messages = []
    for index in sorted(records):
        record = records[index]
        content = record.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        message: Dict[str, Any] = {
            "role": str(record.get("role") or "user"),
            "content": content,
        }
        if record.get("tool_call_id"):
            message["tool_call_id"] = str(record["tool_call_id"])
        if record.get("name"):
            message["name"] = str(record["name"])
        messages.append(message)
    if not messages:
        fallback = attrs.get("input.value")
        decoded = _decode(fallback)
        if isinstance(decoded, list):
            messages = [m for m in decoded if isinstance(m, dict)]
    return messages


def extract_tools(span: Any) -> List[Dict[str, Any]]:
    """Rebuild tool schemas ({name, type, definition}) from span attributes."""
    attrs = _attributes(span)
    records = _indexed(attrs, "neatlogs.llm.tools")
    tools = []
    for index in sorted(records):
        record = records[index]
        definition = _decode(record.get("definition"))
        tools.append(
            {
                "name": str(record.get("name") or ""),
                "type": str(record.get("type") or "function"),
                "definition": definition if definition is not None else dict(record),
            }
        )
    return tools


def extract_llm_request(span: Any) -> Optional[LLMRequest]:
    """Return the invocation for LLM spans, None for other kinds.

    Raises:
        UnsupportedSpanError: span is LLM but provider is not replayable.
        ReplayError: span claims to be LLM but has no usable input.
    """
    kind = span_kind(span)
    if kind != "LLM":
        return None
    provider = span_provider(span).lower()
    if not provider:
        raise UnsupportedSpanError(
            "LLM span carries no provider attribute; refusing to guess OpenAI " "compatibility",
            kind=kind,
            provider="",
        )
    if provider not in SUPPORTED_PROVIDERS and "openai" not in provider:
        raise UnsupportedSpanError(
            f"Provider '{provider}' is not supported for replay in this MVP",
            kind=kind,
            provider=provider,
        )
    attrs = _attributes(span)
    model = str(
        attrs.get("neatlogs.llm.request_model") or attrs.get("neatlogs.llm.model_name") or ""
    )
    if not model:
        raise ReplayError("LLM span has no model (neatlogs.llm.model_name missing)")
    messages = extract_messages(span)
    if not messages:
        raise ReplayError("LLM span has no input messages to replay")
    tools = extract_tools(span)
    temperature = _number(_param(attrs, "temperature"))
    top_p = _number(_param(attrs, "top_p"))
    max_tokens = _integer(
        _param(attrs, "max_tokens")
        if _param(attrs, "max_tokens") is not None
        else attrs.get("neatlogs.llm.invocation_parameters.max_output_tokens")
    )
    return LLMRequest(
        provider=provider,
        model=model,
        messages=messages,
        tools=tools,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        frequency_penalty=_number(_param(attrs, "frequency_penalty")),
        presence_penalty=_number(_param(attrs, "presence_penalty")),
    )
