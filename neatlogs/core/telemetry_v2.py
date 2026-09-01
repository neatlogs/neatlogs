"""Typed canonical-v2 normalization used by diagnostics and contract tests.

This module deliberately operates on the post-processor ``ReadableSpan`` that an
exporter receives.  It therefore describes what NeatLogs is actually about to
send, rather than re-running wrapper logic or backend simplification rules.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Optional

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from ..schema_v2 import TELEMETRY_SCHEMA_VERSION

_KINDS = {
    "WORKFLOW",
    "AGENT",
    "CHAIN",
    "TASK",
    "LLM",
    "TOOL",
    "MCP_TOOL",
    "RETRIEVER",
    "RERANKER",
    "EMBEDDING",
    "VECTOR_STORE",
    "MEMORY",
    "GUARDRAIL",
    "LOG",
    "HTTP",
    "UNKNOWN",
}
_FIDELITIES = {"native", "normalized", "flattened", "synthetic-recovery", "unknown"}


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    return str(value)


def _decode(value: Any) -> Any:
    if not isinstance(value, str):
        return _json_value(value)
    stripped = value.strip()
    if not stripped or stripped[:1] not in {"[", "{"}:
        return value
    try:
        return _json_value(json.loads(stripped))
    except (TypeError, ValueError):
        return value


def _first(attrs: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = attrs.get(key)
        if value is not None and value != "":
            return value
    return default


def _integer(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"true", "1", "yes"}:
            return True
        if value.lower() in {"false", "0", "no"}:
            return False
    return default


def _indexed(attrs: Mapping[str, Any], prefix: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    needle = f"{prefix}."
    for key, value in attrs.items():
        if not key.startswith(needle):
            continue
        suffix = key[len(needle) :]
        index_text, separator, field = suffix.partition(".")
        if not separator or not index_text.isdigit() or not field:
            continue
        result.setdefault(int(index_text), {})[field] = value
    return result


def _kind(attrs: Mapping[str, Any]) -> str:
    value = str(
        attrs.get("neatlogs.span.kind") or attrs.get("openinference.span.kind") or "UNKNOWN"
    )
    if "." in value:
        value = value.rsplit(".", 1)[-1]
    value = value.upper()
    return value if value in _KINDS else "UNKNOWN"


def _typed_value(attrs: Mapping[str, Any], kind: str, direction: str) -> dict[str, Any]:
    lower_kind = kind.lower()
    raw = _first(
        attrs,
        f"neatlogs.{lower_kind}.{direction}",
        f"{direction}.value",
    )
    mime = str(
        _first(
            attrs,
            f"neatlogs.{lower_kind}.{direction}_mime_type",
            f"{direction}.mime_type",
            default="",
        )
    )
    value = _decode(raw)
    media = _media(attrs, f"neatlogs.{lower_kind}.{direction}", direction)
    if raw is None and not media:
        value_type = "none"
    elif media and raw is None:
        value_type = "media"
    elif mime == "application/json" or isinstance(value, (dict, list)):
        value_type = "json"
    else:
        value_type = "text"
    return {"type": value_type, "value": value, "media": media}


def _normalize_media_records(
    records: Mapping[int, Mapping[str, Any]], purpose: str
) -> list[dict[str, Any]]:
    normalized = []
    for index, record in sorted(records.items()):
        reference = str(record.get("reference") or "")
        digest = str(record.get("sha256") or "")
        if len(digest) != 64:
            digest = hashlib.sha256(reference.encode()).hexdigest()
        source = str(record.get("source") or "provider")
        if source not in {"inline", "url", "file", "provider", "uploaded", "generated"}:
            source = "provider"
        state = str(record.get("state") or "available")
        if state not in {"inline", "pending-upload", "available", "failed", "expired"}:
            state = "available"
        normalized.append(
            {
                "id": str(record.get("id") or f"nl_media_{digest[:24]}_{index}"),
                "sha256": digest,
                "mime_type": str(record.get("mime_type") or "application/octet-stream"),
                "byte_length": _integer(record.get("byte_length")) or 0,
                "source": source,
                "purpose": purpose if purpose in {"input", "output"} else "document",
                "state": state,
                "safe_preview": None,
            }
        )
    return normalized


def _media(attrs: Mapping[str, Any], prefix: str, purpose: str) -> list[dict[str, Any]]:
    return _normalize_media_records(_indexed(attrs, f"{prefix}.media"), purpose)


def _content(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, Mapping) and item.get("type") == "text":
                parts.append({"type": "text", "text": str(item.get("text") or "")})
            elif isinstance(item, str):
                parts.append({"type": "text", "text": item})
        if parts:
            return parts
    return [{"type": "text", "text": str(value)}]


def _message(
    record: Mapping[str, Any], tool_calls: list[dict[str, Any]], purpose: str
) -> dict[str, Any]:
    role = str(record.get("role") or "assistant").lower()
    if role == "model":
        role = "assistant"
    if role not in {"system", "developer", "user", "assistant", "tool"}:
        role = "assistant"
    content = _content(_decode(record.get("content")))
    raw_media = _indexed(record, "media")
    for index, media in sorted(raw_media.items()):
        references = _normalize_media_records({index: media}, purpose)
        if not references:
            continue
        reference = references[0]
        media_type = str(media.get("type") or "document")
        if media_type not in {"image", "audio", "document", "video"}:
            media_type = "document"
        content.append({"type": media_type, "reference": reference})
    return {
        "role": role,
        "name": str(record["name"]) if record.get("name") else None,
        "content": content,
        "tool_calls": tool_calls,
        "tool_call_id": str(record["tool_call_id"]) if record.get("tool_call_id") else None,
    }


def _llm_semantic(span: ReadableSpan, attrs: Mapping[str, Any]) -> dict[str, Any]:
    tool_calls_by_choice: dict[int, list[dict[str, Any]]] = {}
    for position, record in sorted(_indexed(attrs, "neatlogs.llm.tool_calls").items()):
        choice_index = _integer(record.get("choice_index")) or 0
        tool_index = _integer(record.get("tool_call_index"))
        tool_index = position if tool_index is None else tool_index
        arguments = _decode(record.get("arguments"))
        call_id = str(record.get("id") or "")
        synthetic = _bool(record.get("id_synthetic"))
        if not call_id:
            material = (
                f"{span.context.trace_id:032x}:{span.context.span_id:016x}:"
                f"{choice_index}:{tool_index}:{record.get('name') or ''}:"
                f"{json.dumps(arguments, sort_keys=True, default=str)}"
            )
            call_id = f"nl_{hashlib.sha256(material.encode()).hexdigest()[:24]}"
            synthetic = True
        tool_calls_by_choice.setdefault(choice_index, []).append(
            {
                "id": call_id,
                "id_origin": "deterministic-synthetic" if synthetic else "provider",
                "type": str(record.get("type") or "function"),
                "name": str(record.get("name") or "unknown_tool"),
                "arguments": arguments,
                "choice_index": choice_index,
                "tool_index": tool_index,
            }
        )

    input_messages = [
        _message(record, [], "input")
        for _, record in sorted(_indexed(attrs, "neatlogs.llm.input_messages").items())
    ]
    output_messages = _indexed(attrs, "neatlogs.llm.output_messages")
    choice_indexes = sorted(set(output_messages) | set(tool_calls_by_choice))
    choices = []
    for index in choice_indexes:
        record = output_messages.get(index, {})
        choices.append(
            {
                "choice_index": index,
                "message": _message(record, tool_calls_by_choice.get(index, []), "output"),
                "finish_reason": (
                    str(attrs[f"neatlogs.llm.choices.{index}.finish_reason"])
                    if attrs.get(f"neatlogs.llm.choices.{index}.finish_reason") is not None
                    else None
                ),
            }
        )

    parameters = {
        "temperature": _number(attrs.get("neatlogs.llm.invocation_parameters.temperature")),
        "top_p": _number(attrs.get("neatlogs.llm.invocation_parameters.top_p")),
        "top_k": _number(attrs.get("neatlogs.llm.invocation_parameters.top_k")),
        "max_output_tokens": _integer(
            _first(
                attrs,
                "neatlogs.llm.invocation_parameters.max_output_tokens",
                "neatlogs.llm.invocation_parameters.max_tokens",
            )
        ),
        "stop": [],
        "seed": _integer(attrs.get("neatlogs.llm.invocation_parameters.seed")),
        "frequency_penalty": _number(
            attrs.get("neatlogs.llm.invocation_parameters.frequency_penalty")
        ),
        "presence_penalty": _number(
            attrs.get("neatlogs.llm.invocation_parameters.presence_penalty")
        ),
        "response_format": _decode(attrs.get("neatlogs.llm.invocation_parameters.response_format")),
        "reasoning": _decode(attrs.get("neatlogs.llm.invocation_parameters.reasoning")),
        "service_tier": _first(
            attrs, "neatlogs.llm.invocation_parameters.service_tier", default=None
        ),
        "provider_options": _decode(attrs.get("neatlogs.llm.invocation_parameters")),
    }
    stream_events = []
    for sequence, event in enumerate(span.events):
        if event.name != "neatlogs.stream.chunk":
            continue
        event_attrs = event.attributes or {}
        stream_events.append(
            {
                "sequence": sequence,
                "time_unix_nano": str(event.timestamp or 0),
                "type": "output_text.delta",
                "choice_index": None,
                "tool_index": None,
                "value": _decode(event_attrs.get("neatlogs.stream.chunk.summary")),
            }
        )
    finish_reasons = [choice["finish_reason"] for choice in choices]
    return {
        "kind": "LLM",
        "request": {
            "provider": _first(attrs, "neatlogs.llm.provider", default=None),
            "model": _first(
                attrs,
                "neatlogs.llm.request_model",
                "neatlogs.llm.model_name",
                default=None,
            ),
            "operation": str(_first(attrs, "neatlogs.llm.operation", default="unknown")),
            "messages": input_messages,
            "tools": [],
            "parameters": parameters,
        },
        "response": {
            "id": _first(attrs, "neatlogs.llm.response_id", default=None),
            "model": _first(attrs, "neatlogs.llm.model_name", default=None),
            "choices": choices,
            "finish_reasons": finish_reasons,
        },
        "usage": _usage(attrs),
        "stream": {
            "time_to_first_token_ms": _number(attrs.get("neatlogs.llm.metrics.ttft_ms")),
            "chunk_count": _integer(attrs.get("neatlogs.stream.chunk_count")) or 0,
            "choice_count": len(choices),
            "events": stream_events,
            "raw_chunks": None,
        },
    }


def _usage(attrs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "input_tokens": _integer(attrs.get("neatlogs.llm.token_count.prompt")),
        "output_tokens": _integer(attrs.get("neatlogs.llm.token_count.completion")),
        "total_tokens": _integer(attrs.get("neatlogs.llm.token_count.total")),
        "reasoning_tokens": _integer(attrs.get("neatlogs.llm.token_count.reasoning")),
        "cache_read_tokens": _integer(attrs.get("neatlogs.llm.token_count.cache_read")),
        "cache_write_tokens": _integer(attrs.get("neatlogs.llm.token_count.cache_write")),
        "cost_usd": _number(attrs.get("neatlogs.llm.cost_usd")),
    }


def _documents(value: Any) -> list[dict[str, Any]]:
    decoded = _decode(value)
    items = decoded if isinstance(decoded, list) else ([] if decoded is None else [decoded])
    result = []
    for item in items:
        record = item if isinstance(item, Mapping) else {"content": item}
        result.append(
            {
                "id": str(record["id"]) if record.get("id") is not None else None,
                "content": _json_value(record.get("content")),
                "score": _number(record.get("score")),
                "metadata": _json_value(record.get("metadata")),
                "media": [],
            }
        )
    return result


def _tool_semantic(span: ReadableSpan, attrs: Mapping[str, Any], kind: str) -> dict[str, Any]:
    prefix = "neatlogs.mcp_tool" if kind == "MCP_TOOL" else "neatlogs.tool"
    name = str(_first(attrs, f"{prefix}.name", default=span.name) or "unknown_tool")
    arguments = _decode(_first(attrs, f"{prefix}.arguments", f"{prefix}.input"))
    call_id = str(_first(attrs, f"{prefix}.call_id", f"{prefix}.id", default="") or "")
    id_origin = "neatlogs-direct"
    if not call_id:
        material = (
            f"{span.context.trace_id:032x}:{span.context.span_id:016x}:{name}:"
            f"{json.dumps(arguments, sort_keys=True, default=str)}"
        )
        call_id = f"nl_{hashlib.sha256(material.encode()).hexdigest()[:24]}"
        id_origin = "deterministic-synthetic"
    output = _decode(_first(attrs, f"{prefix}.output", "output.value"))
    return {
        "kind": kind,
        "definition": {
            "type": str(_first(attrs, f"{prefix}.type", default="function")),
            "name": name,
            "description": _first(attrs, f"{prefix}.description", default=None),
            "schema": _decode(attrs.get(f"{prefix}.schema")),
            "configuration": _decode(attrs.get(f"{prefix}.configuration")),
        },
        "call": {
            "id": call_id,
            "id_origin": id_origin,
            "type": str(_first(attrs, f"{prefix}.type", default="function")),
            "name": name,
            "arguments": arguments,
            "choice_index": _integer(attrs.get(f"{prefix}.choice_index")) or 0,
            "tool_index": _integer(attrs.get(f"{prefix}.tool_index")) or 0,
        },
        "result": {"call_id": call_id, "value": output, "is_error": False, "media": []},
        "requesting_span_id": f"{span.parent.span_id:016x}" if span.parent else None,
        "transport": (
            {"transport": "unknown", "server": None, "method": None, "request_id": None}
            if kind == "MCP_TOOL"
            else None
        ),
    }


def _semantic(span: ReadableSpan, attrs: Mapping[str, Any], kind: str) -> dict[str, Any]:
    if kind == "LLM":
        return _llm_semantic(span, attrs)
    if kind in {"TOOL", "MCP_TOOL"}:
        return _tool_semantic(span, attrs, kind)
    if kind == "RETRIEVER":
        return {
            "kind": kind,
            "query": _decode(_first(attrs, "neatlogs.retriever.query", "neatlogs.retriever.input")),
            "documents": _documents(
                _first(attrs, "neatlogs.retriever.documents", "neatlogs.retriever.output")
            ),
            "top_k": _integer(attrs.get("neatlogs.retriever.top_k")),
            "filters": _decode(attrs.get("neatlogs.retriever.filters")),
        }
    if kind == "RERANKER":
        return {
            "kind": kind,
            "model": _first(attrs, "neatlogs.reranker.model", default=None),
            "query": _decode(attrs.get("neatlogs.reranker.query")),
            "input_documents": _documents(attrs.get("neatlogs.reranker.input_documents")),
            "output_documents": _documents(attrs.get("neatlogs.reranker.output_documents")),
            "top_n": _integer(attrs.get("neatlogs.reranker.top_n")),
        }
    if kind == "EMBEDDING":
        inputs = _decode(_first(attrs, "neatlogs.embedding.input", default=[]))
        vectors = _decode(_first(attrs, "neatlogs.embedding.output", default=[]))
        return {
            "kind": kind,
            "provider": _first(attrs, "neatlogs.embedding.provider", default=None),
            "model": _first(attrs, "neatlogs.embedding.model_name", default=None),
            "inputs": inputs if isinstance(inputs, list) else [inputs],
            "vectors": vectors if isinstance(vectors, list) else [],
            "dimensions": _integer(attrs.get("neatlogs.embedding.dimensions")),
            "usage": _usage(attrs),
        }
    if kind == "VECTOR_STORE":
        return {
            "kind": kind,
            "provider": _first(attrs, "neatlogs.vector_store.provider", default=None),
            "operation": str(_first(attrs, "neatlogs.vector_store.operation", default="unknown")),
            "collection": _first(attrs, "neatlogs.vector_store.collection", default=None),
            "query": _decode(attrs.get("neatlogs.vector_store.query")),
            "documents": _documents(attrs.get("neatlogs.vector_store.documents")),
        }
    if kind == "MEMORY":
        operation = str(_first(attrs, "neatlogs.memory.operation", default="unknown"))
        if operation not in {"read", "write", "update", "delete", "search", "unknown"}:
            operation = "unknown"
        return {
            "kind": kind,
            "operation": operation,
            "memory_id": _first(attrs, "neatlogs.memory.id", default=None),
            "scope": _decode(attrs.get("neatlogs.memory.scope")),
            "value": _decode(_first(attrs, "neatlogs.memory.value", "neatlogs.memory.output")),
        }
    if kind == "GUARDRAIL":
        return {
            "kind": kind,
            "name": str(_first(attrs, "neatlogs.guardrail.name", default=span.name)),
            "action": _first(attrs, "neatlogs.guardrail.action", default=None),
            "triggered": _bool(attrs.get("neatlogs.guardrail.triggered")),
            "score": _number(attrs.get("neatlogs.guardrail.score")),
            "reason": _first(attrs, "neatlogs.guardrail.reason", default=None),
        }
    if kind == "LOG":
        return {
            "kind": kind,
            "severity_text": _first(attrs, "neatlogs.log.severity_text", default=None),
            "severity_number": _integer(attrs.get("neatlogs.log.severity_number")),
            "body": _decode(_first(attrs, "neatlogs.log.body", "neatlogs.log.output")),
            "logger_name": _first(attrs, "neatlogs.log.logger_name", default=None),
        }
    lower_kind = kind.lower()
    return {
        "kind": kind,
        "operation": _first(attrs, f"neatlogs.{lower_kind}.operation", default=span.name),
        "role": _first(attrs, f"neatlogs.{lower_kind}.role", default=None),
        "metadata": _decode(attrs.get(f"neatlogs.{lower_kind}.metadata")),
        "recovery": None,
    }


def _error(span: ReadableSpan) -> Optional[dict[str, Any]]:
    for event in reversed(span.events):
        if event.name != "exception":
            continue
        attrs = event.attributes or {}
        return {
            "type": attrs.get("exception.type"),
            "message": attrs.get("exception.message"),
            "stack": attrs.get("exception.stacktrace"),
            "escaped": _bool(attrs.get("exception.escaped")),
        }
    return None


@dataclass(frozen=True, slots=True)
class OwnershipV2:
    owner: str = "neatlogs-sdk"
    provider_generation: int = 0
    project_key_id: Optional[str] = None
    propagation: str = "local-private-context"


@dataclass(frozen=True, slots=True)
class WrapperV2:
    captured: bool
    integration: Optional[str]
    integration_version: Optional[str]
    capture_fidelity: str


@dataclass(frozen=True, slots=True)
class TypedValueV2:
    type: str
    value: Any
    media: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class StatusV2:
    code: str
    message: Optional[str]
    source: str


@dataclass(frozen=True, slots=True)
class TelemetrySpanV2:
    schema_version: int
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    kind: str
    start_time_unix_nano: str
    end_time_unix_nano: str
    ownership: OwnershipV2
    wrapper: WrapperV2
    input: TypedValueV2
    output: TypedValueV2
    status: StatusV2
    error: Optional[dict[str, Any]]
    code: Optional[dict[str, Any]]
    provenance: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]
    semantic: dict[str, Any]
    attributes: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the language-neutral JSON representation."""

        return asdict(self)


def normalize_span_v2(
    span: ReadableSpan,
    *,
    provider_generation: int = 0,
    project_key_id: Optional[str] = None,
) -> TelemetrySpanV2:
    """Normalize one post-processor span into the canonical telemetry contract."""

    attrs = {str(key): _json_value(value) for key, value in (span.attributes or {}).items()}
    resource_attrs = span.resource.attributes if span.resource else {}
    kind = _kind(attrs)
    scope = getattr(span, "instrumentation_scope", None)
    scope_name = str(getattr(scope, "name", "") or "")
    fidelity = str(attrs.get("neatlogs.capture_fidelity") or "unknown")
    if fidelity not in _FIDELITIES:
        fidelity = "unknown"
    integration = _first(
        attrs,
        "neatlogs.integration.name",
        "neatlogs.instrumentation.name",
        default=scope_name or None,
    )
    integration_version = _first(
        attrs,
        "neatlogs.integration.version",
        default=getattr(scope, "version", None),
    )
    code = None
    if any(
        key in attrs
        for key in ("code.file.path", "code.function.name", "code.line.number", "code.namespace")
    ):
        code = {
            "file": attrs.get("code.file.path"),
            "function": attrs.get("code.function.name"),
            "line": _integer(attrs.get("code.line.number")),
            "column": _integer(attrs.get("code.column.number")),
            "namespace": attrs.get("code.namespace"),
        }
    status_source = "sdk" if attrs.get("neatlogs.trace.interrupted") else "application"
    input_value = _typed_value(attrs, kind, "input")
    output_value = _typed_value(attrs, kind, "output")
    return TelemetrySpanV2(
        schema_version=TELEMETRY_SCHEMA_VERSION,
        trace_id=f"{span.context.trace_id:032x}",
        span_id=f"{span.context.span_id:016x}",
        parent_span_id=f"{span.parent.span_id:016x}" if span.parent else None,
        name=str(span.name or "unknown"),
        kind=kind,
        start_time_unix_nano=str(span.start_time or 0),
        end_time_unix_nano=str(span.end_time or 0),
        ownership=OwnershipV2(
            provider_generation=max(0, int(provider_generation)),
            project_key_id=project_key_id or resource_attrs.get("neatlogs.project.key_id") or None,
        ),
        wrapper=WrapperV2(
            captured=scope_name.startswith("neatlogs") or fidelity != "unknown",
            integration=str(integration) if integration else None,
            integration_version=str(integration_version) if integration_version else None,
            capture_fidelity=fidelity,
        ),
        input=TypedValueV2(**input_value),
        output=TypedValueV2(**output_value),
        status=StatusV2(
            code=span.status.status_code.name,
            message=span.status.description,
            source=status_source,
        ),
        error=_error(span),
        code=code,
        provenance=[],
        conflicts=[],
        semantic=_semantic(span, attrs, kind),
        attributes=attrs,
    )


class InMemoryDiagnosticSpanExporter(SpanExporter):
    """Bounded, thread-safe canonical exporter for doctor/tests; never sends data."""

    def __init__(self, max_spans: int = 1000, *, provider_generation: int = 0) -> None:
        if isinstance(max_spans, bool) or not isinstance(max_spans, int) or max_spans <= 0:
            raise ValueError("max_spans must be a positive integer")
        self._spans: deque[TelemetrySpanV2] = deque(maxlen=max_spans)
        self._provider_generation = provider_generation
        self._lock = threading.RLock()
        self._closed = False
        self._dropped = 0

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        normalized = [
            normalize_span_v2(span, provider_generation=self._provider_generation) for span in spans
        ]
        with self._lock:
            if self._closed:
                return SpanExportResult.FAILURE
            overflow = max(0, len(self._spans) + len(normalized) - self._spans.maxlen)
            self._dropped += overflow
            self._spans.extend(normalized)
        return SpanExportResult.SUCCESS

    def get_finished_envelopes(self) -> tuple[TelemetrySpanV2, ...]:
        with self._lock:
            return tuple(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()
            self._dropped = 0

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
