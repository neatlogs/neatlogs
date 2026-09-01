"""Bound telemetry capture without hiding that content was truncated."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.trace import Link, Status

DEFAULT_MAX_CAPTURE_VALUE_BYTES = 100_000
DEFAULT_MAX_CAPTURE_ITEM_BYTES = 1_000_000
CAPTURE_TRUNCATION_MARKER = "...[neatlogs-truncated"
UPLOAD_UNAVAILABLE_REASON = "backend_upload_contract_unavailable"


def _truncation_marker(original_bytes: int, digest: str) -> bytes:
    return (
        f"{CAPTURE_TRUNCATION_MARKER} original_bytes={original_bytes} "
        f"sha256={digest} overflow={UPLOAD_UNAVAILABLE_REASON}]"
    ).encode("utf-8")


def _join_prefix_and_marker(prefix: bytes, marker: bytes, max_bytes: int) -> str:
    if max_bytes <= len(marker):
        return marker[:max_bytes].decode("utf-8", errors="ignore")
    return prefix[: max_bytes - len(marker)].decode("utf-8", errors="ignore") + marker.decode(
        "utf-8"
    )


def bound_text(value: str, max_bytes: int = DEFAULT_MAX_CAPTURE_VALUE_BYTES) -> str:
    """Return UTF-8 text no larger than ``max_bytes`` with an explicit marker."""

    max_bytes = max(0, max_bytes)
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    digest = hashlib.sha256(encoded).hexdigest()
    return _join_prefix_and_marker(encoded, _truncation_marker(len(encoded), digest), max_bytes)


def bound_bytes(value: bytes, max_bytes: int = DEFAULT_MAX_CAPTURE_VALUE_BYTES) -> bytes:
    """Return bytes no larger than ``max_bytes`` with the same explicit marker."""

    max_bytes = max(0, max_bytes)
    if len(value) <= max_bytes:
        return value
    marker = _truncation_marker(len(value), hashlib.sha256(value).hexdigest())
    if max_bytes <= len(marker):
        return marker[:max_bytes]
    return value[: max_bytes - len(marker)] + marker


class BoundedTextAccumulator:
    """Hash a complete text stream while retaining only a bounded prefix."""

    def __init__(self, max_bytes: int = DEFAULT_MAX_CAPTURE_VALUE_BYTES) -> None:
        self._max_bytes = max_bytes
        self._prefix = bytearray()
        self._digest = hashlib.sha256()
        self.original_bytes = 0

    def append(self, value: str) -> None:
        encoded = value.encode("utf-8")
        self._digest.update(encoded)
        self.original_bytes += len(encoded)
        remaining = self._max_bytes - len(self._prefix)
        if remaining > 0:
            self._prefix.extend(encoded[:remaining])

    @property
    def truncated(self) -> bool:
        return self.original_bytes > self._max_bytes

    def value(self) -> str:
        if not self.truncated:
            return bytes(self._prefix).decode("utf-8", errors="ignore")
        return _join_prefix_and_marker(
            bytes(self._prefix),
            _truncation_marker(self.original_bytes, self._digest.hexdigest()),
            self._max_bytes,
        )

    def __bool__(self) -> bool:
        return self.original_bytes > 0


class _CaptureLimiter:
    def __init__(self, max_item_bytes: int, max_value_bytes: int) -> None:
        self.remaining = max_item_bytes
        self.max_value_bytes = max_value_bytes
        self.truncated = 0

    def _string(self, value: str) -> str:
        encoded_size = len(value.encode("utf-8"))
        limit = max(0, min(self.max_value_bytes, self.remaining))
        bounded = bound_text(value, limit)
        retained_size = len(bounded.encode("utf-8"))
        self.remaining = max(0, self.remaining - retained_size)
        if encoded_size > limit or CAPTURE_TRUNCATION_MARKER in value:
            self.truncated += 1
        return bounded

    def _bytes(self, value: bytes | bytearray) -> bytes:
        raw = bytes(value)
        limit = max(0, min(self.max_value_bytes, self.remaining))
        bounded = bound_bytes(raw, limit)
        self.remaining = max(0, self.remaining - len(bounded))
        if len(raw) > limit or CAPTURE_TRUNCATION_MARKER.encode() in raw:
            self.truncated += 1
        return bounded

    def _key(self, key: Any, used: set[str]) -> str | None:
        original = key if isinstance(key, str) else str(key)
        encoded = original.encode("utf-8")
        limit = max(0, min(self.max_value_bytes, self.remaining))
        if limit <= 0:
            self.truncated += 1
            return None
        if len(encoded) <= limit:
            candidate = original
        else:
            digest = hashlib.sha256(
                f"{type(key).__module__}.{type(key).__qualname__}:{original}".encode("utf-8")
            ).hexdigest()
            suffix = f"~{digest}"
            if limit <= len(suffix):
                candidate = suffix[:limit]
            else:
                candidate = encoded[: limit - len(suffix)].decode("utf-8", errors="ignore") + suffix
            self.truncated += 1
        size = len(candidate.encode("utf-8"))
        self.remaining = max(0, self.remaining - size)
        if candidate in used:
            # Never overwrite a previously retained entry. The item-level
            # truncation diagnostics account for the omitted collision.
            self.truncated += 1
            return None
        return candidate

    def mapping(self, value: Mapping[Any, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        used: set[str] = set()
        for key, item in value.items():
            bounded_key = self._key(key, used)
            if bounded_key is None:
                continue
            used.add(bounded_key)
            result[bounded_key] = self.value(item)
        return result

    def value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._string(value)
        if isinstance(value, (bytes, bytearray)):
            return self._bytes(value)
        if isinstance(value, Mapping):
            return self.mapping(value)
        if isinstance(value, Sequence):
            return tuple(self.value(item) for item in value)
        return value


def _reported_truncations(attributes: Mapping[str, Any]) -> int:
    value = attributes.get("neatlogs.capture.truncated_count", 0)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def limit_span_capture(
    span: ReadableSpan,
    *,
    max_item_bytes: int = DEFAULT_MAX_CAPTURE_ITEM_BYTES,
    max_value_bytes: int = DEFAULT_MAX_CAPTURE_VALUE_BYTES,
) -> tuple[ReadableSpan, int]:
    """Clone a span with bounded post-mask attributes and events."""

    limiter = _CaptureLimiter(max_item_bytes, max_value_bytes)
    previously_reported = _reported_truncations(span.attributes or {})
    name = limiter.value(span.name)
    resource_attributes = limiter.mapping(span.resource.attributes or {})
    status_description = limiter.value(span.status.description)
    attributes = limiter.mapping(span.attributes or {})
    events = [
        Event(
            limiter.value(event.name),
            attributes=limiter.mapping(event.attributes or {}),
            timestamp=event.timestamp,
        )
        for event in span.events
    ]
    links = [
        Link(
            link.context,
            attributes=limiter.mapping(link.attributes or {}),
        )
        for link in span.links
    ]
    truncations = limiter.truncated + previously_reported
    if not truncations:
        return span, 0

    attributes.update(
        {
            "neatlogs.capture.truncated": True,
            "neatlogs.capture.truncated_count": truncations,
            "neatlogs.capture.overflow.state": "disabled",
            "neatlogs.capture.overflow.reason": UPLOAD_UNAVAILABLE_REASON,
        }
    )
    return (
        ReadableSpan(
            name=name,
            context=span.context,
            parent=span.parent,
            resource=Resource(resource_attributes, schema_url=span.resource.schema_url),
            attributes=attributes,
            events=events,
            links=links,
            kind=span.kind,
            status=Status(span.status.status_code, status_description),
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=span.instrumentation_scope,
        ),
        truncations,
    )


def limit_log_capture(
    item: Any,
    *,
    max_item_bytes: int = DEFAULT_MAX_CAPTURE_ITEM_BYTES,
    max_value_bytes: int = DEFAULT_MAX_CAPTURE_VALUE_BYTES,
) -> tuple[Any, int]:
    """Clone a readable log record with bounded post-mask body and attributes."""

    limiter = _CaptureLimiter(max_item_bytes, max_value_bytes)
    clone = copy.copy(item)
    record = copy.copy(item.log_record)
    previously_reported = _reported_truncations(record.attributes or {})
    record.body = limiter.value(record.body)
    record.attributes = limiter.mapping(record.attributes or {})
    record.severity_text = limiter.value(record.severity_text)
    record.event_name = limiter.value(record.event_name)
    original_resource = getattr(item, "resource", None) or getattr(record, "resource", None)
    bounded_resource = None
    if original_resource is not None:
        bounded_resource = Resource(
            limiter.mapping(original_resource.attributes or {}),
            schema_url=original_resource.schema_url,
        )
    truncations = limiter.truncated + previously_reported
    if truncations:
        record.attributes.update(
            {
                "neatlogs.capture.truncated": True,
                "neatlogs.capture.truncated_count": truncations,
                "neatlogs.capture.overflow.state": "disabled",
                "neatlogs.capture.overflow.reason": UPLOAD_UNAVAILABLE_REASON,
            }
        )
    object.__setattr__(clone, "log_record", record)
    if bounded_resource is not None:
        if hasattr(clone, "resource"):
            object.__setattr__(clone, "resource", bounded_resource)
        elif hasattr(record, "resource"):
            object.__setattr__(record, "resource", bounded_resource)
    return clone, truncations
