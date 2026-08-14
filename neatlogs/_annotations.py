"""
Post-hoc trace annotations.

Public API:
    import neatlogs

    applied = neatlogs.annotate(needs_review=True, severity="high", ticket="JIRA-1234")
    applied = neatlogs.add_event("user_feedback", rating=1, comment="wrong answer")

Both functions silently no-op (return False) when:
- neatlogs.init() was not called
- there is no active neatlogs span
- the active span is a foreign (co-tenant) span — prevents annotation leak
- the active span has ended or is sampled out
- the key/name is invalid (dotted, starts with "neatlogs.", empty, non-string)

Both functions return True if at least one attribute/event was applied.

All attributes and event names are namespaced under "neatlogs.annotation." so
they can be filtered in the backend without colliding with instrumentation
attributes emitted by the provider wrappers.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional

from opentelemetry import trace as otel_trace
from opentelemetry.trace import Span

from ._wrap_utils import (
    _current_neatlogs_parent,
    _is_neatlogs_span,
    _isolation_active,
)

ANNOTATION_PREFIX = "neatlogs.annotation."


# Valid key characters: ASCII letters, digits, underscore, hyphen. Mirrors the
# loosest convention OpenTelemetry itself allows for attribute keys; explicitly
# excludes dots (which the backend's ClickHouse mapping would split into nested
# structures) and the "neatlogs." prefix (which would cause double-namespacing).
_VALID_KEY_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def _valid_key(key: Any) -> bool:
    if not isinstance(key, str) or not key:
        return False
    if "." in key:
        return False
    if key.startswith("neatlogs."):
        return False
    return all(c in _VALID_KEY_CHARS for c in key)


def _coerce(value: Any) -> Any:
    """Pass OTel-native primitives through; JSON-serialize complex objects.

    bool is checked before int because bool is a subclass of int in Python.
    NaN/Inf are converted to their string form because json.dumps emits them
    by default and strict backend parsers (e.g. ClickHouse JSON) reject the
    unquoted forms. Pydantic v2 models are dumped via .model_dump() first.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        return value
    if isinstance(value, bytes):
        # OTel supports bytes-as-string; pass through and let the SDK decide
        # whether to base64-encode. utf-8 decode with errors='replace' as a
        # safety net; fall back to repr for non-decodable bytes.
        try:
            return value.decode("utf-8")
        except (UnicodeDecodeError, AttributeError):
            return repr(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=_pydantic_default)
    # datetime, Pydantic v2 models, or anything else: try .model_dump() first.
    if hasattr(value, "model_dump"):
        try:
            return json.dumps(value.model_dump(), default=str)
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _pydantic_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    return str(obj)


def _resolve_neatlogs_span() -> Optional[Span]:
    """Isolation-aware active-span resolver.

    Returns the active neatlogs scope span in any mode, or None if:
    - neatlogs.init() was not called (no provider, no tracer)
    - there is no active span
    - the active span is foreign (co-tenant) — we must not annotate it
    - the active span is sampling/ended

    In isolated mode, the active neatlogs span lives on a private context key
    set by attach_as_current. In default mode, the active neatlogs span IS
    the OTel current span (because all neatlogs spans use start_as_current_span).
    """
    try:
        if _isolation_active():
            parent = _current_neatlogs_parent()
            if parent is None or not parent.is_recording():
                return None
            return parent
        current = otel_trace.get_current_span()
        if not current or not current.is_recording():
            return None
        if not _is_neatlogs_span(current):
            # Foreign span (e.g. openlit/langfuse co-tenant) — refuse to
            # annotate so the annotation doesn't leak into a co-tenant pipeline.
            return None
        return current
    except Exception:
        # The fail-open contract: never raise from an annotation call.
        return None


def annotate(**attrs: Any) -> bool:
    """Add filterable attributes to the currently active neatlogs span.

    Each kwarg is stored as a span attribute under the
    ``neatlogs.annotation.`` namespace. Returns True if at least one attribute
    was applied; False if there is no active neatlogs span, or all supplied
    keys/values were invalid, or no kwargs were supplied.

    The function is fail-open by design: it never raises, even if the
    underlying tracer or span machinery is in an unexpected state. To detect
    no-op, callers can branch on the bool return.

    Examples:

        >>> neatlogs.annotate(needs_review=True, ticket="JIRA-1234")
        True
        >>> neatlogs.annotate()  # no kwargs
        False
    """
    span = _resolve_neatlogs_span()
    if span is None:
        return False
    if not attrs:
        return False
    applied = False
    for key, value in attrs.items():
        if value is None:
            continue
        if not _valid_key(key):
            continue
        try:
            coerced = _coerce(value)
        except Exception:
            # Coercion failure on this value (e.g. circular ref, hostile
            # __str__): skip just this value, continue with the rest.
            # The fail-open contract: never raise from an annotation call.
            continue
        try:
            span.set_attribute(f"{ANNOTATION_PREFIX}{key}", coerced)
        except Exception:
            # SDK-level failure on set_attribute: skip just this value,
            # continue with the rest. Same fail-open contract.
            continue
        applied = True
    return applied


def add_event(name: str, **attrs: Any) -> bool:
    """Add a time-stamped event marker to the currently active neatlogs span.

    The event name is prefixed with ``neatlogs.annotation.`` so events from
    annotations are filterable in the backend alongside attributes. Returns
    True if the event was added; False if there is no active neatlogs span,
    the name is invalid, or the active span has ended.

    The function is fail-open by design: it never raises.

    Examples:

        >>> neatlogs.add_event("user_feedback", rating=1)
        True
        >>> neatlogs.add_event("reviewed_by", reviewer="alice")
        True
    """
    if not isinstance(name, str) or not name:
        return False
    if "." in name or name.startswith("neatlogs."):
        return False
    span = _resolve_neatlogs_span()
    if span is None:
        return False
    coerced: Dict[str, Any] = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if not _valid_key(key):
            continue
        try:
            coerced[key] = _coerce(value)
        except Exception:
            # Coercion failure on this attr: skip just this attr, continue.
            # The fail-open contract: never raise from an annotation call.
            continue
    event_name = f"{ANNOTATION_PREFIX}{name}"
    try:
        span.add_event(event_name, attributes=coerced)
    except Exception:
        # The fail-open contract: never raise from add_event.
        return False
    return True
