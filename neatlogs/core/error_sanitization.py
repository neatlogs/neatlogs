"""Credential removal from diagnostic error text."""

import re
from collections.abc import Mapping
from typing import Any

_SEPARATOR = r"(?:[_\-\s]|%5f|%2d|%20)?"
_KEY = rf"(?:access{_SEPARATOR}token|refresh{_SEPARATOR}token|api{_SEPARATOR}key|client{_SEPARATOR}secret|authorization|password|passwd)"
_PREFIX = rf"(?:\b|(?<=%22)|(?<=%27)){_KEY}\b(?:\\*[\"']|%22|%27)?(?:\s|%20)*(?:provided(?:\s|%20)*)?(?:(?P<encoded>%3[ad])|[:=])(?:\s|%20)*"
_QUOTED = re.compile(
    rf"(?P<prefix>{_PREFIX})(?P<value>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|"
    r"(?P<escaped>\\+[\"']).*?(?P=escaped)|%22.*?%22|%27.*?%27)",
    re.I | re.S,
)
_UNQUOTED = re.compile(
    rf"(?P<prefix>{_PREFIX})(?:(?P<scheme>(?:Bearer|Basic)(?:\s|%20)+)|(?!(?:Bearer|Basic)(?:\s|%20)+))"
    r"(?!\[REDACTED\]|%5bREDACTED%5d)"
    r"(?(encoded)(?:(?!%26|%22|%27|%2c|%7d|%5d)[^\s&\"'\\,;}\])<>])+|[^\s&\"'\\,;}\])<>]+)",
    re.I,
)
_ERROR_FIELDS = frozenset(
    {
        "exception.message",
        "exception.stacktrace",
        "error.message",
        "error.stack",
        "error.stacktrace",
    }
)


def _is_error_field(key: Any) -> bool:
    return key in _ERROR_FIELDS or any(str(key).endswith("." + field) for field in _ERROR_FIELDS)


def is_error_log(severity: Any, attributes: Any) -> bool:
    return str(severity or "").upper() in {"ERROR", "CRITICAL", "FATAL"} or (
        isinstance(attributes, Mapping) and any(_is_error_field(key) for key in attributes)
    )


def sanitize_error_text(value: str) -> str:
    """Redact key-labelled credentials, including escaped JSON and encoded queries."""

    def replacement(match):
        value = match.groupdict().get("value") or ""
        quote = match.groupdict().get("escaped") or (
            value[:3] if value.lower().startswith(("%22", "%27")) else value[:1]
        )
        marker = "%5BREDACTED%5D" if match["encoded"] or quote.startswith("%") else "[REDACTED]"
        return match["prefix"] + (match.groupdict().get("scheme") or "") + quote + marker + quote

    return _UNQUOTED.sub(replacement, _QUOTED.sub(replacement, value))


def sanitize_error_snapshot(snapshot: dict[str, Any]) -> bool:
    """Sanitize diagnostic fields in an export snapshot, leaving captured I/O alone."""
    changed = False

    def clean(mapping, key):
        nonlocal changed
        if not isinstance(mapping, dict):
            return
        value = mapping.get(key)
        if isinstance(value, str):
            sanitized = sanitize_error_text(value)
            if sanitized != value:
                mapping[key] = sanitized
                changed = True

    def clean_attributes(attributes):
        if not isinstance(attributes, dict):
            return
        for key in list(attributes):
            if _is_error_field(key):
                clean(attributes, key)

    clean(snapshot.get("status") or {}, "description")
    attributes = snapshot.get("attributes") or {}
    clean_attributes(attributes)
    for event in snapshot.get("events") or ():
        if isinstance(event, dict):
            clean_attributes(event.get("attributes") or {})
    if is_error_log(snapshot.get("severity_text"), attributes):
        clean(snapshot, "body")
    return changed
