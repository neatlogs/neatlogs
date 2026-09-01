"""Post-mask typed-media upload and canonical reference replacement."""

from __future__ import annotations

import copy
import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

try:
    from opentelemetry.sdk._logs import ReadableLogRecord
    from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult
except ImportError:  # OpenTelemetry 1.35-1.38 compatibility
    from opentelemetry.sdk._logs import LogData as ReadableLogRecord
    from opentelemetry.sdk._logs.export import (
        LogExporter as LogRecordExporter,
        LogExportResult as LogRecordExportResult,
    )

from .delivery import DeliveryDiagnostics
from .media import PendingMediaStore, sanitize_media_attributes, sanitize_media_payload
from .upload_authority import MediaExportReceipt, UploadAuthority, UploadError

logger = logging.getLogger(__name__)


def _json_container(value: str) -> Any | None:
    stripped = value.lstrip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, (dict, list)) else None


def _is_pending_media_record(value: Mapping[str, Any]) -> bool:
    identifier = value.get("id")
    digest = value.get("sha256")
    return (
        value.get("state") == "pending-upload"
        and isinstance(identifier, str)
        and identifier.startswith("nl_media_")
        and isinstance(digest, str)
        and len(digest) == 64
    )


def _token_counts(value: Any, *, pending_only: bool) -> Counter[str]:
    found: Counter[str] = Counter()

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            token = node.get("upload_token")
            if (
                isinstance(token, str)
                and token.startswith("nl_pending_media_")
                and (not pending_only or _is_pending_media_record(node))
            ):
                found[token] += 1
            for key, item in node.items():
                prefix = key[: -len(".upload_token")] if isinstance(key, str) else ""
                if (
                    isinstance(key, str)
                    and key.endswith(".upload_token")
                    and isinstance(item, str)
                    and item.startswith("nl_pending_media_")
                    and (not pending_only or node.get(f"{prefix}.state") == "pending-upload")
                    and isinstance(node.get(f"{prefix}.id"), str)
                    and node[f"{prefix}.id"].startswith("nl_media_")
                ):
                    found[item] += 1
                visit(item)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for item in node:
                visit(item)
        elif isinstance(node, str):
            parsed = _json_container(node)
            if parsed is not None:
                visit(parsed)

    visit(value)
    return found


def release_removed_media(store: PendingMediaStore | None, before: Any, after: Any | None) -> None:
    """Release staged bytes whose internal token was removed by masking."""

    if store is None:
        return
    before_counts = _token_counts(before, pending_only=False)
    after_counts = _token_counts(after, pending_only=False) if after is not None else Counter()
    for token, count in (before_counts - after_counts).items():
        store.release(token, count)


def _has_unresolved_pending(value: Any, known_tokens: set[str]) -> bool:
    if isinstance(value, Mapping):
        token = value.get("upload_token")
        if (
            _is_pending_media_record(value)
            and "upload_token" in value
            and token not in known_tokens
        ):
            return True
        if _is_pending_media_record(value) and token not in known_tokens:
            return True
        for key, item in value.items():
            if isinstance(key, str) and key.endswith(".upload_token"):
                if item not in known_tokens:
                    return True
            if isinstance(key, str) and key.endswith(".state") and item == "pending-upload":
                prefix = key[: -len(".state")]
                identifier = value.get(f"{prefix}.id")
                if (
                    isinstance(identifier, str)
                    and identifier.startswith("nl_media_")
                    and value.get(f"{prefix}.upload_token") not in known_tokens
                ):
                    return True
            if _has_unresolved_pending(item, known_tokens):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_has_unresolved_pending(item, known_tokens) for item in value)
    if isinstance(value, str):
        parsed = _json_container(value)
        return parsed is not None and _has_unresolved_pending(parsed, known_tokens)
    return False


def _reference_record(receipt: MediaExportReceipt) -> dict[str, Any]:
    reference = receipt.reference
    return {
        "id": reference.id,
        "source": "uploaded",
        "mime_type": reference.mime_type,
        "byte_length": reference.byte_length,
        "sha256": reference.sha256,
        "state": "available",
        "safe_preview": "authenticated upload ready",
    }


def _failed_record(original: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        key: value
        for key, value in original.items()
        if key != "upload_token" and key != "safe_preview"
    } | {"state": "failed", "safe_preview": f"upload failed: {reason}"}


def _replace(
    value: Any,
    replacements: Mapping[str, dict[str, Any]],
    failures: Mapping[str, str],
) -> Any:
    if isinstance(value, Mapping):
        token = value.get("upload_token")
        if isinstance(token, str) and token.startswith("nl_pending_media_"):
            if token in replacements:
                replacement = dict(value)
                replacement.pop("upload_token", None)
                replacement.pop("safe_preview", None)
                replacement.update(replacements[token])
                return replacement
            if token in failures:
                return _failed_record(value, failures[token])
        if _is_pending_media_record(value) or (
            isinstance(token, str) and token.startswith("nl_pending_media_")
        ):
            return _failed_record(value, "upload_token_missing")
        return {key: _replace(item, replacements, failures) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_replace(item, replacements, failures) for item in value]
    if isinstance(value, str):
        if (
            "nl_pending_media_" not in value
            and "upload_token" not in value
            and "pending-upload" not in value
        ):
            return value
        parsed = _json_container(value)
        if parsed is None:
            return value
        transformed = _replace(parsed, replacements, failures)
        return json.dumps(transformed, ensure_ascii=False, separators=(",", ":"))
    return value


def _replace_attributes(
    attributes: Mapping[str, Any],
    replacements: Mapping[str, dict[str, Any]],
    failures: Mapping[str, str],
) -> dict[str, Any]:
    result = {
        str(key): _replace(value, replacements, failures) for key, value in attributes.items()
    }
    for key, token in list(result.items()):
        if (
            not key.endswith(".upload_token")
            or not isinstance(token, str)
            or not token.startswith("nl_pending_media_")
        ):
            continue
        prefix = key[: -len(".upload_token")]
        identifier = result.get(f"{prefix}.id")
        if not isinstance(identifier, str) or not identifier.startswith("nl_media_"):
            continue
        result.pop(key, None)
        if result.get(f"{prefix}.state") != "pending-upload":
            result[f"{prefix}.state"] = "failed"
            result[f"{prefix}.safe_preview"] = "upload failed: masked_pending_state"
        elif token in replacements:
            result.pop(f"{prefix}.safe_preview", None)
            for field, item in replacements[token].items():
                result[f"{prefix}.{field}"] = item
        elif token in failures:
            result[f"{prefix}.state"] = "failed"
            result[f"{prefix}.safe_preview"] = f"upload failed: {failures[token]}"
        else:
            result[f"{prefix}.state"] = "failed"
            result[f"{prefix}.safe_preview"] = "upload failed: upload_token_missing"
    for key, state in list(result.items()):
        if not key.endswith(".state") or state != "pending-upload":
            continue
        prefix = key[: -len(".state")]
        identifier = result.get(f"{prefix}.id")
        if (
            isinstance(identifier, str)
            and identifier.startswith("nl_media_")
            and f"{prefix}.upload_token" not in result
        ):
            result[key] = "failed"
            result[f"{prefix}.safe_preview"] = "upload failed: upload_token_missing"
    return result


class _MediaResolver:
    def __init__(
        self,
        authority: UploadAuthority,
        store: PendingMediaStore,
        diagnostics: DeliveryDiagnostics | None,
        signal: str,
    ) -> None:
        self._authority = authority
        self._store = store
        self._diagnostics = diagnostics
        self._signal = signal

    def resolve(self, value: Any) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        replacements: dict[str, dict[str, Any]] = {}
        failures: dict[str, str] = {}
        all_tokens = _token_counts(value, pending_only=False)
        pending_tokens = set(_token_counts(value, pending_only=True))
        for token in sorted(all_tokens):
            if token not in pending_tokens:
                failures[token] = "masked_pending_state"
                self._store.release(token, all_tokens[token])
                continue
            payload = self._store.get(token)
            if payload is None:
                failures[token] = "staged_payload_missing"
            else:
                try:
                    receipt = self._authority.export_media(payload)
                    if not isinstance(receipt, MediaExportReceipt) or not receipt.complete:
                        failures[token] = "incomplete_receipt"
                    else:
                        replacements[token] = _reference_record(receipt)
                except UploadError as exc:
                    failures[token] = f"{exc.stage}:{exc.reason_code}"
                except Exception as exc:
                    logger.error("[neatlogs] typed media upload failed (%s)", type(exc).__name__)
                    failures[token] = "unexpected_error"
                finally:
                    self._store.release(token, all_tokens[token])
            if self._diagnostics is not None:
                self._diagnostics.record_media_upload(
                    self._signal,
                    succeeded=token in replacements,
                    reason=failures.get(token, ""),
                )
        if _has_unresolved_pending(value, set(all_tokens)):
            failures["__unresolved_pending__"] = "upload_token_missing"
            if self._diagnostics is not None:
                self._diagnostics.record_media_upload(
                    self._signal,
                    succeeded=False,
                    reason="upload_token_missing",
                )
        return replacements, failures


class TypedMediaSpanExporter(SpanExporter):
    """Sanitize and resolve media only after the outer mask has run."""

    def __init__(
        self,
        inner: SpanExporter,
        authority: UploadAuthority | None,
        store: PendingMediaStore | None,
        diagnostics: DeliveryDiagnostics | None = None,
    ) -> None:
        self._inner = inner
        self._store = store
        self._resolver = (
            _MediaResolver(authority, store, diagnostics, "span")
            if authority is not None and authority.available and store is not None
            else None
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        sanitized: list[tuple[ReadableSpan, dict[str, Any], list[dict[str, Any]]]] = []
        for span in spans:
            attributes = sanitize_media_attributes(span.attributes or {}, store=self._store)
            events = [
                sanitize_media_attributes(event.attributes or {}, store=self._store)
                for event in span.events
            ]
            sanitized.append((span, attributes, events))
        candidates = [
            {"attributes": attributes, "events": events} for _, attributes, events in sanitized
        ]
        replacements, failures = (
            self._resolver.resolve(candidates) if self._resolver is not None else ({}, {})
        )
        prepared: list[ReadableSpan] = []
        for span, attributes, events in sanitized:
            prepared.append(
                ReadableSpan(
                    name=span.name,
                    context=span.context,
                    parent=span.parent,
                    resource=span.resource,
                    attributes=_replace_attributes(attributes, replacements, failures),
                    events=[
                        Event(
                            event.name,
                            attributes=_replace_attributes(
                                event_attributes, replacements, failures
                            ),
                            timestamp=event.timestamp,
                        )
                        for event, event_attributes in zip(span.events, events)
                    ],
                    links=span.links,
                    kind=span.kind,
                    status=span.status,
                    start_time=span.start_time,
                    end_time=span.end_time,
                    instrumentation_scope=span.instrumentation_scope,
                )
            )
        result = self._inner.export(prepared)
        return SpanExportResult.FAILURE if failures else result

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        result = self._inner.force_flush(timeout_millis)
        return True if result is None else bool(result)

    def shutdown(self) -> None:
        self._inner.shutdown()


class TypedMediaLogExporter(LogRecordExporter):
    def __init__(
        self,
        inner: LogRecordExporter,
        authority: UploadAuthority | None,
        store: PendingMediaStore | None,
        diagnostics: DeliveryDiagnostics | None = None,
    ) -> None:
        self._inner = inner
        self._resolver = (
            _MediaResolver(authority, store, diagnostics, "log")
            if authority is not None and authority.available and store is not None
            else None
        )
        self._store = store

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        sanitized: list[tuple[ReadableLogRecord, Any, dict[str, Any]]] = []
        for item in batch:
            body = item.log_record.body
            if isinstance(body, str):
                parsed = _json_container(body)
                if parsed is not None:
                    body = json.dumps(
                        sanitize_media_payload(parsed, store=self._store),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                else:
                    body = sanitize_media_payload(body, store=self._store)
            else:
                body = sanitize_media_payload(body, store=self._store)
            attributes = sanitize_media_attributes(
                item.log_record.attributes or {}, store=self._store
            )
            sanitized.append((item, body, attributes))
        candidates = [{"body": body, "attributes": attributes} for _, body, attributes in sanitized]
        replacements, failures = (
            self._resolver.resolve(candidates) if self._resolver is not None else ({}, {})
        )
        prepared = []
        for item, body, attributes in sanitized:
            record = item.log_record
            clone = copy.copy(item)
            clone_record = copy.copy(record)
            clone_record.body = _replace(body, replacements, failures)
            clone_record.attributes = _replace_attributes(attributes, replacements, failures)
            object.__setattr__(clone, "log_record", clone_record)
            prepared.append(clone)
        result = self._inner.export(prepared)
        return LogRecordExportResult.FAILURE if failures else result

    def force_flush(self, timeout_millis: int = 10000) -> bool:
        force_flush = getattr(self._inner, "force_flush", None)
        if force_flush is None:
            return True
        result = force_flush(timeout_millis)
        return True if result is None else bool(result)

    def shutdown(self) -> None:
        self._inner.shutdown()
