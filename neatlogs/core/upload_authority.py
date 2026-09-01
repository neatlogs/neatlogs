"""Authenticated, bounded upload authority for typed media and OTLP overflow."""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Protocol
from urllib.parse import quote, urlparse
from uuid import UUID

import requests

from .capture import UPLOAD_UNAVAILABLE_REASON

logger = logging.getLogger(__name__)

UploadPurpose = Literal["typed_media", "otlp_overflow"]
ContentEncoding = Literal["identity", "gzip"]

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MIME_TYPE = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")
_REASON_CODE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_RETRYABLE_STATUS = {429, 502, 503, 504}
DEFAULT_UPLOAD_DEADLINE_SECONDS = 10.0
DEFAULT_MAX_MEDIA_UPLOAD_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_OVERFLOW_UPLOAD_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_UPLOAD_BYTES = DEFAULT_MAX_MEDIA_UPLOAD_BYTES
DEFAULT_MAX_UPLOAD_ATTEMPTS = 3


class UploadError(RuntimeError):
    """A safe upload failure containing no URL, credentials, or payload bytes."""

    def __init__(self, stage: str, reason_code: str, *, retryable: bool = False) -> None:
        super().__init__(f"{stage}:{reason_code}")
        self.stage = stage
        self.reason_code = reason_code
        self.retryable = retryable


@dataclass(frozen=True)
class OverflowPayload:
    """A complete, masked OTLP item offered to an upload implementation."""

    content: bytes
    sha256: str
    byte_length: int
    signal: str
    purpose: str = "otlp_overflow"
    mime_type: str = "application/x-protobuf"
    content_encoding: str = "identity"

    @property
    def encoding(self) -> str:
        return self.content_encoding


@dataclass(frozen=True)
class MediaPayload:
    """Canonical typed-media bytes retained by the bounded staging store."""

    content: bytes
    sha256: str
    byte_length: int
    mime_type: str
    media_purpose: str
    purpose: str = "typed_media"
    content_encoding: str = "identity"

    @property
    def encoding(self) -> str:
        return self.content_encoding


@dataclass(frozen=True)
class UploadReference:
    id: str
    purpose: str
    sha256: str
    byte_length: int
    mime_type: str
    content_encoding: str
    state: str


@dataclass(frozen=True)
class OverflowExportReceipt:
    """Proof that backend validation completed and accepted the OTLP item."""

    upload_id: str
    project_id: str = ""
    state: str = "ready"
    reference_exported: bool = True
    reference: UploadReference | None = None

    @property
    def complete(self) -> bool:
        return bool(self.upload_id and self.state == "ready" and self.reference_exported)


@dataclass(frozen=True)
class MediaExportReceipt:
    upload_id: str
    state: str
    reference: UploadReference

    @property
    def complete(self) -> bool:
        return bool(self.upload_id and self.state == "ready" and self.reference.state == "ready")


class UploadAuthority(Protocol):
    available: bool
    unavailable_reason: str

    def export_overflow(self, payload: OverflowPayload) -> OverflowExportReceipt | None: ...

    def export_media(self, payload: MediaPayload) -> MediaExportReceipt | None: ...


class DisabledUploadAuthority:
    """Default until the explicitly gated backend contract is enabled."""

    available = False
    unavailable_reason = UPLOAD_UNAVAILABLE_REASON

    def export_overflow(self, payload: OverflowPayload) -> None:
        del payload
        return None

    def export_media(self, payload: MediaPayload) -> None:
        del payload
        return None


def uploads_enabled(option: bool | None, environment: str | None) -> bool:
    """Resolve the typed option before the default-off environment gate."""

    if option is not None:
        if not isinstance(option, bool):
            raise ValueError("uploads_enabled must be a boolean or None")
        return option
    return str(environment or "").strip().lower() in {"1", "true", "yes"}


def _strict_string(value: Any, field: str, *, maximum: int | None = None) -> str:
    if not isinstance(value, str) or not value or (maximum is not None and len(value) > maximum):
        raise UploadError("response", f"invalid_{field}")
    return value


def _strict_uuid(value: Any, field: str) -> str:
    text = _strict_string(value, field)
    try:
        UUID(text)
    except (TypeError, ValueError, AttributeError) as exc:
        raise UploadError("response", f"invalid_{field}") from exc
    return text


def _strict_reference(value: Any, expected: Mapping[str, Any]) -> UploadReference:
    if not isinstance(value, Mapping):
        raise UploadError("response", "invalid_reference")
    reference = UploadReference(
        id=_strict_uuid(value.get("id"), "reference_id"),
        purpose=_strict_string(value.get("purpose"), "reference_purpose"),
        sha256=_strict_string(value.get("sha256"), "reference_sha256"),
        byte_length=value.get("byte_length"),
        mime_type=_strict_string(value.get("mime_type"), "reference_mime_type"),
        content_encoding=_strict_string(
            value.get("content_encoding"), "reference_content_encoding"
        ),
        state=_strict_string(value.get("state"), "reference_state"),
    )
    if (
        isinstance(reference.byte_length, bool)
        or not isinstance(reference.byte_length, int)
        or reference.byte_length <= 0
    ):
        raise UploadError("response", "invalid_reference_byte_length")
    if (
        reference.purpose != expected["purpose"]
        or reference.sha256 != expected["sha256"]
        or reference.byte_length != expected["byte_length"]
        or reference.mime_type != expected["mime_type"]
        or reference.content_encoding != expected["content_encoding"]
    ):
        raise UploadError("response", "reference_mismatch")
    return reference


def _strict_diagnostic(
    value: Any,
    *,
    stage: str,
    fallback_reason: str,
    fallback_retryable: bool,
) -> tuple[str, bool]:
    if value is None:
        return fallback_reason, fallback_retryable
    if not isinstance(value, Mapping):
        raise UploadError(stage, "invalid_diagnostic")
    diagnostic_stage = value.get("stage")
    reason = value.get("reason_code")
    retryable = value.get("retryable")
    if (
        not isinstance(diagnostic_stage, str)
        or _REASON_CODE.fullmatch(diagnostic_stage) is None
        or not isinstance(reason, str)
        or _REASON_CODE.fullmatch(reason) is None
        or not isinstance(retryable, bool)
    ):
        raise UploadError(stage, "invalid_diagnostic")
    return reason, retryable


class AuthenticatedUploadAuthority:
    """Perform prepare → object PUT → complete with existing API-key auth."""

    available = True
    unavailable_reason = ""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        session: requests.Session | None = None,
        deadline_seconds: float = DEFAULT_UPLOAD_DEADLINE_SECONDS,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
        max_overflow_bytes: int = DEFAULT_MAX_OVERFLOW_UPLOAD_BYTES,
        max_attempts: int = DEFAULT_MAX_UPLOAD_ATTEMPTS,
    ) -> None:
        parsed = urlparse(str(base_url).rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if not str(api_key).strip():
            raise ValueError("api_key is required for uploads")
        if (
            deadline_seconds <= 0
            or max_upload_bytes <= 0
            or max_overflow_bytes <= 0
            or max_attempts <= 0
        ):
            raise ValueError("upload bounds must be greater than zero")
        self._base_url = f"{parsed.scheme}://{parsed.netloc}"
        self._api_headers = {"x-api-key": str(api_key).strip()}
        self._session = session or requests.Session()
        self._owns_session = session is None
        self._deadline_seconds = float(deadline_seconds)
        self.max_upload_bytes = int(max_upload_bytes)
        self.max_overflow_bytes = int(max_overflow_bytes)
        self._max_attempts = int(max_attempts)
        self._cache: OrderedDict[str, MediaExportReceipt | OverflowExportReceipt] = OrderedDict()
        self._lock = threading.RLock()

    def export_overflow(self, payload: OverflowPayload) -> OverflowExportReceipt:
        self._validate_payload(payload)
        if payload.signal != "span":
            # The v1 backend contract intentionally defines only an OTLP trace
            # envelope. Never guess a log schema or submit it as trace bytes.
            raise UploadError("prepare", "overflow_signal_unsupported")
        receipt = self._upload(payload, payload_schema="otlp.traces.v1")
        return OverflowExportReceipt(
            upload_id=receipt.upload_id,
            state=receipt.state,
            reference_exported=True,
            reference=receipt.reference,
        )

    def export_media(self, payload: MediaPayload) -> MediaExportReceipt:
        self._validate_payload(payload)
        return self._upload(payload, payload_schema="neatlogs.media.v1")

    def _validate_payload(self, payload: Any) -> None:
        if payload.purpose not in {"typed_media", "otlp_overflow"}:
            raise UploadError("prepare", "invalid_purpose")
        if not isinstance(payload.content, bytes) or not payload.content:
            raise UploadError("prepare", "invalid_content")
        if (
            isinstance(payload.byte_length, bool)
            or not isinstance(payload.byte_length, int)
            or payload.byte_length <= 0
            or payload.byte_length != len(payload.content)
            or payload.byte_length
            > (
                self.max_overflow_bytes
                if payload.purpose == "otlp_overflow"
                else self.max_upload_bytes
            )
        ):
            raise UploadError("prepare", "invalid_byte_length")
        if not _DIGEST.fullmatch(payload.sha256) or (
            hashlib.sha256(payload.content).hexdigest() != payload.sha256
        ):
            raise UploadError("prepare", "invalid_sha256")
        if (
            not isinstance(payload.mime_type, str)
            or len(payload.mime_type) > 160
            or _MIME_TYPE.fullmatch(payload.mime_type) is None
        ):
            raise UploadError("prepare", "invalid_mime_type")
        if payload.content_encoding not in {"identity", "gzip"}:
            raise UploadError("prepare", "invalid_content_encoding")

    def _upload(self, payload: Any, *, payload_schema: str | None) -> MediaExportReceipt:
        material = ":".join(
            (
                "1",
                payload.purpose,
                payload.sha256,
                str(payload.byte_length),
                payload.mime_type,
                payload.content_encoding,
                payload_schema or "",
            )
        )
        idempotency_key = f"nl-py-v1:{hashlib.sha256(material.encode()).hexdigest()}"
        with self._lock:
            cached = self._cache.get(idempotency_key)
            if cached is not None:
                self._cache.move_to_end(idempotency_key)
                if isinstance(cached, MediaExportReceipt):
                    return cached
                if cached.reference is None:
                    raise UploadError("complete", "cached_reference_missing")
                return MediaExportReceipt(cached.upload_id, cached.state, cached.reference)

        deadline = time.monotonic() + self._deadline_seconds
        request_body: dict[str, Any] = {
            "version": 1,
            "purpose": payload.purpose,
            "sha256": payload.sha256,
            "byte_length": payload.byte_length,
            "mime_type": payload.mime_type,
            "content_encoding": payload.content_encoding,
            "idempotency_key": idempotency_key,
        }
        if payload_schema is not None:
            request_body["payload_schema"] = payload_schema

        prepare_status, prepared = self._json_request(
            "POST",
            f"{self._base_url}/v1/telemetry/uploads",
            deadline,
            stage="prepare",
            expected_status={200, 201, 202},
            return_status=True,
            headers=self._api_headers,
            json=request_body,
        )
        upload_id, upload, prepared_reference, replay_receipt = self._validate_prepare(
            prepare_status, prepared, request_body
        )
        if replay_receipt is not None:
            with self._lock:
                self._cache[idempotency_key] = replay_receipt
            return replay_receipt
        assert upload is not None
        self._put_object(upload, payload, deadline)
        completed = self._json_request(
            "POST",
            f"{self._base_url}/v1/telemetry/uploads/{quote(upload_id, safe='')}/complete",
            deadline,
            stage="complete",
            expected_status={200, 202},
            headers=self._api_headers,
            json={"sha256": payload.sha256, "byte_length": payload.byte_length},
        )
        receipt = self._validate_complete(completed, upload_id, request_body)
        if receipt.reference.id != prepared_reference.id:
            raise UploadError("complete", "reference_id_mismatch")
        with self._lock:
            self._cache[idempotency_key] = receipt
            self._cache.move_to_end(idempotency_key)
            while len(self._cache) > 256:
                self._cache.popitem(last=False)
        return receipt

    def _validate_prepare(self, status: int, value: Any, expected: Mapping[str, Any]) -> tuple[
        str,
        Mapping[str, Any] | None,
        UploadReference,
        MediaExportReceipt | None,
    ]:
        if not isinstance(value, Mapping):
            raise UploadError("prepare", "invalid_response")
        upload_id = _strict_uuid(value.get("upload_id"), "upload_id")
        state = value.get("state")
        if status == 200 and state == "ready":
            _strict_diagnostic(
                value.get("diagnostic"),
                stage="prepare",
                fallback_reason="upload_ready",
                fallback_retryable=False,
            )
            reference = _strict_reference(value.get("reference"), expected)
            if reference.id != upload_id or reference.state != "ready":
                raise UploadError("prepare", "reference_mismatch")
            receipt = MediaExportReceipt(upload_id, "ready", reference)
            return upload_id, None, reference, receipt
        if status != 201 or state != "prepared":
            if status not in {200, 202} or state not in {"uploaded", "validating", "rejected"}:
                raise UploadError("prepare", "invalid_state")
            reference = _strict_reference(value.get("reference"), expected)
            if reference.id != upload_id or reference.state != state:
                raise UploadError("prepare", "reference_mismatch")
            reason, retryable = _strict_diagnostic(
                value.get("diagnostic"),
                stage="prepare",
                fallback_reason=f"upload_{state or 'not_prepared'}",
                fallback_retryable=state in {"uploaded", "validating"},
            )
            raise UploadError("prepare", reason, retryable=retryable)
        expires_at = _strict_string(value.get("expires_at"), "expires_at")
        try:
            parsed_expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if parsed_expiry.tzinfo is None or parsed_expiry <= datetime.now(timezone.utc):
                raise ValueError
        except ValueError as exc:
            raise UploadError("prepare", "invalid_expiry") from exc
        upload = value.get("upload")
        if not isinstance(upload, Mapping) or upload.get("method") != "PUT":
            raise UploadError("prepare", "invalid_upload_authority")
        url = _strict_string(upload.get("url"), "upload_url")
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise UploadError("prepare", "invalid_upload_url")
        headers = upload.get("headers")
        if not isinstance(headers, Mapping) or not all(
            isinstance(key, str)
            and isinstance(item, str)
            and "\r" not in key
            and "\n" not in key
            and "\r" not in item
            and "\n" not in item
            and key.lower() not in {"authorization", "cookie", "proxy-authorization", "x-api-key"}
            for key, item in headers.items()
        ):
            raise UploadError("prepare", "invalid_upload_headers")
        reference = _strict_reference(value.get("reference"), expected)
        if reference.state != "prepared" or reference.id != upload_id:
            raise UploadError("prepare", "invalid_reference_state")
        return upload_id, upload, reference, None

    def _validate_complete(
        self, value: Any, upload_id: str, expected: Mapping[str, Any]
    ) -> MediaExportReceipt:
        if not isinstance(value, Mapping) or value.get("upload_id") != upload_id:
            raise UploadError("complete", "invalid_response")
        state = value.get("state")
        if state not in {"ready", "validating", "rejected"}:
            raise UploadError("complete", "invalid_state")
        reference = _strict_reference(value.get("reference"), expected)
        if reference.id != upload_id or reference.state != state:
            raise UploadError("complete", "reference_mismatch")
        if state != "ready":
            reason, retryable = _strict_diagnostic(
                value.get("diagnostic"),
                stage="complete",
                fallback_reason=f"upload_{state}",
                fallback_retryable=state == "validating",
            )
            raise UploadError("complete", reason, retryable=retryable)
        _strict_diagnostic(
            value.get("diagnostic"),
            stage="complete",
            fallback_reason="upload_ready",
            fallback_retryable=False,
        )
        return MediaExportReceipt(upload_id=upload_id, state=state, reference=reference)

    def _put_object(self, upload: Mapping[str, Any], payload: Any, deadline: float) -> None:
        response = self._request(
            "PUT",
            upload["url"],
            deadline,
            stage="object_put",
            headers=dict(upload["headers"]),
            data=payload.content,
        )
        try:
            if response.status_code < 200 or response.status_code >= 300:
                raise UploadError(
                    "object_put",
                    f"http_{response.status_code}",
                    retryable=response.status_code in _RETRYABLE_STATUS,
                )
        finally:
            response.close()

    def _json_request(
        self,
        method: str,
        url: str,
        deadline: float,
        *,
        stage: str,
        expected_status: set[int],
        return_status: bool = False,
        **kwargs: Any,
    ) -> Any:
        response = self._request(method, url, deadline, stage=stage, **kwargs)
        try:
            if response.status_code not in expected_status:
                reason_code = f"http_{response.status_code}"
                retryable = response.status_code in _RETRYABLE_STATUS
                try:
                    error = response.json()
                    if isinstance(error, Mapping):
                        candidate = error.get("reason_code")
                        if isinstance(candidate, str) and _REASON_CODE.fullmatch(candidate):
                            reason_code = candidate
                        if isinstance(error.get("retryable"), bool):
                            retryable = error["retryable"]
                except (TypeError, ValueError):
                    pass
                raise UploadError(
                    stage,
                    reason_code,
                    retryable=retryable,
                )
            try:
                value = response.json()
                return (response.status_code, value) if return_status else value
            except (TypeError, ValueError) as exc:
                raise UploadError(stage, "invalid_json") from exc
        finally:
            response.close()

    def _request(
        self, method: str, url: str, deadline: float, *, stage: str, **kwargs: Any
    ) -> requests.Response:
        for attempt in range(self._max_attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise UploadError(stage, "deadline_exceeded", retryable=True)
            try:
                response = self._session.request(method, url, timeout=remaining, **kwargs)
            except requests.RequestException as exc:
                if attempt + 1 == self._max_attempts:
                    raise UploadError(stage, "transport_error", retryable=True) from exc
                continue
            if response.status_code not in _RETRYABLE_STATUS or attempt + 1 == self._max_attempts:
                return response
            response.close()
        raise UploadError(stage, "retry_exhausted", retryable=True)

    def close(self) -> None:
        if self._owns_session:
            self._session.close()
