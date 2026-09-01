import hashlib
from datetime import datetime, timedelta, timezone

import pytest
import requests

from neatlogs.core.upload_authority import (
    AuthenticatedUploadAuthority,
    MediaPayload,
    OverflowPayload,
    UploadError,
    uploads_enabled,
)

UPLOAD_ID = "123e4567-e89b-12d3-a456-426614174000"


class Response:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = body
        self.closed = False

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body

    def close(self):
        self.closed = True


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.closed = False

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.closed = True


def _reference(payload, state):
    return {
        "id": UPLOAD_ID,
        "purpose": payload.purpose,
        "sha256": payload.sha256,
        "byte_length": payload.byte_length,
        "mime_type": payload.mime_type,
        "content_encoding": payload.content_encoding,
        "state": state,
    }


def _prepared(payload, **extra):
    return {
        "upload_id": UPLOAD_ID,
        "state": "prepared",
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        "upload": {
            "method": "PUT",
            "url": "https://object.example/signed?secret=never-serialize",
            "headers": {"x-object-token": "signed-secret", "content-type": payload.mime_type},
        },
        "reference": _reference(payload, "prepared"),
        **extra,
    }


def _completed(payload, state="ready", **extra):
    return {
        "upload_id": UPLOAD_ID,
        "state": state,
        "reference": _reference(payload, state),
        **extra,
    }


def _media(content=b"private-image"):
    return MediaPayload(
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        byte_length=len(content),
        mime_type="image/png",
        media_purpose="input",
    )


def _authority(session, **kwargs):
    return AuthenticatedUploadAuthority(
        base_url="https://ingest.example/v1/traces",
        api_key="project-secret",
        session=session,
        deadline_seconds=1,
        **kwargs,
    )


def test_upload_activation_is_typed_and_default_off():
    assert uploads_enabled(None, None) is False
    assert uploads_enabled(None, "true") is True
    assert uploads_enabled(False, "true") is False
    assert uploads_enabled(True, None) is True
    with pytest.raises(ValueError):
        uploads_enabled("true", None)


def test_media_prepare_put_complete_uses_exact_contract_and_scoped_headers():
    payload = _media()
    responses = [Response(201, _prepared(payload, ignored="forward-compatible")), Response(204)]
    responses.append(Response(200, _completed(payload, ignored="forward-compatible")))
    session = Session(responses)
    authority = _authority(session)

    receipt = authority.export_media(payload)

    assert receipt.complete
    assert receipt.reference.id == UPLOAD_ID
    assert len(session.calls) == 3
    prepare_method, prepare_url, prepare_call = session.calls[0]
    assert (prepare_method, prepare_url) == (
        "POST",
        "https://ingest.example/v1/telemetry/uploads",
    )
    assert prepare_call["headers"] == {"x-api-key": "project-secret"}
    assert prepare_call["json"] == {
        "version": 1,
        "purpose": "typed_media",
        "sha256": payload.sha256,
        "byte_length": payload.byte_length,
        "mime_type": "image/png",
        "content_encoding": "identity",
        "idempotency_key": prepare_call["json"]["idempotency_key"],
        "payload_schema": "neatlogs.media.v1",
    }
    assert len(prepare_call["json"]["idempotency_key"]) <= 128
    assert session.calls[1][0] == "PUT"
    assert session.calls[1][2]["headers"] == {
        "x-object-token": "signed-secret",
        "content-type": "image/png",
    }
    assert "x-api-key" not in session.calls[1][2]["headers"]
    assert session.calls[1][2]["data"] == payload.content
    assert session.calls[2][2]["json"] == {
        "sha256": payload.sha256,
        "byte_length": payload.byte_length,
    }
    assert "secret" not in repr(receipt)


def test_ready_prepare_replay_short_circuits_put_and_complete_and_is_cached():
    payload = _media()
    session = Session([Response(200, _completed(payload))])
    authority = _authority(session)

    first = authority.export_media(payload)
    second = authority.export_media(payload)

    assert first == second
    assert first.complete
    assert len(session.calls) == 1


@pytest.mark.parametrize("status,state", [(200, "validating"), (202, "validating")])
def test_prepare_in_progress_is_retryable_but_never_success(status, state):
    payload = _media()
    body = _completed(
        payload,
        state,
        diagnostic={"stage": "validation", "reason_code": "still_validating", "retryable": True},
    )
    authority = _authority(Session([Response(status, body)]))

    with pytest.raises(UploadError) as caught:
        authority.export_media(payload)

    assert caught.value.stage == "prepare"
    assert caught.value.reason_code == "still_validating"
    assert caught.value.retryable is True


def test_prepare_in_progress_still_requires_a_matching_reference():
    payload = _media()
    body = _completed(
        payload,
        "uploaded",
        diagnostic={"stage": "validation", "reason_code": "retry", "retryable": True},
    )
    body["reference"]["sha256"] = "0" * 64
    authority = _authority(Session([Response(200, body)]))

    with pytest.raises(UploadError) as caught:
        authority.export_media(payload)

    assert caught.value.reason_code == "reference_mismatch"


def test_rejected_completion_surfaces_backend_diagnostic():
    payload = _media()
    rejected = _completed(
        payload,
        "rejected",
        diagnostic={"stage": "scan", "reason_code": "unsupported_scanner", "retryable": False},
    )
    authority = _authority(
        Session([Response(201, _prepared(payload)), Response(200), Response(200, rejected)])
    )

    with pytest.raises(UploadError) as caught:
        authority.export_media(payload)

    assert (caught.value.stage, caught.value.reason_code, caught.value.retryable) == (
        "complete",
        "unsupported_scanner",
        False,
    )


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda body: body.update(upload_id="not-a-uuid"), "invalid_upload_id"),
        (lambda body: body.update(expires_at="yesterday"), "invalid_expiry"),
        (lambda body: body["upload"].update(method="POST"), "invalid_upload_authority"),
        (
            lambda body: body["upload"].update(url="http://object.example/unsafe"),
            "invalid_upload_url",
        ),
        (
            lambda body: body["upload"]["headers"].update(authorization="secret"),
            "invalid_upload_headers",
        ),
        (lambda body: body["reference"].update(sha256="0" * 64), "reference_mismatch"),
    ],
)
def test_prepare_response_is_strict(mutate, reason):
    payload = _media()
    body = _prepared(payload)
    mutate(body)
    authority = _authority(Session([Response(201, body)]))

    with pytest.raises(UploadError) as caught:
        authority.export_media(payload)

    assert caught.value.reason_code == reason


def test_retryable_http_is_bounded_and_closes_intermediate_responses():
    payload = _media()
    first = Response(503, {})
    second = Response(503, {})
    third = Response(503, {})
    session = Session([first, second, third])
    authority = _authority(session, max_attempts=3)

    with pytest.raises(UploadError) as caught:
        authority.export_media(payload)

    assert caught.value.reason_code == "http_503"
    assert caught.value.retryable is True
    assert len(session.calls) == 3
    assert first.closed and second.closed and third.closed


def test_payload_bounds_and_digest_are_checked_before_network():
    session = Session([])
    authority = _authority(session, max_upload_bytes=4)
    payload = _media(b"12345")
    with pytest.raises(UploadError) as caught:
        authority.export_media(payload)
    assert caught.value.reason_code == "invalid_byte_length"
    assert session.calls == []

    bad = MediaPayload(b"abc", "0" * 64, 3, "image/png", "input")
    with pytest.raises(UploadError) as caught:
        _authority(Session([])).export_media(bad)
    assert caught.value.reason_code == "invalid_sha256"


def test_overflow_declares_trace_schema_and_never_uses_ordinary_ingest():
    content = b"complete-masked-otlp"
    payload = OverflowPayload(
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        byte_length=len(content),
        signal="span",
    )
    session = Session(
        [Response(201, _prepared(payload)), Response(200), Response(200, _completed(payload))]
    )
    receipt = _authority(session).export_overflow(payload)

    assert receipt.complete
    assert session.calls[0][2]["json"]["purpose"] == "otlp_overflow"
    assert session.calls[0][2]["json"]["payload_schema"] == "otlp.traces.v1"
    assert all(not call[1].endswith("/v1/traces") for call in session.calls)


def test_transport_exceptions_are_safely_classified():
    authority = _authority(
        Session([requests.ConnectionError("signed-url-and-key-must-not-escape")] * 3)
    )
    with pytest.raises(UploadError) as caught:
        authority.export_media(_media())
    assert str(caught.value) == "prepare:transport_error"
    assert "signed-url" not in str(caught.value)


def test_backend_error_diagnostic_is_preserved_without_accepting_unsafe_text():
    payload = _media()
    authority = _authority(
        Session(
            [
                Response(
                    415,
                    {
                        "reason_code": "MEDIA_TYPE_UNSUPPORTED",
                        "retryable": False,
                        "error": "must not be copied into diagnostics",
                    },
                )
            ]
        )
    )
    with pytest.raises(UploadError) as caught:
        authority.export_media(payload)
    assert str(caught.value) == "prepare:MEDIA_TYPE_UNSUPPORTED"
    assert caught.value.retryable is False

    unsafe = _authority(Session([Response(400, {"reason_code": "secret=https://signed.example"})]))
    with pytest.raises(UploadError) as caught:
        unsafe.export_media(payload)
    assert str(caught.value) == "prepare:http_400"


def test_backend_specific_media_and_overflow_limits_are_enforced_locally():
    authority = _authority(Session([]))
    assert authority.max_upload_bytes == 25 * 1024 * 1024
    assert authority.max_overflow_bytes == 20 * 1024 * 1024


def test_log_overflow_is_not_mislabeled_as_the_trace_schema():
    content = b"complete-masked-otlp-log"
    payload = OverflowPayload(
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        byte_length=len(content),
        signal="log",
    )
    session = Session([])
    with pytest.raises(UploadError) as caught:
        _authority(session).export_overflow(payload)
    assert caught.value.reason_code == "overflow_signal_unsupported"
    assert session.calls == []


def test_diagnostic_fields_cannot_smuggle_signed_material_into_failure_telemetry():
    payload = _media()
    body = _completed(
        payload,
        "rejected",
        diagnostic={
            "stage": "validation",
            "reason_code": "https://signed.example/?credential=secret",
            "retryable": False,
        },
    )
    authority = _authority(
        Session([Response(201, _prepared(payload)), Response(200), Response(200, body)])
    )
    with pytest.raises(UploadError) as caught:
        authority.export_media(payload)
    assert str(caught.value) == "complete:invalid_diagnostic"
    assert "signed" not in str(caught.value)
