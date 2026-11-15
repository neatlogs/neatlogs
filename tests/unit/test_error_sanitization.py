import io
import json

import pytest
from opentelemetry._logs import SeverityNumber
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import neatlogs
from neatlogs._wrap_utils import set_neatlogs_provider
from neatlogs.core.error_sanitization import sanitize_error_snapshot, sanitize_error_text
from neatlogs.core.masking_exporter import MaskingLogExporter, MaskingSpanExporter
from neatlogs.core.span_processor import NeatlogsSpanProcessor

FAKE_TOKEN = "FAKE_CREDENTIAL_123+/="
MESSAGE = (
    "Status: 403 missing instagram_manage_messages\n"
    f"Params: {{'access_token': '{FAKE_TOKEN}', 'recipient': 'demo'}}"
)


@pytest.mark.parametrize(
    "text",
    [
        MESSAGE,
        f'{{"access_token": "{FAKE_TOKEN}"}}',
        r"{\"access_token\": \"FAKE_CREDENTIAL_123\"}",
        "https://example.invalid/messages?access_token=FAKE_CREDENTIAL_123%2B%2F%3D&fields=id",
        "access_token%3DFAKE_CREDENTIAL_123%26fields%3Did",
        "%22access_token%22%3A%20%22FAKE_CREDENTIAL_123%22",
        "Authorization: Bearer FAKE_CREDENTIAL_123",
        "Incorrect API key provided: FAKE_CREDENTIAL_123",
        repr({"password": "FAKE_CREDENTIAL_123'\"AFTER_QUOTE"}),
        json.dumps({"password": "FAKE_CREDENTIAL_123'\"AFTER_QUOTE"}),
    ],
)
def test_error_text_redacts_common_credential_encodings_idempotently(text):
    sanitized = sanitize_error_text(text)
    assert "FAKE_CREDENTIAL" not in sanitized
    assert "AFTER_QUOTE" not in sanitized
    assert "REDACTED" in sanitized
    assert sanitize_error_text(sanitized) == sanitized


def test_error_snapshot_keeps_captured_io_and_non_error_log_body():
    snapshot = {
        "attributes": {"input.value": "api_key=business-input", "output.value": False},
        "body": "api_key=business-log",
        "severity_text": "INFO",
    }
    assert not sanitize_error_snapshot(snapshot)
    assert snapshot["attributes"]["input.value"] == "api_key=business-input"
    assert snapshot["attributes"]["output.value"] is False
    assert snapshot["body"] == "api_key=business-log"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("with_mask", [False, True])
async def test_decorated_errors_are_sanitized_without_changing_exception_or_io(
    asynchronous, with_mask
):
    originals, exported = InMemorySpanExporter(), InMemorySpanExporter()
    provider = TracerProvider()
    normalizer = NeatlogsSpanProcessor(own_all_spans=True)
    raw, processed = io.StringIO(), io.StringIO()
    normalizer._raw_log_file_handle = raw
    normalizer._processed_log_file_handle = processed
    masks = []

    def mask(snapshot):
        masks.append(snapshot["name"])
        return snapshot

    normalizer.mask = mask if with_mask else None
    provider.add_span_processor(SimpleSpanProcessor(originals))
    provider.add_span_processor(normalizer)
    provider.add_span_processor(
        SimpleSpanProcessor(MaskingSpanExporter(exported, mask if with_mask else None))
    )
    set_neatlogs_provider(provider)
    error = RuntimeError(MESSAGE)
    calls = []

    @neatlogs.span(kind="TOOL", name="dm.send")
    def send_sync(question):
        calls.append(question)
        raise error

    @neatlogs.span(kind="TOOL", name="dm.send")
    async def send_async(question):
        calls.append(question)
        raise error

    @neatlogs.span(kind="WORKFLOW", name="send-workflow")
    def workflow_sync(question):
        return send_sync(question)

    @neatlogs.span(kind="WORKFLOW", name="send-workflow")
    async def workflow_async(question):
        return await send_async(question)

    try:
        with pytest.raises(RuntimeError) as raised:
            if asynchronous:
                await workflow_async("Can you help?")
            else:
                workflow_sync("Can you help?")
        assert raised.value is error
        assert str(error) == MESSAGE
        assert calls == ["Can you help?"]
        spans = [
            span
            for span in exported.get_finished_spans()
            if span.name in {"dm.send", "send-workflow"}
        ]
        assert len(spans) == 2
        for span in spans:
            assert FAKE_TOKEN not in span.status.description
            assert "403" in span.status.description
            assert "instagram_manage_messages" in span.status.description
            assert span.attributes["input.value"] == '{"question": "Can you help?"}'
            assert FAKE_TOKEN not in json.dumps([dict(event.attributes) for event in span.events])
        assert all(
            FAKE_TOKEN in span.status.description
            for span in originals.get_finished_spans()
            if span.name in {"dm.send", "send-workflow"}
        )
        assert FAKE_TOKEN not in raw.getvalue()
        assert FAKE_TOKEN not in processed.getvalue()
        if with_mask:
            assert masks.count("dm.send") == masks.count("send-workflow") == 1
        else:
            assert masks == []
    finally:
        provider.shutdown()


@pytest.mark.parametrize("with_mask", [False, True])
def test_error_logs_are_sanitized_with_or_without_custom_mask(with_mask):
    inner = InMemoryLogRecordExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(
        SimpleLogRecordProcessor(
            MaskingLogExporter(inner, (lambda item: item) if with_mask else None)
        )
    )
    try:
        provider.get_logger("app").emit(
            severity_number=SeverityNumber.ERROR,
            severity_text="ERROR",
            body=MESSAGE,
            attributes={"exception.message": MESSAGE, "exception.stacktrace": MESSAGE},
        )
        record = inner.get_finished_logs()[0].log_record
        assert FAKE_TOKEN not in record.body
        assert "instagram_manage_messages" in record.body
        assert FAKE_TOKEN not in json.dumps(dict(record.attributes))
    finally:
        provider.shutdown()


def test_ordinary_logs_keep_fast_pass_without_snapshot(monkeypatch):
    import neatlogs.core.masking_exporter as exporter_module

    def unexpected_snapshot(_item):
        raise AssertionError("ordinary INFO logs should bypass snapshots")

    monkeypatch.setattr(exporter_module, "_log_snapshot", unexpected_snapshot)
    inner = InMemoryLogRecordExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(MaskingLogExporter(inner, None)))
    try:
        provider.get_logger("app").emit(
            severity_number=SeverityNumber.INFO, severity_text="INFO", body="ordinary business log"
        )
        assert inner.get_finished_logs()[0].log_record.body == "ordinary business log"
    finally:
        provider.shutdown()


def test_error_sanitization_keeps_mask_drop_contract():
    inner = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(MaskingSpanExporter(inner, lambda _: None)))
    try:
        with provider.get_tracer("neatlogs.test").start_as_current_span("failure") as span:
            span.record_exception(RuntimeError(MESSAGE))
        assert not inner.get_finished_spans()
    finally:
        provider.shutdown()
