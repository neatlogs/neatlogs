"""Final-boundary masking contracts using real OTel providers/exporters."""

import asyncio
import importlib
import time

from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

from neatlogs.core.mask import register_mask
from neatlogs.core.masking_exporter import MaskingLogExporter, MaskingSpanExporter
from neatlogs.core.span_processor import NeatlogsSpanProcessor

SENTINEL = "privacy-sentinel-never-export"


def _pipeline(mask=None, exporter=None, timeout=5.0):
    provider = TracerProvider(resource=Resource.create({"secret.resource": SENTINEL}))
    inner = exporter or InMemorySpanExporter()
    wrapper = MaskingSpanExporter(inner, mask, timeout_seconds=timeout)
    provider.add_span_processor(NeatlogsSpanProcessor(mask=mask, own_all_spans=True))
    provider.add_span_processor(SimpleSpanProcessor(wrapper))
    return provider, inner, wrapper


def _span(provider, name="call"):
    span = provider.get_tracer("neatlogs.test").start_span(name)
    span.set_attribute("openinference.span.kind", "LLM")
    span.set_attribute("input.value", SENTINEL)
    return span


def test_canonicalize_then_mask_clone_without_mutating_source():
    def mask(snapshot):
        # The processor has completed before the exporter takes its canonical
        # snapshot; typed input is therefore present at the mask boundary.
        assert snapshot["attributes"]["input.value"] == SENTINEL
        snapshot["attributes"]["input.value"] = "***"
        snapshot["resource"]["attributes"]["secret.resource"] = "***"
        return snapshot

    provider, inner, _ = _pipeline(mask)
    span = _span(provider)
    span.end()
    provider.shutdown()
    exported = inner.get_finished_spans()[0]
    assert exported.attributes["input.value"] == "***"
    assert exported.resource.attributes["secret.resource"] == "***"
    assert span.attributes["input.value"] == SENTINEL


def test_events_exceptions_and_status_are_maskable_without_source_mutation():
    def mask(snapshot):
        for event in snapshot["events"]:
            event["attributes"] = {key: "***" for key in event["attributes"]}
        snapshot["status"]["description"] = "***"
        return snapshot

    provider, inner, _ = _pipeline(mask)
    span = _span(provider, "failure")
    try:
        raise RuntimeError(SENTINEL)
    except RuntimeError as exc:
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, SENTINEL))
    span.end()
    provider.shutdown()
    exported = inner.get_finished_spans()[0]
    assert all(SENTINEL not in str(dict(event.attributes)) for event in exported.events)
    assert exported.status.description == "***"
    assert any(SENTINEL in str(dict(event.attributes)) for event in span.events)


def test_per_span_precedence_and_internal_id_removal():
    calls = []

    def global_mask(snapshot):
        calls.append("global")
        return snapshot

    def local_mask(snapshot):
        calls.append("local")
        snapshot["attributes"]["input.value"] = "***"
        return snapshot

    provider, inner, _ = _pipeline(global_mask)
    span = _span(provider)
    span.set_attribute("neatlogs.mask_id", register_mask(local_mask))
    span.end()
    provider.shutdown()
    exported = inner.get_finished_spans()[0]
    assert calls == ["local"]
    assert exported.attributes["input.value"] == "***"
    assert "neatlogs.mask_id" not in exported.attributes


def test_async_mask_and_null_drop():
    async def mask(snapshot):
        await asyncio.sleep(0)
        snapshot["attributes"]["input.value"] = "***"
        return snapshot

    provider, inner, wrapper = _pipeline(mask)
    _span(provider).end()
    provider.shutdown()
    assert inner.get_finished_spans()[0].attributes["input.value"] == "***"
    assert wrapper.health.healthy

    provider, inner, wrapper = _pipeline(lambda _: None)
    _span(provider).end()
    provider.shutdown()
    assert inner.get_finished_spans() == ()
    assert not wrapper.health.healthy
    assert wrapper.health.drops == 1


def test_exception_and_timeout_fail_closed():
    def broken(_):
        raise RuntimeError(SENTINEL)

    provider, inner, wrapper = _pipeline(broken)
    _span(provider).end()
    provider.shutdown()
    assert inner.get_finished_spans() == ()
    assert not wrapper.health.healthy

    def slow(snapshot):
        time.sleep(0.1)
        return snapshot

    provider, inner, wrapper = _pipeline(slow, timeout=0.01)
    _span(provider).end()
    provider.shutdown()
    assert inner.get_finished_spans() == ()
    assert not wrapper.health.healthy


class _FailingExporter(InMemorySpanExporter):
    def export(self, spans):
        return SpanExportResult.FAILURE


def test_transport_failure_exposed_by_health_and_flush():
    provider, _, wrapper = _pipeline(lambda snapshot: snapshot, _FailingExporter())
    _span(provider).end()
    assert not wrapper.health.healthy
    assert wrapper.force_flush() is False
    provider.shutdown()


def test_public_flush_reports_mask_drop_health(monkeypatch):
    init_module = importlib.import_module("neatlogs.init")
    provider, _, wrapper = _pipeline(lambda _: None)
    _span(provider).end()
    monkeypatch.setattr(init_module, "_tracer_provider", None)
    monkeypatch.setattr(init_module, "_meter_provider", None)
    monkeypatch.setattr(init_module, "_log_provider", None)
    monkeypatch.setattr(init_module, "_export_health", [wrapper])
    assert init_module.flush() is False
    provider.shutdown()


def test_log_body_attributes_and_resource_are_masked():
    def mask(snapshot):
        snapshot["body"] = "***"
        snapshot["attributes"]["secret"] = "***"
        snapshot["resource"]["attributes"]["secret.resource"] = "***"
        return snapshot

    provider = LoggerProvider(resource=Resource.create({"secret.resource": SENTINEL}))
    inner = InMemoryLogRecordExporter()
    wrapper = MaskingLogExporter(inner, mask)
    provider.add_log_record_processor(SimpleLogRecordProcessor(wrapper))
    provider.get_logger("test").emit(body=SENTINEL, attributes={"secret": SENTINEL})
    provider.shutdown()
    exported = inner.get_finished_logs()[0]
    assert exported.log_record.body == "***"
    assert exported.log_record.attributes["secret"] == "***"
    assert exported.resource.attributes["secret.resource"] == "***"
