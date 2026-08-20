"""Masking is applied once, off the normalizer, at the final exporter boundary."""

import asyncio

from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from neatlogs.core.mask import register_mask
from neatlogs.core.masking_exporter import MaskingLogExporter, MaskingSpanExporter
from neatlogs.core.span_processor import NeatlogsSpanProcessor


def _pipeline(mask=None):
    provider = TracerProvider(resource=Resource.create({"secret.resource": "resource-secret"}))
    inner = InMemorySpanExporter()
    provider.add_span_processor(NeatlogsSpanProcessor(mask=mask, own_all_spans=True))
    provider.add_span_processor(SimpleSpanProcessor(MaskingSpanExporter(inner, mask)))
    return provider, inner


def _counting_mask(counter, tag):
    def mask(snapshot):
        counter["n"] += 1
        attrs = snapshot.get("attributes", {})
        for key in list(attrs):
            if "value" in key:
                attrs[key] = f"[{tag}#{counter['n']}]" + str(attrs[key])
        return snapshot

    return mask


def test_global_mask_applied_once_to_exported_clone():
    counter = {"n": 0}
    provider, inner = _pipeline(_counting_mask(counter, "G"))
    span = provider.get_tracer("neatlogs.test").start_span("child")
    span.set_attribute("openinference.span.kind", "CHAIN")
    span.set_attribute("input.value", "IN")
    span.set_attribute("output.value", "OUT")
    span.end()
    provider.shutdown()

    assert counter["n"] == 1
    exported = inner.get_finished_spans()[0]
    assert exported.attributes["input.value"] == "[G#1]IN"
    assert exported.attributes["output.value"] == "[G#1]OUT"


def test_per_span_mask_takes_precedence_and_internal_id_is_not_exported():
    global_count, span_count = {"n": 0}, {"n": 0}
    provider, inner = _pipeline(_counting_mask(global_count, "G"))
    mask_id = register_mask(_counting_mask(span_count, "S"))
    span = provider.get_tracer("neatlogs.test").start_span("child")
    span.set_attribute("openinference.span.kind", "CHAIN")
    span.set_attribute("neatlogs.mask_id", mask_id)
    span.set_attribute("input.value", "IN")
    span.end()
    provider.shutdown()

    exported = inner.get_finished_spans()[0]
    assert span_count["n"] == 1
    assert global_count["n"] == 0
    assert exported.attributes["input.value"] == "[S#1]IN"
    assert "neatlogs.mask_id" not in exported.attributes


def test_mask_covers_events_resources_and_name():
    def mask(snapshot):
        snapshot["name"] = "masked-name"
        snapshot["events"][0]["attributes"]["secret"] = "***"
        snapshot["resource"]["attributes"]["secret.resource"] = "***"
        return snapshot

    provider, inner = _pipeline(mask)
    span = provider.get_tracer("neatlogs.test").start_span("secret-name")
    span.set_attribute("openinference.span.kind", "CHAIN")
    span.add_event("chunk", {"secret": "event-secret"})
    span.end()
    provider.shutdown()

    exported = inner.get_finished_spans()[0]
    assert exported.name == "masked-name"
    assert exported.events[0].attributes["secret"] == "***"
    assert exported.resource.attributes["secret.resource"] == "***"


def test_awaitable_mask_runs_and_callback_failure_fails_closed():
    async def async_mask(snapshot):
        await asyncio.sleep(0)
        snapshot["attributes"]["input.value"] = "***"
        return snapshot

    provider, inner = _pipeline(async_mask)
    span = provider.get_tracer("neatlogs.test").start_span("async")
    span.set_attribute("openinference.span.kind", "CHAIN")
    span.set_attribute("input.value", "secret")
    span.end()
    provider.shutdown()
    assert inner.get_finished_spans()[0].attributes["input.value"] == "***"

    def broken_mask(_snapshot):
        raise RuntimeError("do not leak")

    provider, inner = _pipeline(broken_mask)
    span = provider.get_tracer("neatlogs.test").start_span("broken")
    span.set_attribute("openinference.span.kind", "CHAIN")
    span.set_attribute("input.value", "must-not-export")
    span.end()
    provider.shutdown()
    assert inner.get_finished_spans() == ()


def test_global_mask_covers_logs_and_drops_failed_items():
    def mask(snapshot):
        snapshot["body"] = "***"
        snapshot["attributes"]["secret"] = "***"
        return snapshot

    provider = LoggerProvider()
    inner = InMemoryLogRecordExporter()
    provider.add_log_record_processor(SimpleLogRecordProcessor(MaskingLogExporter(inner, mask)))
    provider.get_logger("test").emit(body="secret body", attributes={"secret": "value"})
    provider.shutdown()
    exported = inner.get_finished_logs()[0].log_record
    assert exported.body == "***"
    assert exported.attributes["secret"] == "***"

    def broken(_snapshot):
        raise RuntimeError("drop")

    provider = LoggerProvider()
    inner = InMemoryLogRecordExporter()
    provider.add_log_record_processor(SimpleLogRecordProcessor(MaskingLogExporter(inner, broken)))
    provider.get_logger("test").emit(body="must-not-export")
    provider.shutdown()
    assert inner.get_finished_logs() == ()
