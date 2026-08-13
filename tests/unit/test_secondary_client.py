import asyncio

from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import neatlogs
from neatlogs._wrap_utils import get_neatlogs_provider, get_tracer


def _client(name: str, *, capture_logs: bool = False):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client = neatlogs.Client(
        api_key=f"{name}-key",
        workflow_name=name,
        capture_logs=capture_logs,
        disable_export=True,
        tracer_provider=provider,
    )
    return client, provider, exporter


def test_client_is_additive_to_default_init():
    neatlogs.init(
        api_key="default-key",
        workflow_name="default",
        disable_export=True,
        instrumentations=[],
    )
    default_provider = get_neatlogs_provider()
    secondary, secondary_provider, _ = _client("secondary")

    assert get_neatlogs_provider() is default_provider
    with secondary.activate():
        assert get_neatlogs_provider() is secondary_provider
    assert get_neatlogs_provider() is default_provider

    secondary.shutdown()


def test_concurrent_clients_do_not_cross_projects():
    first, _, first_exporter = _client("first")
    second, _, second_exporter = _client("second")

    async def run(client, span_name):
        with client.activate():
            with neatlogs.trace(span_name, kind="WORKFLOW"):
                await asyncio.sleep(0)

    async def main():
        await asyncio.gather(run(first, "first-run"), run(second, "second-run"))

    asyncio.run(main())

    first_names = {span.name for span in first_exporter.get_finished_spans()}
    second_names = {span.name for span in second_exporter.get_finished_spans()}
    assert "first-run" in first_names
    assert "second-run" not in first_names
    assert "second-run" in second_names
    assert "first-run" not in second_names

    first.shutdown()
    second.shutdown()


def test_wrapper_lookup_uses_the_active_client():
    first, _, first_exporter = _client("first")
    second, _, second_exporter = _client("second")

    with first.activate():
        span = get_tracer().start_span("first-wrapper")
        span.end()
    with second.activate():
        span = get_tracer().start_span("second-wrapper")
        span.end()

    first_names = {span.name for span in first_exporter.get_finished_spans()}
    second_names = {span.name for span in second_exporter.get_finished_spans()}
    assert "first-wrapper" in first_names
    assert "second-wrapper" not in first_names
    assert "second-wrapper" in second_names
    assert "first-wrapper" not in second_names

    first.shutdown()
    second.shutdown()


def test_structured_logs_use_the_active_client():
    first, _, _ = _client("first", capture_logs=True)
    second, _, _ = _client("second", capture_logs=True)
    first_logs = InMemoryLogRecordExporter()
    second_logs = InMemoryLogRecordExporter()
    first.log_provider.add_log_record_processor(SimpleLogRecordProcessor(first_logs))
    second.log_provider.add_log_record_processor(SimpleLogRecordProcessor(second_logs))

    with first.activate():
        with neatlogs.trace("first-run", kind="WORKFLOW"):
            neatlogs.log("first message")
    with second.activate():
        with neatlogs.trace("second-run", kind="WORKFLOW"):
            neatlogs.log("second message")

    first_bodies = {str(item.log_record.body) for item in first_logs.get_finished_logs()}
    second_bodies = {str(item.log_record.body) for item in second_logs.get_finished_logs()}
    assert first_bodies == {"first message"}
    assert second_bodies == {"second message"}

    first.shutdown()
    second.shutdown()


def test_flushing_one_client_does_not_flush_another(monkeypatch):
    first, first_provider, _ = _client("first")
    second, second_provider, _ = _client("second")
    first_flushes = []
    second_flushes = []
    monkeypatch.setattr(first_provider, "force_flush", lambda **_: first_flushes.append(1) or True)
    monkeypatch.setattr(
        second_provider, "force_flush", lambda **_: second_flushes.append(1) or True
    )

    first.flush()

    assert first_flushes == [1]
    assert second_flushes == []

    first.shutdown()
    second.shutdown()
