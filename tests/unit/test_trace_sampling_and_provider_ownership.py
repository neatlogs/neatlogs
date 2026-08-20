import math

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import neatlogs
from neatlogs._wrap_utils import get_neatlogs_provider


def _export_from(provider: TracerProvider) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


def test_init_never_claims_or_reuses_the_global_provider():
    global_provider = TracerProvider()
    global_exporter = _export_from(global_provider)
    otel_trace.set_tracer_provider(global_provider)

    neatlogs.init(
        api_key="unused",
        disable_export=True,
        instrumentations=[],
    )
    private_provider = get_neatlogs_provider()
    assert private_provider is not None
    assert private_provider is not global_provider
    private_exporter = _export_from(private_provider)

    with neatlogs.trace("private-root", kind="WORKFLOW"):
        pass
    global_span = global_provider.get_tracer("host").start_span("host-root")
    global_span.end()

    assert {span.name for span in private_exporter.get_finished_spans()} == {"private-root"}
    assert {span.name for span in global_exporter.get_finished_spans()} == {"host-root"}


def test_zero_sample_rate_drops_the_whole_trace_not_individual_children():
    neatlogs.init(
        api_key="unused",
        disable_export=True,
        instrumentations=[],
        sample_rate=0.0,
    )
    provider = get_neatlogs_provider()
    assert provider is not None
    exporter = _export_from(provider)

    with neatlogs.trace("root", kind="WORKFLOW"):
        with neatlogs.trace("child", kind="CHAIN"):
            pass

    assert exporter.get_finished_spans() == ()


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf, -math.inf, True, "1"])
def test_invalid_sample_rate_is_a_typed_configuration_error(value):
    with pytest.raises(neatlogs.NeatlogsConfigurationError, match="sample_rate"):
        neatlogs.init(
            api_key="unused",
            disable_export=True,
            instrumentations=[],
            sample_rate=value,
        )


def test_caller_owned_provider_rejects_an_unenforceable_sample_rate():
    with pytest.raises(neatlogs.NeatlogsConfigurationError, match="caller-owned"):
        neatlogs.Client(
            api_key="unused",
            workflow_name="test",
            disable_export=True,
            tracer_provider=TracerProvider(),
            sample_rate=0.5,
        )


def test_explicit_global_provider_is_rejected():
    provider = TracerProvider()
    otel_trace.set_tracer_provider(provider)
    with pytest.raises(neatlogs.NeatlogsConfigurationError, match="process-global"):
        neatlogs.init(
            api_key="unused",
            disable_export=True,
            instrumentations=[],
            tracer_provider=provider,
        )
