import math

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, TraceState

import neatlogs
from neatlogs._wrap_utils import get_neatlogs_provider


def _export_from(provider: TracerProvider) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


def _remote_parent(*, sampled: bool):
    flags = TraceFlags(TraceFlags.SAMPLED if sampled else TraceFlags.DEFAULT)
    span = NonRecordingSpan(
        SpanContext(
            trace_id=0x1234567890ABCDEF1234567890ABCDEF,
            span_id=0x1234567890ABCDEF,
            is_remote=True,
            trace_flags=flags,
            trace_state=TraceState(),
        )
    )
    return otel_trace.set_span_in_context(span)


def test_init_never_claims_or_reuses_the_global_provider():
    global_provider = TracerProvider()
    global_exporter = _export_from(global_provider)
    otel_trace.set_tracer_provider(global_provider)

    neatlogs.init(api_key="unused", disable_export=True, instrumentations=[])
    private_provider = get_neatlogs_provider()
    assert private_provider is not None
    assert private_provider is not global_provider
    private_exporter = _export_from(private_provider)

    with neatlogs.trace("private-root", kind="WORKFLOW"):
        pass
    global_provider.get_tracer("foreign.host").start_span("foreign-root").end()

    assert {span.name for span in private_exporter.get_finished_spans()} == {"private-root"}
    assert {span.name for span in global_exporter.get_finished_spans()} == {"foreign-root"}


def test_deprecated_isolate_false_cannot_disable_private_provider_isolation():
    global_provider = TracerProvider()
    otel_trace.set_tracer_provider(global_provider)

    neatlogs.init(
        api_key="unused",
        disable_export=True,
        instrumentations=[],
        isolate=False,
    )

    assert get_neatlogs_provider() is not global_provider


@pytest.mark.parametrize("value", [0.0, 1e-12, 0.5, 1.0 - 1e-12, 1.0])
def test_finite_sampling_boundaries_are_accepted_and_parent_based(value):
    neatlogs.init(
        api_key="unused",
        disable_export=True,
        instrumentations=[],
        sample_rate=value,
    )

    assert isinstance(get_neatlogs_provider().sampler, ParentBased)


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf, -math.inf, True, "1", None])
def test_invalid_sample_rate_is_a_typed_configuration_error(value):
    with pytest.raises(neatlogs.NeatlogsConfigurationError, match="sample_rate"):
        neatlogs.init(
            api_key="unused",
            disable_export=True,
            instrumentations=[],
            sample_rate=value,
        )


def test_zero_sample_rate_drops_the_whole_local_trace():
    neatlogs.init(
        api_key="unused",
        disable_export=True,
        instrumentations=[],
        sample_rate=0.0,
    )
    exporter = _export_from(get_neatlogs_provider())

    with neatlogs.trace("root", kind="WORKFLOW"):
        with neatlogs.trace("child", kind="CHAIN"):
            pass

    assert exporter.get_finished_spans() == ()


@pytest.mark.parametrize(
    ("sample_rate", "remote_sampled", "expected_names"),
    [(0.0, True, {"remote-child"}), (1.0, False, set())],
)
def test_remote_parent_sampling_decision_is_preserved(sample_rate, remote_sampled, expected_names):
    neatlogs.init(
        api_key="unused",
        disable_export=True,
        instrumentations=[],
        sample_rate=sample_rate,
    )
    provider = get_neatlogs_provider()
    exporter = _export_from(provider)

    span = provider.get_tracer("neatlogs.remote-test").start_span(
        "remote-child",
        context=_remote_parent(sampled=remote_sampled),
    )
    span.end()

    assert {span.name for span in exporter.get_finished_spans()} == expected_names


@pytest.mark.parametrize("factory", ["init", "client"])
def test_caller_owned_provider_rejects_an_unenforceable_sample_rate(factory):
    provider = TracerProvider()
    with pytest.raises(neatlogs.NeatlogsConfigurationError, match="caller-owned"):
        if factory == "init":
            neatlogs.init(
                api_key="unused",
                disable_export=True,
                instrumentations=[],
                tracer_provider=provider,
                sample_rate=0.5,
            )
        else:
            neatlogs.Client(
                api_key="unused",
                workflow_name="test",
                disable_export=True,
                tracer_provider=provider,
                sample_rate=0.5,
            )


@pytest.mark.parametrize("factory", ["init", "client"])
def test_explicit_process_global_provider_is_rejected(factory):
    provider = TracerProvider()
    otel_trace.set_tracer_provider(provider)

    with pytest.raises(neatlogs.NeatlogsConfigurationError, match="process-global"):
        if factory == "init":
            neatlogs.init(
                api_key="unused",
                disable_export=True,
                instrumentations=[],
                tracer_provider=provider,
            )
        else:
            neatlogs.Client(
                api_key="unused",
                workflow_name="test",
                disable_export=True,
                tracer_provider=provider,
            )


def test_clients_with_different_sample_rates_remain_isolated():
    dropped = neatlogs.Client(
        api_key="unused",
        workflow_name="dropped",
        disable_export=True,
        sample_rate=0.0,
    )
    kept = neatlogs.Client(
        api_key="unused",
        workflow_name="kept",
        disable_export=True,
        sample_rate=1.0,
    )
    dropped_exporter = _export_from(dropped.tracer_provider)
    kept_exporter = _export_from(kept.tracer_provider)

    dropped.get_tracer("client.dropped").start_span("dropped-root").end()
    kept.get_tracer("client.kept").start_span("kept-root").end()

    assert dropped_exporter.get_finished_spans() == ()
    assert {span.name for span in kept_exporter.get_finished_spans()} == {"kept-root"}
    assert dropped.tracer_provider is not kept.tracer_provider

    dropped.shutdown()
    kept.shutdown()
