"""current_trace_id() reads the active Neatlogs trace id in both provider modes.

Raw ``trace.get_current_span()`` only works in DEFAULT mode; in ISOLATED mode
the Neatlogs span is threaded on a private context key, so a naive getter
returns an invalid span context (or the host's span). This helper must resolve
through the SDK's own parent resolution instead.
"""

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from neatlogs._wrap_utils import attach_as_current, detach, set_neatlogs_provider
from neatlogs.core.propagation import current_trace_id


def _provider():
    p = TracerProvider()
    p.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    return p


def test_returns_none_with_no_active_span():
    assert current_trace_id() is None


def test_reads_active_span_in_default_mode():
    provider = _provider()
    set_neatlogs_provider(None)  # default mode: global provider
    trace_api.set_tracer_provider(provider)
    try:
        tracer = provider.get_tracer("neatlogs.test")
        with tracer.start_as_current_span("nl-root") as span:
            expected = format(span.get_span_context().trace_id, "032x")
            token = attach_as_current(span)
            try:
                assert current_trace_id() == expected
            finally:
                detach(token)
    finally:
        provider.shutdown()


def test_reads_private_parent_in_isolated_mode():
    private = _provider()
    set_neatlogs_provider(private)  # private != OTel global → isolated
    try:
        tracer = private.get_tracer("neatlogs.test")
        span = tracer.start_span("nl-root")
        token = attach_as_current(span, force_owned=True)
        try:
            # the OTel global current-span is NOT ours here ...
            assert trace_api.get_current_span().get_span_context().span_id != (
                span.get_span_context().span_id
            )
            # ... but the helper still finds our trace id.
            assert current_trace_id() == format(span.get_span_context().trace_id, "032x")
        finally:
            detach(token)
            span.end()
        assert current_trace_id() is None
    finally:
        set_neatlogs_provider(None)
        private.shutdown()


def test_ignores_foreign_only_span():
    """A co-tenant's span alone is not a Neatlogs trace — return None rather
    than leaking a foreign trace id."""
    private = _provider()
    foreign = _provider()
    set_neatlogs_provider(private)  # isolated
    try:
        with foreign.get_tracer("foreign.datadog").start_as_current_span("f"):
            assert trace_api.get_current_span().is_recording()
            assert current_trace_id() is None
    finally:
        set_neatlogs_provider(None)
        private.shutdown()
        foreign.shutdown()


def test_default_owned_trace_survives_independent_foreign_root():
    from opentelemetry.context import Context

    from neatlogs.core.propagation import inject_trace_context

    own = _provider()
    foreign = _provider()
    set_neatlogs_provider(None)
    root = own.get_tracer("neatlogs.test").start_span("owned-root")
    token = attach_as_current(root)
    try:
        expected = format(root.get_span_context().trace_id, "032x")
        with foreign.get_tracer("foreign.test").start_as_current_span(
            "independent-root", context=Context()
        ) as other:
            assert other.get_span_context().trace_id != root.get_span_context().trace_id
            assert current_trace_id() == expected
            carrier = {}
            assert inject_trace_context(carrier)
            assert carrier["traceparent"].split("-")[1] == expected
        assert current_trace_id() == expected
    finally:
        detach(token)
        root.end()
        own.shutdown()
        foreign.shutdown()
    assert current_trace_id() is None


def test_default_foreign_only_trace_is_not_owned():
    foreign = _provider()
    set_neatlogs_provider(None)
    try:
        with foreign.get_tracer("foreign.test").start_as_current_span("foreign-root"):
            assert current_trace_id() is None
    finally:
        foreign.shutdown()


def test_default_ended_owned_span_does_not_resolve_foreign_trace():
    from opentelemetry.context import Context

    own = _provider()
    foreign = _provider()
    set_neatlogs_provider(None)
    root = own.get_tracer("neatlogs.test").start_span("owned-root")
    token = attach_as_current(root)
    try:
        root.end()
        with foreign.get_tracer("foreign.test").start_as_current_span(
            "independent-root", context=Context()
        ):
            assert current_trace_id() is None
    finally:
        detach(token)
        own.shutdown()
        foreign.shutdown()


def test_default_nested_owned_scope_restores_parent():
    from opentelemetry.context import Context

    own = _provider()
    set_neatlogs_provider(None)
    outer = own.get_tracer("neatlogs.test").start_span("outer", context=Context())
    inner = own.get_tracer("neatlogs.test").start_span("inner", context=Context())
    outer_token = attach_as_current(outer)
    try:
        inner_token = attach_as_current(inner)
        try:
            assert current_trace_id() == format(inner.get_span_context().trace_id, "032x")
        finally:
            detach(inner_token)
            inner.end()
        assert current_trace_id() == format(outer.get_span_context().trace_id, "032x")
    finally:
        detach(outer_token)
        outer.end()
        own.shutdown()
    assert current_trace_id() is None
