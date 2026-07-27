"""Isolation guarantees for inject_trace_context / the outbound HTTP auto-patch.

The private-provider design promises two things at a service boundary:
  1. Neatlogs injects ITS OWN active span as traceparent — even in isolated mode
     where that span is threaded on a private context key, not the OTel global.
  2. Neatlogs NEVER injects a co-tenant's span (Datadog/openlit/langfuse). When
     only a foreign span is active, inject is a no-op and the carrier is clean.
"""
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from neatlogs._wrap_utils import (
    attach_as_current,
    detach,
    set_neatlogs_provider,
    _has_active_recording_parent,
    _neatlogs_root_kwargs,
)
from neatlogs.core.propagation import extract_trace_context, inject_trace_context

_REMOTE_TP = "00-11111111111111111111111111111111-2222222222222222-01"
_REMOTE_TRACE = "11111111111111111111111111111111"


def _provider():
    p = TracerProvider()
    p.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    return p


def _traceparent_trace_id(carrier):
    tp = carrier.get("traceparent")
    if not tp:
        return None
    return tp.split("-")[1]


def test_inject_emits_our_span_in_isolated_mode():
    """A Neatlogs span active on the PRIVATE key is injected, even though the
    OTel global current-span is untouched (isolated mode)."""
    private = _provider()
    set_neatlogs_provider(private)  # private != OTel global → isolated
    try:
        tracer = private.get_tracer("neatlogs.test")
        span = tracer.start_span("nl-root")
        token = attach_as_current(span, force_owned=True)
        try:
            # Global current-span is NOT our span (isolation) ...
            assert trace_api.get_current_span().get_span_context().span_id != (
                span.get_span_context().span_id
            )
            carrier = {}
            ok = inject_trace_context(carrier)
            # ... yet inject still emits OUR trace_id from the private key.
            assert ok is True
            assert _traceparent_trace_id(carrier) == format(
                span.get_span_context().trace_id, "032x"
            )
        finally:
            detach(token)
            span.end()
    finally:
        set_neatlogs_provider(None)
        private.shutdown()


def test_inject_is_noop_with_only_a_foreign_span_active():
    """When only a co-tenant's span is active (no Neatlogs ancestor), inject must
    write nothing — never leak the foreign tracer's context as ours."""
    private = _provider()
    foreign = _provider()
    set_neatlogs_provider(private)  # isolated
    try:
        foreign_tracer = foreign.get_tracer("foreign.datadog")
        with foreign_tracer.start_as_current_span("foreign-root"):
            assert trace_api.get_current_span().is_recording()  # foreign is active
            carrier = {}
            ok = inject_trace_context(carrier)
            assert ok is False
            assert "traceparent" not in carrier  # no foreign leak
    finally:
        set_neatlogs_provider(None)
        private.shutdown()
        foreign.shutdown()


def test_inject_is_noop_when_nothing_active():
    private = _provider()
    set_neatlogs_provider(private)
    try:
        carrier = {}
        assert inject_trace_context(carrier) is False
        assert carrier == {}
    finally:
        set_neatlogs_provider(None)
        private.shutdown()


def test_extract_installs_remote_parent_privately_in_isolated_mode():
    """extract_trace_context makes the remote span a Neatlogs parent WITHOUT
    touching the OTel global current-span (isolation)."""
    private = _provider()
    set_neatlogs_provider(private)  # isolated
    try:
        with extract_trace_context({"traceparent": _REMOTE_TP}, session_id="s"):
            # Global current-span is untouched — a co-tenant reads no remote parent.
            assert not trace_api.get_current_span().get_span_context().is_valid
            # ... but Neatlogs sees an active recording parent, so its next span
            # nests under the remote trace instead of starting a fresh root.
            assert _has_active_recording_parent() is True
            root_kwargs = _neatlogs_root_kwargs()
            ctx = root_kwargs.get("context")
            assert ctx is not None
            parent_sc = trace_api.get_current_span(ctx).get_span_context()
            assert format(parent_sc.trace_id, "032x") == _REMOTE_TRACE
    finally:
        set_neatlogs_provider(None)
        private.shutdown()


def test_extract_child_span_joins_remote_trace_and_is_not_root():
    """A neatlogs child opened inside extract_trace_context shares the remote
    trace_id and is parented to the remote span (not a new root)."""
    private = _provider()
    set_neatlogs_provider(private)
    try:
        with extract_trace_context({"traceparent": _REMOTE_TP}):
            tracer = private.get_tracer("neatlogs.test")
            kwargs = _neatlogs_root_kwargs()  # what get_tracer() applies
            span = tracer.start_span("callee-child", **kwargs)
            sc = span.get_span_context()
            assert format(sc.trace_id, "032x") == _REMOTE_TRACE
            span.end()
    finally:
        set_neatlogs_provider(None)
        private.shutdown()


def test_extract_empty_carrier_is_noop_passthrough():
    private = _provider()
    set_neatlogs_provider(private)
    try:
        with extract_trace_context({}):
            # No remote parent → no neatlogs ancestor forced; next span is a root.
            assert _has_active_recording_parent() is False
    finally:
        set_neatlogs_provider(None)
        private.shutdown()


def test_extract_does_not_leak_into_a_cotenant_tracer():
    """The remote parent installed by extract must NOT become the parent of a
    FOREIGN co-tenant's span (which resolves parent from the OTel global)."""
    private = _provider()
    foreign = _provider()
    foreign_exporter = InMemorySpanExporter()
    foreign.add_span_processor(SimpleSpanProcessor(foreign_exporter))
    set_neatlogs_provider(private)  # isolated
    try:
        foreign_tracer = foreign.get_tracer("foreign.datadog")
        with extract_trace_context({"traceparent": _REMOTE_TP}):
            with foreign_tracer.start_as_current_span("foreign-span"):
                pass
        finished = {s.name: s for s in foreign_exporter.get_finished_spans()}
        # The foreign span is a ROOT of its OWN trace — it never adopted the
        # remote neatlogs parent.
        assert finished["foreign-span"].parent is None
        assert format(finished["foreign-span"].context.trace_id, "032x") != _REMOTE_TRACE
    finally:
        set_neatlogs_provider(None)
        private.shutdown()
        foreign.shutdown()


def test_inject_does_not_clobber_upstream_traceparent():
    private = _provider()
    set_neatlogs_provider(private)
    try:
        tracer = private.get_tracer("neatlogs.test")
        span = tracer.start_span("nl-root")
        token = attach_as_current(span, force_owned=True)
        try:
            upstream = "00-abcdef00000000000000000000000000-1111111111111111-01"
            carrier = {"traceparent": upstream}
            ok = inject_trace_context(carrier)
            assert ok is True
            assert carrier["traceparent"] == upstream  # preserved, not overwritten
        finally:
            detach(token)
            span.end()
    finally:
        set_neatlogs_provider(None)
        private.shutdown()
