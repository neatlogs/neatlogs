"""Explicit Neatlogs W3C propagation does not borrow the host propagator."""

import pytest
from opentelemetry import baggage
from opentelemetry import context as context_api
from opentelemetry import propagate
from opentelemetry import trace as trace_api
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.trace import TracerProvider

from neatlogs._wrap_utils import (
    _neatlogs_root_kwargs,
    attach_as_current,
    detach,
    set_neatlogs_provider,
)
from neatlogs.core.propagation import extract_trace_context, inject_trace_context

REMOTE = "00-11111111111111111111111111111111-2222222222222222-01"


@pytest.fixture(params=[CompositePropagator([]), W3CBaggagePropagator()])
def host(request):
    previous = propagate.get_global_textmap()
    provider = TracerProvider()
    propagate.set_global_textmap(request.param)
    set_neatlogs_provider(provider)
    try:
        yield provider, request.param
        assert propagate.get_global_textmap() is request.param
    finally:
        set_neatlogs_provider(None)
        propagate.set_global_textmap(previous)
        provider.shutdown()


def test_inject_w3c_with_non_w3c_host_propagator(host):
    provider, _ = host
    span = provider.get_tracer("neatlogs.test").start_span("caller")
    token = attach_as_current(span, force_owned=True)
    try:
        carrier = {}
        assert inject_trace_context(carrier) is True
        assert carrier["traceparent"].split("-")[1] == format(
            span.get_span_context().trace_id, "032x"
        )
    finally:
        detach(token)
        span.end()


def test_extract_w3c_with_non_w3c_host_propagator(host):
    with extract_trace_context({"traceparent": REMOTE}):
        context = _neatlogs_root_kwargs()["context"]
        parent = trace_api.get_current_span(context).get_span_context()
        assert format(parent.trace_id, "032x") == REMOTE.split("-")[1]
        assert not trace_api.get_current_span().get_span_context().is_valid


def test_inject_does_not_export_host_baggage(host):
    provider, _ = host
    span = provider.get_tracer("neatlogs.test").start_span("caller")
    baggage_token = context_api.attach(baggage.set_baggage("host-only", "private"))
    token = attach_as_current(span, force_owned=True)
    try:
        carrier = {}
        assert inject_trace_context(carrier) is True
        assert "baggage" not in carrier
    finally:
        detach(token)
        context_api.detach(baggage_token)
        span.end()


def test_extract_preserves_w3c_tracestate_without_host_baggage(host):
    carrier = {"traceparent": REMOTE, "tracestate": "vendor=value", "baggage": "remote=private"}
    with extract_trace_context(carrier):
        context = _neatlogs_root_kwargs()["context"]
        parent = trace_api.get_current_span(context).get_span_context()
        assert parent.trace_state.to_header() == "vendor=value"
        assert baggage.get_baggage("remote", context=context) is None
