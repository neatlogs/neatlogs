"""The tracing wrapper must never retry a provider call on its own."""

import asyncio
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import neatlogs.claude_agent_sdk as integration


@pytest.fixture
def tracing(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(integration, "get_tracer", lambda: provider.get_tracer("fixture"))
    monkeypatch.setattr(integration, "is_suppressed", lambda: False)
    return exporter


@pytest.mark.parametrize(
    "failure", [ValueError("provider"), RuntimeError("provider"), TypeError("provider")]
)
def test_original_failure_is_not_retried(monkeypatch, tracing, failure):
    original = Mock(side_effect=failure)
    with pytest.raises(type(failure)) as caught:
        integration._wrap_query(original)(prompt="fixture")
    assert caught.value is failure
    assert original.call_count == 1
    assert len(tracing.get_finished_spans()) == 1


def test_constructor_failure_returns_the_existing_result(monkeypatch, tracing):
    result = object()
    original = Mock(return_value=result)
    monkeypatch.setattr(integration, "_TracingQuery", Mock(side_effect=RuntimeError("telemetry")))
    assert integration._wrap_query(original)(prompt="fixture") is result
    assert original.call_count == 1
    assert len(tracing.get_finished_spans()) == 1


def test_detach_failure_returns_existing_result(monkeypatch, tracing):
    result = object()
    original = Mock(return_value=result)
    real_detach = integration._nl_detach

    def detach_then_fail(token):
        real_detach(token)
        raise RuntimeError("telemetry")

    monkeypatch.setattr(integration, "_nl_detach", detach_then_fail)
    wrapped = integration._wrap_query(original)(prompt="fixture")
    assert wrapped is result or wrapped._q is result
    assert original.call_count == 1


def test_setup_failure_calls_original_once(monkeypatch, tracing):
    original = Mock(return_value=object())
    monkeypatch.setattr(integration, "get_tracer", Mock(side_effect=RuntimeError("telemetry")))
    assert integration._wrap_query(original)(prompt="fixture") is original.return_value
    assert original.call_count == 1


def test_suppressed_calls_original_once(monkeypatch, tracing):
    original = Mock(return_value=object())
    monkeypatch.setattr(integration, "is_suppressed", lambda: True)
    assert integration._wrap_query(original)(prompt="fixture") is original.return_value
    assert original.call_count == 1
    assert not tracing.get_finished_spans()


def test_success_remains_wrapped(monkeypatch, tracing):
    result = object()
    original = Mock(return_value=result)
    wrapped = integration._wrap_query(original)(prompt="fixture")
    assert wrapped._q is result
    assert original.call_count == 1
    wrapped._finalize("ok")
    assert len(tracing.get_finished_spans()) == 1


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(), asyncio.CancelledError()])
def test_baseexception_is_not_retried_and_span_is_closed(monkeypatch, tracing, failure):
    original = Mock(side_effect=failure)
    with pytest.raises(type(failure)) as caught:
        integration._wrap_query(original)(prompt="fixture")
    assert caught.value is failure
    assert original.call_count == 1
    assert len(tracing.get_finished_spans()) == 1


def test_attach_failure_closes_span_and_calls_once(monkeypatch, tracing):
    original = Mock(return_value=object())
    monkeypatch.setattr(
        integration, "attach_as_current", Mock(side_effect=RuntimeError("telemetry"))
    )
    assert integration._wrap_query(original)(prompt="fixture") is original.return_value
    assert original.call_count == 1
    assert len(tracing.get_finished_spans()) == 1


def test_cleanup_failure_does_not_mask_original_error(monkeypatch, tracing):
    error = ValueError("provider")
    original = Mock(side_effect=error)
    real_detach = integration._nl_detach

    def detach_then_fail(token):
        real_detach(token)
        raise RuntimeError("telemetry")

    monkeypatch.setattr(integration, "_nl_detach", detach_then_fail)
    with pytest.raises(ValueError) as caught:
        integration._wrap_query(original)(prompt="fixture")
    assert caught.value is error
    assert original.call_count == 1
    assert len(tracing.get_finished_spans()) == 1
