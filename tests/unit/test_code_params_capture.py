"""
Tests that @neatlogs.span() decorators auto-capture code location attributes
(code.file.path, code.function.name, code.line.number, code.namespace) on the
resulting span, and that auto-instrumented spans (not decorated) do NOT carry
these attributes.
"""

import asyncio
import inspect

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

import neatlogs
from neatlogs.decorators._base import _decorate_span

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCodeParamsOnDecoratedSpans:
    def test_sync_decorator_sets_code_file_path(self):
        provider, exporter = _make_provider()
        from opentelemetry import trace

        trace.set_tracer_provider(provider)

        @_decorate_span(openinference_kind="CHAIN")
        def my_function():
            return 42

        my_function()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert "code.file.path" in attrs
        assert __file__.rstrip("c") in attrs["code.file.path"].rstrip("c")  # .py or .pyc

    def test_sync_decorator_sets_code_function_name(self):
        provider, exporter = _make_provider()
        from opentelemetry import trace

        trace.set_tracer_provider(provider)

        @_decorate_span(openinference_kind="TOOL")
        def my_tool_function():
            pass

        my_tool_function()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert (
            attrs.get("code.function.name")
            == "TestCodeParamsOnDecoratedSpans.test_sync_decorator_sets_code_function_name.<locals>.my_tool_function"
        )

    def test_sync_decorator_sets_code_line_number(self):
        provider, exporter = _make_provider()
        from opentelemetry import trace

        trace.set_tracer_provider(provider)

        @_decorate_span(openinference_kind="AGENT")
        def my_agent():
            pass

        expected_lineno = inspect.getsourcelines(
            my_agent.__wrapped__ if hasattr(my_agent, "__wrapped__") else my_agent
        )[1]

        my_agent()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert "code.line.number" in attrs
        assert isinstance(attrs["code.line.number"], int)
        assert attrs["code.line.number"] > 0

    def test_sync_decorator_sets_code_namespace(self):
        provider, exporter = _make_provider()
        from opentelemetry import trace

        trace.set_tracer_provider(provider)

        @_decorate_span(openinference_kind="WORKFLOW")
        def my_workflow():
            pass

        my_workflow()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert attrs.get("code.namespace") == __name__

    def test_async_decorator_sets_code_attributes(self):
        provider, exporter = _make_provider()
        from opentelemetry import trace

        trace.set_tracer_provider(provider)

        @_decorate_span(openinference_kind="CHAIN")
        async def my_async_function():
            return "result"

        asyncio.get_event_loop().run_until_complete(my_async_function())

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert "code.file.path" in attrs
        assert "code.function.name" in attrs
        assert "code.line.number" in attrs
        assert "code.namespace" in attrs

    def test_caller_attributes_not_overwritten_by_code_attrs(self):
        """User-supplied attributes should not be overwritten by code params."""
        provider, exporter = _make_provider()
        from opentelemetry import trace

        trace.set_tracer_provider(provider)

        @_decorate_span(openinference_kind="TOOL", attributes={"custom.key": "custom_value"})
        def my_tool():
            pass

        my_tool()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert attrs.get("custom.key") == "custom_value"
        # code attrs are also present
        assert "code.file.path" in attrs
        assert "code.function.name" in attrs

    def test_code_attrs_do_not_appear_on_plain_otel_span(self):
        """Spans created without the neatlogs decorator should not have code params."""
        provider, exporter = _make_provider()
        from opentelemetry import trace

        trace.set_tracer_provider(provider)

        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("plain-span") as span:
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert "code.file.path" not in attrs
        assert "code.function.name" not in attrs
        assert "code.line.number" not in attrs
