import sys
from types import ModuleType

from openinference.instrumentation import OITracer, TraceConfig
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace.propagation import _SPAN_KEY

from neatlogs._wrap_utils import set_neatlogs_provider
from neatlogs.instrumentation.openinference_isolation import (
    provider_for_openinference,
)


def _provider_with_exporter():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_openinference_current_spans_stay_private_from_foreign_pipeline():
    private_provider, private_exporter = _provider_with_exporter()
    foreign_provider, foreign_exporter = _provider_with_exporter()
    set_neatlogs_provider(private_provider)

    try:
        oi_provider = provider_for_openinference(private_provider)
        oi_tracer = OITracer(
            oi_provider.get_tracer("openinference.instrumentation.test"),
            TraceConfig(),
        )
        foreign_tracer = foreign_provider.get_tracer("foreign.openlit")

        with foreign_tracer.start_as_current_span("foreign-root") as foreign_root:
            with oi_tracer.start_as_current_span("oi-root") as oi_root:
                assert trace_api.get_current_span() is foreign_root

                with foreign_tracer.start_as_current_span("foreign-child"):
                    with oi_tracer.start_as_current_span("oi-child"):
                        assert trace_api.get_current_span().get_span_context().span_id != (
                            oi_root.get_span_context().span_id
                        )

        private_spans = {span.name: span for span in private_exporter.get_finished_spans()}
        foreign_spans = {span.name: span for span in foreign_exporter.get_finished_spans()}

        assert set(private_spans) == {"oi-root", "oi-child"}
        assert set(foreign_spans) == {"foreign-root", "foreign-child"}
        assert private_spans["oi-root"].parent is None
        assert private_spans["oi-child"].parent.span_id == private_spans["oi-root"].context.span_id
        assert foreign_spans["foreign-child"].parent.span_id == (
            foreign_spans["foreign-root"].context.span_id
        )
        assert private_spans["oi-root"].context.trace_id != (
            foreign_spans["foreign-root"].context.trace_id
        )
    finally:
        set_neatlogs_provider(None)
        private_provider.shutdown()
        foreign_provider.shutdown()


def test_direct_openinference_use_span_and_set_span_in_context_are_private():
    private_provider, private_exporter = _provider_with_exporter()
    foreign_provider, _ = _provider_with_exporter()
    set_neatlogs_provider(private_provider)

    try:
        oi_provider = provider_for_openinference(private_provider)
        oi_tracer = OITracer(
            oi_provider.get_tracer("openinference.instrumentation.direct"),
            TraceConfig(),
        )
        foreign_tracer = foreign_provider.get_tracer("foreign.langfuse")

        with foreign_tracer.start_as_current_span("foreign-root") as foreign_root:
            parent = oi_tracer.start_span("oi-parent")
            try:
                with trace_api.use_span(parent, end_on_exit=False):
                    assert trace_api.get_current_span() is foreign_root
                    child = oi_tracer.start_span("oi-child")
                    child.end()

                private_context = trace_api.set_span_in_context(parent)
                token = context_api.attach(private_context)
                try:
                    assert trace_api.get_current_span() is foreign_root
                    explicit_child = oi_tracer.start_span("oi-explicit-child")
                    explicit_child.end()
                finally:
                    context_api.detach(token)

                low_level_context = context_api.set_value(_SPAN_KEY, parent)
                token = context_api.attach(low_level_context)
                try:
                    assert trace_api.get_current_span() is foreign_root
                    low_level_child = oi_tracer.start_span("oi-low-level-child")
                    low_level_child.end()
                finally:
                    context_api.detach(token)
            finally:
                parent.end()

        spans = {span.name: span for span in private_exporter.get_finished_spans()}
        assert spans["oi-child"].parent.span_id == spans["oi-parent"].context.span_id
        assert spans["oi-explicit-child"].parent.span_id == spans["oi-parent"].context.span_id
        assert spans["oi-low-level-child"].parent.span_id == spans["oi-parent"].context.span_id
    finally:
        set_neatlogs_provider(None)
        private_provider.shutdown()
        foreign_provider.shutdown()


def test_unmarked_openinference_tracer_keeps_standard_otel_behavior():
    # The isolation patch is process-wide, but its behavior is marker-scoped.
    # An OI tracer not obtained through the private Neatlogs facade must remain
    # visible through OpenTelemetry's ordinary current-span API.
    provider, _ = _provider_with_exporter()
    set_neatlogs_provider(None)

    try:
        tracer = OITracer(
            provider.get_tracer("openinference.instrumentation.foreign"),
            TraceConfig(),
        )
        with tracer.start_as_current_span("ordinary-oi") as span:
            assert trace_api.get_current_span().get_span_context().span_id == (
                span.get_span_context().span_id
            )
    finally:
        provider.shutdown()


def test_openinference_modules_importing_get_current_span_see_the_private_span():
    private_provider, _ = _provider_with_exporter()
    foreign_provider, _ = _provider_with_exporter()
    set_neatlogs_provider(private_provider)
    module_name = "openinference.instrumentation._neatlogs_test_getter"
    adapter = ModuleType(module_name)
    adapter.get_current_span = trace_api.get_current_span
    sys.modules[module_name] = adapter

    try:
        oi_provider = provider_for_openinference(private_provider)
        oi_tracer = OITracer(
            oi_provider.get_tracer("openinference.instrumentation.imported-getter"),
            TraceConfig(),
        )
        foreign_tracer = foreign_provider.get_tracer("foreign.imported-getter")

        with foreign_tracer.start_as_current_span("foreign-root") as foreign_root:
            with oi_tracer.start_as_current_span("private-root") as private_root:
                assert trace_api.get_current_span() is foreign_root
                assert adapter.get_current_span().get_span_context().span_id == (
                    private_root.get_span_context().span_id
                )
    finally:
        sys.modules.pop(module_name, None)
        set_neatlogs_provider(None)
        private_provider.shutdown()
        foreign_provider.shutdown()
