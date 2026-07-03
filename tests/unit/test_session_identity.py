"""
Tests that session identity (``neatlogs.session.id``) is stamped on the trace
ROOT span only, via ``trace(session_id=...)`` and ``@span(session_id=...)``, and
that nested child spans do not carry it.
"""

from opentelemetry import trace

import neatlogs
from neatlogs.core.context import trace as nl_trace
from neatlogs.core.end_user import END_USER_ID_KEY
from neatlogs.core.identity import identify
from neatlogs.core.session import SESSION_ID_KEY, apply_session_attributes
from neatlogs.decorators.orchestration import span as neatlogs_span


def _install(tracer_provider):
    trace.set_tracer_provider(tracer_provider)


# ---------------------------------------------------------------------------
# Helper-level behavior
# ---------------------------------------------------------------------------


def test_apply_session_attributes_root(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("root") as span:
        apply_session_attributes(span, "chat_123", is_root=True)

    spans = in_memory_span_exporter.get_finished_spans()
    assert spans[0].attributes.get(SESSION_ID_KEY) == "chat_123"


def test_apply_session_attributes_non_root_ignored(
    tracer_provider, in_memory_span_exporter
):
    _install(tracer_provider)
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("child") as span:
        apply_session_attributes(span, "chat_123", is_root=False)

    spans = in_memory_span_exporter.get_finished_spans()
    assert SESSION_ID_KEY not in spans[0].attributes


def test_apply_session_attributes_empty_noop(
    tracer_provider, in_memory_span_exporter
):
    _install(tracer_provider)
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("root") as span:
        apply_session_attributes(span, None, is_root=True)

    spans = in_memory_span_exporter.get_finished_spans()
    assert SESSION_ID_KEY not in spans[0].attributes


# ---------------------------------------------------------------------------
# @span decorator
# ---------------------------------------------------------------------------


def test_span_decorator_sets_session_on_root(
    tracer_provider, in_memory_span_exporter
):
    _install(tracer_provider)

    @neatlogs_span(kind="WORKFLOW", session_id="chat_123")
    def handle_turn():
        return 42

    handle_turn()

    spans = in_memory_span_exporter.get_finished_spans()
    root = next(s for s in spans if s.name == "handle_turn")
    assert root.attributes.get(SESSION_ID_KEY) == "chat_123"


def test_span_decorator_child_does_not_inherit_session(
    tracer_provider, in_memory_span_exporter
):
    _install(tracer_provider)

    @neatlogs_span(kind="CHAIN")
    def child():
        return 1

    @neatlogs_span(kind="WORKFLOW", session_id="chat_123")
    def handle_turn():
        return child()

    handle_turn()

    spans = in_memory_span_exporter.get_finished_spans()
    root = next(s for s in spans if s.name == "handle_turn")
    leaf = next(s for s in spans if s.name == "child")
    assert root.attributes.get(SESSION_ID_KEY) == "chat_123"
    assert SESSION_ID_KEY not in leaf.attributes


# ---------------------------------------------------------------------------
# trace() context manager
# ---------------------------------------------------------------------------


def test_trace_sets_session_on_root(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    with nl_trace("chat_turn", session_id="chat_123"):
        pass

    spans = in_memory_span_exporter.get_finished_spans()
    root = next(s for s in spans if s.name == "chat_turn")
    assert root.attributes.get(SESSION_ID_KEY) == "chat_123"


# ---------------------------------------------------------------------------
# identify() context (the wrapper-only path)
# ---------------------------------------------------------------------------


def test_identify_context_stamps_root(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    # No per-call args: a trace() inside identify() inherits from context.
    with identify(session_id="ctx_session", end_user_id="ctx_user"):
        with nl_trace("turn"):
            pass

    spans = in_memory_span_exporter.get_finished_spans()
    root = next(s for s in spans if s.name == "turn")
    assert root.attributes.get(SESSION_ID_KEY) == "ctx_session"
    assert root.attributes.get(END_USER_ID_KEY) == "ctx_user"


def test_percall_arg_wins_over_identify(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    with identify(session_id="ctx_session", end_user_id="ctx_user"):
        with nl_trace("turn", session_id="explicit_session"):
            pass

    spans = in_memory_span_exporter.get_finished_spans()
    root = next(s for s in spans if s.name == "turn")
    # explicit per-call session wins; end-user still inherited from context.
    assert root.attributes.get(SESSION_ID_KEY) == "explicit_session"
    assert root.attributes.get(END_USER_ID_KEY) == "ctx_user"


def test_identify_restores_on_exit(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    with identify(session_id="ctx_session"):
        pass
    # Outside the block, a new trace must NOT carry the session.
    with nl_trace("after"):
        pass

    spans = in_memory_span_exporter.get_finished_spans()
    root = next(s for s in spans if s.name == "after")
    assert SESSION_ID_KEY not in root.attributes


# ---------------------------------------------------------------------------
# Span-processor fallback: identify() reaches ANY root span, including framework
# auto-roots (langchain, openai-agents, strands, ...) whose roots do NOT stamp
# identity themselves. NeatlogsSpanProcessor.on_start fills it in as a fallback;
# explicit trace()/@span values still override.
# ---------------------------------------------------------------------------


def _provider_with_neatlogs_processor():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from neatlogs.core.span_processor import NeatlogsSpanProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(NeatlogsSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_processor_stamps_identify_on_bare_framework_root():
    """A root span created WITHOUT explicit identity (e.g. a framework auto-root)
    picks up identify()'s session + end-user via the processor fallback."""
    provider, exporter = _provider_with_neatlogs_processor()
    _install(provider)
    tracer = trace.get_tracer(__name__)

    with identify(session_id="conv_fw", end_user_id="u_fw"):
        # Mimic a framework auto-root: a plain root span, no apply_* call.
        root = tracer.start_span(
            "LangGraph", attributes={"neatlogs.span.kind": "workflow"}
        )
        root.end()

    root = next(s for s in exporter.get_finished_spans() if s.name == "LangGraph")
    assert root.attributes.get(SESSION_ID_KEY) == "conv_fw"
    assert root.attributes.get(END_USER_ID_KEY) == "u_fw"


def test_processor_skips_child_spans():
    """The processor fallback stamps only roots; child spans never carry identity."""
    provider, exporter = _provider_with_neatlogs_processor()
    _install(provider)
    tracer = trace.get_tracer(__name__)

    with identify(session_id="conv_fw"):
        with tracer.start_as_current_span(
            "root", attributes={"neatlogs.span.kind": "workflow"}
        ):
            with tracer.start_as_current_span(
                "child", attributes={"neatlogs.span.kind": "LLM"}
            ):
                pass

    child = next(s for s in exporter.get_finished_spans() if s.name == "child")
    assert SESSION_ID_KEY not in child.attributes


def test_processor_noop_without_identify():
    """No active identify() → the processor adds no session attribute (no phantom)."""
    provider, exporter = _provider_with_neatlogs_processor()
    _install(provider)
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span(
        "bare", attributes={"neatlogs.span.kind": "workflow"}
    ):
        pass

    root = next(s for s in exporter.get_finished_spans() if s.name == "bare")
    assert SESSION_ID_KEY not in root.attributes
