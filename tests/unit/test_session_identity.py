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
