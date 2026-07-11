"""
Framework-agnostic cross-step ``trace_id=`` grouping.

Every surface — @span, trace(), langchain_handler(), openai_agents_processor() —
emits its OWN real (parentless) root when handed a shared ``trace_id=G`` from
``neatlogs.new_trace_id()``. All roots then share G, so the backend's
``normalizeTopLevelRowsUnderCanonicalRoot`` collapses them into ONE trace.
Grouping is by shared trace_id, NOT by a business key or by API. Asserts on real
spans captured via ``InMemorySpanExporter`` — no mocks, no fake LLM calls.
"""

import asyncio
from uuid import uuid4

from opentelemetry import trace

import neatlogs
from neatlogs.core.context import trace as nl_trace
from neatlogs.decorators.orchestration import span as neatlogs_span


def _install(tracer_provider):
    trace.set_tracer_provider(tracer_provider)


def _hex(span) -> str:
    return format(span.context.trace_id, "032x")


def _roots(spans):
    return [s for s in spans if s.parent is None]


# ---------------------------------------------------------------------------
# @span decorator
# ---------------------------------------------------------------------------


def test_span_root_forced_onto_shared_trace_id(
    tracer_provider, in_memory_span_exporter
):
    _install(tracer_provider)
    G = neatlogs.new_trace_id()

    @neatlogs_span(kind="WORKFLOW", trace_id=G)
    def work():
        @neatlogs_span(kind="CHAIN")
        def inner():
            return 1

        return inner()

    work()

    spans = in_memory_span_exporter.get_finished_spans()
    roots = _roots(spans)
    # One forced root, and the nested child inherits its trace_id.
    assert len(roots) == 1
    assert {_hex(s) for s in spans} == {G}


def test_span_child_ignores_trace_id(tracer_provider, in_memory_span_exporter):
    """trace_id only takes effect on the ROOT span; a nested @span inherits its
    parent's trace regardless of any trace_id passed to it."""
    _install(tracer_provider)
    G = neatlogs.new_trace_id()
    other = neatlogs.new_trace_id()

    @neatlogs_span(kind="WORKFLOW", trace_id=G)
    def outer():
        @neatlogs_span(kind="CHAIN", trace_id=other)
        def inner():
            return 1

        return inner()

    outer()

    spans = in_memory_span_exporter.get_finished_spans()
    assert {_hex(s) for s in spans} == {G}


# ---------------------------------------------------------------------------
# trace() context manager
# ---------------------------------------------------------------------------


def test_trace_root_forced_onto_shared_trace_id(
    tracer_provider, in_memory_span_exporter
):
    _install(tracer_provider)
    G = neatlogs.new_trace_id()

    with nl_trace(name="pipeline", trace_id=G):
        with nl_trace(name="child_step"):  # nested → inherits parent trace
            pass

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(_roots(spans)) == 1
    assert {_hex(s) for s in spans} == {G}


# ---------------------------------------------------------------------------
# langchain_handler()
# ---------------------------------------------------------------------------


def _drive_langchain_llm(handler):
    """Drive a bare llm.invoke run through the callback handler: a chat-model
    start with no parent (→ auto-root) followed by an llm end."""
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    async def run():
        rid = uuid4()
        await handler.on_chat_model_start({"id": ["ChatOpenAI"]}, [[]], run_id=rid)
        res = LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="hi"))]]
        )
        await handler.on_llm_end(res, run_id=rid)

    asyncio.run(run())


def test_langchain_handler_root_forced_onto_shared_trace_id(
    tracer_provider, in_memory_span_exporter
):
    import pytest

    pytest.importorskip("langchain_core")
    _install(tracer_provider)
    G = neatlogs.new_trace_id()

    handler = neatlogs.langchain_handler(trace_id=G)
    _drive_langchain_llm(handler)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(_roots(spans)) == 1
    assert {_hex(s) for s in spans} == {G}


# ---------------------------------------------------------------------------
# openai_agents_processor()
# ---------------------------------------------------------------------------


class _FakeTrace:
    name = "support-bot"
    trace_id = "trace_abc"


class _FakeGenerationData:
    type = "generation"
    model = "gpt-4.1"
    input = [{"role": "user", "content": "hello"}]
    output = [{"role": "assistant", "content": "hi"}]
    usage = {"input_tokens": 5, "output_tokens": 2}


class _FakeSpan:
    span_id = "span_1"
    parent_id = None
    trace_id = "trace_abc"
    span_data = _FakeGenerationData()
    error = None


def test_openai_agents_processor_root_forced_onto_shared_trace_id(
    tracer_provider, in_memory_span_exporter
):
    _install(tracer_provider)
    G = neatlogs.new_trace_id()

    proc = neatlogs.openai_agents_processor(trace_id=G)
    tr = _FakeTrace()
    proc.on_trace_start(tr)
    sp = _FakeSpan()
    proc.on_span_start(sp)
    proc.on_span_end(sp)
    proc.on_trace_end(tr)

    spans = in_memory_span_exporter.get_finished_spans()
    assert {_hex(s) for s in _roots(spans)} == {G}
    assert {_hex(s) for s in spans} == {G}


# ---------------------------------------------------------------------------
# The money test: MULTIPLE surfaces, SAME trace_id → ONE trace, N real roots
# ---------------------------------------------------------------------------


def test_multiple_surfaces_share_one_trace(
    tracer_provider, in_memory_span_exporter
):
    import pytest

    pytest.importorskip("langchain_core")
    _install(tracer_provider)
    G = neatlogs.new_trace_id()

    @neatlogs_span(kind="WORKFLOW", trace_id=G)
    def step_a():
        return "a"

    step_a()
    with nl_trace(name="step_b", trace_id=G):
        pass
    _drive_langchain_llm(neatlogs.langchain_handler(trace_id=G))

    spans = in_memory_span_exporter.get_finished_spans()
    roots = _roots(spans)
    # Three independent surfaces → three real parentless roots, but ALL sharing G
    # → the backend collapses them into ONE trace.
    assert len(roots) == 3
    assert all(_hex(r) == G for r in roots)
    assert {_hex(s) for s in spans} == {G}


# ---------------------------------------------------------------------------
# Grouping is opt-in: no trace_id → independent traces
# ---------------------------------------------------------------------------


def test_no_trace_id_yields_independent_traces(
    tracer_provider, in_memory_span_exporter
):
    _install(tracer_provider)

    @neatlogs_span(kind="WORKFLOW")
    def a():
        return 1

    @neatlogs_span(kind="WORKFLOW")
    def b():
        return 2

    a()
    b()

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(_roots(spans)) == 2
    assert len({_hex(s) for s in spans}) == 2
