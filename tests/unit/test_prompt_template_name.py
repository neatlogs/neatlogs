"""Prompt family identity crosses real trace context and normalization boundaries."""

import asyncio

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from neatlogs._wrap_utils import get_provider_tracer, set_neatlogs_provider
from neatlogs.core.context import trace
from neatlogs.core.span_processor import NeatlogsSpanProcessor

KEY = "neatlogs.llm.prompt_key"


@pytest.fixture
def pipeline():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(
        NeatlogsSpanProcessor(emit_completion_markers=False, own_all_spans=True)
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_neatlogs_provider(provider)
    yield get_provider_tracer(), exporter
    provider.shutdown()


def llm(tracer, name="chat.completions", attributes=None):
    span_attributes = {"openinference.span.kind": "LLM", **(attributes or {})}
    with tracer.start_as_current_span(name, attributes=span_attributes):
        pass


@pytest.mark.parametrize("keyword", ["system_prompt_template", "prompt_template"])
def test_named_system_template_reaches_normalized_llm_child(pipeline, keyword):
    tracer, exporter = pipeline
    with trace("review-document", **{keyword: "Review {{document}}"}):
        llm(tracer)
    spans = exporter.get_finished_spans()
    child = next(s for s in spans if s.name == "chat.completions")
    assert child.attributes[KEY] == "review-document"
    assert child.attributes["neatlogs.llm.prompt_template"] == "Review {{document}}"
    assert all(KEY not in s.attributes for s in spans if s is not child)


def test_name_only_and_user_only_trace_do_not_create_system_prompt_identity(pipeline):
    tracer, exporter = pipeline
    with trace("group"):
        llm(tracer)
    with trace("user-only", user_prompt_template="Question {{question}}"):
        llm(tracer)
    assert all(KEY not in span.attributes for span in exporter.get_finished_spans())


def test_nested_context_restores_outer_identity_and_ignores_group_names(pipeline):
    tracer, exporter = pipeline
    with trace("outer", system_prompt_template="Outer"):
        with trace("group"):
            llm(tracer, "chat.group")
        with trace("inner", system_prompt_template="Inner"):
            llm(tracer, "chat.inner")
        llm(tracer, "chat.outer")
    llm(tracer, "chat.after")
    names = {s.name: s.attributes.get(KEY) for s in exporter.get_finished_spans()}
    assert {
        key: names[key] for key in ("chat.group", "chat.inner", "chat.outer", "chat.after")
    } == {
        "chat.group": "outer",
        "chat.inner": "inner",
        "chat.outer": "outer",
        "chat.after": None,
    }


def test_async_tasks_keep_independent_prompt_names(pipeline):
    tracer, exporter = pipeline

    async def run(name):
        with trace(name, system_prompt_template="Shared"):
            await asyncio.sleep(0)
            llm(tracer, "chat." + name)

    async def both():
        await asyncio.gather(run("first"), run("second"))

    asyncio.run(both())
    names = {s.name: s.attributes.get(KEY) for s in exporter.get_finished_spans()}
    assert names["chat.first"] == "first"
    assert names["chat.second"] == "second"


def test_canonical_system_keyword_wins_over_legacy_alias(pipeline):
    tracer, exporter = pipeline
    with trace("canonical", system_prompt_template="New", prompt_template="Old"):
        llm(tracer)
    child = next(s for s in exporter.get_finished_spans() if s.name == "chat.completions")
    assert child.attributes[KEY] == "canonical"
    assert child.attributes["neatlogs.llm.prompt_template"] == "New"


@pytest.mark.parametrize("attribute", ["neatlogs.llm.prompt_key", "traceloop.prompt.key"])
def test_existing_wire_prompt_key_wins_over_trace_name(pipeline, attribute):
    tracer, exporter = pipeline
    with trace("wrapper-name", system_prompt_template="Template"):
        llm(tracer, attributes={attribute: "provider-key"})
    child = next(s for s in exporter.get_finished_spans() if s.name == "chat.completions")
    assert child.attributes[KEY] == "provider-key"


def test_exception_restores_outer_identity(pipeline):
    tracer, exporter = pipeline
    with trace("outer", system_prompt_template="Outer"):
        with pytest.raises(ValueError), trace("inner", system_prompt_template="Inner"):
            raise ValueError("test")
        llm(tracer)
    child = next(s for s in exporter.get_finished_spans() if s.name == "chat.completions")
    assert child.attributes[KEY] == "outer"


def test_new_template_with_empty_name_does_not_borrow_outer_name(pipeline):
    tracer, exporter = pipeline
    with (
        trace("outer", system_prompt_template="Outer"),
        trace("", system_prompt_template="Independent"),
    ):
        llm(tracer)
    child = next(s for s in exporter.get_finished_spans() if s.name == "chat.completions")
    assert KEY not in child.attributes
    assert child.attributes["neatlogs.llm.prompt_template"] == "Independent"


def test_template_object_and_auto_root_preserve_prompt_identity(pipeline):
    from neatlogs import PromptTemplate

    tracer, exporter = pipeline
    with trace(
        "named-object",
        kind="TOOL",
        system_prompt_template=PromptTemplate("Review {{document}}"),
    ):
        llm(tracer)
    spans = exporter.get_finished_spans()
    child = next(s for s in spans if s.name == "chat.completions")
    assert child.attributes[KEY] == "named-object"
    roots = [s for s in spans if s.attributes.get("neatlogs.auto_root") is True]
    assert len(roots) == 1
    wrapper = next(s for s in spans if s.name == "named-object")
    assert wrapper.parent.span_id == roots[0].context.span_id
    assert child.parent.span_id == wrapper.context.span_id
    assert KEY not in roots[0].attributes


def test_prompt_identity_trims_name_without_renaming_the_trace_span(pipeline):
    tracer, exporter = pipeline
    with trace("  named prompt  ", system_prompt_template="Template"):
        llm(tracer)
    spans = exporter.get_finished_spans()
    child = next(s for s in spans if s.name == "chat.completions")
    assert child.attributes[KEY] == "named prompt"
    assert any(s.name == "  named prompt  " for s in spans)


def test_thread_contexts_keep_names_separate(pipeline):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    tracer, exporter = pipeline
    barrier = Barrier(2)

    def run(name):
        with trace(name, system_prompt_template="Shared"):
            barrier.wait(timeout=5)
            llm(tracer, "chat." + name)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run, name) for name in ("first", "second")]
        for future in futures:
            future.result(timeout=10)
    names = {s.name: s.attributes.get(KEY) for s in exporter.get_finished_spans()}
    assert names["chat.first"] == "first"
    assert names["chat.second"] == "second"


def test_streaming_child_keeps_identity_for_its_full_lifecycle(pipeline):
    tracer, exporter = pipeline
    with (
        trace("streaming", system_prompt_template="Template"),
        tracer.start_as_current_span(
            "chat.completions.stream", attributes={"openinference.span.kind": "LLM"}
        ) as child,
    ):
        for chunk in ("first", "second"):
            child.add_event("chunk", {"text": chunk})
    child = next(s for s in exporter.get_finished_spans() if s.name == "chat.completions.stream")
    assert child.attributes[KEY] == "streaming"
    assert len(child.events) == 2
