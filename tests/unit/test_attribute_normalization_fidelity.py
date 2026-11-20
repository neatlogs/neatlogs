"""Fidelity of indexed messages, tool calls, kinds, tokens and stream order."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

import neatlogs
from neatlogs.core.attribute_processor import (
    MAX_INDEXED_ITEMS,
    UnifiedAttributeProcessor,
)
from neatlogs.core.choice_accumulator import ChoiceAccumulator
from neatlogs.core.telemetry_v2 import normalize_span_v2
from neatlogs.span_kinds.mapping import infer_span_kind_from_name

_MAPPING = json.loads(
    (Path(neatlogs.__file__).resolve().parent / "config" / "attribute-mapping.json").read_text()
)


def _processor() -> UnifiedAttributeProcessor:
    return UnifiedAttributeProcessor(mapping_config=_MAPPING, debug=False)


def _raw_span(attributes: dict, name: str = "op", events: list | None = None):
    return SimpleNamespace(
        kind=SpanKind.INTERNAL,
        name=name,
        attributes=attributes,
        resource=SimpleNamespace(attributes={}),
        instrumentation_scope=None,
        events=events or [],
        links=[],
        start_time=0,
        end_time=1_000_000,
        context=SimpleNamespace(trace_id=1, span_id=2),
        parent=None,
        status=SimpleNamespace(status_code=StatusCode.UNSET, description=None),
    )


def _canonical(unified: dict, name: str = "llm"):
    span = _raw_span(unified, name=name)
    return normalize_span_v2(span).to_dict()


class TestMultiChoiceToolCalls:
    def test_tool_calls_from_different_choices_are_all_preserved(self):
        attrs = {
            "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_zero",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "alpha",
            "llm.output_messages.1.message.tool_calls.0.tool_call.id": "call_one",
            "llm.output_messages.1.message.tool_calls.0.tool_call.function.name": "beta",
            "llm.output_messages.1.message.tool_calls.1.tool_call.id": "call_two",
            "llm.output_messages.1.message.tool_calls.1.tool_call.function.name": "gamma",
        }

        out = _processor()._normalize_conventions(_raw_span({}), dict(attrs))

        calls = {
            key: value
            for key, value in out.items()
            if key.startswith("llm.tool_calls.") and key.endswith(".name")
        }
        assert calls == {
            "llm.tool_calls.0.name": "alpha",
            "llm.tool_calls.1.name": "beta",
            "llm.tool_calls.2.name": "gamma",
        }
        assert out["llm.tool_calls.0.choice_index"] == 0
        assert out["llm.tool_calls.1.choice_index"] == 1
        assert out["llm.tool_calls.2.choice_index"] == 1
        assert out["llm.tool_calls.1.tool_call_index"] == 0
        assert out["llm.tool_calls.2.tool_call_index"] == 1

    def test_single_choice_keeps_previous_flat_index_contract(self):
        attrs = {
            "llm.output_messages.0.message.tool_calls.0.tool_call.id": "first",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "alpha",
            "llm.output_messages.0.message.tool_calls.1.tool_call.id": "second",
            "llm.output_messages.0.message.tool_calls.1.tool_call.function.name": "beta",
        }

        out = _processor()._normalize_conventions(_raw_span({}), dict(attrs))

        assert out["llm.tool_calls.0.name"] == "alpha"
        assert out["llm.tool_calls.1.name"] == "beta"
        assert out["llm.tool_calls.0.choice_index"] == 0

    def test_choice_coordinates_reach_canonical_response(self):
        attrs = {
            "neatlogs.span.kind": "llm",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "first",
            "llm.output_messages.1.message.role": "assistant",
            "llm.output_messages.1.message.content": "second",
            "llm.output_messages.1.message.tool_calls.0.tool_call.id": "call_one",
            "llm.output_messages.1.message.tool_calls.0.tool_call.function.name": "beta",
        }

        unified = _processor().process(_raw_span(dict(attrs)))
        choices = _canonical(unified)["semantic"]["response"]["choices"]

        assert [choice["choice_index"] for choice in choices] == [0, 1]
        assert [tool["name"] for tool in choices[1]["message"]["tool_calls"]] == ["beta"]


class TestIndexedMappingGapsAndBounds:
    def test_sole_message_at_index_one_is_mapped(self):
        attrs = {
            "openinference.span.kind": "LLM",
            "llm.output_messages.1.message.role": "assistant",
            "llm.output_messages.1.message.content": "only msg",
        }

        out = _processor().process(_raw_span(attrs))

        assert out["neatlogs.llm.output_messages.1.content"] == "only msg"

    def test_items_after_a_gap_are_mapped(self):
        attrs = {
            "openinference.span.kind": "LLM",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "zero",
            "llm.output_messages.2.message.role": "assistant",
            "llm.output_messages.2.message.content": "two",
        }

        out = _processor().process(_raw_span(attrs))

        assert out["neatlogs.llm.output_messages.0.content"] == "zero"
        assert out["neatlogs.llm.output_messages.2.content"] == "two"
        assert "neatlogs.llm.output_messages.1.content" not in out

    def test_long_conversations_are_not_truncated_at_twenty(self):
        attrs = {"openinference.span.kind": "LLM"}
        for index in range(25):
            attrs[f"llm.input_messages.{index}.message.role"] = "user"
            attrs[f"llm.input_messages.{index}.message.content"] = f"msg-{index}"

        out = _processor().process(_raw_span(attrs))

        contents = [key for key in out if key.endswith(".content")]
        assert len(contents) == 25
        assert out["neatlogs.llm.input_messages.24.content"] == "msg-24"

    def test_non_indexed_sources_still_map_once(self):
        attrs = {
            "openinference.span.kind": "RETRIEVER",
            "db.query.result.id": "doc-1",
            "db.query.result.document": "text",
        }

        out = _processor().process(_raw_span(attrs))

        assert out["neatlogs.retriever.documents.0.id"] == "doc-1"
        assert out["neatlogs.retriever.documents.0.content"] == "text"

    def test_index_fanout_is_bounded(self):
        assert MAX_INDEXED_ITEMS == 128
        attrs = {"openinference.span.kind": "LLM"}
        for index in range(MAX_INDEXED_ITEMS + 5):
            attrs[f"llm.input_messages.{index}.message.role"] = "user"
            attrs[f"llm.input_messages.{index}.message.content"] = f"msg-{index}"

        out = _processor().process(_raw_span(attrs))

        contents = [key for key in out if key.endswith(".content")]
        assert len(contents) == MAX_INDEXED_ITEMS
        assert f"neatlogs.llm.input_messages.{MAX_INDEXED_ITEMS}.content" not in out


class TestKindKeyedIO:
    @pytest.mark.parametrize(
        "name,kind",
        [
            ("score_relevance", "evaluator"),
            ("answer_question_agent", "agent"),
            ("call_weather_tool", "tool"),
        ],
    )
    def test_name_inferred_kinds_produce_namespaced_io(self, name, kind):
        attrs = {"input.value": '{"q": "hi"}', "output.value": '{"a": "ok"}'}

        out = _processor().process(_raw_span(attrs, name=name))

        assert out["neatlogs.span.kind"] == kind
        assert out[f"neatlogs.{kind}.input"] == '{"q": "hi"}'
        assert out[f"neatlogs.{kind}.output"] == '{"a": "ok"}'

    def test_explicit_kind_takes_precedence_over_name(self):
        attrs = {
            "openinference.span.kind": "LLM",
            "input.value": "hello",
            "output.value": "world",
        }

        out = _processor().process(_raw_span(attrs, name="score_relevance"))

        assert out["neatlogs.span.kind"] == "llm"
        assert out["neatlogs.llm.input"] == "hello"
        assert out["neatlogs.llm.output"] == "world"


class TestNumericStringIndices:
    def _names(self, indices):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        span = provider.get_tracer("test").start_span("llm")
        accumulator = ChoiceAccumulator()
        for index, name in zip(indices, ("first", "second")):
            accumulator.add_chunk(
                span,
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "id": name,
                                        "type": "function",
                                        "function": {"name": name, "arguments": "{}"},
                                    }
                                ]
                            },
                        }
                    ]
                },
            )
        accumulator.apply(span)
        span.end()
        attributes = dict(exporter.get_finished_spans()[0].attributes)
        return [
            value
            for key, value in attributes.items()
            if key.startswith("neatlogs.llm.tool_calls.") and key.endswith(".name")
        ]

    def test_numeric_string_tool_indices_are_preserved(self):
        assert self._names(("0", "1")) == ["first", "second"]

    def test_non_numeric_index_falls_back_to_position(self):
        assert self._names(("bogus", "bogus")) == ["second"]

    def test_numeric_string_choice_index_is_preserved(self):
        accumulator = ChoiceAccumulator()
        accumulator.add_response(
            {
                "choices": [
                    {
                        "index": "1",
                        "message": {"role": "assistant", "content": "second choice"},
                    }
                ]
            }
        )
        span = SimpleNamespace(set_attribute=lambda *_: None)
        accumulator.apply(span)
        assert 1 in accumulator.choices
        assert 0 not in accumulator.choices


class TestSpanKindNameInferencePrecedence:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("chat_tool_call", "TOOL"),
            ("chat_history_retrieval", "RETRIEVER"),
            ("rerank_search_results", "RERANKER"),
            ("chat_completion", "LLM"),
            ("openai_chat", "LLM"),
            ("embed_query", "EMBEDDING"),
            ("score_answers", "EVALUATOR"),
        ],
    )
    def test_specific_operations_outscore_generic_llm_keywords(self, name, expected):
        assert infer_span_kind_from_name(name) == expected

    def test_processor_uses_inferred_kind(self):
        out = _processor().process(_raw_span({}, name="chat_tool_call"))
        assert out["neatlogs.span.kind"] == "tool"


class TestCrewAITokenFallback:
    def test_missing_counts_are_filled_without_overwriting_authoritative_values(self):
        attrs = {
            "neatlogs.crew.token_usage": (
                "total_tokens=67305 prompt_tokens=46983 "
                "cached_prompt_tokens=7 completion_tokens=20322 successful_requests=27"
            ),
            "llm.token_count.total": 999,
        }

        _processor()._add_crewai_token_usage_fallback(attrs)

        assert attrs["llm.token_count.prompt"] == 46983
        assert attrs["llm.token_count.completion"] == 20322
        assert attrs["llm.token_count.prompt_details.cache_read"] == 7
        assert attrs["llm.token_count.total"] == 999

    def test_existing_prompt_value_is_preserved(self):
        attrs = {
            "neatlogs.crew.token_usage": "prompt_tokens=1 completion_tokens=2 total_tokens=3",
            "llm.token_count.prompt": 42,
        }

        _processor()._add_crewai_token_usage_fallback(attrs)

        assert attrs["llm.token_count.prompt"] == 42
        assert attrs["llm.token_count.completion"] == 2
        assert attrs["llm.token_count.total"] == 3


class TestStreamSequence:
    def test_recorded_chunk_index_is_used(self):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        span = provider.get_tracer("test").start_span("llm")
        span.set_attribute("neatlogs.span.kind", "llm")
        span.add_event("exception", {})
        span.add_event(
            "neatlogs.stream.chunk",
            {"neatlogs.stream.chunk.index": 0, "neatlogs.stream.chunk.summary": "a"},
        )
        span.add_event(
            "neatlogs.stream.chunk",
            {"neatlogs.stream.chunk.index": 1, "neatlogs.stream.chunk.summary": "b"},
        )
        span.end()

        events = normalize_span_v2(exporter.get_finished_spans()[0]).to_dict()["semantic"][
            "stream"
        ]["events"]
        assert [(event["sequence"], event["value"]) for event in events] == [(0, "a"), (1, "b")]

    def test_event_position_fallback_when_index_is_missing(self):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        span = provider.get_tracer("test").start_span("llm")
        span.set_attribute("neatlogs.span.kind", "llm")
        span.add_event("neatlogs.stream.chunk", {"neatlogs.stream.chunk.summary": "a"})
        span.end()

        events = normalize_span_v2(exporter.get_finished_spans()[0]).to_dict()["semantic"][
            "stream"
        ]["events"]
        assert events[0]["sequence"] == 0
