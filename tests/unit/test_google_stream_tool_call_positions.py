import json

from neatlogs.core.choice_accumulator import GoogleStreamFinalizer


def _function_chunk(*calls):
    return {
        "candidates": [
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [{"function_call": call} for call in calls],
                },
            }
        ]
    }


def _stream(tracer_provider, in_memory_span_exporter, chunks):
    span = tracer_provider.get_tracer("neatlogs.google_genai").start_span("gemini-stream")
    finalizer = GoogleStreamFinalizer()
    for chunk in chunks:
        finalizer.on_chunk(span, chunk)
    finalizer.finish(span, duration_ms=10.0, ttft_ms=1.0)
    attrs = dict(in_memory_span_exporter.get_finished_spans()[-1].attributes)
    return {k: v for k, v in attrs.items() if k.startswith("neatlogs.llm.tool_calls.")}


def test_function_calls_in_separate_chunks_stay_separate(tracer_provider, in_memory_span_exporter):
    tools = _stream(
        tracer_provider,
        in_memory_span_exporter,
        [
            _function_chunk(
                {"id": "call_weather", "name": "get_weather", "args": {"city": "Paris"}}
            ),
            _function_chunk(
                {"id": "call_time", "name": "get_time", "args": {"tz": "Europe/Paris"}}
            ),
            {"candidates": [{"index": 0, "finish_reason": "STOP"}]},
        ],
    )

    assert tools["neatlogs.llm.tool_calls.0.id"] == "call_weather"
    assert tools["neatlogs.llm.tool_calls.0.name"] == "get_weather"
    assert json.loads(tools["neatlogs.llm.tool_calls.0.arguments"]) == {"city": "Paris"}
    assert tools["neatlogs.llm.tool_calls.0.tool_call_index"] == 0
    assert tools["neatlogs.llm.tool_calls.1.id"] == "call_time"
    assert tools["neatlogs.llm.tool_calls.1.name"] == "get_time"
    assert json.loads(tools["neatlogs.llm.tool_calls.1.arguments"]) == {"tz": "Europe/Paris"}
    assert tools["neatlogs.llm.tool_calls.1.tool_call_index"] == 1


def test_id_less_function_calls_in_separate_chunks_stay_separate(
    tracer_provider, in_memory_span_exporter
):
    tools = _stream(
        tracer_provider,
        in_memory_span_exporter,
        [
            _function_chunk({"name": "get_weather", "args": {"city": "Paris"}}),
            _function_chunk({"name": "get_time", "args": {"tz": "Europe/Paris"}}),
        ],
    )

    assert tools["neatlogs.llm.tool_calls.0.name"] == "get_weather"
    assert json.loads(tools["neatlogs.llm.tool_calls.0.arguments"]) == {"city": "Paris"}
    assert tools["neatlogs.llm.tool_calls.1.name"] == "get_time"
    assert json.loads(tools["neatlogs.llm.tool_calls.1.arguments"]) == {"tz": "Europe/Paris"}


def test_repeated_provider_id_updates_one_call_with_valid_arguments(
    tracer_provider, in_memory_span_exporter
):
    tools = _stream(
        tracer_provider,
        in_memory_span_exporter,
        [
            _function_chunk({"id": "call_weather", "name": "get_weather", "args": {"city": "Par"}}),
            _function_chunk({"id": "call_weather", "args": {"city": "Paris"}}),
        ],
    )

    assert "neatlogs.llm.tool_calls.1.id" not in tools
    assert tools["neatlogs.llm.tool_calls.0.name"] == "get_weather"
    assert json.loads(tools["neatlogs.llm.tool_calls.0.arguments"]) == {"city": "Paris"}


def test_parallel_calls_in_one_chunk_keep_their_order(tracer_provider, in_memory_span_exporter):
    tools = _stream(
        tracer_provider,
        in_memory_span_exporter,
        [
            _function_chunk(
                {"id": "a", "name": "first", "args": {"n": 1}},
                {"id": "b", "name": "second", "args": {"n": 2}},
            ),
        ],
    )

    assert tools["neatlogs.llm.tool_calls.0.name"] == "first"
    assert tools["neatlogs.llm.tool_calls.1.name"] == "second"
