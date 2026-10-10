from types import SimpleNamespace

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from neatlogs.bedrock import wrap_bedrock_client

THINKING = "neatlogs.llm.output_messages.0.thinking"
CONTENT = "neatlogs.llm.output_messages.0.content"
MODEL = "anthropic.claude-3-7-sonnet-20250219-v1:0"
MESSAGES = [{"role": "user", "content": [{"text": "2+2?"}]}]


def _client(exporter, **methods):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    import neatlogs._wrap_utils as wu

    wu._wrapper_tracer = None
    meta = SimpleNamespace(service_model=SimpleNamespace(service_name="bedrock-runtime"))
    return wrap_bedrock_client(SimpleNamespace(meta=meta, **methods))


def _converse_response(blocks):
    return {
        "output": {"message": {"role": "assistant", "content": blocks}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 5, "outputTokens": 9, "totalTokens": 14},
    }


def _last_attrs(exporter):
    return dict(exporter.get_finished_spans()[-1].attributes)


def test_converse_keeps_reasoning_text(in_memory_span_exporter):
    blocks = [
        {"reasoningContent": {"reasoningText": {"text": "let me think", "signature": "s"}}},
        {"text": "the answer is 4"},
    ]
    client = _client(in_memory_span_exporter, converse=lambda **kw: _converse_response(blocks))
    client.converse(modelId=MODEL, messages=MESSAGES)
    attrs = _last_attrs(in_memory_span_exporter)
    assert attrs[THINKING] == "let me think"
    assert attrs[CONTENT] == "the answer is 4"


def test_converse_without_reasoning_has_no_thinking(in_memory_span_exporter):
    client = _client(
        in_memory_span_exporter,
        converse=lambda **kw: _converse_response([{"text": "the answer is 4"}]),
    )
    client.converse(modelId=MODEL, messages=MESSAGES)
    attrs = _last_attrs(in_memory_span_exporter)
    assert THINKING not in attrs
    assert attrs[CONTENT] == "the answer is 4"


def test_converse_stream_keeps_reasoning_text(in_memory_span_exporter):
    events = [
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": "let me"}},
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": " think"}},
            }
        },
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "the answer is 4"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    client = _client(in_memory_span_exporter, converse_stream=lambda **kw: {"stream": iter(events)})
    list(client.converse_stream(modelId=MODEL, messages=MESSAGES)["stream"])
    attrs = _last_attrs(in_memory_span_exporter)
    assert attrs[THINKING] == "let me think"
    assert attrs[CONTENT] == "the answer is 4"


def test_converse_stream_without_reasoning_has_no_thinking(in_memory_span_exporter):
    events = [
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "the answer is 4"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    client = _client(in_memory_span_exporter, converse_stream=lambda **kw: {"stream": iter(events)})
    list(client.converse_stream(modelId=MODEL, messages=MESSAGES)["stream"])
    attrs = _last_attrs(in_memory_span_exporter)
    assert THINKING not in attrs
    assert attrs[CONTENT] == "the answer is 4"
