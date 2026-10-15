from types import SimpleNamespace

from neatlogs.agno import _finalize_model, _finalize_model_stream


class FakeSpan:
    def __init__(self):
        self.attrs = {}

    def set_attribute(self, key, value):
        self.attrs[key] = value

    def set_status(self, *args, **kwargs):
        pass

    def end(self, *args, **kwargs):
        pass


THINKING = "neatlogs.llm.output_messages.0.thinking"


def test_model_response_keeps_reasoning_content():
    span = FakeSpan()
    _finalize_model(
        span, SimpleNamespace(content="Paris", tool_calls=None, reasoning_content="thinking")
    )
    assert span.attrs[THINKING] == "thinking"
    assert span.attrs["neatlogs.llm.output_messages.0.content"] == "Paris"


def test_model_response_without_reasoning_has_no_thinking():
    span = FakeSpan()
    _finalize_model(span, SimpleNamespace(content="Paris", tool_calls=None))
    assert THINKING not in span.attrs


def test_stream_joins_reasoning_chunks():
    span = FakeSpan()
    chunks = [
        SimpleNamespace(content=None, reasoning_content="think "),
        SimpleNamespace(content="Pa", reasoning_content="more"),
        SimpleNamespace(content="ris", reasoning_content=None),
    ]
    _finalize_model_stream(span, chunks)
    assert span.attrs[THINKING] == "think more"
    assert span.attrs["neatlogs.llm.output_messages.0.content"] == "Paris"
