"""Anthropic ``system`` content blocks must capture as readable text.

When prompt caching is enabled, callers pass ``system`` as a list of content
blocks (``{"type": "text", "text": ..., "cache_control": ...}``). Capturing
the raw JSON string made the system prompt render as an unreadable blob (or
empty) downstream; the text of the blocks is what should land on the span.
"""

from neatlogs.anthropic import _set_input_attributes


class _Span:
    def __init__(self):
        self.attrs = {}

    def set_attribute(self, key, value):
        self.attrs[key] = value


def test_cached_system_blocks_capture_joined_text():
    span = _Span()
    system = [
        {
            "type": "text",
            "text": "You are a careful assistant.",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "Answer briefly."},
    ]
    _set_input_attributes(span, [{"role": "user", "content": "hi"}], system, {})
    assert span.attrs["neatlogs.llm.input_messages.0.role"] == "system"
    assert (
        span.attrs["neatlogs.llm.input_messages.0.content"]
        == "You are a careful assistant.\nAnswer briefly."
    )
    assert span.attrs["neatlogs.llm.input_messages.1.content"] == "hi"


def test_string_system_unchanged():
    span = _Span()
    _set_input_attributes(span, [], "be nice", {})
    assert span.attrs["neatlogs.llm.input_messages.0.content"] == "be nice"


def test_non_text_system_falls_back_to_serialized():
    span = _Span()
    system = [{"type": "image", "source": {"data": "x"}}]
    _set_input_attributes(span, [], system, {})
    content = span.attrs["neatlogs.llm.input_messages.0.content"]
    assert '"type": "image"' in content
