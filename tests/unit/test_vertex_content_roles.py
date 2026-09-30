from types import SimpleNamespace
import pytest
from neatlogs.vertex_ai import _set_input_attributes


class Span:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


@pytest.mark.parametrize("in_list", [False, True])
@pytest.mark.parametrize("typed", [False, True])
def test_model_history_preserves_role_and_text(in_list, typed):
    part = SimpleNamespace(text="prior answer", function_call=None, function_response=None)
    content = (
        SimpleNamespace(role="model", parts=[part])
        if typed
        else {"role": "model", "parts": [{"text": "prior answer"}]}
    )
    span = Span()
    _set_input_attributes(span, [content] if in_list else content, {})
    assert span.attributes["neatlogs.llm.input_messages.0.role"] == "model"
    assert span.attributes["neatlogs.llm.input_messages.0.content"] == "prior answer"


@pytest.mark.parametrize("system", [False, True])
def test_multiple_history_turns_and_system_index(system):
    span = Span()
    turns = [
        SimpleNamespace(role="user", parts=[SimpleNamespace(text="question")]),
        SimpleNamespace(role="model", parts=[SimpleNamespace(text="answer")]),
    ]
    _set_input_attributes(
        span, turns, {"config": {"system_instruction": "be helpful"}} if system else {}
    )
    offset = int(system)
    assert span.attributes[f"neatlogs.llm.input_messages.{offset}.role"] == "user"
    assert span.attributes[f"neatlogs.llm.input_messages.{offset}.content"] == "question"
    assert span.attributes[f"neatlogs.llm.input_messages.{offset+1}.role"] == "model"
    assert span.attributes[f"neatlogs.llm.input_messages.{offset+1}.content"] == "answer"


def test_plain_string_stays_user():
    span = Span()
    _set_input_attributes(span, "hello", {})
    assert span.attributes["neatlogs.llm.input_messages.0.role"] == "user"
    assert span.attributes["neatlogs.llm.input_messages.0.content"] == "hello"


def test_typed_function_call_preserves_model_role():
    span = Span()
    item = SimpleNamespace(
        role="model",
        parts=[
            SimpleNamespace(
                text=None,
                function_call=SimpleNamespace(name="weather", args={"city": "Paris"}),
                function_response=None,
            )
        ],
    )
    _set_input_attributes(span, [item], {})
    assert span.attributes["neatlogs.llm.input_messages.0.role"] == "model"
    assert "weather" in span.attributes["neatlogs.llm.input_messages.0.content"]
