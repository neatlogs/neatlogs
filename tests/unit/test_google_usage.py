from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from neatlogs.core.google_usage import COMPLETION_INCLUDES_REASONING, google_completion_tokens


@pytest.mark.parametrize(
    ("candidates", "thoughts", "expected"),
    [
        (645, 3981, 4626),  # thinking model: answer + thinking
        (5, None, 5),  # no thinking reported
        (5, 0, 5),
        (None, 12, 12),  # stopped while thinking, no answer tokens
        (None, None, None),
    ],
)
def test_completion_includes_thinking(candidates, thoughts, expected):
    assert google_completion_tokens(candidates, thoughts) == expected


def test_non_integer_values_are_left_as_reported():
    sentinel = MagicMock()
    assert google_completion_tokens(sentinel, None) is sentinel
    assert google_completion_tokens(7, sentinel) == 7
    assert google_completion_tokens(True, None) is True


def test_vertex_usage_attributes_report_thinking_inside_completion():
    from neatlogs.vertex_ai import _set_usage_attributes

    span = MagicMock()
    usage = SimpleNamespace(
        prompt_token_count=3134,
        candidates_token_count=645,
        total_token_count=7760,
        cached_content_token_count=None,
        thoughts_token_count=3981,
    )
    _set_usage_attributes(span, usage)

    attributes = {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}
    assert attributes["neatlogs.llm.token_count.completion"] == 4626
    assert attributes["neatlogs.llm.token_count.reasoning"] == 3981
    assert attributes["neatlogs.llm.token_count.total"] == 7760
    assert attributes[COMPLETION_INCLUDES_REASONING] is True
    assert "neatlogs.llm.token_count.reasoning_is_separate" not in attributes
    assert (
        attributes["neatlogs.llm.token_count.prompt"]
        + attributes["neatlogs.llm.token_count.completion"]
        == attributes["neatlogs.llm.token_count.total"]
    )


def test_google_genai_usage_attributes_report_thinking_inside_completion():
    from neatlogs.google_genai import _set_usage_attributes

    span = MagicMock()
    usage = SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=5,
        total_token_count=35,
        cached_content_token_count=None,
        thoughts_token_count=20,
    )
    _set_usage_attributes(span, usage)

    attributes = {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}
    assert attributes["neatlogs.llm.token_count.completion"] == 25
    assert attributes[COMPLETION_INCLUDES_REASONING] is True
    assert "neatlogs.llm.token_count.reasoning_is_separate" not in attributes


def test_tool_use_prompt_tokens_keep_the_marker_so_thinking_is_not_added_twice():
    """Google's total also counts tool-use prompt tokens (review repro on #2346)."""
    from neatlogs.google_genai import _set_usage_attributes

    span = MagicMock()
    usage = SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=5,
        total_token_count=55,  # 10 prompt + 20 tool-use prompt + 20 thinking + 5 answer
        cached_content_token_count=None,
        thoughts_token_count=20,
    )
    _set_usage_attributes(span, usage)

    attributes = {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}
    assert attributes["neatlogs.llm.token_count.completion"] == 25
    assert attributes[COMPLETION_INCLUDES_REASONING] is True
    assert "neatlogs.llm.token_count.reasoning_is_separate" not in attributes


def test_no_marker_without_a_completion_count():
    from neatlogs.google_genai import _set_usage_attributes

    span = MagicMock()
    usage = SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=None,
        total_token_count=10,
        cached_content_token_count=None,
        thoughts_token_count=None,
    )
    _set_usage_attributes(span, usage)

    attributes = {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}
    assert COMPLETION_INCLUDES_REASONING not in attributes
