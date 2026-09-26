"""Generation config parameters should survive Google GenAI instrumentation."""

import json
from types import SimpleNamespace

import pytest

from neatlogs.google_genai import _set_input_attributes


@pytest.mark.parametrize("config_type", [dict, SimpleNamespace])
def test_google_genai_captures_penalty_parameters(
    config_type, tracer_provider, in_memory_span_exporter
):
    config = {
        "temperature": 0.37,
        "top_p": 0.8,
        "top_k": 20,
        "max_output_tokens": 57,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.4,
    }
    span = tracer_provider.get_tracer("neatlogs.google_genai").start_span("generate")
    _set_input_attributes(
        span,
        "hello",
        {"config": config_type(**config) if config_type is SimpleNamespace else config},
    )
    span.end()

    attrs = in_memory_span_exporter.get_finished_spans()[0].attributes
    expected = {
        "temperature": 0.37,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": 57,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.4,
    }
    for name, value in expected.items():
        assert attrs[f"neatlogs.llm.{name}"] == value
    assert json.loads(attrs["neatlogs.llm.invocation_parameters"]) == expected


def test_google_genai_preserves_zero_and_omits_unset_penalties(
    tracer_provider, in_memory_span_exporter
):
    span = tracer_provider.get_tracer("neatlogs.google_genai").start_span("generate")
    _set_input_attributes(span, "hello", {"config": {"frequency_penalty": 0.0}})
    span.end()

    attrs = in_memory_span_exporter.get_finished_spans()[0].attributes
    assert attrs["neatlogs.llm.frequency_penalty"] == 0.0
    assert "neatlogs.llm.presence_penalty" not in attrs
    assert json.loads(attrs["neatlogs.llm.invocation_parameters"]) == {"frequency_penalty": 0.0}
