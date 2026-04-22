import json
from types import SimpleNamespace

from opentelemetry.trace import SpanKind

from neatlogs.core.attribute_processor import UnifiedAttributeProcessor


def _mk_processor() -> UnifiedAttributeProcessor:
    # Mapping not needed for these unit tests; we only exercise normalization helpers.
    return UnifiedAttributeProcessor(mapping_config={}, debug=False)


def test_intermediate_steps_extracted_from_output_messages() -> None:
    proc = _mk_processor()

    unified = {
        "neatlogs.span.kind": "llm",
        "neatlogs.llm.output_messages.0.role": "assistant",
        "neatlogs.llm.output_messages.0.content": """
Thought: I should do X
Action: Search Web
Action Input: {"query":"llm adoption"}
Observation: result text here

Thought: I now know the final answer
Final Answer: Done
""".strip(),
    }

    proc._add_intermediate_steps(unified)

    assert "neatlogs.llm.intermediate_steps" in unified
    steps = json.loads(unified["neatlogs.llm.intermediate_steps"])
    assert isinstance(steps, list)
    assert len(steps) == 2
    assert steps[0]["thought"].startswith("I should do X")
    assert steps[0]["action"] == "Search Web"
    assert "query" in steps[0]["action_input"]
    assert "observation" in steps[0]
    assert "final_answer" in steps[1]


def test_intermediate_steps_not_added_when_no_react_markers() -> None:
    proc = _mk_processor()

    unified = {
        "neatlogs.span.kind": "llm",
        "neatlogs.llm.output_messages.0.role": "assistant",
        "neatlogs.llm.output_messages.0.content": "Hello world (no Thought markers).",
    }

    proc._add_intermediate_steps(unified)
    assert "neatlogs.llm.intermediate_steps" not in unified


def test_intermediate_steps_ignored_for_non_llm_spans() -> None:
    proc = _mk_processor()

    unified = {
        "neatlogs.span.kind": "tool",
        "neatlogs.llm.output_messages.0.role": "assistant",
        "neatlogs.llm.output_messages.0.content": "Thought: should not be parsed here",
    }

    proc._add_intermediate_steps(unified)
    assert "neatlogs.llm.intermediate_steps" not in unified


def test_tool_calls_normalized_and_source_keys_removed() -> None:
    proc = _mk_processor()
    span = SimpleNamespace(kind=SpanKind.INTERNAL, events=[])

    attrs = {
        # OpenInference-style tool call keys
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "mcp_add",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"a": 1, "b": 2}',
    }

    out = proc._normalize_conventions(span, dict(attrs))

    assert out["llm.tool_calls.0.name"] == "mcp_add"
    assert out["llm.tool_calls.0.arguments"] == '{"a": 1, "b": 2}'

    # Original source keys are removed to avoid duplication.
    for k in attrs.keys():
        assert k not in out


def test_crewai_token_usage_fallback_parsed() -> None:
    proc = _mk_processor()
    attrs = {
        "neatlogs.crew.token_usage": (
            "total_tokens=67305 prompt_tokens=46983 cached_prompt_tokens=7 completion_tokens=20322 successful_requests=27"
        )
    }

    proc._add_crewai_token_usage_fallback(attrs)

    assert attrs["llm.token_count.total"] == 67305
    assert attrs["llm.token_count.prompt"] == 46983
    assert attrs["llm.token_count.completion"] == 20322
    assert attrs["llm.token_count.prompt_details.cache_read"] == 7
