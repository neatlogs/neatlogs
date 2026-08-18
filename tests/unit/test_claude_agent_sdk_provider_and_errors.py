"""
Claude Agent SDK: provider resolution + tool-error attributes.

The Agent SDK is a proxy — the model id it reports may be served by Bedrock, Vertex or a
LiteLLM gateway, so the provider cannot be hardcoded to ``anthropic``. These tests pin the
id -> provider table, the precedence between its rules, and the way the resolved provider
reaches the emitted LLM span. They also cover the OTel-standard ``error.*`` attributes
stamped on a failed tool_result.
"""

import asyncio

import pytest
from opentelemetry import trace as otel_trace

from neatlogs.claude_agent_sdk import _provider_for_model, _TracingQuery

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _make_query(tracer_provider, prompt="hello"):
    """A _TracingQuery wired to a real tracer, with no underlying SDK object."""
    tracer = tracer_provider.get_tracer("test")
    agent_span = tracer.start_span("claude_agent.query")
    agent_ctx = otel_trace.set_span_in_context(agent_span)
    return _TracingQuery(None, agent_span, agent_ctx, tracer, {"text": prompt})


def _spans_by_kind(exporter, kind):
    return [
        s for s in exporter.get_finished_spans() if s.attributes.get("neatlogs.span.kind") == kind
    ]


def _assistant(model=None, text="ok", tool_calls=()):
    content = []
    if text:
        content.append({"type": "text", "text": text})
    for tc in tool_calls:
        content.append({"type": "tool_use", **tc})
    message = {"content": content}
    if model:
        message["model"] = model
    return {"type": "assistant", "message": message}


def _tool_result(tool_use_id, content, is_error=None):
    block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error is not None:
        block["is_error"] = is_error
    return {"type": "user", "message": {"content": [block]}}


# ---------------------------------------------------------------------------
# _provider_for_model — the id -> provider table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        # Gemini / Vertex
        ("gemini-2.5-pro", "vertex_ai"),
        ("gemini-1.5-flash", "vertex_ai"),
        ("publishers/google/models/gemini-2.0-flash", "vertex_ai"),
        ("GEMINI-2.5-PRO", "vertex_ai"),
        # Anthropic — bare, Bedrock and Vertex flavoured ids
        ("claude-sonnet-5", "anthropic"),
        ("claude-opus-4-8", "anthropic"),
        ("us.anthropic.claude-opus-4-8-v1:0", "anthropic"),
        ("CLAUDE-HAIKU-4-5", "anthropic"),
        # OpenAI — prefix rules
        ("gpt-4o", "openai"),
        ("gpt-5", "openai"),
        ("GPT-4O-MINI", "openai"),
        ("gpt-oss-120b", "openai"),
        ("o1-preview", "openai"),
        ("o3-mini", "openai"),
        ("o4-mini", "openai"),
        ("azure/gpt-4o", "openai"),
        # Mistral
        ("mistral-large-latest", "mistral"),
        ("mixtral-8x7b-instruct", "mistral"),
        ("open-mistral-nemo", "mistral"),
        ("MISTRAL-SMALL", "mistral"),
        # Meta
        ("llama-3.1-70b-instruct", "meta"),
        ("meta-llama/Llama-3-8b", "meta"),
        ("LLAMA-4-SCOUT", "meta"),
        # Unrecognised
        ("command-r-plus", "unknown"),
        ("deepseek-chat", "unknown"),
        ("qwen-2.5-72b", "unknown"),
        ("", "unknown"),
    ],
)
def test_provider_for_model_table(model, expected):
    assert _provider_for_model(model) == expected


@pytest.mark.parametrize("falsy", [None, ""])
def test_provider_for_model_handles_missing_model(falsy):
    """A missing model id must resolve, not raise — the caller passes str(model or '')."""
    assert _provider_for_model(falsy) == "unknown"


def test_provider_for_model_is_case_insensitive():
    assert _provider_for_model("Claude-Sonnet-5") == _provider_for_model("claude-sonnet-5")


@pytest.mark.parametrize(
    "model,expected,why",
    [
        # gemini is checked first, so it wins over a co-occurring "claude"
        ("claude-proxy-via-gemini", "vertex_ai", "gemini rule precedes claude"),
        # claude is checked before the gpt- prefix rule
        ("claude-gpt-bridge", "anthropic", "claude rule precedes openai"),
        # claude precedes mistral/llama too
        ("claude-llama-router", "anthropic", "claude rule precedes meta"),
        # mistral precedes llama
        ("mistral-llama-merge", "mistral", "mistral rule precedes meta"),
    ],
)
def test_provider_for_model_rule_precedence(model, expected, why):
    assert _provider_for_model(model) == expected, why


@pytest.mark.parametrize(
    "model",
    [
        "azure/gpt-4o",
        "openai/gpt-4o",
        "litellm_proxy/gpt-4o",
        "my-gpt-4o-deployment",
        "openai.gpt-oss-120b-1:0",
        "bedrock/openai.gpt-oss-20b",
    ],
)
def test_gateway_namespaced_openai_ids_resolve(model):
    """A gateway or Bedrock/Azure namespace in front of an OpenAI id must not hide the provider —
    this is the shape LiteLLM-proxied runs report."""
    assert _provider_for_model(model) == "openai"


@pytest.mark.parametrize(
    "model", ["o1-preview", "o3", "o4-mini", "azure/o3-mini", "litellm_proxy/o1"]
)
def test_openai_o_series_ids(model):
    assert _provider_for_model(model) == "openai"


@pytest.mark.parametrize("model", ["chatgpt-4o-latest", "azure/chatgpt-4o-latest"])
def test_chatgpt_ids_resolve(model):
    assert _provider_for_model(model) == "openai"


@pytest.mark.parametrize("model", ["cohere-o1", "command-o1", "some-model-o3"])
def test_o_series_does_not_overmatch_other_vendors(model):
    """The o-series rule anchors to a namespace segment, so a trailing ``-o1`` on another vendor's id
    must not be claimed for OpenAI."""
    assert _provider_for_model(model) == "unknown"


# ---------------------------------------------------------------------------
# The resolved provider reaches the LLM span
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-sonnet-5", "anthropic"),
        ("gemini-2.5-pro", "vertex_ai"),
        ("gpt-4o", "openai"),
        ("llama-3.1-70b", "meta"),
        ("some-unlisted-model", "unknown"),
    ],
)
def test_llm_span_carries_resolved_provider(
    model, expected, tracer_provider, in_memory_span_exporter
):
    q = _make_query(tracer_provider)
    q._handle_message(_assistant(model=model))
    q._finalize("ok")

    [llm] = _spans_by_kind(in_memory_span_exporter, "llm")
    assert llm.attributes["neatlogs.llm.provider"] == expected
    assert llm.attributes["neatlogs.llm.system"] == expected
    assert llm.attributes["neatlogs.llm.model_name"] == model


def test_llm_span_provider_is_not_hardcoded_anthropic(tracer_provider, in_memory_span_exporter):
    """Regression: the instrumentor used to stamp ``anthropic`` on every turn regardless of model."""
    q = _make_query(tracer_provider)
    q._handle_message(_assistant(model="gemini-2.5-pro"))
    q._finalize("ok")

    [llm] = _spans_by_kind(in_memory_span_exporter, "llm")
    assert llm.attributes["neatlogs.llm.provider"] != "anthropic"


def test_provider_falls_back_to_system_message_model(tracer_provider, in_memory_span_exporter):
    """When an assistant turn omits the model, the run-level model from the system message is used."""
    q = _make_query(tracer_provider)
    q._handle_message({"type": "system", "session_id": "s-1", "model": "gemini-2.5-pro"})
    q._handle_message(_assistant(model=None))
    q._finalize("ok")

    [llm] = _spans_by_kind(in_memory_span_exporter, "llm")
    assert llm.attributes["neatlogs.llm.provider"] == "vertex_ai"
    assert llm.attributes["neatlogs.llm.model_name"] == "gemini-2.5-pro"


def test_provider_unknown_when_no_model_anywhere(tracer_provider, in_memory_span_exporter):
    """No model on the turn and none on the run: provider is ``unknown`` and model_name is omitted."""
    q = _make_query(tracer_provider)
    q._handle_message(_assistant(model=None))
    q._finalize("ok")

    [llm] = _spans_by_kind(in_memory_span_exporter, "llm")
    assert llm.attributes["neatlogs.llm.provider"] == "unknown"
    assert llm.attributes["neatlogs.llm.system"] == "unknown"
    assert "neatlogs.llm.model_name" not in llm.attributes


def test_provider_resolved_per_turn_when_model_switches(tracer_provider, in_memory_span_exporter):
    """A run that switches models mid-stream gets a per-turn provider, not one run-level guess."""
    q = _make_query(tracer_provider)
    q._handle_message(_assistant(model="claude-sonnet-5", text="first"))
    q._handle_message({"type": "user", "message": {"content": "next"}})
    q._handle_message(_assistant(model="gemini-2.5-pro", text="second"))
    q._finalize("ok")

    providers = [
        s.attributes["neatlogs.llm.provider"]
        for s in _spans_by_kind(in_memory_span_exporter, "llm")
    ]
    assert providers == ["anthropic", "vertex_ai"]


def test_subagent_llm_span_also_resolves_provider(tracer_provider, in_memory_span_exporter):
    """Subagent turns run through the same flush path, so they get the same provider treatment."""
    q = _make_query(tracer_provider)
    q._handle_message(
        _assistant(
            model="claude-sonnet-5",
            text="delegating",
            tool_calls=[{"id": "task-1", "name": "Task", "input": {"prompt": "go"}}],
        )
    )
    q._handle_message(
        {
            "type": "assistant",
            "parent_tool_use_id": "task-1",
            "subagent_type": "explorer",
            "message": {"content": [{"type": "text", "text": "sub"}], "model": "gemini-2.5-pro"},
        }
    )
    q._finalize("ok")

    providers = {
        s.attributes.get("neatlogs.llm.model_name"): s.attributes["neatlogs.llm.provider"]
        for s in _spans_by_kind(in_memory_span_exporter, "llm")
    }
    assert providers == {"claude-sonnet-5": "anthropic", "gemini-2.5-pro": "vertex_ai"}


# ---------------------------------------------------------------------------
# Tool error attributes
# ---------------------------------------------------------------------------


def _one_tool_span(tracer_provider, exporter, result_msg):
    q = _make_query(tracer_provider)
    q._handle_message(
        _assistant(
            model="claude-sonnet-5",
            text="",
            tool_calls=[{"id": "t1", "name": "Bash", "input": {"cmd": "ls"}}],
        )
    )
    q._handle_message(result_msg)
    q._finalize("ok")
    [tool] = _spans_by_kind(exporter, "tool")
    return tool


def test_failed_tool_result_sets_standard_error_attributes(
    tracer_provider, in_memory_span_exporter
):
    tool = _one_tool_span(
        tracer_provider, in_memory_span_exporter, _tool_result("t1", "boom: no such file", True)
    )

    assert tool.status.status_code is otel_trace.StatusCode.ERROR
    assert tool.attributes["neatlogs.tool.is_error"] is True
    assert tool.attributes["error.type"] == "tool_error"
    assert tool.attributes["error.message"] == "boom: no such file"
    # the output blob is still recorded alongside the error
    assert tool.attributes["output.value"] == "boom: no such file"


def test_successful_tool_result_sets_no_error_attributes(tracer_provider, in_memory_span_exporter):
    tool = _one_tool_span(
        tracer_provider, in_memory_span_exporter, _tool_result("t1", "file-a\nfile-b", False)
    )

    assert tool.status.status_code is otel_trace.StatusCode.OK
    assert "error.type" not in tool.attributes
    assert "error.message" not in tool.attributes
    assert "neatlogs.tool.is_error" not in tool.attributes
    assert tool.attributes["output.value"] == "file-a\nfile-b"


def test_tool_result_without_is_error_flag_is_success(tracer_provider, in_memory_span_exporter):
    """``is_error`` absent entirely (the common success shape) must not stamp error attributes."""
    tool = _one_tool_span(tracer_provider, in_memory_span_exporter, _tool_result("t1", "done"))

    assert tool.status.status_code is otel_trace.StatusCode.OK
    assert "error.type" not in tool.attributes
    assert "error.message" not in tool.attributes


def test_error_message_serialises_non_string_tool_output(tracer_provider, in_memory_span_exporter):
    """Structured tool_result content is JSON-serialised into error.message, matching output.value."""
    payload = [{"type": "text", "text": "exit 1"}]
    tool = _one_tool_span(
        tracer_provider, in_memory_span_exporter, _tool_result("t1", payload, True)
    )

    assert tool.attributes["error.type"] == "tool_error"
    assert tool.attributes["error.message"] == tool.attributes["output.value"]
    assert "exit 1" in tool.attributes["error.message"]


def test_error_message_tolerates_empty_tool_output(tracer_provider, in_memory_span_exporter):
    """An error with no content still marks the span failed rather than dropping the error flag."""
    tool = _one_tool_span(tracer_provider, in_memory_span_exporter, _tool_result("t1", "", True))

    assert tool.status.status_code is otel_trace.StatusCode.ERROR
    assert tool.attributes["neatlogs.tool.is_error"] is True
    assert tool.attributes["error.type"] == "tool_error"
    assert tool.attributes["error.message"] == ""


def test_only_the_failing_tool_is_marked_in_a_mixed_turn(tracer_provider, in_memory_span_exporter):
    """One failed tool call must not contaminate its siblings in the same turn."""
    q = _make_query(tracer_provider)
    q._handle_message(
        _assistant(
            model="claude-sonnet-5",
            text="",
            tool_calls=[
                {"id": "ok-1", "name": "Read", "input": {"path": "a"}},
                {"id": "bad-1", "name": "Bash", "input": {"cmd": "false"}},
            ],
        )
    )
    q._handle_message(
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "ok-1", "content": "contents"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "bad-1",
                        "content": "exit status 1",
                        "is_error": True,
                    },
                ]
            },
        }
    )
    q._finalize("ok")

    by_name = {
        s.attributes["neatlogs.tool.name"]: s
        for s in _spans_by_kind(in_memory_span_exporter, "tool")
    }
    assert by_name["Read"].status.status_code is otel_trace.StatusCode.OK
    assert "error.type" not in by_name["Read"].attributes
    assert by_name["Bash"].status.status_code is otel_trace.StatusCode.ERROR
    assert by_name["Bash"].attributes["error.message"] == "exit status 1"


def test_error_attributes_survive_full_async_iteration(tracer_provider, in_memory_span_exporter):
    """End-to-end through __anext__, not just the internal handler."""

    class _FakeQuery:
        def __init__(self, messages):
            self._messages = list(messages)

        def __aiter__(self):
            async def gen():
                for m in self._messages:
                    yield m

            return gen()

    messages = [
        {"type": "system", "session_id": "s-1", "model": "claude-sonnet-5"},
        _assistant(
            model="claude-sonnet-5",
            text="",
            tool_calls=[{"id": "t1", "name": "Bash", "input": {"cmd": "false"}}],
        ),
        _tool_result("t1", "exit status 1", True),
        {"type": "result", "result": "finished", "session_id": "s-1"},
    ]

    tracer = tracer_provider.get_tracer("test")
    agent_span = tracer.start_span("claude_agent.query")
    q = _TracingQuery(
        _FakeQuery(messages),
        agent_span,
        otel_trace.set_span_in_context(agent_span),
        tracer,
        {"text": "run it"},
    )

    async def drain():
        return [m async for m in q]

    assert len(asyncio.run(drain())) == len(messages)

    [tool] = _spans_by_kind(in_memory_span_exporter, "tool")
    assert tool.attributes["error.type"] == "tool_error"
    assert tool.attributes["error.message"] == "exit status 1"
    [llm] = _spans_by_kind(in_memory_span_exporter, "llm")
    assert llm.attributes["neatlogs.llm.provider"] == "anthropic"
