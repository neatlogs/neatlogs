import importlib
import importlib.util
import json
import sys
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

import neatlogs

EXAMPLE_PATH = Path(__file__).parents[2] / "examples" / "sdk_examples" / "crewai_prompt_fidelity.py"


def _load_example(monkeypatch):
    for name in (
        "NEATLOGS_API_KEY",
        "NEATLOGS_ENDPOINT",
        "CREWAI_MODEL",
        "OPENAI_API_KEY",
    ):
        monkeypatch.setenv(name, f"test-{name.lower()}")
    monkeypatch.setattr(neatlogs, "init", lambda **_kwargs: None)

    module_name = "crewai_prompt_fidelity_example_under_test"
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def test_example_uses_generic_document_analysis_contract(monkeypatch) -> None:
    example = _load_example(monkeypatch)

    assert example.AGENT_ROLE == "Long Document Analyst"
    assert example.AGENT_GOAL == (
        "Summarize long synthetic documents without losing details near the end"
    )
    assert example.AGENT_BACKSTORY == (
        "You inspect technical documents carefully and return concise, structured findings."
    )

    description = example.build_long_task_description()
    assert (
        "**Objective:** Analyze the synthetic document and return a structured summary."
        in description
    )
    assert "**Input document:**" in description
    assert "**Required output schema:**" in description
    assert '"document_id": "synthetic-document-001"' in description
    assert '"tail_marker_seen": "boolean"' in description


def test_example_builds_deterministic_task_beyond_12000_characters(monkeypatch) -> None:
    example = _load_example(monkeypatch)

    description = example.build_long_task_description()
    near_offset = description.index(example.NEAR_10000_MARKER)
    tail_offset = description.index(example.TAIL_AFTER_12000_MARKER)

    assert len(description) > 12_000
    assert 9_500 < near_offset < 10_000
    assert tail_offset > 12_000


def test_example_uses_automatic_crewai_instrumentation() -> None:
    source = EXAMPLE_PATH.read_text()

    assert 'workflow_name="crewai-long-prompt-fidelity"' in source
    assert 'instrumentations=["crewai"]' in source
    assert 'name="long_prompt_fidelity"' in source
    assert 'kind="WORKFLOW"' in source
    assert "await crew.kickoff_async()" in source
    assert "bind_templates(" not in source
    assert "register_crewai_task(" not in source


def _install_test_tracer(in_memory_span_exporter) -> None:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    trace.set_tracer_provider(provider)
    neatlogs.init(api_key="test-key", disable_export=True, instrumentations=[])


def test_crewai_string_input_preserves_content_after_10000_characters(
    in_memory_span_exporter,
) -> None:
    _install_test_tracer(in_memory_span_exporter)
    crewai_instrumentation = importlib.import_module("neatlogs.crewai")

    class FakeStringLlm:
        model = "test-model"

        def call(self, messages, *args, **kwargs):
            return "ok"

    crewai_instrumentation._patch_llm_call(FakeStringLlm)

    tail_sentinel = "TAIL_SENTINEL_AFTER_10000"
    user_prompt = f"USER_PROMPT_START\n{'u' * 12_000}\n{tail_sentinel}"
    FakeStringLlm().call(user_prompt)

    llm_span = next(
        span
        for span in in_memory_span_exporter.get_finished_spans()
        if span.name == "crewai.llm.call"
    )
    captured = llm_span.attributes["neatlogs.llm.input_messages.0.content"]

    assert captured == user_prompt
    assert tail_sentinel in captured
    structured_input = json.loads(llm_span.attributes["input.value"])
    assert structured_input == {"messages": [{"role": "user", "content": user_prompt}]}


def test_crewai_message_list_preserves_roles_history_and_long_prompts(
    in_memory_span_exporter,
) -> None:
    _install_test_tracer(in_memory_span_exporter)
    crewai_instrumentation = importlib.import_module("neatlogs.crewai")

    class FakeMessageLlm:
        model = "test-model"

        def call(self, messages, *args, **kwargs):
            return "ok"

    crewai_instrumentation._patch_llm_call(FakeMessageLlm)

    system_tail = "SYSTEM_TAIL_AFTER_10000"
    user_tail = "TASK_TAIL_AFTER_10000"
    messages = [
        {
            "role": "system",
            "content": f"SYSTEM_START\n{'s' * 12_000}\n{system_tail}",
        },
        {"role": "assistant", "content": "prior answer"},
        {
            "role": "user",
            "content": f"Current Task:\n{'t' * 12_000}\n{user_tail}",
        },
    ]
    FakeMessageLlm().call(messages)

    llm_span = next(
        span
        for span in in_memory_span_exporter.get_finished_spans()
        if span.name == "crewai.llm.call"
    )
    assert llm_span.attributes["neatlogs.llm.input_messages.0.role"] == "system"
    assert llm_span.attributes["neatlogs.llm.input_messages.1.role"] == "assistant"
    assert llm_span.attributes["neatlogs.llm.input_messages.2.role"] == "user"
    assert system_tail in llm_span.attributes["neatlogs.llm.input_messages.0.content"]
    assert user_tail in llm_span.attributes["neatlogs.llm.input_messages.2.content"]
    structured_input = json.loads(llm_span.attributes["input.value"])
    assert structured_input == {"messages": messages}
