"""
Counterfactual trace replay example (offline, no API keys required).

Demonstrates the MVP vertical slice:

1. Run a small support agent and capture its trace (WORKFLOW + LLM spans).
   The LLM step uses the same ``ChoiceAccumulator`` the OpenAI wrapper uses,
   so the captured attributes match production traces — the "provider call"
   itself is stubbed so this runs without network access. Both spans are
   emitted on one plain tracer with explicit nesting (in production the LLM
   span comes from ``neatlogs.wrap(openai.OpenAI())`` and nests automatically).
2. Store the trace in ``InMemoryTraceStore``.
3. Replay it with a changed system prompt (the counterfactual).
4. Compare original vs replay, print what changed, and build an eval-case
   handoff (rejected vs preferred output + provenance).

Safe by default: replay re-issues the LLM invocation only. Tool calls in the
replayed output are recorded as data and never executed.

Run:
    python examples/sdk_examples/counterfactual_replay_basic.py
"""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import neatlogs
from neatlogs import InMemoryTraceStore, ReplayOverrides
from neatlogs.core.choice_accumulator import ChoiceAccumulator
from neatlogs.replay import LLMRequest, LLMResponse, ToolCallData

MODEL = "gpt-4o-mini"
QUESTION = "Where is my order #123?"
ORIGINAL_SYSTEM = "You are a billing support agent. Be helpful."
COUNTERFACTUAL_SYSTEM = (
    "You are a billing support agent. Never issue refunds; "
    "cancel the order instead and cite policy v13."
)


def _emit_llm_span(tracer, *, model, messages, content, tool_calls):
    """Mimic what the OpenAI wrapper records, without network access."""
    attributes = {
        "neatlogs.span.kind": "llm",
        "openinference.span.kind": "LLM",
        "neatlogs.llm.provider": "openai",
        "neatlogs.llm.system": "openai",
        "neatlogs.llm.model_name": model,
        "neatlogs.llm.temperature": 0.3,
    }
    for i, message in enumerate(messages):
        attributes[f"neatlogs.llm.input_messages.{i}.role"] = message["role"]
        attributes[f"neatlogs.llm.input_messages.{i}.content"] = message["content"]
    attributes["neatlogs.llm.tools.0.name"] = "refund_order"
    attributes["neatlogs.llm.tools.0.definition"] = (
        '{"type": "function", "function": {"name": "refund_order"}}'
    )
    with tracer.start_as_current_span(
        "openai.chat.completions.create", attributes=attributes
    ) as span:
        accumulator = ChoiceAccumulator()
        accumulator.add_single_response(
            content,
            tool_calls=[
                {
                    "index": i,
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                }
                for i, call in enumerate(tool_calls)
            ],
            usage={"prompt_tokens": 42, "completion_tokens": 18, "total_tokens": 60},
            model=model,
        )
        accumulator.apply(span)
        span.set_attribute("neatlogs.llm.metrics.duration_ms", 210.0)


def support_agent(tracer, question: str):
    """Run the (failing) agent: workflow root + one LLM call, one trace."""
    with tracer.start_as_current_span(
        "support_agent",
        attributes={
            "neatlogs.span.kind": "workflow",
            "openinference.span.kind": "WORKFLOW",
        },
    ):
        messages = [
            {"role": "system", "content": ORIGINAL_SYSTEM},
            {"role": "user", "content": question},
        ]
        _emit_llm_span(
            tracer,
            model=MODEL,
            messages=messages,
            content="I'll issue a refund for order #123.",
            tool_calls=[{"id": "call-1", "name": "refund_order", "arguments": '{"order_id": 123}'}],
        )
        return "refund issued (wrong)"


def stub_caller(request: LLMRequest) -> LLMResponse:
    """Counterfactual provider: the fixed prompt cancels instead of refunding."""
    assert request.messages[0]["content"] == COUNTERFACTUAL_SYSTEM
    return LLMResponse(
        content="I'll cancel order #123 per policy v13.",
        tool_calls=[ToolCallData(name="cancel_order", arguments='{"order_id": 123}', id="call-2")],
        model=request.model,
        prompt_tokens=48,
        completion_tokens=20,
        total_tokens=68,
        latency_ms=190.0,
        finish_reason="tool_calls",
    )


def main() -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    neatlogs.init(
        workflow_name="counterfactual-replay-demo",
        tracer_provider=provider,
        disable_export=True,
    )

    # 1. Run the agent and capture the (failed) production trace.
    agent_tracer = provider.get_tracer("support-agent-demo")
    print(f"Agent result: {support_agent(agent_tracer, QUESTION)}")
    original_spans = exporter.get_finished_spans()
    print(f"Captured {len(original_spans)} spans")

    # 2. Index the trace for replay.
    store = InMemoryTraceStore()
    trace_id = store.put_spans(original_spans)
    print(f"Original trace: {trace_id}")

    # 3. Replay with one change: the system prompt.
    overrides = ReplayOverrides(
        messages=[
            {"role": "system", "content": COUNTERFACTUAL_SYSTEM},
            {"role": "user", "content": QUESTION},
        ]
    )
    result = neatlogs.replay_trace(trace_id, overrides, store=store, llm_caller=stub_caller)
    print(f"Replay trace:   {result.replay_trace_id} (replay_id={result.replay_id})")
    print(f"Skipped (safe mode, not re-executed): {result.skipped}")

    # 4. Compare and report what changed.
    comparison = neatlogs.compare_traces(store.get_trace(trace_id), result.spans, result.overrides)
    print("\n" + comparison.summary())
    for change in comparison.output_changes:
        print(f"\nOutput changed in {change['span']}:")
        print(f"  original: {change['original_output']!r}")
        print(f"  replay:   {change['replay_output']!r}")
    for change in comparison.tool_call_changes:
        print(f"\nTool call changed in {change['span']}:")
        print(f"  original: {change['original_tool_calls']}")
        print(f"  replay:   {change['replay_tool_calls']}")
    print(f"\nLatency delta (ms): {comparison.latency_delta_ms}")
    print(
        f"Token deltas (prompt/completion/total): "
        f"{comparison.prompt_tokens_delta}/{comparison.completion_tokens_delta}/"
        f"{comparison.total_tokens_delta}"
    )

    # 5. Turn the fixed failure into a regression/eval case handoff.
    case = comparison.to_eval_case()
    print(f"\nEval case: rejected={case['rejected_output']!r}")
    print(f"Eval case: preferred={case['preferred_output']!r}")
    print(f"Eval case provenance: {case['provenance']}")

    neatlogs.shutdown()


if __name__ == "__main__":
    main()
