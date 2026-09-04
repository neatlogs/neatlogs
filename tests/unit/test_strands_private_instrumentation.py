import asyncio

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import neatlogs

pytest.importorskip("strands")
pytest.importorskip("openinference.instrumentation.strands_agents")


def _local_agent(
    name: str,
    *,
    with_tool: bool = False,
    fail: bool = False,
    started: asyncio.Event | None = None,
    release: asyncio.Event | None = None,
):
    from strands import Agent, tool
    from strands.models.model import Model

    class LocalModel(Model):
        def __init__(self) -> None:
            self.config = {"model_id": f"model-{name}", "context_window_limit": 8192}
            self.calls = 0

        def update_config(self, **model_config):
            self.config.update(model_config)

        def get_config(self):
            return dict(self.config)

        async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
            del tool_specs, system_prompt, kwargs
            if self.calls == 0:
                assert messages[-1]["content"][-1]["text"] == f"question-{name}"
            self.calls += 1
            if started is not None:
                started.set()
            if release is not None:
                await release.wait()
            if fail:
                raise RuntimeError(f"model-failure-{name}")
            yield {"messageStart": {"role": "assistant"}}
            if with_tool and self.calls == 1:
                yield {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "name": "lookup_temperature",
                                "toolUseId": f"tool-{name}",
                            }
                        }
                    }
                }
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"city":"Paris"}'}}}}
                yield {"contentBlockStop": {}}
                yield {"messageStop": {"stopReason": "tool_use"}}
                yield {
                    "metadata": {
                        "usage": {"inputTokens": 9, "outputTokens": 1, "totalTokens": 10},
                        "metrics": {"latencyMs": 1},
                    }
                }
                return
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": f"answer-{name}"}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
            yield {
                "metadata": {
                    "usage": {"inputTokens": 9, "outputTokens": 4, "totalTokens": 13},
                    "metrics": {"latencyMs": 1},
                }
            }

        async def structured_output(self, output_model, prompt, **kwargs):
            del output_model, prompt, kwargs
            raise NotImplementedError

    @tool
    def lookup_temperature(city: str) -> str:
        """Return deterministic weather for one city."""
        return f"{city}: 20 C"

    return Agent(
        model=LocalModel(),
        name=f"agent-{name}",
        callback_handler=None,
        tools=[lookup_temperature] if with_tool else [],
        retry_strategy=None,
    )


def _providers():
    foreign_provider = TracerProvider()
    foreign_exporter = InMemorySpanExporter()
    foreign_provider.add_span_processor(SimpleSpanProcessor(foreign_exporter))
    trace_api.set_tracer_provider(foreign_provider)

    private_provider = TracerProvider()
    private_exporter = InMemorySpanExporter()
    return private_provider, private_exporter, foreign_provider, foreign_exporter


def _semantic_spans(exporter):
    return [
        span
        for span in exporter.get_finished_spans()
        if span.attributes.get("openinference.span.kind") in {"AGENT", "CHAIN", "LLM", "TOOL"}
    ]


def test_automatic_strands_instrumentation_uses_the_private_provider():
    private_provider, private_exporter, _, foreign_exporter = _providers()
    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        instrumentations=["strands"],
        tracer_provider=private_provider,
        register_shutdown_handlers=False,
    )
    private_provider.add_span_processor(SimpleSpanProcessor(private_exporter))

    result = _local_agent("automatic")("question-automatic")

    assert str(result).strip() == "answer-automatic"
    private_spans = _semantic_spans(private_exporter)
    kinds = [span.attributes.get("openinference.span.kind") for span in private_spans]
    assert kinds.count("AGENT") == 1
    assert kinds.count("CHAIN") == 1
    assert kinds.count("LLM") == 1
    by_kind = {span.attributes.get("openinference.span.kind"): span for span in private_spans}
    assert by_kind["AGENT"].parent is None
    assert by_kind["CHAIN"].parent.span_id == by_kind["AGENT"].context.span_id
    assert by_kind["LLM"].parent.span_id == by_kind["CHAIN"].context.span_id
    assert by_kind["LLM"].attributes["input.value"] == "question-automatic"
    assert "answer-automatic" in by_kind["LLM"].attributes["output.value"]
    assert by_kind["LLM"].attributes["llm.token_count.total"] == 13
    assert not _semantic_spans(foreign_exporter)


def test_strands_tool_call_has_one_complete_private_tool_span():
    private_provider, private_exporter, _, foreign_exporter = _providers()
    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        instrumentations=["strands"],
        tracer_provider=private_provider,
        register_shutdown_handlers=False,
    )
    private_provider.add_span_processor(SimpleSpanProcessor(private_exporter))

    result = _local_agent("tool", with_tool=True)("question-tool")

    assert str(result).strip() == "answer-tool"
    semantic = _semantic_spans(private_exporter)
    tools = [span for span in semantic if span.attributes.get("openinference.span.kind") == "TOOL"]
    llms = [span for span in semantic if span.attributes.get("openinference.span.kind") == "LLM"]
    assert len(tools) == 1
    assert len(llms) == 2
    assert "Paris" in tools[0].attributes["input.value"]
    assert "20 C" in tools[0].attributes["output.value"]
    assert not _semantic_spans(foreign_exporter)


@pytest.mark.asyncio
async def test_concurrent_strands_agents_keep_content_on_the_private_provider():
    private_provider, private_exporter, _, foreign_exporter = _providers()
    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        instrumentations=["strands"],
        tracer_provider=private_provider,
        register_shutdown_handlers=False,
    )
    private_provider.add_span_processor(SimpleSpanProcessor(private_exporter))
    alpha = _local_agent("alpha")
    beta = _local_agent("beta")

    results = await asyncio.gather(
        alpha.invoke_async("question-alpha"),
        beta.invoke_async("question-beta"),
    )

    assert [str(result).strip() for result in results] == ["answer-alpha", "answer-beta"]
    llms = [
        span
        for span in _semantic_spans(private_exporter)
        if span.attributes.get("openinference.span.kind") == "LLM"
    ]
    assert len(llms) == 2
    assert {span.attributes["input.value"] for span in llms} == {
        "question-alpha",
        "question-beta",
    }
    assert not _semantic_spans(foreign_exporter)


@pytest.mark.asyncio
async def test_cancelled_strands_run_finishes_private_interrupted_spans():
    private_provider, private_exporter, _, foreign_exporter = _providers()
    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        instrumentations=["strands"],
        tracer_provider=private_provider,
        register_shutdown_handlers=False,
    )
    private_provider.add_span_processor(SimpleSpanProcessor(private_exporter))
    started = asyncio.Event()
    release = asyncio.Event()
    agent = _local_agent("cancelled", started=started, release=release)
    task = asyncio.create_task(agent.invoke_async("question-cancelled"))
    await asyncio.wait_for(started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert neatlogs.shutdown(termination_reason="cancelled")
    semantic = _semantic_spans(private_exporter)
    assert {span.attributes.get("openinference.span.kind") for span in semantic} >= {
        "AGENT",
        "CHAIN",
        "LLM",
    }
    assert all(span.status.status_code.name == "UNSET" for span in semantic)
    assert all(span.attributes["neatlogs.trace.interrupted"] is True for span in semantic)
    assert all(
        span.attributes["neatlogs.trace.termination.reason"] == "cancelled" for span in semantic
    )
    assert not _semantic_spans(foreign_exporter)


def test_strands_model_failure_is_exported_once_as_an_error():
    private_provider, private_exporter, _, foreign_exporter = _providers()
    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        instrumentations=["strands"],
        tracer_provider=private_provider,
        register_shutdown_handlers=False,
    )
    private_provider.add_span_processor(SimpleSpanProcessor(private_exporter))

    with pytest.raises(RuntimeError, match="model-failure-error"):
        _local_agent("error", fail=True)("question-error")

    error_spans = [
        span
        for span in _semantic_spans(private_exporter)
        if span.status.status_code.name == "ERROR"
    ]
    assert {span.attributes.get("openinference.span.kind") for span in error_spans} >= {
        "AGENT",
        "CHAIN",
        "LLM",
    }
    assert not _semantic_spans(foreign_exporter)


def test_repeated_strands_hooks_do_not_duplicate_semantic_spans():
    private_provider, private_exporter, _, foreign_exporter = _providers()
    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        instrumentations=["strands"],
        tracer_provider=private_provider,
        register_shutdown_handlers=False,
    )
    private_provider.add_span_processor(SimpleSpanProcessor(private_exporter))
    agent = _local_agent("repeated")

    assert neatlogs.strands_hooks(agent) is agent
    assert neatlogs.strands_hooks(agent) is agent
    result = agent("question-repeated")

    assert str(result).strip() == "answer-repeated"
    kinds = [
        span.attributes.get("openinference.span.kind") for span in _semantic_spans(private_exporter)
    ]
    assert kinds.count("AGENT") == 1
    assert kinds.count("CHAIN") == 1
    assert kinds.count("LLM") == 1
    assert not _semantic_spans(foreign_exporter)


def test_precreated_strands_agent_is_redirected_when_explicitly_wrapped():
    private_provider, private_exporter, _, foreign_exporter = _providers()
    agent = _local_agent("precreated")
    old_tracer = agent.tracer
    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        tracer_provider=private_provider,
        register_shutdown_handlers=False,
    )
    private_provider.add_span_processor(SimpleSpanProcessor(private_exporter))

    assert neatlogs.strands_hooks(agent) is agent
    result = agent("question-precreated")

    assert str(result).strip() == "answer-precreated"
    assert agent.tracer is not old_tracer
    assert {
        span.attributes.get("openinference.span.kind") for span in _semantic_spans(private_exporter)
    } >= {
        "AGENT",
        "CHAIN",
        "LLM",
    }
    assert not _semantic_spans(foreign_exporter)

    assert neatlogs.shutdown()
    from strands.telemetry import tracer as tracer_module

    assert agent.tracer is old_tracer
    assert tracer_module._tracer_instance is old_tracer


def test_strands_shutdown_and_reinitialize_bind_to_the_new_private_provider():
    first_provider, first_exporter, _, _ = _providers()
    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        instrumentations=["strands"],
        tracer_provider=first_provider,
        register_shutdown_handlers=False,
    )
    first_provider.add_span_processor(SimpleSpanProcessor(first_exporter))
    _local_agent("first")("question-first")
    assert neatlogs.shutdown()

    second_provider = TracerProvider()
    second_exporter = InMemorySpanExporter()
    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        instrumentations=["strands"],
        tracer_provider=second_provider,
        register_shutdown_handlers=False,
    )
    second_provider.add_span_processor(SimpleSpanProcessor(second_exporter))
    _local_agent("second")("question-second")

    assert (
        len(
            [
                span
                for span in _semantic_spans(first_exporter)
                if span.attributes.get("openinference.span.kind") == "LLM"
            ]
        )
        == 1
    )
    assert (
        len(
            [
                span
                for span in _semantic_spans(second_exporter)
                if span.attributes.get("openinference.span.kind") == "LLM"
            ]
        )
        == 1
    )
