import asyncio
import gc
import weakref

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import neatlogs


class _SnapshotExporter(SpanExporter):
    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(
            {"name": span.name, "attributes": dict(span.attributes or {})} for span in spans
        )
        return SpanExportResult.SUCCESS


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


def _processor_names(provider):
    return [
        type(processor).__name__ for processor in provider._active_span_processor._span_processors
    ]


def _llm_spans(exporter):
    return [
        span
        for span in _semantic_spans(exporter)
        if span.attributes.get("openinference.span.kind") == "LLM"
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


def test_strands_processors_are_installed_only_when_strands_is_selected():
    unselected_provider, _, _, _ = _providers()
    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        instrumentations=[],
        tracer_provider=unselected_provider,
        register_shutdown_handlers=False,
    )

    assert "_NeatlogsStrandsProcessor" not in _processor_names(unselected_provider)

    assert neatlogs.shutdown()
    selected_provider = TracerProvider()
    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        instrumentations=["strands"],
        tracer_provider=selected_provider,
        register_shutdown_handlers=False,
    )

    assert _processor_names(selected_provider)[:2] == [
        "ProviderPreprocessor",
        "NeatlogsSpanProcessor",
    ]
    hub = selected_provider._active_span_processor._span_processors[0]
    assert type(hub._processors["strands"]).__name__ == "_NeatlogsStrandsProcessor"


def test_active_client_routes_explicit_strands_wrap_to_client_provider():
    default_provider, default_exporter, _, foreign_exporter = _providers()
    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        instrumentations=["strands"],
        tracer_provider=default_provider,
        register_shutdown_handlers=False,
    )
    default_provider.add_span_processor(SimpleSpanProcessor(default_exporter))

    client_provider = TracerProvider()
    client_exporter = _SnapshotExporter()
    client = neatlogs.Client(
        api_key="client-key",
        workflow_name="client-workflow",
        disable_export=True,
        tracer_provider=client_provider,
    )
    client_provider.add_span_processor(SimpleSpanProcessor(client_exporter))

    with client.activate():
        agent = neatlogs.strands_hooks(_local_agent("client"))
        result = agent("question-client")

    assert str(result).strip() == "answer-client"
    client_llm = [
        span
        for span in client_exporter.spans
        if span["attributes"].get("openinference.span.kind") == "LLM"
    ]
    assert len(client_llm) == 1
    assert not _llm_spans(default_exporter)
    assert not _semantic_spans(foreign_exporter)
    assert client.shutdown()


@pytest.mark.asyncio
async def test_concurrent_active_clients_keep_strands_spans_isolated():
    default_provider, default_exporter, _, foreign_exporter = _providers()
    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        instrumentations=["strands"],
        tracer_provider=default_provider,
        register_shutdown_handlers=False,
    )
    default_provider.add_span_processor(SimpleSpanProcessor(default_exporter))

    first_provider = TracerProvider()
    second_provider = TracerProvider()
    first = neatlogs.Client(
        api_key="first-key",
        workflow_name="first-workflow",
        disable_export=True,
        tracer_provider=first_provider,
    )
    second = neatlogs.Client(
        api_key="second-key",
        workflow_name="second-workflow",
        disable_export=True,
        tracer_provider=second_provider,
    )
    first_exporter = InMemorySpanExporter()
    second_exporter = InMemorySpanExporter()
    first_provider.add_span_processor(SimpleSpanProcessor(first_exporter))
    second_provider.add_span_processor(SimpleSpanProcessor(second_exporter))
    first_agent = _local_agent("first-client")
    second_agent = _local_agent("second-client")

    async def invoke(client, agent, prompt):
        with client.activate():
            return await agent.invoke_async(prompt)

    try:
        results = await asyncio.gather(
            invoke(first, first_agent, "question-first-client"),
            invoke(second, second_agent, "question-second-client"),
        )
    finally:
        first.shutdown()
        second.shutdown()

    assert [str(result).strip() for result in results] == [
        "answer-first-client",
        "answer-second-client",
    ]
    first_inputs = {span.attributes.get("input.value") for span in _llm_spans(first_exporter)}
    second_inputs = {span.attributes.get("input.value") for span in _llm_spans(second_exporter)}
    assert first_inputs == {"question-first-client"}
    assert second_inputs == {"question-second-client"}
    assert not _llm_spans(default_exporter)
    assert not _semantic_spans(foreign_exporter)


def test_strands_hooks_before_init_binds_when_neatlogs_initializes():
    private_provider, private_exporter, _, foreign_exporter = _providers()
    agent = _local_agent("deferred")
    old_tracer = agent.tracer

    assert neatlogs.strands_hooks(agent) is agent
    assert agent.tracer is not old_tracer

    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        tracer_provider=private_provider,
        register_shutdown_handlers=False,
    )
    private_provider.add_span_processor(SimpleSpanProcessor(private_exporter))
    result = agent("question-deferred")

    assert str(result).strip() == "answer-deferred"
    assert agent.tracer is not old_tracer
    assert len(_llm_spans(private_exporter)) == 1
    assert not _semantic_spans(foreign_exporter)


def test_wrapped_strands_agent_retention_is_bounded_after_gc():
    from neatlogs import strands as strands_module

    private_provider, _, _, _ = _providers()
    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        tracer_provider=private_provider,
        register_shutdown_handlers=False,
    )

    agent = neatlogs.strands_hooks(_local_agent("gc"))
    ref = weakref.ref(agent)
    assert len(strands_module._WRAPPED_AGENTS) == 1

    del agent
    gc.collect()

    assert ref() is None
    assert len(strands_module._WRAPPED_AGENTS) == 0


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


def test_default_shutdown_does_not_disable_a_live_client_strands_agent():
    default_provider, _, _, _ = _providers()
    neatlogs.init(
        api_key="default-key",
        disable_export=True,
        instrumentations=["strands"],
        tracer_provider=default_provider,
        register_shutdown_handlers=False,
    )

    client_provider = TracerProvider()
    client_exporter = _SnapshotExporter()
    client_provider.add_span_processor(SimpleSpanProcessor(client_exporter))
    client = neatlogs.Client(
        api_key="client-key",
        workflow_name="client-workflow",
        disable_export=True,
        tracer_provider=client_provider,
    )
    agent = _local_agent("live-client")
    with client.activate():
        neatlogs.strands_hooks(agent)
        agent("question-live-client")

    assert neatlogs.shutdown()

    with client.activate():
        agent("question-live-client")

    llm_spans = [
        span
        for span in client_exporter.spans
        if span["attributes"].get("openinference.span.kind") == "LLM"
    ]
    assert len(llm_spans) == 2
    assert client.shutdown()


def test_client_shutdown_removes_only_its_strands_state():
    from strands.telemetry import tracer as tracer_module

    from neatlogs import strands as strands_module

    provider, _, _, _ = _providers()
    exporter = _SnapshotExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client = neatlogs.Client(
        api_key="client-key",
        workflow_name="client-workflow",
        disable_export=True,
        tracer_provider=provider,
    )
    agent = _local_agent("client-shutdown")
    original_agent_tracer = agent.tracer
    original_singleton = tracer_module._tracer_instance

    with client.activate():
        neatlogs.strands_hooks(agent)
        agent("question-client-shutdown")

    assert client.shutdown()
    assert agent.tracer is original_agent_tracer
    assert tracer_module._tracer_instance is original_singleton
    assert provider not in strands_module._PROVIDER_PROCESSORS
    assert provider not in strands_module._PROVIDER_TRACERS
    assert provider not in strands_module._PROVIDER_OWNERS


def test_strands_provider_tracer_cache_does_not_retain_provider():
    from neatlogs import strands as strands_module

    provider = TracerProvider(shutdown_on_exit=False)
    tracer = strands_module._tracer_for_provider(provider)
    provider_ref = weakref.ref(provider)

    strands_module.release_default_strands(provider)
    del tracer
    del provider
    gc.collect()

    assert provider_ref() is None


def test_strands_conversion_precedes_existing_provider_exporters():
    provider = TracerProvider()
    exporter = _SnapshotExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client = neatlogs.Client(
        api_key="client-key",
        workflow_name="client-workflow",
        disable_export=True,
        tracer_provider=provider,
    )
    agent = _local_agent("existing-exporter")

    with client.activate():
        neatlogs.strands_hooks(agent)
        agent("question-existing-exporter")

    assert {span["attributes"].get("openinference.span.kind") for span in exporter.spans} >= {
        "AGENT",
        "CHAIN",
        "LLM",
    }
    assert client.shutdown()
