import asyncio
import threading
from types import SimpleNamespace

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import neatlogs
from neatlogs.client import Client
from neatlogs.instrumentation.manager import _ManagedSpanProcessor

pytest.importorskip("pydantic_ai")
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel


class _SnapshotExporter(SpanExporter):
    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(
            SimpleNamespace(
                name=span.name,
                attributes=dict(span.attributes or {}),
                context=span.context,
                parent=span.parent,
                status=span.status,
                events=tuple(span.events),
            )
            for span in spans
        )
        return SpanExportResult.SUCCESS

    def get_finished_spans(self):
        return tuple(self.spans)


class _BlockingProcessor:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.stopped = threading.Event()

    def on_start(self, span, parent_context=None):
        pass

    def on_end(self, span):
        self.started.set()
        assert self.release.wait(timeout=2)

    def shutdown(self):
        self.stopped.set()

    def force_flush(self, timeout_millis=30000):
        return True


@pytest.fixture(autouse=True)
def _restore_pydantic_ai_default():
    previous = Agent._instrument_default
    yield
    Agent.instrument_all(previous)


def _init_with_exporter(*, foreign_exporter=None):
    private_provider = TracerProvider()
    private_exporter = _SnapshotExporter()

    if foreign_exporter is not None:
        foreign_provider = TracerProvider()
        foreign_provider.add_span_processor(SimpleSpanProcessor(foreign_exporter))
        trace_api.set_tracer_provider(foreign_provider)

    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        instrumentations=["pydantic_ai"],
        tracer_provider=private_provider,
        register_shutdown_handlers=False,
    )
    private_provider.add_span_processor(SimpleSpanProcessor(private_exporter))
    return private_provider, private_exporter


def _spans_by_kind(spans):
    by_kind = {}
    for span in spans:
        kind = span.attributes.get("openinference.span.kind")
        if kind is not None:
            by_kind.setdefault(kind, []).append(span)
    return by_kind


def _only_span_by_kind(spans, kind):
    matching = _spans_by_kind(spans).get(kind, [])
    assert len(matching) == 1
    return matching[0]


def _is_descendant_of(span, ancestor, spans):
    if span.context.trace_id != ancestor.context.trace_id:
        return False

    spans_by_context = {(item.context.trace_id, item.context.span_id): item for item in spans}
    parent = span.parent
    while parent is not None:
        if (
            parent.trace_id == ancestor.context.trace_id
            and parent.span_id == ancestor.context.span_id
        ):
            return True
        parent_span = spans_by_context.get((parent.trace_id, parent.span_id))
        parent = parent_span.parent if parent_span is not None else None
    return False


def test_native_pydantic_ai_spans_are_normalized_and_parented_to_workflow():
    private_provider, exporter = _init_with_exporter()
    agent = Agent(TestModel(custom_output_text="deterministic answer"), name="support-agent")

    with neatlogs.trace("support-workflow", kind="WORKFLOW"):
        result = agent.run_sync("answer this")

    assert result.output == "deterministic answer"
    spans = exporter.get_finished_spans()
    workflow = next(span for span in spans if span.name == "support-workflow")
    agent_span = _only_span_by_kind(spans, "AGENT")
    model_span = _only_span_by_kind(spans, "LLM")

    processor_names = [
        type(processor).__name__
        for processor in private_provider._active_span_processor._span_processors
    ]
    assert processor_names[:2] == ["ProviderPreprocessor", "NeatlogsSpanProcessor"]
    assert _is_descendant_of(agent_span, workflow, spans)
    assert _is_descendant_of(model_span, agent_span, spans)
    assert agent_span.attributes["openinference.span.kind"] == "AGENT"
    assert agent_span.attributes["input.value"] == "answer this"
    assert agent_span.attributes["output.value"] == "deterministic answer"
    assert model_span.attributes["openinference.span.kind"] == "LLM"
    assert model_span.attributes["llm.model_name"] == "test"
    assert model_span.attributes["input.value"] == "answer this"
    assert model_span.attributes["output.value"] == "deterministic answer"
    assert agent_span.attributes["neatlogs.agent.input"] == "answer this"
    assert model_span.attributes["neatlogs.llm.output"] == "deterministic answer"


def test_native_normalization_precedes_existing_provider_exporters():
    provider = TracerProvider()
    exporter = _SnapshotExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        instrumentations=["pydantic_ai"],
        tracer_provider=provider,
        register_shutdown_handlers=False,
    )
    agent = Agent(TestModel(custom_output_text="normalized first"), name="ordered-agent")
    with neatlogs.trace("ordered-workflow", kind="WORKFLOW"):
        agent.run_sync("normalize before export")

    spans = exporter.get_finished_spans()
    agent_span = _only_span_by_kind(spans, "AGENT")
    model_span = _only_span_by_kind(spans, "LLM")
    assert agent_span.attributes["input.value"] == "normalize before export"
    assert model_span.attributes["output.value"] == "normalized first"


def test_native_pydantic_ai_spans_do_not_reach_the_global_provider():
    foreign_exporter = InMemorySpanExporter()
    _, private_exporter = _init_with_exporter(foreign_exporter=foreign_exporter)
    foreign_tracer = trace_api.get_tracer("foreign.application")
    agent = Agent(TestModel(custom_output_text="private"), name="private-agent")

    with foreign_tracer.start_as_current_span("foreign-root"):
        with neatlogs.trace("private-workflow", kind="WORKFLOW"):
            agent.run_sync("stay private")

    private_spans = private_exporter.get_finished_spans()
    foreign_names = {span.name for span in foreign_exporter.get_finished_spans()}

    assert any(span.name == "private-workflow" for span in private_spans)
    assert len(_spans_by_kind(private_spans)["AGENT"]) == 1
    assert len(_spans_by_kind(private_spans)["LLM"]) == 1
    assert foreign_names == {"foreign-root"}


def test_native_pydantic_ai_spans_follow_the_active_client_provider():
    _, default_exporter = _init_with_exporter()
    client_provider = TracerProvider()
    client_exporter = _SnapshotExporter()
    client = Client(
        api_key="client-key",
        workflow_name="client-workflow",
        disable_export=True,
        tracer_provider=client_provider,
    )
    client_provider.add_span_processor(SimpleSpanProcessor(client_exporter))
    agent = neatlogs.wrap(Agent(TestModel(custom_output_text="client answer"), name="client-agent"))

    try:
        with client.activate():
            with neatlogs.trace("client-run", kind="WORKFLOW"):
                result = agent.run_sync("route to client")
    finally:
        client.shutdown()

    assert result.output == "client answer"
    assert {span.name for span in default_exporter.get_finished_spans()} == set()

    client_spans = client_exporter.get_finished_spans()
    workflow = next(span for span in client_spans if span.name == "client-run")
    agent_span = _only_span_by_kind(client_spans, "AGENT")
    model_span = _only_span_by_kind(client_spans, "LLM")
    assert agent_span.attributes["input.value"] == "route to client"
    assert model_span.attributes["output.value"] == "client answer"
    assert _is_descendant_of(agent_span, workflow, client_spans)
    assert _is_descendant_of(model_span, agent_span, client_spans)


def test_managed_processor_shutdown_waits_for_active_normalization():
    delegate = _BlockingProcessor()
    managed = _ManagedSpanProcessor(delegate)
    callback = threading.Thread(target=managed.on_end, args=(object(),))
    shutdown = threading.Thread(target=managed.shutdown)

    callback.start()
    assert delegate.started.wait(timeout=1)
    shutdown.start()
    assert not delegate.stopped.wait(timeout=0.05)

    delegate.release.set()
    callback.join(timeout=1)
    shutdown.join(timeout=1)

    assert not callback.is_alive()
    assert not shutdown.is_alive()
    assert delegate.stopped.is_set()


def test_global_agent_opt_out_after_init_is_respected():
    _, exporter = _init_with_exporter()
    Agent.instrument_all(False)
    agent = Agent(TestModel(custom_output_text="not instrumented"), name="opted-out-agent")

    with neatlogs.trace("opt-out-workflow", kind="WORKFLOW"):
        agent.run_sync("skip native spans")

    assert [span.name for span in exporter.get_finished_spans()] == ["opt-out-workflow"]


def test_shutdown_restores_preexisting_global_opt_out():
    Agent.instrument_all(False)

    _init_with_exporter()
    assert isinstance(Agent._instrument_default, InstrumentationSettings)

    assert neatlogs.shutdown()
    assert Agent._instrument_default is False


def test_shutdown_restores_the_previous_pydantic_ai_default():
    previous = InstrumentationSettings(tracer_provider=TracerProvider(), include_content=False)
    Agent.instrument_all(previous)

    _init_with_exporter()

    installed = Agent._instrument_default
    assert isinstance(installed, InstrumentationSettings)
    assert installed is not previous

    assert neatlogs.shutdown()
    assert Agent._instrument_default is previous


def test_shutdown_preserves_a_user_override_made_after_init():
    _init_with_exporter()
    replacement = InstrumentationSettings(tracer_provider=TracerProvider(), include_content=False)
    Agent.instrument_all(replacement)

    assert neatlogs.shutdown()
    assert Agent._instrument_default is replacement


def test_wrap_does_not_duplicate_native_pydantic_ai_spans():
    _, exporter = _init_with_exporter()
    agent = neatlogs.wrap(Agent(TestModel(custom_output_text="one answer"), name="wrapped-agent"))

    with neatlogs.trace("wrapped-workflow", kind="WORKFLOW"):
        agent.run_sync("one question")

    spans = exporter.get_finished_spans()
    assert len(_spans_by_kind(spans)["AGENT"]) == 1
    assert len(_spans_by_kind(spans)["LLM"]) == 1
    assert not any(span.name.startswith("pydantic_ai.") for span in spans)


def test_agent_wrapped_before_init_does_not_duplicate_native_spans():
    agent = neatlogs.wrap(
        Agent(TestModel(custom_output_text="one answer"), name="prewrapped-agent")
    )
    _, exporter = _init_with_exporter()

    with neatlogs.trace("prewrapped-workflow", kind="WORKFLOW"):
        agent.run_sync("one question")

    spans = exporter.get_finished_spans()
    assert len(_spans_by_kind(spans)["AGENT"]) == 1
    assert len(_spans_by_kind(spans)["LLM"]) == 1
    assert not any(span.name.startswith("pydantic_ai.") for span in spans)


def test_native_pydantic_ai_tool_span_has_expected_hierarchy():
    _, exporter = _init_with_exporter()
    agent = Agent(TestModel(), name="tool-agent")

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        return a + b

    with neatlogs.trace("tool-workflow", kind="WORKFLOW"):
        agent.run_sync("add two numbers")

    spans = exporter.get_finished_spans()
    by_kind = _spans_by_kind(spans)

    agent_span = _only_span_by_kind(spans, "AGENT")
    tool_span = _only_span_by_kind(spans, "TOOL")
    llm_spans = by_kind["LLM"]

    assert len(llm_spans) == 2
    assert _is_descendant_of(tool_span, agent_span, spans)
    assert all(_is_descendant_of(llm_span, agent_span, spans) for llm_span in llm_spans)
    for chain_span in by_kind.get("CHAIN", []):
        assert _is_descendant_of(chain_span, agent_span, spans)
    assert tool_span.attributes["tool.name"] == "add"
    assert tool_span.attributes["input.value"] == '{"a":0,"b":0}'
    assert tool_span.attributes["output.value"] == "0"


@pytest.mark.asyncio
async def test_native_pydantic_ai_async_run_emits_one_agent_and_model_span():
    _, exporter = _init_with_exporter()
    agent = Agent(TestModel(custom_output_text="async answer"), name="async-agent")

    with neatlogs.trace("async-workflow", kind="WORKFLOW"):
        result = await agent.run("async question")

    assert result.output == "async answer"
    spans = exporter.get_finished_spans()
    assert sum(span.attributes.get("openinference.span.kind") == "AGENT" for span in spans) == 1
    assert sum(span.attributes.get("openinference.span.kind") == "LLM" for span in spans) == 1


@pytest.mark.asyncio
async def test_native_pydantic_ai_model_error_finishes_error_spans():
    class FailingModel(TestModel):
        async def request(self, *args, **kwargs):
            raise RuntimeError("controlled model failure")

    _, exporter = _init_with_exporter()
    agent = Agent(FailingModel(), name="failing-agent")

    with pytest.raises(RuntimeError, match="controlled model failure"):
        with neatlogs.trace("failure-workflow", kind="WORKFLOW"):
            await agent.run("fail")

    spans = exporter.get_finished_spans()
    assert sum(span.status.status_code == StatusCode.ERROR for span in spans) >= 1
    assert any(event.name == "exception" for span in spans for event in span.events)


@pytest.mark.asyncio
async def test_native_pydantic_ai_cancellation_leaves_no_unfinished_spans():
    started = asyncio.Event()

    class BlockingModel(TestModel):
        async def request(self, *args, **kwargs):
            started.set()
            await asyncio.Event().wait()

    _, exporter = _init_with_exporter()
    agent = Agent(BlockingModel(), name="cancelled-agent")
    task = asyncio.create_task(agent.run("wait forever"))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    by_kind = _spans_by_kind(exporter.get_finished_spans())
    assert len(by_kind["AGENT"]) == 1
    assert len(by_kind["LLM"]) == 1


def test_shutdown_then_reinitialize_uses_the_new_private_provider():
    _, first_exporter = _init_with_exporter()
    Agent(TestModel(custom_output_text="first"), name="first-agent").run_sync("first")
    assert neatlogs.shutdown()

    _, second_exporter = _init_with_exporter()
    Agent(TestModel(custom_output_text="second"), name="second-agent").run_sync("second")

    first_by_kind = _spans_by_kind(first_exporter.get_finished_spans())
    second_by_kind = _spans_by_kind(second_exporter.get_finished_spans())
    assert set(first_by_kind) == {"AGENT", "LLM"}
    assert set(second_by_kind) == {"AGENT", "LLM"}
    assert len(first_by_kind["AGENT"]) == len(first_by_kind["LLM"]) == 1
    assert len(second_by_kind["AGENT"]) == len(second_by_kind["LLM"]) == 1


def test_repeated_identical_init_does_not_duplicate_native_processors():
    provider, exporter = _init_with_exporter()
    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        instrumentations=["pydantic_ai"],
        tracer_provider=provider,
        register_shutdown_handlers=False,
    )

    Agent(TestModel(custom_output_text="once"), name="single-agent").run_sync("once")

    spans = exporter.get_finished_spans()
    assert len(_spans_by_kind(spans)["AGENT"]) == 1
    assert len(_spans_by_kind(spans)["LLM"]) == 1


@pytest.mark.asyncio
async def test_manual_wrapper_ends_agent_span_when_run_is_cancelled():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        instrumentations=[],
        tracer_provider=provider,
        register_shutdown_handlers=False,
    )
    started = asyncio.Event()

    class BlockingModel(TestModel):
        async def request(self, *args, **kwargs):
            started.set()
            await asyncio.Event().wait()

    agent = neatlogs.wrap(Agent(BlockingModel(), name="cancelled-agent"))
    task = asyncio.create_task(agent.run("wait forever"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["pydantic_ai.agent.run"]
    assert spans[0].attributes["neatlogs.stream.cancelled"] is True
    assert spans[0].status.status_code is StatusCode.UNSET
