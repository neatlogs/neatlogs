import asyncio
import types

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import neatlogs

pytest.importorskip("google.adk")


async def _run_local_adk(
    *,
    wrapped: bool,
    suffix: str = "one",
    with_tool: bool = False,
    fail: bool = False,
    started: asyncio.Event | None = None,
    release: asyncio.Event | None = None,
    sync: bool = False,
    before_wrap=None,
    after_wrap=None,
) -> str:
    from google.adk.agents import LlmAgent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    call_count = 0

    class LocalLlm(BaseLlm):
        async def generate_content_async(self, llm_request, stream=False):
            nonlocal call_count
            if call_count == 0:
                assert llm_request.contents[-1].parts[-1].text == f"question-{suffix}"
            del stream
            call_count += 1
            if started is not None:
                started.set()
            if release is not None:
                await release.wait()
            if fail:
                raise RuntimeError(f"model-failure-{suffix}")
            if with_tool and call_count == 1:
                yield LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part(
                                function_call=types.FunctionCall(
                                    name="lookup_temperature",
                                    args={"city": "Paris"},
                                )
                            )
                        ],
                    ),
                    turn_complete=False,
                    finish_reason=types.FinishReason.STOP,
                )
                return
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=f"answer-{suffix}")],
                ),
                turn_complete=True,
                finish_reason=types.FinishReason.STOP,
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=8,
                    candidates_token_count=5,
                    total_token_count=13,
                ),
            )

    def lookup_temperature(city: str) -> dict[str, str]:
        """Return deterministic weather for one city."""
        return {"city": city, "temperature": "20 C"}

    app_name = f"private-adk-{suffix}"
    user_id = f"user-{suffix}"
    session_id = f"session-{suffix}"
    sessions = InMemorySessionService()
    await sessions.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )
    runner = Runner(
        app_name=app_name,
        agent=LlmAgent(
            name=f"agent_{suffix}",
            model=LocalLlm(model=f"model-{suffix}"),
            instruction=f"instruction-{suffix}",
            tools=[lookup_temperature] if with_tool else [],
        ),
        session_service=sessions,
    )
    if before_wrap is not None:
        before_wrap(runner)
    if wrapped:
        runner = neatlogs.wrap(runner)
    if after_wrap is not None:
        after_wrap()

    message = types.Content(
        role="user",
        parts=[types.Part(text=f"question-{suffix}")],
    )

    def event_text(event) -> str:
        result = ""
        for part in getattr(getattr(event, "content", None), "parts", None) or []:
            result += getattr(part, "text", None) or ""
        return result

    output = ""
    if sync:
        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        ):
            output += event_text(event)
        return output

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
    ):
        output += event_text(event)
    return output


def _init_adk(*, automatic: bool):
    foreign_provider = TracerProvider()
    foreign_exporter = InMemorySpanExporter()
    foreign_provider.add_span_processor(SimpleSpanProcessor(foreign_exporter))
    trace_api.set_tracer_provider(foreign_provider)

    private_provider = TracerProvider()
    private_exporter = InMemorySpanExporter()
    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        instrumentations=["google_adk"] if automatic else [],
        tracer_provider=private_provider,
        register_shutdown_handlers=False,
    )
    private_provider.add_span_processor(SimpleSpanProcessor(private_exporter))
    return private_exporter, foreign_exporter


def _semantic_spans(exporter):
    return [
        span
        for span in exporter.get_finished_spans()
        if span.attributes.get("openinference.span.kind") in {"CHAIN", "AGENT", "LLM", "TOOL"}
    ]


@pytest.mark.asyncio
async def test_automatic_google_adk_uses_private_provider_with_full_hierarchy():
    private_exporter, foreign_exporter = _init_adk(automatic=True)

    with neatlogs.trace("automatic-workflow", kind="WORKFLOW"):
        output = await _run_local_adk(wrapped=False, suffix="automatic")

    assert output == "answer-automatic"
    spans = private_exporter.get_finished_spans()
    kinds = [span.attributes.get("openinference.span.kind") for span in spans]
    assert {"CHAIN", "AGENT", "LLM"}.issubset(kinds)
    assert not _semantic_spans(foreign_exporter)

    by_kind = {
        span.attributes.get("openinference.span.kind"): span
        for span in spans
        if span.attributes.get("openinference.span.kind")
    }
    workflow = next(span for span in spans if span.name == "automatic-workflow")
    assert by_kind["CHAIN"].parent.span_id == workflow.context.span_id
    assert by_kind["AGENT"].parent.span_id == by_kind["CHAIN"].context.span_id
    assert by_kind["LLM"].parent.span_id == by_kind["AGENT"].context.span_id
    assert "question-automatic" in by_kind["LLM"].attributes["input.value"]
    assert "answer-automatic" in by_kind["LLM"].attributes["output.value"]


@pytest.mark.asyncio
async def test_explicit_google_adk_wrap_activates_the_same_private_instrumentor():
    private_exporter, foreign_exporter = _init_adk(automatic=False)

    with neatlogs.trace("explicit-workflow", kind="WORKFLOW"):
        output = await _run_local_adk(wrapped=True, suffix="explicit")

    assert output == "answer-explicit"
    kinds = {
        span.attributes.get("openinference.span.kind")
        for span in private_exporter.get_finished_spans()
    }
    assert {"CHAIN", "AGENT", "LLM"}.issubset(kinds)
    assert not _semantic_spans(foreign_exporter)


@pytest.mark.asyncio
async def test_google_adk_wrap_before_init_binds_private_provider_at_invocation():
    private_provider = TracerProvider()
    private_exporter = InMemorySpanExporter()
    foreign_provider = TracerProvider()
    foreign_exporter = InMemorySpanExporter()
    foreign_provider.add_span_processor(SimpleSpanProcessor(foreign_exporter))
    trace_api.set_tracer_provider(foreign_provider)

    def initialize_after_wrap():
        neatlogs.init(
            api_key="test-key",
            disable_export=True,
            instrumentations=[],
            tracer_provider=private_provider,
            register_shutdown_handlers=False,
        )
        private_provider.add_span_processor(SimpleSpanProcessor(private_exporter))

    output = await _run_local_adk(
        wrapped=True,
        suffix="wrap_before_init",
        after_wrap=initialize_after_wrap,
    )

    assert output == "answer-wrap_before_init"
    spans = private_exporter.get_finished_spans()
    kinds = {
        span.attributes.get("openinference.span.kind")
        for span in spans
        if span.attributes.get("openinference.span.kind")
    }
    assert {"CHAIN", "AGENT", "LLM"}.issubset(kinds)
    assert any(span.name == "google_adk.runner.run_async" for span in spans)
    assert not _semantic_spans(foreign_exporter)


@pytest.mark.asyncio
async def test_google_adk_client_activation_owns_root_and_semantic_children():
    default_exporter, foreign_exporter = _init_adk(automatic=True)
    client_provider = TracerProvider()
    client_exporter = InMemorySpanExporter()
    client_provider.add_span_processor(SimpleSpanProcessor(client_exporter))
    client = neatlogs.Client(
        api_key="client-key",
        workflow_name="client-workflow",
        disable_export=True,
        tracer_provider=client_provider,
    )

    try:
        with client.activate():
            output = await _run_local_adk(wrapped=True, suffix="client")
    finally:
        client.shutdown()

    assert output == "answer-client"
    client_spans = client_exporter.get_finished_spans()
    assert any(span.name == "google_adk.runner.run_async" for span in client_spans)
    kinds = {
        span.attributes.get("openinference.span.kind")
        for span in client_spans
        if span.attributes.get("openinference.span.kind")
    }
    assert {"CHAIN", "AGENT", "LLM"}.issubset(kinds)

    wrapper = next(span for span in client_spans if span.name == "google_adk.runner.run_async")
    chain = next(
        span for span in client_spans if span.attributes.get("openinference.span.kind") == "CHAIN"
    )
    assert chain.parent.span_id == wrapper.context.span_id
    assert not _semantic_spans(default_exporter)
    assert not _semantic_spans(foreign_exporter)


@pytest.mark.asyncio
async def test_automatic_google_adk_uses_the_active_client_provider():
    default_exporter, foreign_exporter = _init_adk(automatic=True)
    client_provider = TracerProvider()
    client_exporter = InMemorySpanExporter()
    client_provider.add_span_processor(SimpleSpanProcessor(client_exporter))
    client = neatlogs.Client(
        api_key="client-key",
        workflow_name="client-workflow",
        disable_export=True,
        tracer_provider=client_provider,
    )

    try:
        with client.activate():
            with neatlogs.trace("automatic-client-workflow", kind="WORKFLOW"):
                output = await _run_local_adk(wrapped=False, suffix="automatic_client")
    finally:
        client.shutdown()

    assert output == "answer-automatic_client"
    kinds = {
        span.attributes.get("openinference.span.kind")
        for span in client_exporter.get_finished_spans()
        if span.attributes.get("openinference.span.kind")
    }
    assert {"CHAIN", "AGENT", "LLM"}.issubset(kinds)
    assert not _semantic_spans(default_exporter)
    assert not _semantic_spans(foreign_exporter)


@pytest.mark.asyncio
async def test_concurrent_google_adk_clients_keep_semantic_children_context_local():
    _init_adk(automatic=True)
    first_provider = TracerProvider()
    second_provider = TracerProvider()
    first_exporter = InMemorySpanExporter()
    second_exporter = InMemorySpanExporter()
    first_provider.add_span_processor(SimpleSpanProcessor(first_exporter))
    second_provider.add_span_processor(SimpleSpanProcessor(second_exporter))
    first = neatlogs.Client(
        api_key="first-key",
        workflow_name="first-client",
        disable_export=True,
        tracer_provider=first_provider,
    )
    second = neatlogs.Client(
        api_key="second-key",
        workflow_name="second-client",
        disable_export=True,
        tracer_provider=second_provider,
    )

    async def run(client, suffix):
        with client.activate():
            return await _run_local_adk(wrapped=True, suffix=suffix)

    try:
        outputs = await asyncio.gather(run(first, "first_client"), run(second, "second_client"))
    finally:
        first.shutdown()
        second.shutdown()

    assert outputs == ["answer-first_client", "answer-second_client"]
    first_llm = [
        span
        for span in first_exporter.get_finished_spans()
        if span.attributes.get("openinference.span.kind") == "LLM"
    ]
    second_llm = [
        span
        for span in second_exporter.get_finished_spans()
        if span.attributes.get("openinference.span.kind") == "LLM"
    ]
    assert len(first_llm) == 1
    assert len(second_llm) == 1
    assert "question-first_client" in first_llm[0].attributes["input.value"]
    assert "question-second_client" in second_llm[0].attributes["input.value"]


@pytest.mark.asyncio
async def test_concurrent_google_adk_sessions_keep_their_content_separate():
    private_exporter, _ = _init_adk(automatic=True)

    with neatlogs.trace("concurrent-workflow", kind="WORKFLOW"):
        outputs = await asyncio.gather(
            _run_local_adk(wrapped=False, suffix="alpha"),
            _run_local_adk(wrapped=False, suffix="beta"),
        )

    assert outputs == ["answer-alpha", "answer-beta"]
    llm_spans = [
        span
        for span in private_exporter.get_finished_spans()
        if span.attributes.get("openinference.span.kind") == "LLM"
    ]
    assert len(llm_spans) == 2
    pairs = {
        (
            "question-alpha" in span.attributes["input.value"],
            "answer-alpha" in span.attributes["output.value"],
        )
        for span in llm_spans
    }
    assert pairs == {(True, True), (False, False)}


@pytest.mark.asyncio
async def test_google_adk_shutdown_and_reinitialize_do_not_duplicate_spans():
    first_exporter, _ = _init_adk(automatic=True)
    await _run_local_adk(wrapped=False, suffix="first")
    assert neatlogs.shutdown()

    second_provider = TracerProvider()
    second_exporter = InMemorySpanExporter()
    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        instrumentations=["google_adk"],
        tracer_provider=second_provider,
        register_shutdown_handlers=False,
    )
    second_provider.add_span_processor(SimpleSpanProcessor(second_exporter))
    await _run_local_adk(wrapped=False, suffix="second")

    assert len([span for span in _semantic_spans(first_exporter) if span.name == "call_llm"]) == 1
    assert len([span for span in _semantic_spans(second_exporter) if span.name == "call_llm"]) == 1


@pytest.mark.asyncio
async def test_google_adk_tool_call_keeps_complete_private_hierarchy():
    private_exporter, foreign_exporter = _init_adk(automatic=True)

    with neatlogs.trace("tool-workflow", kind="WORKFLOW"):
        output = await _run_local_adk(wrapped=False, suffix="tool", with_tool=True)

    assert output == "answer-tool"
    semantic = _semantic_spans(private_exporter)
    kinds = [span.attributes.get("openinference.span.kind") for span in semantic]
    assert kinds.count("CHAIN") == 1
    assert kinds.count("AGENT") == 1
    assert kinds.count("LLM") == 2
    assert kinds.count("TOOL") == 1
    tool = next(
        span for span in semantic if span.attributes.get("openinference.span.kind") == "TOOL"
    )
    assert "Paris" in tool.attributes["input.value"]
    assert "20 C" in tool.attributes["output.value"]
    assert not _semantic_spans(foreign_exporter)


@pytest.mark.asyncio
async def test_automatic_and_explicit_google_adk_instrumentation_do_not_duplicate_spans():
    private_exporter, foreign_exporter = _init_adk(automatic=True)

    output = await _run_local_adk(wrapped=True, suffix="combined")

    assert output == "answer-combined"
    semantic = _semantic_spans(private_exporter)
    assert (
        len(
            [span for span in semantic if span.attributes.get("openinference.span.kind") == "CHAIN"]
        )
        == 1
    )
    assert (
        len(
            [span for span in semantic if span.attributes.get("openinference.span.kind") == "AGENT"]
        )
        == 1
    )
    assert (
        len([span for span in semantic if span.attributes.get("openinference.span.kind") == "LLM"])
        == 1
    )
    assert (
        len(
            [
                span
                for span in private_exporter.get_finished_spans()
                if span.name == "google_adk.runner.run_async"
            ]
        )
        == 1
    )
    assert not _semantic_spans(foreign_exporter)


@pytest.mark.asyncio
async def test_google_adk_sync_runner_uses_the_private_provider():
    private_exporter, foreign_exporter = _init_adk(automatic=True)

    output = await _run_local_adk(wrapped=True, suffix="sync", sync=True)

    assert output == "answer-sync"
    semantic = _semantic_spans(private_exporter)
    assert (
        len([span for span in semantic if span.attributes.get("openinference.span.kind") == "LLM"])
        == 1
    )
    assert (
        len(
            [
                span
                for span in private_exporter.get_finished_spans()
                if span.name == "google_adk.runner.run"
            ]
        )
        == 1
    )
    assert not _semantic_spans(foreign_exporter)


@pytest.mark.asyncio
async def test_google_adk_wrap_preserves_instance_async_override():
    private_exporter, foreign_exporter = _init_adk(automatic=True)
    override_calls = 0

    def install_override(runner):
        original = runner.run_async

        async def instance_run_async(self, *args, **kwargs):
            nonlocal override_calls
            del self
            override_calls += 1
            async for event in original(*args, **kwargs):
                yield event

        runner.run_async = types.MethodType(instance_run_async, runner)

    output = await _run_local_adk(
        wrapped=True,
        suffix="instance_override",
        before_wrap=install_override,
    )

    assert output == "answer-instance_override"
    assert override_calls == 1
    assert any(
        span.name == "google_adk.runner.run_async" for span in private_exporter.get_finished_spans()
    )
    assert not _semantic_spans(foreign_exporter)


@pytest.mark.asyncio
async def test_google_adk_model_failure_finishes_error_spans_on_private_provider():
    private_exporter, foreign_exporter = _init_adk(automatic=True)

    with pytest.raises(RuntimeError, match="model-failure-error"):
        await _run_local_adk(wrapped=True, suffix="error", fail=True)

    spans = private_exporter.get_finished_spans()
    wrapper = next(span for span in spans if span.name == "google_adk.runner.run_async")
    llm = next(span for span in spans if span.name == "call_llm")
    assert wrapper.status.status_code == StatusCode.ERROR
    assert llm.status.status_code == StatusCode.ERROR
    assert not _semantic_spans(foreign_exporter)


@pytest.mark.asyncio
async def test_google_adk_cancellation_finishes_the_wrapper_span():
    private_exporter, foreign_exporter = _init_adk(automatic=True)
    started = asyncio.Event()
    release = asyncio.Event()
    task = asyncio.create_task(
        _run_local_adk(
            wrapped=True,
            suffix="cancelled",
            started=started,
            release=release,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    wrappers = [
        span
        for span in private_exporter.get_finished_spans()
        if span.name == "google_adk.runner.run_async"
    ]
    assert len(wrappers) == 1
    assert wrappers[0].status.status_code == StatusCode.ERROR
    assert not _semantic_spans(foreign_exporter)
