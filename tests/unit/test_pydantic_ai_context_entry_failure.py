"""A failed context entry must not disable tracing for the rest of the task."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import neatlogs.pydantic_ai as instrumentation


@pytest.fixture
def telemetry(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("offline-entry-probe")
    monkeypatch.setattr(instrumentation, "get_tracer", lambda: tracer)
    monkeypatch.setattr(instrumentation, "_NATIVE_AUTO_INSTRUMENTATION_ACTIVE", False)
    monkeypatch.setattr(instrumentation, "_CLASS_HOOKS_INSTALLED", True)
    yield tracer, exporter
    provider.shutdown()


class Agent:
    def __init__(self, error=None):
        self.error = error

    @asynccontextmanager
    async def run_stream(self, prompt):
        if self.error is not None:
            raise self.error
        yield SimpleNamespace(output="stream result")

    @asynccontextmanager
    async def iter(self, prompt):
        if self.error is not None:
            raise self.error
        yield SimpleNamespace(result=SimpleNamespace(output="iter result"))

    async def run(self, prompt):
        return SimpleNamespace(output="next run")


def assert_failed_span(span, error):
    if isinstance(error, asyncio.CancelledError):
        assert span.status.status_code == StatusCode.UNSET
        assert span.attributes["neatlogs.stream.cancelled"] is True
    else:
        assert span.status.status_code == StatusCode.ERROR
        assert span.events[0].name == "exception"


@pytest.mark.parametrize("method", ["run_stream", "iter"])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
def test_failed_agent_entry_restores_guard_parent_and_ends_span(telemetry, method, error_type):
    tracer, exporter = telemetry
    error = error_type("entry failed")

    async def go():
        agent = instrumentation.wrap_pydantic_ai(Agent(error))
        with tracer.start_as_current_span("outer") as parent:
            with pytest.raises(error_type) as caught:
                async with getattr(agent, method)("fail"):
                    pytest.fail("failed entry yielded a result")
            assert caught.value is error
            assert instrumentation._agent_span_active.get() is False
            assert trace.get_current_span().get_span_context() == parent.get_span_context()
            failed = exporter.get_finished_spans()
            assert len(failed) == 1
            assert_failed_span(failed[0], error)
            assert (await agent.run("recover")).output == "next run"
        spans = exporter.get_finished_spans()
        assert [span.name for span in spans] == [
            f"pydantic_ai.agent.{method}",
            "pydantic_ai.agent.run",
            "outer",
        ]
        assert all(span.parent.span_id == parent.get_span_context().span_id for span in spans[:2])

    asyncio.run(go())


@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
def test_failed_model_entry_restores_parent_and_ends_span(telemetry, error_type):
    tracer, exporter = telemetry
    error = error_type("model entry failed")

    @asynccontextmanager
    async def context():
        raise error
        yield  # pragma: no cover

    async def go():
        with tracer.start_as_current_span("outer") as parent:
            cm = instrumentation._ModelStreamCtx(context(), {}, [], lambda *_: None)
            with pytest.raises(error_type) as caught:
                async with cm:
                    pytest.fail("failed entry yielded a result")
            assert caught.value is error
            assert trace.get_current_span().get_span_context() == parent.get_span_context()
            spans = exporter.get_finished_spans()
            assert len(spans) == 1
            assert spans[0].name == "pydantic_ai.model.request_stream"
            assert_failed_span(spans[0], error)

    asyncio.run(go())


@pytest.mark.parametrize("method", ["run_stream", "iter", "model"])
def test_successful_entry_keeps_context_until_exit(telemetry, method):
    tracer, exporter = telemetry

    @asynccontextmanager
    async def context():
        yield SimpleNamespace(get=lambda: "model result")

    async def go():
        with tracer.start_as_current_span("outer") as parent:
            if method == "model":
                cm = instrumentation._ModelStreamCtx(context(), {}, [], lambda *_: None)
            else:
                agent = instrumentation.wrap_pydantic_ai(Agent())
                cm = getattr(agent, method)("ok")
            async with cm as value:
                assert value is not None
                assert trace.get_current_span().get_span_context() != parent.get_span_context()
                assert not exporter.get_finished_spans()
            assert trace.get_current_span().get_span_context() == parent.get_span_context()
            assert instrumentation._agent_span_active.get() is False
            spans = exporter.get_finished_spans()
            assert len(spans) == 1
            assert spans[0].status.status_code == StatusCode.OK

    asyncio.run(go())


def test_real_testmodel_stream_entry_failure_does_not_disable_next_run(telemetry):
    pytest.importorskip("pydantic_ai")
    from pydantic_ai import Agent as PydanticAgent
    from pydantic_ai.models.test import TestModel

    tracer, exporter = telemetry
    error = RuntimeError("offline model refused stream")

    class RejectingStreamModel(TestModel):
        @asynccontextmanager
        async def request_stream(self, *args, **kwargs):
            raise error
            yield  # pragma: no cover

    async def go():
        agent = instrumentation.wrap_pydantic_ai(
            PydanticAgent(RejectingStreamModel(custom_output_text="recovered"))
        )
        with tracer.start_as_current_span("outer") as parent:
            with pytest.raises(RuntimeError) as caught:
                async with agent.run_stream("fail"):
                    pytest.fail("failed entry yielded a result")
            assert caught.value is error
            assert instrumentation._agent_span_active.get() is False
            assert trace.get_current_span().get_span_context() == parent.get_span_context()
            assert len(exporter.get_finished_spans()) == 2
            await agent.run("recover")
            assert len(exporter.get_finished_spans()) == 3

    asyncio.run(go())
