import asyncio
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from neatlogs import agno
from neatlogs._wrap_utils import set_neatlogs_provider


@pytest.fixture
def span_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_neatlogs_provider(provider)
    yield exporter
    set_neatlogs_provider(None)
    provider.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow", [False, True])
async def test_arun_cancellation_ends_span_immediately(span_exporter, workflow):
    class Entity:
        name = "cancelled"
        model = None

        async def arun(self, *args, **kwargs):
            await asyncio.Event().wait()

    entity = Entity()
    if workflow:
        entity.steps = []
        agno._patch_workflow(entity)
        expected_name = "agno.workflow.arun"
    else:
        agno._patch_agent(entity)
        expected_name = "agno.entity.arun"

    task = asyncio.create_task(entity.arun("wait forever"))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [expected_name]
    assert spans[0].status.status_code.name == "ERROR"
