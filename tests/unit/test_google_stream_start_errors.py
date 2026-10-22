import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from neatlogs._wrap_utils import set_neatlogs_provider
from neatlogs.google_genai import wrap_google_genai_client
from neatlogs.vertex_ai import wrap_vertex_ai_client


class _Models:
    def generate_content(self, *args, **kwargs):
        return None

    def generate_content_stream(self, *args, **kwargs):
        raise RuntimeError("provider rejected stream")


class _AsyncModels:
    async def generate_content(self, *args, **kwargs):
        return None

    async def generate_content_stream(self, *args, **kwargs):
        raise RuntimeError("provider rejected stream")


class _Namespace:
    pass


@pytest.mark.parametrize("wrapper", [wrap_google_genai_client, wrap_vertex_ai_client])
def test_sync_stream_start_failure_ends_error_span(wrapper):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_neatlogs_provider(provider)

    client = _Namespace()
    client.models = _Models()
    client.aio = _Namespace()
    client.aio.models = _AsyncModels()
    wrapper(client)

    with pytest.raises(RuntimeError, match="provider rejected stream"):
        client.models.generate_content_stream(model="test", contents="hi")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code.name == "ERROR"
