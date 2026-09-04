import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import aiohttp
import httpx
import pytest
import requests
import urllib3
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from neatlogs.instrumentation.manager import InstrumentationManager


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"ok"
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def local_http_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/health"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


async def _aiohttp_get(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            assert await response.text() == "ok"


def _request(library, url):
    if library == "requests":
        assert requests.get(url, timeout=2).text == "ok"
    elif library == "httpx":
        assert httpx.get(url, timeout=2).text == "ok"
    elif library == "urllib3":
        assert urllib3.PoolManager().request("GET", url, timeout=2).data == b"ok"
    else:
        asyncio.run(_aiohttp_get(url))


@pytest.mark.parametrize("library", ["requests", "httpx", "urllib3", "aiohttp"])
def test_explicit_http_instrumentation_emits_one_nested_client_span(library, local_http_url):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    manager = InstrumentationManager(provider, excluded_urls="dev-cloud.neatlogs.com")

    try:
        manager.instrument(libraries=[library])
        tracer = provider.get_tracer("neatlogs.http-test")
        with tracer.start_as_current_span("workflow") as root:
            _request(library, local_http_url)

        children = [
            span
            for span in exporter.get_finished_spans()
            if span.parent and span.parent.span_id == root.context.span_id
        ]
        assert len(children) == 1
        assert children[0].kind.name == "CLIENT"
    finally:
        manager.uninstrument_all()
        provider.shutdown()


def test_empty_instrumentation_list_emits_no_client_span(local_http_url):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    manager = InstrumentationManager(provider)

    try:
        manager.instrument(libraries=[])
        with provider.get_tracer("neatlogs.http-test").start_as_current_span("workflow"):
            assert requests.get(local_http_url, timeout=2).text == "ok"

        assert [span.name for span in exporter.get_finished_spans()] == ["workflow"]
    finally:
        manager.uninstrument_all()
        provider.shutdown()
