from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider

from neatlogs.instrumentation.manager import InstrumentationManager


@pytest.mark.parametrize(
    ("library", "module_name", "class_name"),
    [
        ("requests", "opentelemetry.instrumentation.requests", "RequestsInstrumentor"),
        ("httpx", "opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor"),
        ("urllib3", "opentelemetry.instrumentation.urllib3", "URLLib3Instrumentor"),
        ("aiohttp", "opentelemetry.instrumentation.aiohttp_client", "AioHttpClientInstrumentor"),
    ],
)
def test_explicit_http_key_activates_otel_instrumentor(
    monkeypatch, library, module_name, class_name
):
    calls = []

    class FakeInstrumentor:
        def instrument(self, **kwargs):
            calls.append(kwargs)

    manager = InstrumentationManager(
        TracerProvider(), excluded_urls="https://dev-cloud.neatlogs.com"
    )
    monkeypatch.setattr(manager, "_is_library_installed", lambda name: name == library)
    monkeypatch.setattr(
        "neatlogs.instrumentation.manager.importlib.import_module",
        lambda name: (
            SimpleNamespace(**{class_name: FakeInstrumentor}) if name == module_name else None
        ),
    )

    manager.instrument(libraries=[library])

    assert manager.instrumented == {library}
    assert calls == [
        {
            "tracer_provider": manager.provider,
            "excluded_urls": "https://dev-cloud.neatlogs.com",
        }
    ]


def test_empty_instrumentation_list_does_not_activate_any_client(monkeypatch):
    manager = InstrumentationManager(TracerProvider())
    monkeypatch.setattr(
        manager,
        "_instrument_library",
        lambda *args, **kwargs: pytest.fail("HTTP clients must remain opt-in"),
    )

    manager.instrument(libraries=[])

    assert manager.instrumented == set()
