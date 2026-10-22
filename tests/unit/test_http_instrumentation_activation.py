import pytest

import neatlogs
from neatlogs.errors import NeatlogsConfigurationError
from neatlogs.instrumentation.registry import INSTRUMENTATION_REGISTRY


@pytest.mark.parametrize("library", ["http", "requests", "httpx", "urllib3", "aiohttp"])
def test_http_instrumentation_is_rejected(library):
    with pytest.raises(NeatlogsConfigurationError, match="HTTP client instrumentation"):
        neatlogs.init(
            api_key="unused",
            disable_export=True,
            instrumentations=[library],
            register_shutdown_handlers=False,
        )


def test_http_instrumentors_are_not_published_in_registry():
    assert "http" not in INSTRUMENTATION_REGISTRY["tags"]
    for library in ("requests", "httpx", "urllib3", "aiohttp"):
        assert library not in INSTRUMENTATION_REGISTRY["libraries"]


@pytest.mark.parametrize(
    "kind, attributes",
    [
        ("HTTP", {}),
        ("TOOL", {"neatlogs.span.kind": "http"}),
        ("TOOL", {"openinference.span.kind": "HTTP"}),
    ],
)
def test_manual_trace_rejects_http_kind(kind, attributes):
    with pytest.raises(NeatlogsConfigurationError, match="HTTP spans are not supported"):
        with neatlogs.trace("transport", kind=kind, **attributes):
            pass
