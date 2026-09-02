"""One-peer-at-a-time observability compatibility smoke test."""

import importlib
import os

from opentelemetry import trace as otel_trace

import neatlogs
from neatlogs._wrap_utils import get_neatlogs_provider


def test_peer_import_does_not_change_neatlogs_private_provider_ownership():
    module_name = os.environ["NEATLOGS_COMPAT_MODULE"]
    global_before = otel_trace.get_tracer_provider()

    importlib.import_module(module_name)
    neatlogs.init(
        api_key="compatibility-only",
        instrumentations=[],
        disable_export=True,
        register_shutdown_handlers=False,
    )
    try:
        private = get_neatlogs_provider()
        assert private is not None
        assert private is not global_before
        assert otel_trace.get_tracer_provider() is global_before
        tracer = private.get_tracer("neatlogs.compatibility")
        tracer.start_span(f"coexists-with-{module_name}").end()
        assert neatlogs.flush(timeout_millis=1000)
    finally:
        assert neatlogs.shutdown(timeout_millis=1000)
