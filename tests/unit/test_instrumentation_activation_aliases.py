import importlib
from types import SimpleNamespace

from opentelemetry.sdk.trace import TracerProvider

from neatlogs.instrumentation.manager import InstrumentationManager
from neatlogs.instrumentation.registry import INSTRUMENTATION_REGISTRY


def test_autogen_uses_agentchat_adapter_and_class():
    manager = InstrumentationManager(TracerProvider())

    assert (
        INSTRUMENTATION_REGISTRY["libraries"]["autogen"]["openinference"]
        == "openinference.instrumentation.autogen_agentchat"
    )
    assert (
        manager._get_instrumentor_class_name("autogen", "openinference")
        == "AutogenAgentChatInstrumentor"
    )


def test_application_import_aliases_match_published_modules(monkeypatch):
    imported = []

    def fake_import(name):
        imported.append(name)
        if name not in {"autogen_agentchat", "portkey_ai"}:
            raise ImportError(name)
        return object()

    monkeypatch.setattr(importlib, "import_module", fake_import)
    manager = InstrumentationManager(TracerProvider())

    assert manager._is_library_installed("autogen") is True
    assert manager._is_library_installed("portkey") is True
    assert imported == ["autogen_agentchat", "portkey_ai"]


def test_application_import_aliases_report_missing_dependencies(monkeypatch):
    def missing_import(name):
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", missing_import)
    manager = InstrumentationManager(TracerProvider())

    assert manager._is_library_installed("autogen") is False
    assert manager._is_library_installed("portkey") is False


def test_qdrant_uses_configured_openllmetry_instrumentor(monkeypatch):
    calls = []

    class QdrantInstrumentor:
        def instrument(self, **kwargs):
            calls.append(("instrument", kwargs))

        def uninstrument(self):
            calls.append(("uninstrument", {}))

    module = SimpleNamespace(QdrantInstrumentor=QdrantInstrumentor)
    monkeypatch.setattr(importlib, "import_module", lambda name: module)

    manager = InstrumentationManager(TracerProvider())
    monkeypatch.setattr(manager, "_is_library_installed", lambda library: True)

    manager.instrument(libraries=["qdrant"])
    assert manager.instrumented == {"qdrant"}
    assert calls == [("instrument", {"tracer_provider": manager.provider})]

    manager.uninstrument_all()
    assert calls[-1] == ("uninstrument", {})
