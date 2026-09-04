import importlib

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
