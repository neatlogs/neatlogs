from neatlogs.core.exporter import NeatlogsExporter


def test_disable_export_flag_defaults_to_false(monkeypatch):
    monkeypatch.delenv("NEATLOGS_DISABLE_EXPORT", raising=False)
    exp = NeatlogsExporter(api_key="x", endpoint="http://example", disable_export=False)
    assert exp.disable_export is False


def test_disable_export_flag_from_env(monkeypatch):
    monkeypatch.setenv("NEATLOGS_DISABLE_EXPORT", "true")
    exp = NeatlogsExporter(api_key="x", endpoint="http://example")
    assert exp.disable_export is True
