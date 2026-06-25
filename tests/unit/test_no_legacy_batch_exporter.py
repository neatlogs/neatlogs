import importlib.util


def test_legacy_json_batch_exporter_module_is_removed():
    assert importlib.util.find_spec("neatlogs.core.exporter") is None
