from neatlogs.core.exporter import NeatlogsExporter


def test_legacy_exporter_accepts_base_endpoint():
    exporter = NeatlogsExporter(
        api_key="x",
        endpoint="http://localhost:4100",
        disable_export=True,
    )

    assert exporter.endpoint == "http://localhost:4100/api/data/v4/batch"


def test_legacy_exporter_does_not_duplicate_batch_path():
    exporter = NeatlogsExporter(
        api_key="x",
        endpoint="http://localhost:4100/api/data/v4/batch",
        disable_export=True,
    )

    assert exporter.endpoint == "http://localhost:4100/api/data/v4/batch"
