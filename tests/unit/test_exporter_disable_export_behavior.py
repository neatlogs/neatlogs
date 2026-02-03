from unittest.mock import Mock, patch

from neatlogs.core.exporter import NeatlogsExporter


def test_flush_combined_batch_skips_http_when_disable_export_true() -> None:
    exporter = NeatlogsExporter(
        api_key="test",
        endpoint="http://localhost:3000/api/data/v4/batch",
        flush_interval=0.01,  # keep shutdown fast in unit tests
        disable_export=True,
    )
    try:
        with patch("neatlogs.core.exporter.requests.post") as post:
            with exporter._lock:
                with exporter._metrics_lock:
                    exporter._batch.append({"span": 1})
                    exporter._metrics_batch.append({"metric": 1})
                    exporter._flush_combined_batch()

            post.assert_not_called()
            assert exporter._batch == []
            assert exporter._metrics_batch == []
    finally:
        exporter.shutdown()


def test_flush_combined_batch_posts_when_disable_export_false(monkeypatch) -> None:
    # tests/conftest.py sets NEATLOGS_DISABLE_EXPORT=true for all tests; override for this test.
    monkeypatch.setenv("NEATLOGS_DISABLE_EXPORT", "false")

    exporter = NeatlogsExporter(
        api_key="test",
        endpoint="http://localhost:3000/api/data/v4/batch",
        flush_interval=0.01,
        disable_export=False,
    )
    try:
        with patch("neatlogs.core.exporter.requests.post") as post:
            resp = Mock()
            resp.status_code = 200
            post.return_value = resp

            with exporter._lock:
                with exporter._metrics_lock:
                    exporter._batch.append({"span": 1})
                    exporter._flush_combined_batch()

            post.assert_called_once()
    finally:
        exporter.shutdown()
