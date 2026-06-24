import pytest

from neatlogs._wrap_utils import _normalize_traces_endpoint


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("https://ingest.neatlogs.com", "https://ingest.neatlogs.com/v1/traces"),
        ("https://ingest.neatlogs.com/", "https://ingest.neatlogs.com/v1/traces"),
        (
            "https://ingest.neatlogs.com/v1/traces",
            "https://ingest.neatlogs.com/v1/traces",
        ),
        (
            "https://ingest.neatlogs.com/api/data/v4/batch",
            "https://ingest.neatlogs.com/v1/traces",
        ),
        ("http://localhost:4100/api/data/v2", "http://localhost:4100/v1/traces"),
    ],
)
def test_normalize_traces_endpoint_strips_legacy_paths(endpoint, expected):
    assert _normalize_traces_endpoint(endpoint) == expected
