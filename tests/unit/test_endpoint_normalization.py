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
        ("", "https://ingest.neatlogs.com/v1/traces"),
    ],
)
def test_normalize_traces_endpoint_accepts_base_or_traces_endpoint(endpoint, expected):
    assert _normalize_traces_endpoint(endpoint) == expected


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://ingest.neatlogs.com/api/data/v4/batch",
        "https://ingest.neatlogs.com/api/data/v2",
        "ingest.neatlogs.com",
    ],
)
def test_normalize_traces_endpoint_rejects_path_endpoints(endpoint):
    with pytest.raises(ValueError, match="NEATLOGS_ENDPOINT"):
        _normalize_traces_endpoint(endpoint)
