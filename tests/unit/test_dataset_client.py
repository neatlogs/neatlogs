import json
from uuid import uuid4

import httpx
import pytest

from neatlogs import datasets

PROJECT, DATASET, VERSION, CASE, REVISION = [str(uuid4()) for _ in range(5)]
CONFIG = {"project_id": PROJECT, "token": "public-only", "base_url": "https://app.example.com"}


@pytest.fixture
def requests(monkeypatch):
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return httpx.Response(
            201,
            json={"success": True, "data": {"datasetId": DATASET, "versionId": VERSION}},
        )

    monkeypatch.setattr("neatlogs.datasets._api.httpx.request", request)
    return calls


def test_create_empty_dataset_does_not_need_telemetry(requests):
    result = datasets.create("  Regression  ", **CONFIG)
    assert result == {"datasetId": DATASET, "versionId": VERSION}
    method, url, request = requests[0]
    body = json.loads(request["content"])
    assert method == "POST" and url.endswith("/api/v1/public/datasets")
    assert body["name"] == "Regression" and body["cases"] == []
    assert body["idempotencyKey"] == request["headers"]["Idempotency-Key"]
    assert request["headers"]["Authorization"] == "Bearer public-only"
    assert request["headers"]["x-project-id"] == PROJECT
    assert request["follow_redirects"] is False


def test_create_preserves_case_outcome_and_json_input(requests):
    case = {
        "input": {"question": "What is 2+2?"},
        "observedOutput": "4",
        "expected": {"value": "Answer four", "status": "draft", "source": "human"},
    }
    datasets.create("Math", cases=[case], **CONFIG)
    assert json.loads(requests[0][2]["content"])["cases"] == [case]


def test_update_outcomes_creates_version_without_removals(requests):
    datasets.update_outcomes(
        DATASET,
        base_version_id=VERSION,
        changes=[{"caseId": CASE, "revisionId": REVISION, "outcome": {"answer": "4"}}],
        **CONFIG,
    )
    body = json.loads(requests[0][2]["content"])
    assert requests[0][1].endswith(f"/datasets/{DATASET}/versions")
    assert body["baseVersionId"] == VERSION and body["removedCaseIds"] == []
    assert body["expectedChanges"] == [
        {
            "caseId": CASE,
            "revisionId": REVISION,
            "expected": {"value": {"answer": "4"}, "status": "draft", "source": "human"},
        }
    ]


def test_null_outcome_clears_and_absent_outcome_is_invalid(requests):
    datasets.update_outcomes(
        DATASET,
        base_version_id=VERSION,
        changes=[{"caseId": CASE, "revisionId": REVISION, "outcome": None}],
        **CONFIG,
    )
    assert json.loads(requests[0][2]["content"])["expectedChanges"][0]["expected"] is None
    with pytest.raises(ValueError, match="outcome"):
        datasets.update_outcomes(
            DATASET,
            base_version_id=VERSION,
            changes=[{"caseId": CASE, "revisionId": REVISION}],
            **CONFIG,
        )


@pytest.mark.parametrize("code", [401, 403, 409, 413])
def test_api_failure_is_visible_and_body_does_not_leak(requests, monkeypatch, code):
    monkeypatch.setattr(
        "neatlogs.datasets._api.httpx.request",
        lambda *a, **k: httpx.Response(code, text="private response body"),
    )
    with pytest.raises(datasets.DatasetError) as exc:
        datasets.create("Test", **CONFIG)
    assert exc.value.status_code == code
    assert exc.value.idempotency_key
    assert "private" not in str(exc.value) and "public-only" not in str(exc.value)


def test_retries_freeze_request_bytes_and_key(requests, monkeypatch):
    original = __import__("neatlogs.datasets._api", fromlist=["httpx"]).httpx.request
    calls = []

    def flaky(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return httpx.Response(503)
        return original(*args, **kwargs)

    monkeypatch.setattr("neatlogs.datasets._api.httpx.request", flaky)
    monkeypatch.setattr("neatlogs.datasets._api.time.sleep", lambda _: None)
    key = str(uuid4())
    datasets.create("Test", idempotency_key=key, **CONFIG)
    assert calls[0]["content"] == calls[1]["content"]
    assert calls[0]["headers"]["Idempotency-Key"] == key


@pytest.mark.parametrize("value", [float("nan"), {1: "bad"}, object(), (1, 2)])
def test_non_json_values_rejected_before_request(requests, value):
    with pytest.raises(ValueError):
        datasets.create("Test", cases=[{"input": value}], **CONFIG)
    assert not requests


def test_byte_cap_uses_actual_serialized_utf8_bytes(requests):
    with pytest.raises(ValueError, match="64 KiB"):
        datasets.create("Test", cases=[{"input": "🙂" * 10000}], **CONFIG)
    assert not requests


@pytest.mark.parametrize(
    "origin",
    [
        "http://example.com",
        "https://token@example.com",
        "https://app.example.com/path",
        "https://app.example.com?token=x",
    ],
)
def test_unsafe_origin_rejected(requests, origin):
    with pytest.raises(ValueError, match="origin"):
        datasets.create("Test", **{**CONFIG, "base_url": origin})
    assert not requests


def test_duplicate_case_changes_rejected(requests):
    change = {"caseId": CASE, "revisionId": REVISION, "outcome": "ok"}
    with pytest.raises(ValueError, match="only once"):
        datasets.update_outcomes(
            DATASET, base_version_id=VERSION, changes=[change, change], **CONFIG
        )
    assert not requests


def test_invalid_publication_receipt_is_rejected(requests, monkeypatch):
    monkeypatch.setattr(
        "neatlogs.datasets._api.httpx.request",
        lambda *a, **k: httpx.Response(200, json={"success": True, "data": {"datasetId": DATASET}}),
    )
    with pytest.raises(datasets.DatasetError, match="receipt"):
        datasets.create("Test", **CONFIG)
