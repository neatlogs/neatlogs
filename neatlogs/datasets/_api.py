"""Authenticated dataset operations, separate from telemetry ingestion."""

from __future__ import annotations

import json
import time
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
from opentelemetry.instrumentation.utils import suppress_instrumentation


class DatasetError(RuntimeError):
    """A dataset request failed. Response bodies and credentials are never included."""

    def __init__(self, message, *, status_code=None, idempotency_key=None):
        super().__init__(message)
        self.status_code = status_code
        self.idempotency_key = idempotency_key


def encode_json(payload):
    def check(value):
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("Dataset JSON object keys must be strings")
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
        elif value is not None and not isinstance(value, (str, bool, int, float)):
            raise ValueError("Dataset values must be JSON values")

    check(payload)
    content = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    if len(content) > 64 * 1024:
        raise ValueError("Dataset request exceeds 64 KiB; use fewer or smaller cases")
    return content


def publication(data, dataset_id=None):
    try:
        receipt = {key: str(UUID(data[key])) for key in ("datasetId", "versionId")}
        if dataset_id is not None and receipt["datasetId"] != dataset_id:
            raise ValueError()
        return receipt
    except (KeyError, TypeError, ValueError, AttributeError):
        raise DatasetError("Dataset API returned an invalid publication receipt") from None


class DatasetApi:
    def __init__(self, *, project_id, token, base_url):
        self.project_id = str(UUID(project_id))
        if not isinstance(token, str) or not token.strip() or "\n" in token or "\r" in token:
            raise ValueError("An explicit public API user/service-account token is required")
        parsed = urlsplit(base_url)
        if (
            not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
            or (
                parsed.scheme != "https"
                and not (
                    parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
                )
            )
        ):
            raise ValueError("base_url must be an HTTPS origin; HTTP is allowed on loopback")
        self.base_url = base_url.rstrip("/") + "/api/v1/public"
        self._token = token

    def request(self, method, path, *, content=None, key=None):
        headers = {"Authorization": "Bearer " + self._token, "x-project-id": self.project_id}
        if key is not None:
            headers["Idempotency-Key"] = key
        if content is not None:
            headers["Content-Type"] = "application/json"
        for attempt in range(3):
            try:
                with suppress_instrumentation():
                    response = httpx.request(
                        method,
                        self.base_url + path,
                        headers=headers,
                        content=content,
                        timeout=30,
                        follow_redirects=False,
                    )
            except httpx.TransportError:
                if attempt < 2:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                raise DatasetError(
                    "Dataset API transport failed; retry with the same idempotency key",
                    idempotency_key=key,
                ) from None
            if (response.status_code == 429 or response.status_code >= 500) and attempt < 2:
                time.sleep(0.2 * (attempt + 1))
                continue
            if not response.is_success:
                raise DatasetError(
                    f"Dataset API request failed (HTTP {response.status_code})",
                    status_code=response.status_code,
                    idempotency_key=key,
                )
            try:
                envelope = response.json()
                if envelope.get("success") is not True or not isinstance(
                    envelope.get("data"), dict
                ):
                    raise ValueError()
                return envelope["data"]
            except (ValueError, AttributeError):
                raise DatasetError("Dataset API returned an invalid response") from None
        raise AssertionError("request retry exhausted")


def create(
    name,
    *,
    project_id,
    token,
    base_url,
    cases=None,
    description=None,
    idempotency_key=None,
):
    """Create an empty dataset or import up to 500 cases without initializing telemetry.

    Cases use the existing dataset JSON contract: ``input``, ``observedOutput``,
    optional ``expected``, ``context``, ``metadata`` and ``tags``. An optional
    outcome uses ``expected={"value": outcome, "status": "draft", "source": "human"}``.
    """
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 255:
        raise ValueError("name must contain 1–255 characters")
    if description is not None and (not isinstance(description, str) or len(description) > 2000):
        raise ValueError("description must be at most 2000 characters")
    cases = [] if cases is None else cases
    if (
        not isinstance(cases, list)
        or len(cases) > 500
        or any(not isinstance(c, dict) for c in cases)
    ):
        raise ValueError("cases must be a list of at most 500 case objects")
    key = str(UUID(idempotency_key)) if idempotency_key is not None else str(uuid4())
    content = encode_json(
        {"name": name.strip(), "description": description, "cases": cases, "idempotencyKey": key}
    )
    api = DatasetApi(project_id=project_id, token=token, base_url=base_url)
    return publication(api.request("POST", "/datasets", content=content, key=key))


def update_outcomes(
    dataset_id,
    *,
    base_version_id,
    changes,
    project_id,
    token,
    base_url,
    idempotency_key=None,
):
    """Publish per-case outcomes as a new version; ``outcome=None`` clears an outcome.

    Each change contains ``caseId``, ``revisionId`` and ``outcome``. Only supplied
    cases change. A stale base or revision raises ``DatasetError`` with HTTP 409.
    """
    dataset_id = str(UUID(dataset_id))
    if not isinstance(changes, list) or not 1 <= len(changes) <= 2000:
        raise ValueError("changes must contain 1–2000 outcome changes")
    expected_changes = []
    for change in changes:
        if not isinstance(change, dict) or set(change) != {"caseId", "revisionId", "outcome"}:
            raise ValueError("Each change must contain caseId, revisionId and outcome")
        outcome = change["outcome"]
        expected_changes.append(
            {
                "caseId": str(UUID(change["caseId"])),
                "revisionId": str(UUID(change["revisionId"])),
                "expected": (
                    None
                    if outcome is None
                    else {"value": outcome, "status": "draft", "source": "human"}
                ),
            }
        )
    if len({change["caseId"] for change in expected_changes}) != len(expected_changes):
        raise ValueError("Each case may appear only once in outcome changes")
    key = str(UUID(idempotency_key)) if idempotency_key is not None else str(uuid4())
    content = encode_json(
        {
            "baseVersionId": str(UUID(base_version_id)),
            "idempotencyKey": key,
            "expectedChanges": expected_changes,
            "removedCaseIds": [],
        }
    )
    api = DatasetApi(project_id=project_id, token=token, base_url=base_url)
    return publication(
        api.request("POST", f"/datasets/{dataset_id}/versions", content=content, key=key),
        dataset_id,
    )
