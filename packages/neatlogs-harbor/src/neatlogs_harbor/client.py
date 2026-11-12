"""Small HTTP client for the Neatlogs endpoints the integration uses.

* Public API (``/api/v1/public/...``): datasets and runs. Authenticated with a
  service-account token and an ``x-project-id`` header.
* OTLP ingest (``/v1/traces``): authenticated with the project's ingest key.

Every write carries a deterministic ``Idempotency-Key`` derived from Harbor's
identities, so retries, the plugin and the importer all land on one record.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import logging
import time
from collections.abc import Mapping
from typing import Any

import httpx

from .version import __version__

logger = logging.getLogger("neatlogs_harbor")

_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_RETRYABLE_CONFLICTS = frozenset({"IDEMPOTENCY_REQUEST_IN_PROGRESS", "TRACE_NOT_READY"})
USER_AGENT = f"neatlogs-harbor/{__version__}"


class NeatlogsApiError(Exception):
    def __init__(self, status: int, code: str | None, detail: str | None) -> None:
        self.status = status
        self.code = code
        self.detail = detail
        super().__init__(f"HTTP {status} {code or ''}: {detail or ''}".strip())

    @property
    def is_conflict(self) -> bool:
        return self.status == 409

    @property
    def is_auth(self) -> bool:
        return self.status in (401, 403)


def idempotency_key(*parts: object) -> str:
    """A valid ``Idempotency-Key``: 32 bytes, base64url, derived from stable parts."""

    digest = hashlib.sha256(
        ("neatlogs-harbor:" + ":".join(str(p) for p in parts)).encode()
    ).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


class NeatlogsClient:
    def __init__(
        self,
        *,
        endpoint: str,
        otlp_endpoint: str,
        token: str | None,
        project_id: str | None,
        ingest_key: str | None,
        timeout: float = 30.0,
        retries: int = 4,
        transport: httpx.BaseTransport | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.otlp_endpoint = otlp_endpoint
        self.token = token
        self.project_id = project_id
        self.ingest_key = ingest_key
        self.retries = retries
        self._sleep = sleep
        self._http = httpx.Client(
            timeout=timeout,
            transport=transport,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=False,
        )

    def close(self) -> None:
        self._http.close()

    @property
    def can_use_public_api(self) -> bool:
        return bool(self.token)

    @property
    def can_send_traces(self) -> bool:
        return bool(self.ingest_key)

    # ---------------------------------------------------------------- public API

    def resolve_project_id(self) -> str | None:
        """The project for the token: the configured one, or the only one it can see."""

        if self.project_id or not self.token:
            return self.project_id
        data = self._request("GET", "/api/v1/public/projects", project=False)
        projects = (data or {}).get("projects") or []
        if len(projects) == 1:
            self.project_id = str(projects[0]["id"])
        return self.project_id

    def create_dataset(self, body: Mapping[str, Any], key: str) -> dict[str, Any]:
        return self._request("POST", "/api/v1/public/datasets", json=body, key=key)

    def list_datasets(self) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        before: str | None = None
        for _ in range(50):
            params = {"before": before} if before else None
            data = self._request(
                "GET",
                "/api/v1/public/datasets",
                params=params,
                headers={"Prefer": "pagination=cursor"},
            )
            found.extend((data or {}).get("datasets") or [])
            before = (data or {}).get("nextCursor")
            if not before:
                break
        return found

    def list_cases(self, dataset_id: str, version_id: str) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        offset = 0
        while True:
            data = self._request(
                "GET",
                f"/api/v1/public/datasets/{dataset_id}/cases",
                params={"versionId": version_id, "offset": offset, "limit": 100},
            )
            page = (data or {}).get("cases") or []
            cases.extend(page)
            total = int((data or {}).get("total") or 0)
            offset += len(page)
            if not page or offset >= total:
                return cases

    def register_run(self, body: Mapping[str, Any], key: str) -> dict[str, Any]:
        return self._request("POST", "/api/v1/public/runs", json=body, key=key)

    def upload_output(self, run_id: str, body: Mapping[str, Any], key: str) -> dict[str, Any]:
        return self._request("POST", f"/api/v1/public/runs/{run_id}/outputs", json=body, key=key)

    def complete_run(self, run_id: str, key: str) -> dict[str, Any]:
        return self._request("POST", f"/api/v1/public/runs/{run_id}/complete", json={}, key=key)

    def get_run(self, run_id: str, include_grading: bool = False) -> dict[str, Any]:
        params = {"includeGrading": "true"} if include_grading else None
        return self._request("GET", f"/api/v1/public/runs/{run_id}", params=params)

    def comparison(self, run_id: str, baseline: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": 100}
        if baseline:
            params["baseline"] = baseline
        return self._request("GET", f"/api/v1/public/runs/{run_id}/comparison", params=params)

    def tag_run(self, run_id: str, tag: str, key: str) -> dict[str, Any]:
        return self._request(
            "POST", f"/api/v1/public/runs/{run_id}/tags", json={"tag": tag}, key=key
        )

    def trace_exists(self, trace_id: str) -> bool | None:
        """True or False when the API can tell; None when this token cannot read traces."""

        try:
            self._request("GET", f"/api/v1/public/traces/{trace_id}", retry=False)
            return True
        except NeatlogsApiError as exc:
            if exc.status == 404:
                return False
            return None

    # ---------------------------------------------------------------- OTLP

    def send_traces(self, payload: bytes) -> None:
        if not self.ingest_key:
            raise NeatlogsApiError(401, "NO_INGEST_KEY", "NEATLOGS_API_KEY is not set")
        body = gzip.compress(payload)
        headers = {
            "Content-Type": "application/x-protobuf",
            "Content-Encoding": "gzip",
            "x-api-key": self.ingest_key,
        }
        self._send(lambda: self._http.post(self.otlp_endpoint, content=body, headers=headers))

    # ---------------------------------------------------------------- transport

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: Mapping[str, Any] | None = None,
        key: str | None = None,
        headers: Mapping[str, str] | None = None,
        project: bool = True,
        retry: bool = True,
    ) -> Any:
        if not self.token:
            raise NeatlogsApiError(401, "NO_TOKEN", "NEATLOGS_SERVICE_ACCOUNT_TOKEN is not set")
        all_headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}
        if project and self.project_id:
            all_headers["x-project-id"] = self.project_id
        if key:
            all_headers["Idempotency-Key"] = key
        all_headers.update(headers or {})
        url = f"{self.endpoint}{path}"
        response = self._send(
            lambda: self._http.request(method, url, json=json, params=params, headers=all_headers),
            retry=retry,
        )
        payload = response.json() if response.content else {}
        if isinstance(payload, dict) and "data" in payload and payload.get("success") is True:
            return payload["data"]
        return payload

    def _send(self, call: Any, retry: bool = True) -> httpx.Response:
        attempts = self.retries + 1 if retry else 1
        for attempt in range(attempts):
            last = attempt == attempts - 1
            try:
                response = call()
            except httpx.HTTPError as exc:
                if last:
                    raise NeatlogsApiError(0, "NETWORK_ERROR", str(exc)) from exc
                self._sleep(self._backoff(attempt, None))
                continue
            if response.status_code < 400:
                return response
            error = _error_from(response)
            retryable = response.status_code in _RETRYABLE_STATUS or (
                response.status_code == 409 and error.code in _RETRYABLE_CONFLICTS
            )
            if not retryable or last:
                raise error
            self._sleep(self._backoff(attempt, response.headers.get("Retry-After")))
        raise RuntimeError("unreachable")  # pragma: no cover

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
        return min(1.0 * (2**attempt), 30.0)


def _error_from(response: httpx.Response) -> NeatlogsApiError:
    code: str | None = None
    detail: str | None = None
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        code = payload.get("code") or (payload.get("error") or {}).get("code")
        detail = payload.get("detail") or payload.get("message") or payload.get("title")
    if detail is None:
        detail = response.text[:300]
    return NeatlogsApiError(response.status_code, code, detail)
