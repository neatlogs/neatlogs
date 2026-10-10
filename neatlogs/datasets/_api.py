"""Public dataset API, separate from ingestion credentials and transport."""

from __future__ import annotations

import asyncio
import json
import time
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
from opentelemetry.instrumentation.utils import suppress_instrumentation


class DatasetCaptureError(RuntimeError):
    """A capture could not be registered, exported, or published."""


_JOB_STATES = {"recording", "queued", "running", "needs_review", "completed", "failed", "cancelled"}


def encode_manifest(payload):
    content = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    if len(content) > 64 * 1024:
        raise DatasetCaptureError("Capture manifest exceeds 64 KiB; use a smaller block")
    return content


class CaptureHandle:
    """Durable capture identity and publication status. Credentials are never displayed."""

    def __init__(self, *, project_id: str, token: str, base_url: str):
        UUID(project_id)
        if not isinstance(token, str) or not token.strip():
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
            raise ValueError("base_url must be an HTTPS origin (HTTP is allowed on loopback)")
        self._base_url = base_url.rstrip("/") + "/api/v1/public/dataset-capture-jobs"
        self._token = token
        self._project_id = project_id
        self._keys = {phase: str(uuid4()) for phase in ("register", "complete", "abort")}
        self.capture_id: str | None = None
        self.dataset_id: str | None = None
        self.state: str | None = None
        self.error: BaseException | None = None
        self.last_status: dict | None = None
        self._manifest: bytes | None = None

    def _request(self, method, suffix="", *, content=None, phase=None):
        headers = {"Authorization": "Bearer " + self._token, "x-project-id": self._project_id}
        if phase is not None:
            headers["Idempotency-Key"] = self._keys[phase]
        if content is not None:
            headers["Content-Type"] = "application/json"
        for attempt in range(3):
            try:
                with suppress_instrumentation():
                    response = httpx.request(
                        method,
                        self._base_url + suffix,
                        headers=headers,
                        content=content,
                        timeout=30,
                        follow_redirects=False,
                    )
            except httpx.TransportError:
                if attempt < 2:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                raise DatasetCaptureError(
                    "Dataset API transport failed; retry using this capture handle"
                ) from None
            if (response.status_code == 429 or response.status_code >= 500) and attempt < 2:
                time.sleep(0.2 * (attempt + 1))
                continue
            if not response.is_success:
                raise DatasetCaptureError(
                    f"Dataset API request failed (HTTP {response.status_code})"
                )
            try:
                envelope = response.json()
                if envelope.get("success") is not True or not isinstance(
                    envelope.get("data"), dict
                ):
                    raise ValueError()
                return envelope["data"]
            except (ValueError, AttributeError):
                raise DatasetCaptureError("Dataset API returned an invalid response") from None
        raise AssertionError("request retry exhausted")

    def _register(self, destination):
        data = self._request(
            "POST", content=encode_manifest({"destination": destination}), phase="register"
        )
        try:
            self.capture_id = str(UUID(data["captureId"]))
            self.dataset_id = str(UUID(data["datasetId"]))
            if data["state"] != "recording":
                raise ValueError()
            self.state = data["state"]
        except (KeyError, ValueError, TypeError):
            raise DatasetCaptureError(
                "Dataset API returned an invalid capture registration"
            ) from None

    def retry_completion(self):
        """Retry an ambiguous completion with its original manifest and idempotency key."""
        if self.capture_id is None or self._manifest is None:
            raise DatasetCaptureError("This capture has no exported completion manifest")
        data = self._request(
            "POST", f"/{self.capture_id}/complete", content=self._manifest, phase="complete"
        )
        self._apply_receipt(data)
        return data

    def _apply_receipt(self, data):
        try:
            capture_id = str(UUID(data["captureId"]))
            dataset_id = str(UUID(data["datasetId"]))
            state = data["state"]
            if (
                capture_id != self.capture_id
                or dataset_id != self.dataset_id
                or not isinstance(state, str)
                or state not in _JOB_STATES
            ):
                raise ValueError()
        except (AttributeError, KeyError, TypeError, ValueError):
            raise DatasetCaptureError("Dataset API returned an invalid capture receipt") from None
        if self.state == "completed" and state in {"recording", "queued", "running"}:
            return
        self.state = state

    def _abort(self, reason):
        if self.capture_id is not None:
            data = self._request(
                "POST",
                f"/{self.capture_id}/abort",
                content=encode_manifest({"reason": reason}),
                phase="abort",
            )
            self._apply_receipt(data)

    def status(self):
        """Read the job, including captured/failed counts and published version ID."""
        if self.capture_id is None:
            raise DatasetCaptureError("Capture has not been registered")
        data = self._request("GET", f"/{self.capture_id}")
        if not isinstance(data.get("job"), dict):
            raise DatasetCaptureError("Dataset API returned an invalid job")
        self.last_status = data
        self.state = data["job"].get("state")
        return data

    def wait(self, timeout: float = 120, poll_interval: float = 1):
        """Wait for publication; failed/review-required jobs raise instead of looking successful."""
        if timeout <= 0 or poll_interval <= 0:
            raise ValueError("timeout and poll_interval must be positive")
        deadline = time.monotonic() + timeout
        while True:
            data = self.status()
            if self.state == "completed":
                return data["job"]
            if self.state in {"failed", "cancelled", "needs_review"}:
                raise DatasetCaptureError(
                    f"Dataset capture ended in {self.state}; inspect handle.last_status"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Dataset capture is still pending; inspect or wait on this handle later"
                )
            time.sleep(min(poll_interval, remaining))

    async def astatus(self):
        return await asyncio.to_thread(self.status)

    async def await_completion(self, timeout: float = 120, poll_interval: float = 1):
        if timeout <= 0 or poll_interval <= 0:
            raise ValueError("timeout and poll_interval must be positive")
        deadline = time.monotonic() + timeout
        while True:
            data = await self.astatus()
            if self.state == "completed":
                return data["job"]
            if self.state in {"failed", "cancelled", "needs_review"}:
                raise DatasetCaptureError(
                    f"Dataset capture ended in {self.state}; inspect handle.last_status"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Dataset capture is still pending; inspect or wait on this handle later"
                )
            await asyncio.sleep(min(poll_interval, remaining))
