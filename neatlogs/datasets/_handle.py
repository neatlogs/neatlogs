"""Durable capture receipts and publication polling."""

from __future__ import annotations

import asyncio
import math
import time
from uuid import UUID, uuid4

from ._api import DatasetApi, DatasetError, encode_json


class DatasetCaptureError(DatasetError):
    """A capture could not be registered, exported, or published."""


_JOB_STATES = {"recording", "queued", "running", "needs_review", "completed", "failed", "cancelled"}


class CaptureHandle:
    """Durable capture identity and publication status. Credentials are never displayed."""

    def __init__(self, *, project_id: str, token: str, base_url: str):
        self._api = DatasetApi(project_id=project_id, token=token, base_url=base_url)
        self._keys = {phase: str(uuid4()) for phase in ("register", "complete", "abort")}
        self.capture_id: str | None = None
        self.dataset_id: str | None = None
        self.state: str | None = None
        self.error: BaseException | None = None
        self.last_status: dict | None = None
        self._manifest: bytes | None = None

    def _request(self, method, suffix="", *, content=None, phase=None):
        try:
            return self._api.request(
                method,
                "/dataset-capture-jobs" + suffix,
                content=content,
                key=self._keys[phase] if phase else None,
            )
        except DatasetError as error:
            raise DatasetCaptureError(
                str(error), status_code=error.status_code, idempotency_key=error.idempotency_key
            ) from None

    def _register(self, destination):
        data = self._request(
            "POST", content=encode_json({"destination": destination}), phase="register"
        )
        try:
            self.capture_id = str(UUID(data["captureId"]))
            self.dataset_id = str(UUID(data["datasetId"]))
            if data["state"] != "recording":
                raise ValueError()
            self.state = data["state"]
        except (KeyError, ValueError, TypeError, AttributeError):
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

    def abort(self, reason="cancelled"):
        """Abort a recording before it is sealed, without publishing a version."""
        if reason not in {"cancelled", "export_failed", "incomplete_execution"}:
            raise ValueError("Invalid capture abort reason")
        return self._abort(reason)

    async def aabort(self, reason="cancelled"):
        return await asyncio.to_thread(self.abort, reason)

    async def aretry_completion(self):
        return await asyncio.to_thread(self.retry_completion)

    def _abort(self, reason):
        if self.capture_id is not None:
            data = self._request(
                "POST",
                f"/{self.capture_id}/abort",
                content=encode_json({"reason": reason}),
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
        job = data["job"]
        try:
            if (
                str(UUID(job["id"])) != self.capture_id
                or str(UUID(job["datasetId"])) != self.dataset_id
                or job["state"] not in _JOB_STATES
            ):
                raise ValueError()
            version = job.get("datasetVersionId")
            if job["state"] == "completed" and version is None:
                raise ValueError()
            if version is not None:
                UUID(version)
        except (KeyError, TypeError, ValueError, AttributeError):
            raise DatasetCaptureError("Dataset API returned an invalid job") from None
        self.last_status = data
        self.state = job["state"]
        return data

    def wait(self, timeout: float = 120, poll_interval: float = 1):
        """Wait for publication; failed/review-required jobs raise instead of looking successful."""
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
            for v in (timeout, poll_interval)
        ):
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
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
            for v in (timeout, poll_interval)
        ):
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
