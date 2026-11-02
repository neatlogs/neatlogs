"""Bounded public API reads and idempotent local-run acceptance."""

from __future__ import annotations

import time
from datetime import datetime
from uuid import uuid4

import httpx
from opentelemetry.instrumentation.utils import suppress_instrumentation

from ..datasets._api import DatasetApi
from .contracts import ReplayError, parse_json_bytes, uuid
from .provider import ecmascript_json


class LocalRunApiError(ReplayError):
    def __init__(self, message, *, status=None, code=None):
        super().__init__(message)
        self.status = status
        self.code = code


class LocalRunApi:
    def __init__(self, *, project_id, token, base_url):
        validated = DatasetApi(project_id=project_id, token=token, base_url=base_url)
        self.project_id = validated.project_id
        self.base_url = validated.base_url
        self._token = token.strip()

    def request(self, method, path, body=None, *, key=None, limit=64 * 1024, timeout=30):
        if not (path == "/datasets" or path.startswith(("/datasets/", "/runs"))) or ".." in path:
            raise ReplayError("Invalid local replay API path.")
        content = ecmascript_json(body) if body is not None else None
        if content is not None and len(content) > 64 * 1024:
            raise ReplayError("Local run request exceeds 64 KiB.")
        headers = {"Authorization": "Bearer " + self._token, "x-project-id": self.project_id}
        if content is not None:
            headers["Content-Type"] = "application/json"
        if key is not None:
            headers["Idempotency-Key"] = uuid(key)
        deadline = time.monotonic() + timeout
        for attempt in range(2):
            try:
                with (
                    suppress_instrumentation(),
                    httpx.stream(
                        method,
                        self.base_url + path,
                        headers=headers,
                        content=content,
                        timeout=max(0.001, deadline - time.monotonic()),
                        follow_redirects=False,
                    ) as response,
                ):
                    if attempt == 0 and response.status_code in {500, 502, 503, 504}:
                        continue
                    data = bytearray()
                    for chunk in response.iter_bytes():
                        if time.monotonic() >= deadline:
                            raise LocalRunApiError("Local replay API request timed out.")
                        data.extend(chunk)
                        if len(data) > limit:
                            raise LocalRunApiError(
                                "Local replay API response exceeds its size limit."
                            )
                    raw = parse_json_bytes(bytes(data), limit=limit)
                    if not response.is_success:
                        code = raw.get("code") if type(raw) is dict else None
                        if type(raw) is dict and type(raw.get("error")) is dict:
                            code = raw["error"].get("code", code)
                        code = code if type(code) is str and len(code) <= 100 else None
                        raise LocalRunApiError(
                            f"Local replay API request failed with HTTP {response.status_code}.",
                            status=response.status_code,
                            code=code,
                        )
                    if (
                        type(raw) is not dict
                        or raw.get("success") is not True
                        or type(raw.get("data")) is not dict
                    ):
                        raise LocalRunApiError("Local replay API returned an invalid envelope.")
                    return raw["data"]
            except httpx.TransportError:
                if attempt == 0 and time.monotonic() < deadline:
                    continue
                raise LocalRunApiError("Local replay API request failed or timed out.") from None
        raise LocalRunApiError("Local replay API request failed.")

    def frozen_case(self, dataset_id, version_id, case_id):
        return self.request(
            "GET",
            f"/datasets/{uuid(dataset_id)}/cases/{uuid(case_id)}?versionId={uuid(version_id)}",
            limit=17 * 1024 * 1024,
        )

    def register(self, input, *, key=None):
        receipt = self.request("POST", "/runs", input, key=key or str(uuid4()))
        self._receipt(receipt)
        return receipt

    def status(self, run_id):
        receipt = self.request("GET", f"/runs/{uuid(run_id)}")
        self._receipt(receipt)
        if receipt["runId"] != run_id:
            raise LocalRunApiError("Local run status belongs to a different run.")
        return receipt

    @staticmethod
    def _receipt(receipt):
        uuid(receipt.get("runId"))
        try:
            deadline = datetime.fromisoformat(receipt["deadlineAt"].replace("Z", "+00:00"))
            if (
                deadline.tzinfo is None
                or type(receipt["itemCount"]) is not int
                or receipt["itemCount"] < 0
                or type(receipt["state"]) is not str
            ):
                raise ValueError()
        except (KeyError, TypeError, ValueError, AttributeError):
            raise LocalRunApiError("Local replay API returned an invalid run receipt.") from None

    def upload(self, run_id, *, case_id, trial, trace_id, key=None, wait_millis=60_000):
        import re

        run_id, case_id = uuid(run_id), uuid(case_id)
        if (
            type(trial) is not int
            or not 1 <= trial <= 10
            or type(trace_id) is not str
            or not re.fullmatch(r"[a-f0-9]{32}", trace_id)
        ):
            raise ReplayError("Run upload requires one trace and a trial from 1 to 10.")
        if type(wait_millis) is not int or not 1 <= wait_millis <= 300_000:
            raise ReplayError("Run upload wait must be from 1 ms to 5 minutes.")
        key = uuid(key or str(uuid4()))
        deadline = time.monotonic() + wait_millis / 1000
        attempt = 0
        while True:
            try:
                receipt = self.request(
                    "POST",
                    f"/runs/{run_id}/outputs",
                    {"caseId": case_id, "trial": trial, "traceId": trace_id},
                    key=key,
                    timeout=max(0.001, min(30, deadline - time.monotonic())),
                )
                if (
                    receipt.get("runId") != run_id
                    or receipt.get("traceId") != trace_id
                    or receipt.get("state") != "succeeded"
                ):
                    raise LocalRunApiError(
                        "The upload receipt belongs to a different run or trace."
                    )
                uuid(receipt.get("outputId"))
                uuid(receipt.get("itemId"))
                return receipt
            except LocalRunApiError as error:
                remaining = deadline - time.monotonic()
                if error.status != 409 or error.code != "TRACE_NOT_READY" or remaining <= 0:
                    raise
                time.sleep(min(remaining, 2 ** min(attempt, 2)))
                attempt += 1
