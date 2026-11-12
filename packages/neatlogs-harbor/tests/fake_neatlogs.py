"""An in-memory Neatlogs that follows the public API's write rules.

It mirrors the public API behaviour the integration relies on:

* every POST needs an ``Idempotency-Key``; the same key with a different body
  is a 409 ``IDEMPOTENCY_KEY_CONFLICT``;
* ``POST /datasets`` dedupes on the body ``idempotencyKey`` and content;
* ``POST /runs`` returns the existing run for the same key and body;
* one output per (case, trial) slot: the same body returns the same receipt, a
  different one is a 409 ``OUTPUT_CONFLICT``;
* comparisons need a completed baseline, tags move per dataset.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import uuid
from collections import Counter
from typing import Any

import httpx
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from . import contract

NOW = "2026-11-13T00:00:00.000Z"
TOKEN = "nlsa_test_token_value_0123456789"
INGEST_KEY = "projectingestkey0123456789abcdef"
PROJECT_ID = "6f1c2c1e-5d0b-4d8a-9a51-0f4c1b2a3d4e"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class FakeNeatlogs:
    def __init__(self) -> None:
        self.down = False
        # Trial names whose output uploads fail with a 503, to simulate a blip.
        self.failing_trials: set[str] = set()
        self.datasets: dict[str, dict[str, Any]] = {}
        self.dataset_keys: dict[str, tuple[str, str]] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.run_keys: dict[str, str] = {}
        self.header_keys: dict[str, str] = {}
        self.outputs: dict[tuple[str, str, int], dict[str, Any]] = {}
        self.tags: dict[tuple[str, str], str] = {}
        self.spans: Counter[tuple[str, str]] = Counter()
        self.trace_requests = 0
        self.requests: list[tuple[str, str]] = []
        # Requests or responses that break the vendored public API contract.
        self.violations: list[str] = []

    # ------------------------------------------------------------ helpers

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    @property
    def trace_ids(self) -> set[str]:
        return {trace for trace, _ in self.spans}

    @property
    def duplicate_spans(self) -> int:
        return sum(count - 1 for count in self.spans.values() if count > 1)

    def all_outputs(self) -> list[dict[str, Any]]:
        return list(self.outputs.values())

    @staticmethod
    def _ok(data: Any, status: int = 200) -> httpx.Response:
        request_id = "req_" + uuid.uuid4().hex
        return httpx.Response(status, json={"success": True, "data": data, "requestId": request_id})

    @staticmethod
    def _problem(status: int, code: str, detail: str = "") -> httpx.Response:
        return httpx.Response(
            status,
            json={"status": status, "code": code, "detail": detail or code},
            headers={"content-type": "application/problem+json"},
        )

    # ------------------------------------------------------------ routing

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, request.url.path))
        if self.down:
            return self._problem(503, "UNAVAILABLE", "down for the test")
        path = request.url.path
        if path == "/v1/traces":
            return self._traces(request)
        if request.headers.get("authorization") != f"Bearer {TOKEN}":
            return self._problem(401, "UNAUTHORIZED")
        if path == "/api/v1/public/projects":
            project = {
                "id": PROJECT_ID,
                "name": "Harbor",
                "orgId": "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d",
                "createdAt": NOW,
                "updatedAt": NOW,
                "isArchived": False,
                "isDefault": True,
                "tier": "pro",
                "timezone": None,
                "access": {"role": "project_admin"},
            }
            page = {"hasMore": False, "limit": 100, "nextCursor": None}
            return self._ok({"projects": [project], "page": page})
        if request.headers.get("x-project-id") != PROJECT_ID:
            return self._problem(403, "FORBIDDEN", "select a project")
        problems = contract.check_request(request)
        if problems:
            self.violations.extend(problems)
            return self._problem(400, "INVALID_REQUEST", "; ".join(problems))
        body = json.loads(request.content) if request.content else None
        key = request.headers.get("idempotency-key")
        if request.method == "POST":
            if not key:
                return self._problem(400, "IDEMPOTENCY_KEY_REQUIRED")
            seen = self.header_keys.get(key)
            if seen is not None and seen != _digest([path, body]):
                return self._problem(409, "IDEMPOTENCY_KEY_CONFLICT")
        response = self._route(request, path, body)
        if response.status_code < 300 and response.content:
            self.violations.extend(
                contract.check_response(
                    request.method, path, response.status_code, json.loads(response.content)
                )
            )
        if request.method == "POST" and response.status_code < 300:
            # Like the real API, only successful results are kept for replay.
            self.header_keys[key] = _digest([path, body])
        return response

    def _route(self, request: httpx.Request, path: str, body: Any) -> httpx.Response:
        parts = path.removeprefix("/api/v1/public/").split("/")
        if parts[0] == "datasets":
            return self._datasets(request, parts, body)
        if parts[0] == "runs":
            return self._runs(request, parts, body)
        if parts[0] == "traces" and request.method == "GET":
            spans = sum(n for (trace, _), n in self.spans.items() if trace == parts[1])
            if not spans:
                return self._problem(404, "NOT_FOUND")
            return self._ok(self._trace_summary(parts[1], spans))
        return self._problem(404, "NOT_FOUND", path)

    def _traces(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("x-api-key") != INGEST_KEY:
            return httpx.Response(401)
        self.trace_requests += 1
        content = request.content
        if request.headers.get("content-encoding") == "gzip":
            content = gzip.decompress(content)
        export = ExportTraceServiceRequest()
        export.ParseFromString(content)
        for resource_spans in export.resource_spans:
            for scope in resource_spans.scope_spans:
                for span in scope.spans:
                    self.spans[(span.trace_id.hex(), span.span_id.hex())] += 1
        self.last_export = export
        return httpx.Response(200, content=b"")

    def _datasets(self, request: httpx.Request, parts: list[str], body: Any) -> httpx.Response:
        if request.method == "POST" and len(parts) == 1:
            content = _digest([body["name"], body.get("description"), body["cases"]])
            existing = self.dataset_keys.get(body["idempotencyKey"])
            if existing:
                if existing[1] != content:
                    return self._problem(409, "DATASET_REQUEST_FAILED", "different publication")
                dataset = self.datasets[existing[0]]
                return self._ok(
                    {"datasetId": dataset["id"], "versionId": dataset["versionId"]}, 201
                )
            dataset_id, version_id = str(uuid.uuid4()), str(uuid.uuid4())
            self.datasets[dataset_id] = {
                "id": dataset_id,
                "versionId": version_id,
                "name": body["name"],
                "description": body.get("description"),
                "cases": [
                    {
                        "caseId": str(uuid.uuid4()),
                        "revisionId": str(uuid.uuid4()),
                        "position": i,
                        **case,
                    }
                    for i, case in enumerate(body["cases"])
                ],
            }
            self.dataset_keys[body["idempotencyKey"]] = (dataset_id, content)
            return self._ok({"datasetId": dataset_id, "versionId": version_id}, 201)
        if request.method == "GET" and len(parts) == 1:
            return self._ok(
                {
                    "datasets": [
                        {
                            "id": d["id"],
                            "name": d["name"],
                            "description": d["description"],
                            "latestVersionId": d["versionId"],
                            "version": 1,
                            "caseCount": len(d["cases"]),
                            "createdAt": NOW,
                        }
                        for d in self.datasets.values()
                    ],
                    "nextCursor": None,
                }
            )
        if request.method == "GET" and len(parts) == 3 and parts[2] == "cases":
            dataset = self.datasets[parts[1]]
            offset = int(request.url.params.get("offset", 0))
            limit = int(request.url.params.get("limit", 25))
            page = dataset["cases"][offset : offset + limit]
            return self._ok(
                {
                    "cases": [
                        {
                            "caseId": c["caseId"],
                            "revisionId": c["revisionId"],
                            "position": c["position"],
                            "inputPreview": json.dumps(c["input"])[:80],
                            "observedOutputPreview": "",
                            "expectedStatus": None,
                            "hasExpected": False,
                            "criteriaStatus": None,
                            "criteriaCount": 0,
                            "source": {"kind": "import"},
                        }
                        for c in page
                    ],
                    "total": len(dataset["cases"]),
                }
            )
        return self._problem(404, "NOT_FOUND")

    def _runs(self, request: httpx.Request, parts: list[str], body: Any) -> httpx.Response:
        if request.method == "POST" and len(parts) == 1:
            key = request.headers["idempotency-key"]
            if key in self.run_keys:
                run = self.runs[self.run_keys[key]]
                if run["requestHash"] != _digest(body):
                    return self._problem(409, "IDEMPOTENCY_CONFLICT")
                return self._ok(self._receipt(run), 201)
            run_id = str(uuid.uuid4())
            dataset = next(
                d for d in self.datasets.values() if d["versionId"] == body["datasetVersionId"]
            )
            self.runs[run_id] = {
                "id": run_id,
                "datasetId": dataset["id"],
                "body": body,
                "requestHash": _digest(body),
                "state": "queued",
            }
            self.run_keys[key] = run_id
            return self._ok(self._receipt(self.runs[run_id]), 201)
        run = self.runs.get(parts[1])
        if run is None:
            return self._problem(404, "NOT_FOUND")
        action = parts[2] if len(parts) > 2 else None
        if action == "outputs":
            if body.get("output", {}).get("harbor", {}).get("trialName") in self.failing_trials:
                return self._problem(503, "UNAVAILABLE", "blip")
            slot = (run["id"], body["caseId"], body["trial"])
            if (
                body["caseId"] not in run["body"]["caseIds"]
                or body["trial"] > run["body"]["trials"]
            ):
                return self._problem(404, "NOT_FOUND", "slot outside run")
            existing = self.outputs.get(slot)
            if existing is not None:
                if _digest(existing["body"]) != _digest(body):
                    return self._problem(409, "OUTPUT_CONFLICT")
                return self._ok(existing["receipt"])
            if run["state"] in ("completed", "partial", "failed"):
                return self._problem(409, "RUN_CLOSED")
            receipt = {
                "runId": run["id"],
                "itemId": str(uuid.uuid4()),
                "outputId": str(uuid.uuid4()),
                "traceId": None,
                "state": "succeeded",
            }
            self.outputs[slot] = {"body": body, "receipt": receipt}
            return self._ok(receipt)
        if action == "complete":
            if run["state"] not in ("completed", "partial", "failed"):
                filled = sum(1 for (rid, _, _) in self.outputs if rid == run["id"])
                total = len(run["body"]["caseIds"]) * run["body"]["trials"]
                run["state"] = "completed" if filled == total else "partial"
            return self._ok(self._status(run))
        if action == "comparison":
            baseline = request.url.params.get("baseline")
            if baseline and baseline.startswith("tag:"):
                tagged = self.tags.get((run["datasetId"], baseline[4:]))
                if tagged is None or tagged == run["id"]:
                    return self._problem(404, "BASELINE_NOT_FOUND")
            return self._ok({"state": "pending", "summary": {}})
        if action == "tags":
            if run["state"] != "completed":
                return self._problem(409, "RUN_NOT_COMPLETED")
            self.tags[(run["datasetId"], body["tag"])] = run["id"]
            return self._ok({"runId": run["id"], "tag": body["tag"]})
        return self._problem(404, "NOT_FOUND")

    def _status(self, run: dict[str, Any]) -> dict[str, Any]:
        filled = sum(1 for (rid, _, _) in self.outputs if rid == run["id"])
        total = len(run["body"]["caseIds"]) * run["body"]["trials"]
        return {
            **self._receipt(run),
            "completedCount": filled,
            "failedCount": total - filled if run["state"] != "queued" else 0,
            "pendingCount": total - filled if run["state"] == "queued" else 0,
            "gradingHandoffState": "pending",
        }

    @staticmethod
    def _trace_summary(trace_id: str, spans: int) -> dict[str, Any]:
        return {
            "traceId": trace_id,
            "traceName": "harbor.trial",
            "workflowName": None,
            "sessionId": "session",
            "status": "success",
            "finalizationStatus": "finalized",
            "framework": None,
            "startedAt": NOW,
            "completedAt": NOW,
            "ingestedAt": NOW,
            "durationMs": 0,
            "spansCount": spans,
            "llmCallsCount": 0,
            "toolCallsCount": 0,
            "totalTokens": 0,
            "totalCostUsd": 0,
            "tags": [],
            "metadataRevision": 0,
            "metadataUpdatedAt": None,
        }

    @staticmethod
    def _receipt(run: dict[str, Any]) -> dict[str, Any]:
        body = run["body"]
        return {
            "runId": run["id"],
            "state": run["state"],
            "deadlineAt": "2026-11-13T00:00:00Z",
            "itemCount": len(body["caseIds"]) * body["trials"],
        }
