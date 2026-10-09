import asyncio
import importlib
import json
import threading
import warnings
from uuid import uuid4

import httpx
import pytest
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

import neatlogs
from neatlogs.datasets import DatasetCaptureError

PROJECT = "22222222-2222-4222-8222-222222222222"
DATASET = "33333333-3333-4333-8333-333333333333"


@pytest.fixture
def api(monkeypatch):
    calls = []
    jobs = {}
    lock = threading.Lock()

    def request(method, url, **kwargs):
        body = json.loads(kwargs["content"]) if kwargs.get("content") else None
        with lock:
            calls.append((method, url, body, kwargs["headers"]))
            if url.endswith("dataset-capture-jobs"):
                capture_id = jobs.setdefault(kwargs["headers"]["Idempotency-Key"], str(uuid4()))
                data = {"captureId": capture_id, "datasetId": DATASET, "state": "recording"}
            elif method == "GET":
                data = {
                    "job": {
                        "id": url.rsplit("/", 1)[-1],
                        "datasetId": DATASET,
                        "state": "completed",
                        "capturedCount": 1,
                        "datasetVersionId": str(uuid4()),
                    },
                    "failures": [],
                }
            else:
                data = {
                    "captureId": url.split("/")[-2],
                    "datasetId": DATASET,
                    "state": "cancelled" if url.endswith("abort") else "queued",
                }
        return httpx.Response(200, json={"success": True, "data": data, "requestId": "r"})

    monkeypatch.setattr("neatlogs.datasets._api.httpx.request", request)
    return calls


@pytest.fixture
def client(monkeypatch):
    exporter = InMemorySpanExporter()
    monkeypatch.setattr("neatlogs.client.OTLPSpanExporter", lambda **kwargs: exporter)
    client = neatlogs.Client(
        api_key="ingestion-only", workflow_name="capture", uploads_enabled=False
    )
    yield client, exporter
    client.shutdown()


def capture(**kwargs):
    return neatlogs.datasets.capture(
        "Regression",
        project_id=PROJECT,
        token="public-only",
        base_url="https://app.example.com",
        environment="local",
        **kwargs,
    )


def completions(calls):
    return [entry for entry in calls if entry[1].endswith("/complete")]


def test_registers_before_execution_and_keeps_normal_export(client, api):
    sdk, exporter = client
    with sdk.activate():
        with capture(create_if_missing=True) as job:
            assert api[0][2]["destination"] == {
                "kind": "name",
                "name": "Regression",
                "createIfMissing": True,
            }
            assert job.state == "recording"
            with neatlogs.trace("root", kind="WORKFLOW"):
                with neatlogs.trace("child", kind="TOOL"):
                    pass
        manifest = completions(api)[0][2]["traces"]
        exported = [s for s in exporter.get_finished_spans() if s.name != "neatlogs.trace.complete"]
        assert len(manifest) == 1
        assert set(manifest[0]["spanIds"]) == {f"{s.context.span_id:016x}" for s in exported}
        assert len(exported) == 2
        assert all(c[3]["Authorization"] == "Bearer public-only" for c in api)
        assert all("ingestion-only" not in str(c) for c in api)
        assert api[0][3]["Idempotency-Key"] != completions(api)[0][3]["Idempotency-Key"]
        assert job.wait()["capturedCount"] == 1


def test_scope_excludes_existing_root_and_outside_execution(client, api):
    sdk, exporter = client
    with sdk.activate():
        with neatlogs.trace("outside", kind="WORKFLOW"):
            with capture():
                with neatlogs.trace("existing-child", kind="TOOL"):
                    pass
        assert completions(api)[0][2] == {"traces": []}
        sdk.flush()
        assert len(exporter.get_finished_spans()) >= 2


def test_http_transport_and_completion_markers_are_not_required(client, api):
    sdk, exporter = client
    with sdk.activate(), capture():
        with neatlogs.trace("root", kind="WORKFLOW"):
            tracer = sdk.tracer_provider.get_tracer("opentelemetry.instrumentation.httpx")
            with tracer.start_as_current_span(
                "HTTP", kind=SpanKind.CLIENT, attributes={"http.method": "GET"}
            ):
                pass
    assert len(completions(api)[0][2]["traces"][0]["spanIds"]) == 1
    assert "HTTP" not in {s.name for s in exporter.get_finished_spans()}


def test_nested_capture_rejected_before_second_registration(client, api):
    with client[0].activate(), capture():
        with pytest.raises(DatasetCaptureError, match="Nested"):
            with capture():
                pass
    assert len([c for c in api if c[1].endswith("dataset-capture-jobs")]) == 1


def test_unfinished_span_aborts_without_completing(client, api):
    with client[0].activate():
        with pytest.raises(DatasetCaptureError, match="unfinished"):
            with capture():
                span = client[0].get_tracer("neatlogs.test").start_span("unfinished")
        span.end()
    assert not completions(api)
    assert api[-1][2] == {"reason": "incomplete_execution"}


def test_agent_exception_is_preserved_and_error_trace_still_captured(client, api):
    error = RuntimeError("agent failed")
    with client[0].activate(), pytest.raises(RuntimeError) as raised:
        with capture():
            with neatlogs.trace("failed root", kind="WORKFLOW"):
                raise error
    assert raised.value is error
    assert completions(api)


def test_export_loss_aborts_and_preserves_agent_error(client, api, monkeypatch):
    monkeypatch.setattr(client[1], "export", lambda spans: SpanExportResult.FAILURE)
    error = RuntimeError("agent failed")
    with (
        client[0].activate(),
        pytest.warns(RuntimeWarning, match="preserving"),
        pytest.raises(RuntimeError) as raised,
    ):
        with capture() as job:
            with neatlogs.trace("failed root", kind="WORKFLOW"):
                raise error
    assert raised.value is error
    assert isinstance(job.error, DatasetCaptureError)
    assert api[-1][2] == {"reason": "export_failed"}
    assert not completions(api)


def test_export_loss_without_agent_error_is_visible(client, api, monkeypatch):
    monkeypatch.setattr(client[1], "export", lambda spans: SpanExportResult.FAILURE)
    with client[0].activate(), pytest.raises(DatasetCaptureError, match="lost data"):
        with capture():
            with neatlogs.trace("root", kind="WORKFLOW"):
                pass
    assert api[-1][2] == {"reason": "export_failed"}


@pytest.mark.asyncio
async def test_awaited_concurrent_contexts_are_isolated(client, api):
    async def run(name):
        with client[0].activate():
            async with capture() as job:
                with neatlogs.trace(name, kind="WORKFLOW"):
                    await asyncio.sleep(0.01)
                    with neatlogs.trace(name + " child", kind="TOOL"):
                        await asyncio.sleep(0)
            return job

    first, second = await asyncio.gather(run("first"), run("second"))
    manifests = [c[2]["traces"] for c in completions(api)]
    assert len(manifests) == 2
    assert all(len(m) == 1 and len(m[0]["spanIds"]) == 2 for m in manifests)
    assert manifests[0][0]["traceId"] != manifests[1][0]["traceId"]
    assert first.capture_id != second.capture_id
    assert (await first.await_completion())["capturedCount"] == 1


@pytest.mark.asyncio
async def test_cancellation_aborts_without_replacing_cancellation(client, api):
    with client[0].activate(), pytest.raises(asyncio.CancelledError):
        async with capture():
            with neatlogs.trace("cancelled", kind="WORKFLOW"):
                raise asyncio.CancelledError()
    assert api[-1][2] == {"reason": "cancelled"}
    assert not completions(api)


def test_completion_retries_keep_exact_manifest_and_key(client, api, monkeypatch):
    actual = httpx.request
    attempts = []

    def request(method, url, **kwargs):
        if url.endswith("complete"):
            attempts.append((kwargs["content"], kwargs["headers"]["Idempotency-Key"]))
            if len(attempts) < 3:
                return httpx.Response(503)
        return actual(method, url, **kwargs)

    monkeypatch.setattr("neatlogs.datasets._api.httpx.request", request)
    with client[0].activate(), capture():
        with neatlogs.trace("root", kind="WORKFLOW"):
            pass
    assert len(attempts) == 3
    assert len(set(attempts)) == 1


def test_manifest_overflow_is_not_truncated(client, api):
    with client[0].activate(), pytest.raises(DatasetCaptureError, match="manifest limit"):
        with capture():
            with neatlogs.trace("root", kind="WORKFLOW"):
                for _ in range(3001):
                    client[0].get_tracer("neatlogs.test").start_span("child").end()
    assert not completions(api)
    assert api[-1][2] == {"reason": "incomplete_execution"}


def test_sampling_and_disabled_exports_fail_before_registration(api):
    sdk = neatlogs.Client(api_key="key", workflow_name="disabled", disable_export=True)
    try:
        with sdk.activate(), pytest.raises(ValueError, match="export"):
            with capture():
                pass
    finally:
        sdk.shutdown()
    assert not api


def test_environment_guard_is_independent_of_cloud_backend():
    options = dict(project_id=PROJECT, token="public-only", base_url="https://app.example.com")
    with pytest.raises(ValueError, match="environment"):
        neatlogs.datasets.capture("name", **options)
    with pytest.raises(ValueError, match="Production"):
        neatlogs.datasets.capture("name", environment="production", **options)
    neatlogs.datasets.capture("name", environment="ci", **options)
    neatlogs.datasets.capture("name", environment="production", allow_production=True, **options)


def test_default_init_pipeline_supported(monkeypatch, api):
    default = importlib.import_module("neatlogs.init")
    exporter = InMemorySpanExporter()
    monkeypatch.setenv("NEATLOGS_DISABLE_EXPORT", "false")
    monkeypatch.setattr(default, "OTLPSpanExporter", lambda **kwargs: exporter)
    neatlogs.init(
        api_key="ingestion", workflow_name="default", instrumentations=[], uploads_enabled=False
    )
    with capture():
        with neatlogs.trace("root", kind="WORKFLOW"):
            pass
    assert len(completions(api)[0][2]["traces"]) == 1


def test_mask_drop_aborts_instead_of_publishing_missing_evidence(client, api):
    sdk = neatlogs.Client(
        api_key="key", workflow_name="masked", mask=lambda _: None, uploads_enabled=False
    )
    try:
        with sdk.activate(), pytest.raises(DatasetCaptureError, match="lost data"):
            with capture():
                with neatlogs.trace("masked-root", kind="WORKFLOW"):
                    pass
        assert not completions(api)
        assert api[-1][2] == {"reason": "export_failed"}
    finally:
        sdk.shutdown()


def test_other_client_does_not_enter_selected_pipeline_manifest(client, api):
    other = neatlogs.Client(api_key="other-key", workflow_name="other", uploads_enabled=False)
    try:
        with client[0].activate(), capture():
            with other.activate(), neatlogs.trace("other-root", kind="WORKFLOW"):
                pass
            with neatlogs.trace("selected-root", kind="WORKFLOW"):
                pass
        assert len(completions(api)[0][2]["traces"]) == 1
        assert len(completions(api)[0][2]["traces"][0]["spanIds"]) == 1
        other.flush()
        assert {s.name for s in client[1].get_finished_spans()} >= {"other-root", "selected-root"}
    finally:
        other.shutdown()


def test_registration_failure_prevents_execution_and_same_context_can_retry(
    client, api, monkeypatch
):
    actual = httpx.request
    keys = []

    def request(method, url, **kwargs):
        if url.endswith("dataset-capture-jobs"):
            keys.append(kwargs["headers"]["Idempotency-Key"])
            if len(keys) <= 3:
                return httpx.Response(503)
        return actual(method, url, **kwargs)

    monkeypatch.setattr("neatlogs.datasets._api.httpx.request", request)
    context = capture()
    with client[0].activate():
        with pytest.raises(DatasetCaptureError, match="503"):
            with context:
                pytest.fail("Execution cannot run before registration")
        with context:
            pass
    assert len(keys) == 4 and len(set(keys)) == 1


def test_ambiguous_completion_can_retry_without_abort(client, api, monkeypatch):
    actual = httpx.request
    keys = []

    def request(method, url, **kwargs):
        if url.endswith("complete"):
            keys.append(kwargs["headers"]["Idempotency-Key"])
            if len(keys) <= 3:
                raise httpx.ReadTimeout("secret response should not escape")
        return actual(method, url, **kwargs)

    monkeypatch.setattr("neatlogs.datasets._api.httpx.request", request)
    with client[0].activate(), pytest.raises(DatasetCaptureError, match="transport"):
        with capture() as job:
            with neatlogs.trace("root", kind="WORKFLOW"):
                pass
    assert not any(c[1].endswith("abort") for c in api)
    job.retry_completion()
    assert len(keys) == 4 and len(set(keys)) == 1


def test_payload_limit_counts_actual_encoded_bytes():
    from neatlogs.datasets._api import encode_manifest

    assert len(encode_manifest({"x": "a" * (65536 - 8)})) == 65536
    with pytest.raises(DatasetCaptureError, match="64 KiB"):
        encode_manifest({"x": "a" * (65536 - 7)})


def test_sampled_down_provider_rejected_before_registration(api):
    sdk = neatlogs.Client(
        api_key="key", workflow_name="sampled", disable_export=True, sample_rate=0.5
    )
    try:
        with sdk.activate(), pytest.raises(ValueError, match="sampled"):
            with capture():
                pass
    finally:
        sdk.shutdown()
    assert not api


def test_wait_never_treats_partial_capture_as_success(client, api, monkeypatch):
    with client[0].activate(), capture() as job:
        pass
    monkeypatch.setattr(
        job,
        "_request",
        lambda *args, **kwargs: {"job": {"state": "needs_review", "failedCount": 1}},
    )
    with pytest.raises(DatasetCaptureError, match="needs_review"):
        job.wait()
    assert job.last_status["job"]["failedCount"] == 1


def test_manifest_follows_final_post_mask_http_filter(client, api):
    def mask(snapshot):
        if snapshot["name"] == "becomes-http":
            snapshot["attributes"]["neatlogs.span.kind"] = "HTTP"
        return snapshot

    sdk = neatlogs.Client(
        api_key="key", workflow_name="post-mask", mask=mask, uploads_enabled=False
    )
    try:
        with sdk.activate(), capture():
            with neatlogs.trace("root", kind="WORKFLOW"):
                with sdk.get_tracer("neatlogs.test").start_as_current_span(
                    "becomes-http", kind=SpanKind.CLIENT
                ):
                    pass
        assert len(completions(api)[0][2]["traces"][0]["spanIds"]) == 1
        assert "becomes-http" not in {s.name for s in client[1].get_finished_spans()}
    finally:
        sdk.shutdown()


def test_warning_policy_cannot_replace_agent_exception(client, api, monkeypatch):
    monkeypatch.setattr(client[1], "export", lambda spans: SpanExportResult.FAILURE)
    error = RuntimeError("agent failure")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with client[0].activate(), pytest.raises(RuntimeError) as raised:
            with capture() as job:
                with neatlogs.trace("root", kind="WORKFLOW"):
                    raise error
    assert raised.value is error
    assert job.error is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["mask", "export"])
async def test_overlapping_capture_failure_does_not_poison_healthy_capture(
    client, api, monkeypatch, failure
):
    exporter = client[1]
    original_export = exporter.export
    if failure == "export":

        def export(spans):
            return (
                SpanExportResult.FAILURE
                if any(s.name == "bad-root" for s in spans)
                else original_export(spans)
            )

        monkeypatch.setattr(exporter, "export", export)
    sdk = neatlogs.Client(
        api_key="key",
        workflow_name="shared",
        uploads_enabled=False,
        mask=(
            (lambda snapshot: None if snapshot["name"] == "bad-root" else snapshot)
            if failure == "mask"
            else None
        ),
    )
    healthy_started = asyncio.Event()
    bad_finished = asyncio.Event()

    async def healthy():
        with sdk.activate():
            async with capture() as job:
                with neatlogs.trace("healthy-root", kind="WORKFLOW"):
                    healthy_started.set()
                    await bad_finished.wait()
            return job

    async def failing():
        await healthy_started.wait()
        try:
            with sdk.activate(), pytest.raises(DatasetCaptureError, match="lost data"):
                async with capture():
                    with neatlogs.trace("bad-root", kind="WORKFLOW"):
                        pass
        finally:
            bad_finished.set()

    try:
        good, _ = await asyncio.gather(healthy(), failing())
        assert good.error is None
        assert len(completions(api)) == 1
        assert len([c for c in api if c[1].endswith("abort")]) == 1
        assert (
            sdk.get_delivery_diagnostics()[
                "masked_span_drops" if failure == "mask" else "span_export_failures"
            ]
            > 0
        )
        captured_ids = completions(api)[0][2]["traces"][0]["spanIds"]
        assert captured_ids == [
            f"{s.context.span_id:016x}"
            for s in exporter.get_finished_spans()
            if s.name == "healthy-root"
        ]
    finally:
        sdk.shutdown()


@pytest.mark.parametrize("dropped_name", ["child", "neatlogs.trace.complete"])
def test_own_missing_business_span_or_completion_marker_still_aborts(client, api, dropped_name):
    sdk = neatlogs.Client(
        api_key="key",
        workflow_name="partial",
        uploads_enabled=False,
        mask=lambda snapshot: None if snapshot["name"] == dropped_name else snapshot,
    )
    try:
        with sdk.activate(), pytest.raises(DatasetCaptureError, match="lost data"):
            with capture():
                with neatlogs.trace("root", kind="WORKFLOW"):
                    with neatlogs.trace("child", kind="TOOL"):
                        pass
        assert not completions(api)
        assert api[-1][2] == {"reason": "export_failed"}
        assert "root" in {s.name for s in client[1].get_finished_spans()}
    finally:
        sdk.shutdown()


@pytest.mark.parametrize("raise_error", [False, True])
def test_shared_flush_failure_is_ignored_only_with_own_export_receipts(
    client, api, monkeypatch, raise_error
):
    original_flush = client[0].flush

    def flush():
        original_flush()
        if raise_error:
            raise RuntimeError("unrelated pipeline work failed")
        return False

    monkeypatch.setattr(client[0], "flush", flush)
    with client[0].activate(), capture() as job:
        with neatlogs.trace("root", kind="WORKFLOW"):
            pass
    assert job.error is None
    assert len(completions(api)) == 1


def test_queued_idempotent_receipt_preserves_known_completed_status(client, api):
    with client[0].activate(), capture() as job:
        with neatlogs.trace("root", kind="WORKFLOW"):
            pass
    status = job.status()
    assert job.state == "completed"
    assert job.retry_completion()["state"] == "queued"
    assert job.state == "completed"
    assert job.last_status is status


@pytest.mark.parametrize("operation", ["complete", "abort"])
@pytest.mark.parametrize(
    "invalid",
    [
        {"captureId": str(uuid4())},
        {"datasetId": str(uuid4())},
        {"state": "unknown"},
        {"state": None},
        {"captureId": 42},
    ],
)
def test_invalid_capture_receipt_does_not_mutate_handle(
    client, api, monkeypatch, operation, invalid
):
    with client[0].activate(), capture() as job:
        with neatlogs.trace("root", kind="WORKFLOW"):
            pass
    status = job.status()
    receipt = {
        "captureId": job.capture_id,
        "datasetId": job.dataset_id,
        "state": "queued",
        **invalid,
    }
    monkeypatch.setattr(job, "_request", lambda *args, **kwargs: receipt)
    with pytest.raises(DatasetCaptureError, match="invalid capture receipt"):
        job.retry_completion() if operation == "complete" else job._abort("cancelled")
    assert job.state == "completed"
    assert job.last_status is status
