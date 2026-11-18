from types import SimpleNamespace
from uuid import uuid4

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import neatlogs._wrap_utils as wrappers
import neatlogs.evals as evals
import neatlogs.evals._recovery as recovery
from neatlogs.replay.api import LocalRunApiError


def test_optional_trace_wait_reserves_value_upload_window(monkeypatch):
    now = [100.0]
    clock = SimpleNamespace(
        monotonic=lambda: now[0], sleep=lambda delay: now.__setitem__(0, now[0] + delay)
    )
    monkeypatch.setattr(evals, "time", clock)
    monkeypatch.setattr(recovery, "time", clock)
    flush_budgets, uploaded, saved = [], [], []

    def flush(timeout_millis=10_000):
        flush_budgets.append(timeout_millis)
        now[0] += timeout_millis / 1000
        return False

    class Api:
        def request(self, _method, _path, body, *, key, timeout):
            now[0] += 0.1
            if body.get("traceId"):
                raise LocalRunApiError("Trace delayed", status=409, code="TRACE_NOT_READY")
            assert now[0] < 107
            uploaded.append(body.copy())
            return {"state": "succeeded"}

    monkeypatch.setattr(evals, "_flush_case_trace", flush)
    slot = {
        "key": str(uuid4()),
        "body": {"caseId": str(uuid4()), "trial": 1, "output": "computed", "traceId": "a" * 32},
    }
    original_key = slot["key"]
    assert (
        evals._upload_case_output(Api(), str(uuid4()), slot, 107, lambda: saved.append(True))
        is None
    )
    assert flush_budgets == [2000]
    assert uploaded == [{"caseId": slot["body"]["caseId"], "trial": 1, "output": "computed"}]
    assert now[0] <= 102.2
    assert slot["key"] != original_key
    assert saved


def test_wrapper_only_trace_flush_exports_its_private_provider(monkeypatch):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter, schedule_delay_millis=600_000))
    monkeypatch.setattr(wrappers, "_bootstrap_provider", provider)
    monkeypatch.setattr(wrappers, "_neatlogs_provider", provider)
    monkeypatch.setattr(evals, "get_active_client", lambda: None)
    monkeypatch.setattr(
        evals,
        "flush",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("default provider selected")),
    )
    try:
        with provider.get_tracer("wrapper-only").start_as_current_span("case"):
            pass
        assert exporter.get_finished_spans() == ()
        assert evals._flush_case_trace(timeout_millis=500)
        assert [span.name for span in exporter.get_finished_spans()] == ["case"]
    finally:
        provider.shutdown()


def test_uncertain_optional_trace_reconciles_same_body_and_key_before_unlinking(monkeypatch):
    now = [100.0]
    clock = SimpleNamespace(
        monotonic=lambda: now[0], sleep=lambda delay: now.__setitem__(0, now[0] + delay)
    )
    monkeypatch.setattr(evals, "time", clock)
    monkeypatch.setattr(recovery, "time", clock)
    monkeypatch.setattr(evals, "_flush_case_trace", lambda **_kwargs: True)
    requests = []

    class Api:
        def status(self, _run_id, *, timeout):
            return {"state": "running"}

        def request(self, _method, _path, body, *, key, timeout):
            requests.append((body.copy(), key, timeout))
            if len(requests) == 1:
                now[0] += timeout
                raise LocalRunApiError("Lost trace receipt", code="RUN_API_TRANSPORT_FAILURE")
            if body.get("traceId"):
                raise LocalRunApiError("Trace delayed", status=409, code="TRACE_NOT_READY")
            return {"state": "succeeded"}

    slot = {
        "key": str(uuid4()),
        "body": {"caseId": str(uuid4()), "trial": 1, "output": "computed", "traceId": "a" * 32},
    }
    assert evals._upload_case_output(Api(), str(uuid4()), slot, 107, lambda: None) is None
    assert requests[0][2] == 2
    assert requests[1][0] == requests[0][0]
    assert requests[1][1] == requests[0][1]
    assert requests[1][2] == 4
    assert "traceId" not in requests[2][0]
    assert requests[2][1] != requests[0][1]
    assert now[0] < 107
