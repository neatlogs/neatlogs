import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.trace import SpanContext, SpanKind, Status, StatusCode, TraceFlags, TraceState

from neatlogs.__main__ import main
from neatlogs.core.choice_accumulator import ChoiceAccumulator
from neatlogs.core.masking_exporter import MaskingSpanExporter
from neatlogs.core.media import set_media_attributes
from neatlogs.doctor_v2 import (
    clear_doctor_capture,
    doctor_captured_local_v2,
    doctor_local_v2,
    doctor_probe_v2,
    doctor_semantic_digest,
    get_captured_envelope,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "doctor-v2"


def envelope():
    return json.loads((FIXTURES / "valid-envelope.json").read_text())


def rich_envelope():
    return json.loads((FIXTURES / "rich-envelope.json").read_text())


def test_canonical_digest_matches_cross_language_golden_value():
    assert doctor_semantic_digest(envelope()) == (
        "sha256:824650f5fbc6d9f8d923813564116092" "63417219eaf7fdafbd2ba94795b6c4f7"
    )


def test_rich_digest_matches_cross_language_golden_with_top_level_tool_requests():
    fixture = rich_envelope()
    expected = "sha256:45b1ebe029b272ceb45edb210978f6600d29ac50ca4d9cd0f4ef5abb3eff063e"
    assert doctor_semantic_digest(fixture) == expected


def test_valid_envelope_conforms_to_doctor_v2_shape():
    result = doctor_local_v2(envelope(), flush_duration_ms=12)
    assert result["format_version"] == "neatlogs.doctor/v2"
    assert result["mode"] == "local"
    assert result["status"] == "pass", result
    assert result["first_failure"] is None
    assert result["runtime"]["language"] == "python"
    assert result["checks"][0]["reason_code"] == "LOCAL_ENVELOPE_VALID"


def test_unknown_future_reason_uses_safe_support_remediation():
    from neatlogs.doctor_v2 import _check

    assert (
        _check("future", "fail", "SERVER_REASON_V99", "future failure")["remediation_code"]
        == "CONTACT_SUPPORT"
    )


def test_validation_reports_first_contract_order_failure():
    invalid = envelope()
    invalid["trace_id"] = "bad"
    invalid["spans"][1]["parent_span_id"] = "missing"
    result = doctor_local_v2(invalid)
    assert result["status"] == "fail"
    assert result["first_failure"] == "TRACE_ID_INVALID"
    assert "capture" not in result


def _span(attributes, *, trace_id=1, span_id=2, parent_id=0):
    context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags.SAMPLED,
        trace_state=TraceState(),
    )
    parent = (
        None
        if not parent_id
        else SpanContext(
            trace_id=trace_id,
            span_id=parent_id,
            is_remote=False,
            trace_flags=TraceFlags.SAMPLED,
            trace_state=TraceState(),
        )
    )
    return ReadableSpan(
        name="root",
        context=context,
        parent=parent,
        resource=Resource.create({}),
        attributes=attributes,
        events=[],
        links=[],
        kind=SpanKind.INTERNAL,
        status=Status(StatusCode.OK),
        start_time=1,
        end_time=2,
        instrumentation_scope=None,
    )


class Exporter:
    def __init__(self):
        self.spans = []
        self.shutdown_calls = 0

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis=30000):
        return True

    def shutdown(self):
        self.shutdown_calls += 1


def persisted_trace(exporter):
    spans = [span for span in exporter.spans if span.name != "neatlogs.trace.complete"]
    kinds = {
        "WORKFLOW": "workflow",
        "AGENT": "agent_action",
        "LLM": "llm",
        "TOOL": "tool_call",
    }
    return {
        "_id": f"{spans[0].get_span_context().trace_id:032x}",
        "status": "success",
        "workflowName": "neatlogs.doctor.v2",
        "spanCount": len(spans),
        "promptTokens": 11,
        "completionTokens": 7,
        "totalTokensUsed": 18,
        "spans": [
            {
                "span_id": f"{item.get_span_context().span_id:016x}",
                **(
                    {"parent_span_id": f"{item.parent.span_id:016x}"}
                    if item.parent and item.parent.is_valid
                    else {}
                ),
                "node_name": item.name,
                "node_type": kinds[
                    str(
                        item.attributes.get("openinference.span.kind")
                        or item.attributes.get("neatlogs.span.kind")
                    )
                    .removeprefix("Neatlogs.")
                    .upper()
                ],
                "data": {
                    "input_value": item.attributes.get("input.value", "{}"),
                    "output_value": item.attributes.get("output.value", "{}"),
                },
                "span_metadata": {
                    "neatlogs.doctor": item.attributes.get("neatlogs.doctor"),
                    "neatlogs.doctor.version": item.attributes.get("neatlogs.doctor.version"),
                    "service.name": item.attributes.get("service.name"),
                    "telemetry.sdk.language": item.attributes.get("telemetry.sdk.language"),
                    "telemetry.sdk.version": item.attributes.get("telemetry.sdk.version"),
                    "neatlogs.span.type": item.attributes.get("neatlogs.span.type"),
                },
            }
            for item in spans
        ],
    }


def materialize_for_v3(trace_data):
    """Mirror the backend's UI-facing simplified trace representation."""
    for item in trace_data["spans"]:
        data = item["data"]
        if item["node_name"] == "doctor.probe.root":
            data["input_value"] = "generated diagnostic input"
            data["output_value"] = "Value: 2"
        elif item["node_name"] == "doctor.probe.llm":
            data["input_value"] = {"prompt": "generated diagnostic input"}
            data["output_value"] = "Text: generated diagnostic output"
        elif item["node_name"] == "doctor.probe.agent":
            data["input_value"] = "Prompt: generated diagnostic input"
            data["output_value"] = "Text: generated diagnostic output"
        elif item["node_name"] == "doctor.probe.tool":
            data["input_value"] = "Value: 1"
            data["output_value"] = "Value: 2"
    return trace_data


def test_ordinary_exporter_path_never_enters_doctor_retention():
    clear_doctor_capture()
    inner = Exporter()
    exporter = MaskingSpanExporter(inner, None)
    exporter.export([_span({"neatlogs.span.kind": "WORKFLOW"})])
    assert get_captured_envelope("00000000000000000000000000000001") is None


def test_explicit_doctor_exporter_captures_only_post_mask_data():
    clear_doctor_capture()
    inner = Exporter()

    def mask(snapshot):
        snapshot["attributes"]["neatlogs.workflow.input"] = '{"email":"[masked]"}'
        return snapshot

    exporter = MaskingSpanExporter(inner, mask, doctor_capture=True)
    assert (
        exporter.export(
            [
                _span(
                    {
                        "neatlogs.span.kind": "WORKFLOW",
                        "neatlogs.workflow.input": '{"email":"secret@example.com"}',
                    }
                )
            ]
        )
        is SpanExportResult.SUCCESS
    )
    captured = get_captured_envelope("00000000000000000000000000000001")
    assert captured["spans"][0]["input"] == {"email": "[masked]"}
    serialized = json.dumps(captured)
    assert "secret@example.com" not in serialized
    assert doctor_captured_local_v2("00000000000000000000000000000001")["status"] == "pass"
    exporter.shutdown()
    assert get_captured_envelope("00000000000000000000000000000001") is None


def test_production_choice_and_media_paths_flow_into_doctor_capture():
    """Exercise the same helpers used by supported OpenAI-compatible wrappers."""

    clear_doctor_capture()
    provider = TracerProvider()
    provider.add_span_processor(
        SimpleSpanProcessor(MaskingSpanExporter(Exporter(), None, doctor_capture=True))
    )
    tracer = provider.get_tracer("doctor-supported-wrapper-path")
    with tracer.start_as_current_span("wrapped.llm") as llm:
        trace_id = f"{llm.get_span_context().trace_id:032x}"
        llm.set_attribute("neatlogs.span.kind", "LLM")
        llm.set_attribute("neatlogs.llm.input", '{"prompt":"hello"}')
        llm.set_attribute("neatlogs.llm.output", '{"text":"AB"}')
        accumulator = ChoiceAccumulator()
        accumulator.add_chunk(
            llm,
            {
                "choices": [
                    {"index": 0, "delta": {"content": "A"}},
                    {
                        "index": 1,
                        "delta": {
                            "content": "X",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "doctor_call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "diagnostic_tool",
                                        "arguments": '{"value":',
                                    },
                                }
                            ],
                        },
                    },
                ]
            },
        )
        accumulator.add_chunk(
            llm,
            {
                "choices": [
                    {"index": 0, "delta": {"content": "B"}, "finish_reason": "stop"},
                    {
                        "index": 1,
                        "delta": {
                            "content": "Y",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": "1}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    },
                ]
            },
        )
        accumulator.apply(llm)
        llm.set_attribute("neatlogs.llm.is_streaming", True)
        raw = b"doctor-image"
        set_media_attributes(
            llm,
            "neatlogs.llm.output_messages.0",
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + base64.b64encode(raw).decode()},
            },
            "output",
        )
        with tracer.start_as_current_span("wrapped.tool") as tool:
            tool.set_attribute("neatlogs.span.kind", "TOOL")
            tool.set_attribute("neatlogs.tool_call.id", "doctor_call_1")
            tool.set_attribute("neatlogs.tool.name", "diagnostic_tool")
            tool.set_attribute("neatlogs.tool.input", '{"value":1}')
            tool.set_attribute("neatlogs.tool.output", '{"value":2}')
    assert provider.force_flush(timeout_millis=5_000)
    captured = get_captured_envelope(trace_id)
    by_name = {item["name"]: item for item in captured["spans"]}
    projected = by_name["wrapped.llm"]
    assert projected["expected_choice_count"] == 2
    assert [choice["message"]["content"] for choice in projected["choices"]] == ["AB", "XY"]
    assert projected["tool_calls"][0]["arguments"] == {"value": 1}
    assert all("tool_calls" not in choice["message"] for choice in projected["choices"])
    assert len(projected["stream_fragments"]) == 2
    assert projected["streaming"] is True
    assert projected["payload_references"] == [
        {
            "digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
            "mime_type": "image/png",
        }
    ]
    assert by_name["wrapped.tool"]["tool_call"]["id"] == "doctor_call_1"
    assert doctor_local_v2(captured)["status"] == "pass"
    provider.shutdown()
    clear_doctor_capture()


def test_public_google_genai_wrapper_flows_through_normalizer_and_doctor_capture():
    from neatlogs import wrap
    from neatlogs.core.context import trace
    from neatlogs.init import flush, init, shutdown

    class Models:
        def generate_content(self, *args, **kwargs):
            raise AssertionError("non-streaming provider method was not expected")

        def generate_content_stream(self, *args, **kwargs):
            return iter(
                [
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "content": {"role": "model", "parts": [{"text": "A"}]},
                            },
                            {
                                "index": 1,
                                "content": {
                                    "role": "model",
                                    "parts": [
                                        {"text": "X"},
                                        {
                                            "function_call": {
                                                "id": "doctor_call_1",
                                                "name": "diagnostic_tool",
                                                "args": {"value": 1},
                                            }
                                        },
                                    ],
                                },
                            },
                        ]
                    },
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "content": {"role": "model", "parts": [{"text": "B"}]},
                            },
                            {
                                "index": 1,
                                "finish_reason": "tool_calls",
                                "content": {"role": "model", "parts": [{"text": "Y"}]},
                            },
                        ]
                    },
                ]
            )

    Client = type("Client", (), {"__module__": "google.genai.client"})
    client = Client()
    client.vertexai = False
    client.models = Models()
    exporter = Exporter()
    init(
        api_key="project-key",
        endpoint="http://localhost:4100",
        workflow_name="wrapper-doctor-test",
        register_shutdown_handlers=False,
        _doctor_probe=True,
        _doctor_probe_exporter=exporter,
    )
    try:
        wrapped = wrap(client)
        with trace(name="wrapped.root", kind="WORKFLOW") as root:
            trace_id = f"{root.get_span_context().trace_id:032x}"
            list(
                wrapped.models.generate_content_stream(
                    model="gemini-test",
                    contents="generated diagnostic input",
                )
            )
            with trace(name="wrapped.tool", kind="TOOL") as tool:
                tool.set_attribute("neatlogs.tool_call.id", "doctor_call_1")
                tool.set_attribute("neatlogs.tool.name", "diagnostic_tool")
                tool.set_attribute("neatlogs.tool.input", '{"value":1}')
                tool.set_attribute("neatlogs.tool.output", '{"value":2}')
        assert flush(timeout_millis=5_000)
        captured = get_captured_envelope(trace_id)
        assert captured is not None
        by_name = {item["name"]: item for item in captured["spans"]}
        llm = by_name["google_genai.models.generate_content"]
        assert [choice["message"].get("content") for choice in llm["choices"]] == ["AB", "XY"]
        assert llm["tool_calls"][0]["id"] == "doctor_call_1"
        assert all("tool_calls" not in choice["message"] for choice in llm["choices"])
        assert llm["streaming"] is True
        assert by_name["wrapped.tool"]["tool_call"]["id"] == "doctor_call_1"
        assert doctor_local_v2(captured)["status"] == "pass"
    finally:
        shutdown(termination_reason="doctor-wrapper-test-complete")
        clear_doctor_capture()


def test_emitted_span_capture_derives_choices_stream_tool_and_payload_fields():
    clear_doctor_capture()
    provider = TracerProvider()
    provider.add_span_processor(
        SimpleSpanProcessor(MaskingSpanExporter(Exporter(), None, doctor_capture=True))
    )
    tracer = provider.get_tracer("doctor-capture-regression")
    digest = "a" * 64
    with tracer.start_as_current_span("doctor.llm") as llm:
        trace_id = f"{llm.get_span_context().trace_id:032x}"
        llm.set_attribute("neatlogs.span.kind", "LLM")
        llm.set_attribute("neatlogs.llm.input", '{"prompt":"hello"}')
        llm.set_attribute("neatlogs.llm.output", '{"text":"done"}')
        llm.set_attribute("neatlogs.llm.output_messages.0.content", "first")
        llm.set_attribute("neatlogs.llm.output_messages.1.content", "second")
        llm.set_attribute("neatlogs.llm.choices.0.finish_reason", "tool_calls")
        llm.set_attribute("neatlogs.llm.choices.1.finish_reason", "stop")
        llm.set_attribute("neatlogs.llm.tool_calls.0.id", "doctor_call_1")
        llm.set_attribute("neatlogs.llm.tool_calls.0.name", "diagnostic_tool")
        llm.set_attribute("neatlogs.llm.tool_calls.0.arguments", '{"value":1}')
        llm.set_attribute("neatlogs.llm.tool_calls.0.choice_index", 0)
        llm.set_attribute("neatlogs.llm.is_streaming", True)
        llm.add_event(
            "neatlogs.stream.chunk",
            {"neatlogs.stream.chunk.summary": '{"text":"done"}'},
        )
        llm.set_attribute("neatlogs.capture.truncated", True)
        llm.set_attribute("neatlogs.llm.output.media.0.sha256", digest)
        llm.set_attribute("neatlogs.llm.output.media.0.byte_length", 1024)
        llm.set_attribute("neatlogs.llm.output.media.0.mime_type", "application/json")
        with tracer.start_as_current_span("doctor.tool") as tool:
            tool.set_attribute("neatlogs.span.kind", "TOOL")
            tool.set_attribute("neatlogs.tool_call.id", "doctor_call_1")
            tool.set_attribute("neatlogs.tool.name", "diagnostic_tool")
            tool.set_attribute("neatlogs.tool.input", '{"value":1}')
            tool.set_attribute("neatlogs.tool.output", '{"value":2}')
    assert provider.force_flush(timeout_millis=5_000)
    captured = get_captured_envelope(trace_id)
    by_name = {item["name"]: item for item in captured["spans"]}
    projected = by_name["doctor.llm"]
    assert projected["expected_choice_count"] == 2
    assert [choice["index"] for choice in projected["choices"]] == [0, 1]
    assert projected["tool_calls"][0]["id"] == "doctor_call_1"
    assert all("tool_calls" not in choice["message"] for choice in projected["choices"])
    assert projected["stream_fragments"] == [{"text": "done"}]
    assert projected["streaming"] is True
    assert projected["oversized"] is True
    assert projected["payload_references"] == [
        {
            "digest": f"sha256:{digest}",
            "size": 1024,
            "mime_type": "application/json",
        }
    ]
    assert by_name["doctor.tool"]["tool_call"] == {
        "id": "doctor_call_1",
        "name": "diagnostic_tool",
        "arguments": {"value": 1},
        "result": {"value": 2},
    }
    assert doctor_local_v2(captured)["status"] == "pass"
    provider.shutdown()
    clear_doctor_capture()


def test_capture_store_is_bounded_to_sixteen_traces():
    clear_doctor_capture()
    from neatlogs.doctor_v2 import capture_prepared_spans

    for trace_id in range(1, 18):
        capture_prepared_spans([_span({"neatlogs.span.kind": "WORKFLOW"}, trace_id=trace_id)])
    assert get_captured_envelope("00000000000000000000000000000001") is None
    assert get_captured_envelope("00000000000000000000000000000011") is not None


def test_capture_store_has_hard_span_and_byte_bounds():
    clear_doctor_capture()
    from neatlogs.doctor_v2 import capture_prepared_spans

    capture_prepared_spans(
        [
            _span(
                {"neatlogs.span.kind": "TOOL"},
                trace_id=2,
                span_id=index + 1,
                parent_id=1 if index else 0,
            )
            for index in range(80)
        ]
    )
    captured = get_captured_envelope("00000000000000000000000000000002")
    assert captured is not None
    assert len(captured["spans"]) == 64

    clear_doctor_capture()
    capture_prepared_spans(
        [
            _span(
                {
                    "neatlogs.span.kind": "WORKFLOW",
                    "neatlogs.input.value": "x" * (300 * 1024),
                },
                trace_id=3,
                span_id=1,
            )
        ]
    )
    assert get_captured_envelope("00000000000000000000000000000003") is None


def test_probe_exports_and_reads_back_the_exact_trace(monkeypatch):
    exporter = Exporter()
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        payload = materialize_for_v3(persisted_trace(exporter))
        return SimpleNamespace(
            ok=True,
            status_code=200,
            json=lambda: payload,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr("neatlogs.doctor_v2.requests.get", get)
    result = doctor_probe_v2(
        api_key="local-key",
        endpoint="http://localhost:4100",
        timeout_seconds=1,
        _exporter=exporter,
    )
    assert result["status"] == "pass", result
    assert result["capture"]["span_count"] == 4
    assert result["probe"] == {
        "ingest_route": "/v1/traces",
        "marker_header": "x-neatlogs-doctor",
        "marker_version": "v1",
        "visible": True,
        "readback_trace_id": result["capture"]["trace_id"],
        "finalized": True,
        "meaningful_root_count": 1,
        "duplicate_span_count": 0,
        "readback_span_count": 4,
        "hierarchy_valid": True,
        "attributes_valid": True,
        "input_output_valid": True,
        "metadata_valid": True,
        "typed_tokens_valid": True,
    }
    assert calls[0][0].startswith("http://localhost:4100/api/traces/v3/")
    assert calls[0][1]["headers"] == {"x-api-key": "local-key"}
    assert "local-key" not in json.dumps(result)


def test_ordinary_runtime_then_probe_fails_closed_without_disturbing_runtime(monkeypatch):
    import importlib

    from neatlogs.core.context import trace

    init_module = importlib.import_module("neatlogs.init")
    ordinary_exporter = Exporter()
    probe_exporter = Exporter()
    monkeypatch.setenv("NEATLOGS_DISABLE_EXPORT", "false")
    monkeypatch.setattr(
        init_module,
        "OTLPSpanExporter",
        lambda **_kwargs: ordinary_exporter,
    )

    init_module.init(
        api_key="local-key",
        endpoint="http://localhost:4100",
        workflow_name="neatlogs.doctor.v2",
        batch_size=32,
        flush_interval=60,
        register_shutdown_handlers=False,
    )
    ordinary_provider = init_module._tracer_provider

    result = doctor_probe_v2(
        api_key="local-key",
        endpoint="http://localhost:4100",
        timeout_seconds=1,
        _exporter=probe_exporter,
    )

    assert result["status"] == "fail"
    assert result["first_failure"] == "BACKEND_PROBE_UNAVAILABLE"
    assert "capture" not in result
    assert init_module._initialized is True
    assert init_module._tracer_provider is ordinary_provider
    assert ordinary_exporter.shutdown_calls == 0
    assert probe_exporter.spans == []

    with trace(name="ordinary.after-probe", kind="WORKFLOW"):
        pass
    assert init_module.flush(timeout_millis=5_000)
    assert any(item.name == "ordinary.after-probe" for item in ordinary_exporter.spans)
    assert init_module.shutdown(termination_reason="ordinary-test-complete")


def test_probe_then_ordinary_runtime_starts_without_doctor_transport(monkeypatch):
    import importlib

    from neatlogs.core.context import trace

    init_module = importlib.import_module("neatlogs.init")
    probe_exporter = Exporter()

    def get(*_args, **_kwargs):
        payload = materialize_for_v3(persisted_trace(probe_exporter))
        return SimpleNamespace(
            ok=True,
            status_code=200,
            json=lambda: payload,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr("neatlogs.doctor_v2.requests.get", get)
    result = doctor_probe_v2(
        api_key="local-key",
        endpoint="http://localhost:4100",
        timeout_seconds=1,
        _exporter=probe_exporter,
    )
    assert result["status"] == "pass", result
    assert init_module._initialized is False

    ordinary_exporter = Exporter()
    constructor_options = {}

    def build_exporter(**kwargs):
        constructor_options.update(kwargs)
        return ordinary_exporter

    monkeypatch.setenv("NEATLOGS_DISABLE_EXPORT", "false")
    monkeypatch.setattr(init_module, "OTLPSpanExporter", build_exporter)
    init_module.init(
        api_key="local-key",
        endpoint="http://localhost:4100",
        workflow_name="neatlogs.doctor.v2",
        batch_size=32,
        flush_interval=60,
        register_shutdown_handlers=False,
    )

    assert constructor_options["headers"] == {"x-api-key": "local-key"}
    assert "neatlogs.doctor" not in init_module._tracer_provider.resource.attributes
    with trace(name="ordinary.after-doctor", kind="WORKFLOW"):
        pass
    assert init_module.flush(timeout_millis=5_000)
    assert any(item.name == "ordinary.after-doctor" for item in ordinary_exporter.spans)
    assert init_module.shutdown(termination_reason="ordinary-test-complete")


def test_probe_rejects_wrong_materialized_input_output(monkeypatch):
    exporter = Exporter()

    def get(*args, **kwargs):
        trace_data = materialize_for_v3(persisted_trace(exporter))
        trace_data["spans"][2]["data"]["output_value"] = "Text: wrong output"
        return SimpleNamespace(
            ok=True,
            status_code=200,
            json=lambda: trace_data,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr("neatlogs.doctor_v2.requests.get", get)
    result = doctor_probe_v2(
        api_key="local-key",
        endpoint="http://localhost:4100",
        timeout_seconds=1,
        _exporter=exporter,
    )
    assert result["status"] == "fail"
    assert result["first_failure"] == "INPUT_OUTPUT_VALID_FAILED"
    assert result["probe"]["input_output_valid"] is False


def test_probe_reports_terminal_correlation_roots_and_duplicates(monkeypatch):
    exporter = Exporter()

    def get(*args, **kwargs):
        trace_data = materialize_for_v3(persisted_trace(exporter))
        trace_data["spans"][3]["span_id"] = trace_data["spans"][1]["span_id"]
        return SimpleNamespace(
            ok=True,
            status_code=200,
            json=lambda: trace_data,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr("neatlogs.doctor_v2.requests.get", get)
    result = doctor_probe_v2(
        api_key="local-key",
        endpoint="http://localhost:4100",
        timeout_seconds=1,
        _exporter=exporter,
    )
    assert result["status"] == "fail"
    assert result["probe"]["readback_trace_id"] == result["capture"]["trace_id"]
    assert result["probe"]["finalized"] is True
    assert result["probe"]["meaningful_root_count"] == 1
    assert result["probe"]["duplicate_span_count"] == 1
    assert result["probe"]["hierarchy_valid"] is False


def test_probe_rejects_failed_or_error_terminal_readback(monkeypatch):
    for terminal_status in ("failed", "error", "completed"):
        exporter = Exporter()

        def get(*args, **kwargs):
            trace_data = materialize_for_v3(persisted_trace(exporter))
            trace_data["status"] = terminal_status
            return SimpleNamespace(
                ok=True,
                status_code=200,
                json=lambda: trace_data,
                raise_for_status=lambda: None,
            )

        monkeypatch.setattr("neatlogs.doctor_v2.requests.get", get)
        result = doctor_probe_v2(
            api_key="local-key",
            endpoint="http://localhost:4100",
            timeout_seconds=1,
            _exporter=exporter,
        )
        assert result["status"] == "fail"
        assert result["first_failure"] == "TRACE_FINALIZED_FAILED"
        assert result["probe"]["finalized"] is False


def test_probe_exporter_uses_normal_trace_route_and_versioned_header(monkeypatch):
    import importlib

    init_module = importlib.import_module("neatlogs.init")
    captured = {}

    class ConstructorExporter(Exporter):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

    monkeypatch.setattr(init_module, "OTLPSpanExporter", ConstructorExporter)
    init_module.init(
        api_key="project-key",
        endpoint="http://localhost:4100",
        workflow_name="neatlogs.doctor.v2",
        register_shutdown_handlers=False,
        _doctor_probe=True,
    )
    try:
        assert captured["endpoint"] == "http://localhost:4100/v1/traces"
        assert captured["headers"] == {
            "x-api-key": "project-key",
            "x-neatlogs-doctor": "v1",
        }
    finally:
        init_module.shutdown(termination_reason="doctor-test-complete")


def test_probe_missing_credential_never_echoes_environment(monkeypatch):
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)
    result = doctor_probe_v2()
    assert result["first_failure"] == "CREDENTIAL_MISSING"
    # Credential validation is a preflight failure: do not create or imply an
    # exported controlled fixture when the normal authenticated path cannot run.
    assert "capture" not in result


def test_probe_invalid_endpoint_returns_stable_configuration_failure():
    result = doctor_probe_v2(api_key="test-key", endpoint="not-a-url")
    assert result["status"] == "fail"
    assert result["first_failure"] == "ENDPOINT_INVALID"
    assert "capture" not in result
    assert result["checks"][-1]["remediation_code"] == "SET_ENDPOINT"
    assert "api_key" not in json.dumps(result).lower()


def test_standalone_local_runs_real_isolated_exporter_pipeline(monkeypatch, capsys):
    clear_doctor_capture()
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("local Doctor attempted backend access")

    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    assert main(["doctor", "--local", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "pass"
    assert result["first_failure"] is None
    assert result["capture"]["span_count"] == 4
    assert result["capture"]["semantic_digest"] == (
        "sha256:7163d2de42c4165f3ae552279fdde2ec0839413ce608c6e5d71f3fb532df319b"
    )
    assert result["checks"] == [
        {
            "name": "local_envelope",
            "status": "pass",
            "reason_code": "LOCAL_ENVELOPE_VALID",
            "message": "The final normalized local envelope is valid",
            "remediation_code": "NONE",
        }
    ]
    assert result["runtime"]["language"] == "python"
    assert result["sampling"]["sampled"] is True
    assert result["ownership"]["provider"] == "private"
    assert result["queue"]["mode"] == "diagnostic_capture"
    assert result["flush"]["outcome"] == "success"


def test_probe_never_treats_processing_readback_as_success(monkeypatch):
    exporter = Exporter()
    response = SimpleNamespace(
        ok=True,
        status_code=202,
        json=lambda: {"status": "processing"},
        raise_for_status=lambda: None,
    )
    monkeypatch.setattr("neatlogs.doctor_v2.requests.get", lambda *a, **k: response)
    monkeypatch.setattr("neatlogs.doctor_v2.time.sleep", lambda *_: None)
    ticks = iter(float(value) for value in range(1_000))
    monkeypatch.setattr("neatlogs.doctor_v2.time.monotonic", lambda: next(ticks))
    result = doctor_probe_v2(
        api_key="local-key",
        endpoint="http://localhost:4100",
        timeout_seconds=1,
        _exporter=exporter,
    )
    assert result["status"] == "fail"
    assert result["first_failure"] == "BACKEND_PROBE_UNAVAILABLE"


def test_probe_rejects_redacted_token_counts(monkeypatch):
    exporter = Exporter()

    def get(*args, **kwargs):
        trace_data = persisted_trace(exporter)
        trace_data["promptTokens"] = "[REDACTED]"
        return SimpleNamespace(
            ok=True,
            status_code=200,
            json=lambda: trace_data,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr("neatlogs.doctor_v2.requests.get", get)
    result = doctor_probe_v2(
        api_key="local-key",
        endpoint="http://localhost:4100",
        timeout_seconds=1,
        _exporter=exporter,
    )
    assert result["status"] == "fail"
    assert result["first_failure"] == "TYPED_TOKENS_VALID_FAILED"
    assert result["probe"]["typed_tokens_valid"] is False


def test_standalone_local_reports_missing_configuration(monkeypatch, capsys):
    clear_doctor_capture()
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)
    assert main(["doctor", "--local", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["first_failure"] is None
    assert all(check["reason_code"] != "CREDENTIAL_MISSING" for check in result["checks"])
