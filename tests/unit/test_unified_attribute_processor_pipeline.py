import json
from pathlib import Path
from types import SimpleNamespace

from opentelemetry.trace import SpanKind

import neatlogs
from neatlogs.core.attribute_processor import UnifiedAttributeProcessor


def _load_mapping() -> dict:
    mapping_path = Path(neatlogs.__file__).resolve().parent / "config" / "attribute-mapping.json"
    return json.loads(mapping_path.read_text(encoding="utf-8"))


def _mk_span(
    *,
    kind: SpanKind,
    attributes: dict,
    resource_attributes: dict | None = None,
    events: list | None = None,
    start_time_ns: int = 0,
    end_time_ns: int = 1_000_000_000,
    trace_id: int = 1,
    span_id: int = 2,
) -> SimpleNamespace:
    resource = SimpleNamespace(attributes=resource_attributes or {})
    ctx = SimpleNamespace(trace_id=trace_id, span_id=span_id)
    return SimpleNamespace(
        kind=kind,
        attributes=attributes,
        resource=resource,
        events=events or [],
        start_time=start_time_ns,
        end_time=end_time_ns,
        context=ctx,
    )


def test_process_marks_http_client_span_as_http_kind() -> None:
    proc = UnifiedAttributeProcessor(mapping_config=_load_mapping(), debug=False)

    span = _mk_span(
        kind=SpanKind.CLIENT,
        attributes={
            "http.method": "POST",
            "http.url": "https://example.com",
            "http.status_code": 200,
        },
        resource_attributes={"service.name": "svc"},
    )

    out = proc.process(span)
    assert out["neatlogs.span.kind"] == "HTTP"


def test_process_infers_retriever_kind_from_db_system() -> None:
    proc = UnifiedAttributeProcessor(mapping_config=_load_mapping(), debug=False)

    span = _mk_span(
        kind=SpanKind.INTERNAL,
        attributes={"db.system": "chroma", "db.operation": "query"},
        resource_attributes={"service.name": "svc"},
    )

    out = proc.process(span)
    assert out["neatlogs.span.kind"] == "retriever"


def test_process_mcp_response_only_when_mcp_signals_present() -> None:
    proc = UnifiedAttributeProcessor(mapping_config=_load_mapping(), debug=False)

    # No MCP signals => should NOT set mcp.response.value
    span_no_signal = _mk_span(
        kind=SpanKind.INTERNAL,
        attributes={"traceloop.entity.output": '{"ok": true}'},
        resource_attributes={"service.name": "svc"},
    )
    out_no_signal = proc.process(span_no_signal)
    assert "neatlogs.mcp.response_value" not in out_no_signal

    # MCP signals present via traceloop.entity.input => should set response
    span_signal = _mk_span(
        kind=SpanKind.INTERNAL,
        attributes={
            "traceloop.entity.input": json.dumps(
                {"method": "tools/call", "params": {"name": "add", "arguments": {"a": 1, "b": 2}}}
            ),
            "traceloop.entity.output": '{"result": 3}',
        },
        resource_attributes={"service.name": "svc"},
    )
    out_signal = proc.process(span_signal)
    assert out_signal["neatlogs.mcp.method"] == "tools/call"
    assert out_signal["neatlogs.mcp.request_argument"]
    assert out_signal["neatlogs.mcp.response_value"] == '{"result": 3}'


def test_process_streaming_chunk_events_produce_time_per_output_token_metric() -> None:
    proc = UnifiedAttributeProcessor(mapping_config=_load_mapping(), debug=False)

    # Avoid depending on OTel meter internals; just ensure record() is called.
    recorded: dict = {}

    class _DummyHist:
        def record(self, value, attributes=None):
            recorded["value"] = value
            recorded["attributes"] = attributes or {}

    proc.time_per_token_histogram = _DummyHist()

    events = [
        SimpleNamespace(name="llm.content.completion.chunk", timestamp=0, attributes={}),
        SimpleNamespace(
            name="llm.content.completion.chunk", timestamp=2_000_000, attributes={}
        ),  # +2ms
        SimpleNamespace(
            name="llm.content.completion.chunk", timestamp=4_000_000, attributes={}
        ),  # +2ms
    ]
    span = _mk_span(
        kind=SpanKind.INTERNAL,
        attributes={"gen_ai.request.model": "gpt-4o-mini"},
        events=events,
        start_time_ns=0,
        end_time_ns=10_000_000,
        trace_id=0xABC,
        span_id=0xDEF,
    )

    out = proc.process(span)
    assert out["neatlogs.llm.metrics.time_per_output_token"] == 2.0
    assert recorded["value"] == 2.0
    assert recorded["attributes"]["trace_id"] == f"{0xABC:032x}"
    assert recorded["attributes"]["span_id"] == f"{0xDEF:016x}"
    assert recorded["attributes"]["llm_model"] == "gpt-4o-mini"


def test_process_drops_vectordb_embedding_model_on_non_embedding_spans() -> None:
    proc = UnifiedAttributeProcessor(mapping_config=_load_mapping(), debug=False)

    span = _mk_span(
        kind=SpanKind.INTERNAL,
        attributes={
            "openinference.span.kind": "LLM",
            # This should be dropped by _apply_namespace_mapping for non-embedding/non-retriever spans.
            "neatlogs.vectordb.embedding_model": "text-embedding-3-small",
        },
        resource_attributes={"service.name": "svc"},
    )

    out = proc.process(span)
    assert out["neatlogs.span.kind"] == "llm"
    assert "neatlogs.vectordb.embedding_model" not in out


def test_process_sets_framework_when_gen_ai_system_is_known_framework() -> None:
    proc = UnifiedAttributeProcessor(mapping_config=_load_mapping(), debug=False)

    span = _mk_span(
        kind=SpanKind.INTERNAL,
        attributes={"gen_ai.system": "langchain"},
        resource_attributes={"service.name": "svc"},
    )
    out = proc.process(span)
    assert out["neatlogs.framework"] == "langchain"
