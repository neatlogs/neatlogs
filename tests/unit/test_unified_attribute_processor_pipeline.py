import json
from pathlib import Path
from types import SimpleNamespace

import pytest
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
    name: str = "",
) -> SimpleNamespace:
    resource = SimpleNamespace(attributes=resource_attributes or {})
    ctx = SimpleNamespace(trace_id=trace_id, span_id=span_id)
    return SimpleNamespace(
        kind=kind,
        name=name,
        attributes=attributes,
        resource=resource,
        instrumentation_scope=None,
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
    assert out["neatlogs.span.kind"] == "http"


@pytest.mark.parametrize(
    "attributes",
    [
        {
            "http.method": "POST",
            "http.url": "https://user:pass@example.com/pay?token=secret#frag",
            "http.status_code": 201,
        },
        {
            "http.request.method": "POST",
            "url.full": "https://user:pass@example.com/pay?token=secret#frag",
            "http.response.status_code": 201,
        },
    ],
)
def test_process_synthesizes_safe_http_canonical_io(attributes: dict) -> None:
    proc = UnifiedAttributeProcessor(mapping_config=_load_mapping(), debug=False)
    span = _mk_span(kind=SpanKind.CLIENT, attributes=attributes)

    out = proc.process(span)

    assert out["neatlogs.span.kind"] == "http"
    assert json.loads(out["neatlogs.http.input"]) == {
        "method": "POST",
        "url": "https://example.com/pay",
    }
    assert json.loads(out["neatlogs.http.output"]) == {"status": 201}
    serialized = json.dumps(out, default=str)
    assert "user:pass" not in serialized
    assert "token=secret" not in serialized
    assert "frag" not in serialized


def test_process_synthesizes_http_input_without_status_or_secret_payloads() -> None:
    proc = UnifiedAttributeProcessor(mapping_config=_load_mapping(), debug=False)
    span = _mk_span(
        kind=SpanKind.CLIENT,
        attributes={
            "http.request.method": "GET",
            "url.scheme": "https",
            "server.address": "api.example.com",
            "server.port": 443,
            "url.path": "/health",
            "url.query": "api_key=secret",
            "url.fragment": "private",
            "input.value": '{"password":"secret-input"}',
            "output.value": '{"token":"secret-output"}',
            "http.request.header.authorization": "Bearer secret",
            "http.response.header.x-internal-token": "secret-response",
            "http.request.header.cookie": "sid=secret",
            "http.request.body": '{"password":"secret"}',
        },
    )

    out = proc.process(span)

    assert json.loads(out["neatlogs.http.input"]) == {
        "method": "GET",
        "url": "https://api.example.com:443/health",
    }
    assert json.loads(out["input.value"]) == {
        "method": "GET",
        "url": "https://api.example.com:443/health",
    }
    assert "neatlogs.http.output" not in out
    assert "output.value" not in out
    serialized = json.dumps(out, default=str)
    assert "Bearer secret" not in serialized
    assert "sid=secret" not in serialized
    assert "password" not in serialized
    assert "secret-response" not in serialized
    assert "secret-input" not in serialized
    assert "secret-output" not in serialized


def test_http_url_sanitization_handles_ipv6_and_malformed_ports() -> None:
    proc = UnifiedAttributeProcessor(mapping_config=_load_mapping(), debug=False)
    ipv6 = _mk_span(
        kind=SpanKind.CLIENT,
        attributes={
            "http.request.method": "GET",
            "url.scheme": "http",
            "server.address": "2001:db8::1",
            "server.port": 8080,
            "url.path": "/health",
        },
    )
    malformed = _mk_span(
        kind=SpanKind.CLIENT,
        attributes={
            "http.request.method": "GET",
            "url.full": "https://example.com:not-a-port/path?token=secret",
        },
    )

    ipv6_out = proc.process(ipv6)
    malformed_out = proc.process(malformed)

    assert json.loads(ipv6_out["neatlogs.http.input"])["url"] == (
        "http://[2001:db8::1]:8080/health"
    )
    assert json.loads(malformed_out["neatlogs.http.input"])["url"] == ("https://example.com/path")


@pytest.mark.parametrize(
    "path_key,path_value",
    [
        ("url.path", "/health?api_key=secret#fragment"),
        ("http.target", "https://user:pass@evil.example/health?api_key=secret#fragment"),
        ("http.route", "/users/{id}?token=secret#fragment"),
    ],
)
def test_http_path_attributes_cannot_preserve_url_credentials(
    path_key: str, path_value: str
) -> None:
    proc = UnifiedAttributeProcessor(mapping_config=_load_mapping(), debug=False)
    attributes = {
        "http.request.method": "GET",
        "server.address": "api.example.com",
        path_key: path_value,
    }
    span = _mk_span(kind=SpanKind.CLIENT, attributes=attributes)

    out = proc.process(span)

    assert json.loads(out["neatlogs.http.input"]) == {
        "method": "GET",
        "url": (
            "http://api.example.com/health"
            if path_key != "http.route"
            else "http://api.example.com/users/{id}"
        ),
    }
    serialized = json.dumps(out, default=str)
    for secret in ("user:pass", "evil.example", "api_key=secret", "token=secret", "fragment"):
        assert secret not in serialized


@pytest.mark.parametrize("kind", ["GUARDRAIL", "EVALUATOR", "MEMORY"])
def test_process_maps_generic_io_to_semantic_kind_namespace(kind: str) -> None:
    proc = UnifiedAttributeProcessor(mapping_config=_load_mapping(), debug=False)
    span = _mk_span(
        kind=SpanKind.INTERNAL,
        attributes={
            "openinference.span.kind": kind,
            "input.value": '{"question":"what happened"}',
            "output.value": '{"answer":"verified"}',
            "input.mime_type": "application/json",
            "output.mime_type": "application/json",
        },
    )

    out = proc.process(span)
    namespace = kind.lower()
    assert out["neatlogs.span.kind"] == namespace
    assert out[f"neatlogs.{namespace}.input"] == '{"question":"what happened"}'
    assert out[f"neatlogs.{namespace}.output"] == '{"answer":"verified"}'
    assert out[f"neatlogs.{namespace}.input_mime_type"] == "application/json"
    assert out[f"neatlogs.{namespace}.output_mime_type"] == "application/json"


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
