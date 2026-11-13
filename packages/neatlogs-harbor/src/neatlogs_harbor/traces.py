"""One OTLP trace per Harbor trial.

The trace has a ``harbor.trial`` root span with one child span per phase Harbor
timed (environment setup, agent setup, agent run, verification). When the agent
wrote an ATIF trajectory, Harbor's own ``harbor-atif2otel`` converter turns it
into agent, model-call and tool-call spans that hang under the agent phase.

Trace and span ids are derived from the trial's stable id, so sending a trial
twice produces the same ids.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, InstrumentationScope, KeyValue
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span, Status

from . import ids
from .records import Outcome, TrialRecord, final_agent_message, parse_time
from .redact import Redactor
from .version import __version__

SERVICE_NAME = "harbor"
SCOPE_NAME = "neatlogs-harbor"
MAX_ATTRIBUTE_BYTES = 10_240

# Marks the trace as test traffic so it stays out of production dashboards,
# alerts and detection rates (``origin = simulation``).
TEST_TRAFFIC_ATTRIBUTES: dict[str, Any] = {
    "neatlogs.origin": "simulation",
    "neatlogs.environment": "test",
    "deployment.environment": "test",
}

_PHASES: tuple[tuple[str, str], ...] = (
    ("environment_setup", "harbor.environment_setup"),
    ("agent_setup", "harbor.agent_setup"),
    ("agent_execution", "harbor.agent"),
    ("verifier", "harbor.verifier"),
)


def _span_id(seed: str, name: str) -> bytes:
    return hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8]


def _nanos(value: Any) -> int:
    parsed = parse_time(value)
    return int(parsed.timestamp() * 1_000_000_000) if parsed else 0


def _kv(key: str, value: Any) -> KeyValue | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return KeyValue(key=key, value=AnyValue(bool_value=value))
    if isinstance(value, int):
        return KeyValue(key=key, value=AnyValue(int_value=value))
    if isinstance(value, float):
        return KeyValue(key=key, value=AnyValue(double_value=value))
    if isinstance(value, (dict, list)):
        value = json.dumps(value)
    return KeyValue(key=key, value=AnyValue(string_value=_truncate(str(value))))


def _attrs(pairs: Mapping[str, Any]) -> list[KeyValue]:
    return [kv for key, value in pairs.items() if (kv := _kv(key, value)) is not None]


def _truncate(text: str, limit: int = MAX_ATTRIBUTE_BYTES) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[: limit - 15].decode("utf-8", errors="ignore") + "...[truncated]"


def _convert_trajectory(trajectory: Mapping[str, Any], seed: str) -> ResourceSpans | None:
    try:
        from harbor_atif2otel import convert_trajectory
    except ImportError:  # pragma: no cover - harbor-atif2otel is a hard dependency
        return None
    return convert_trajectory(dict(trajectory), trace_seed=seed, service_name=SERVICE_NAME)


def build_trial_trace(
    trial: TrialRecord,
    outcome: Outcome,
    *,
    trajectory: Mapping[str, Any] | None,
    instruction: str | None,
    resource_attributes: Mapping[str, Any],
    redactor: Redactor,
) -> ExportTraceServiceRequest:
    """Build the OTLP export request for one trial, redacted."""

    seed = ids.trace_seed(trial.trial_uuid)
    trace_id = bytes.fromhex(ids.trace_id_hex(trial.trial_uuid))
    result = trial.result
    root_id = _span_id(seed, "trial")
    start, end = _nanos(result.get("started_at")), _nanos(result.get("finished_at"))
    end = max(end, start)

    status = Status(code=Status.STATUS_CODE_OK)
    if outcome.error_class is not None:
        status = Status(
            code=Status.STATUS_CODE_ERROR,
            message=f"{outcome.exception_type}: {outcome.exception_message}",
        )

    output_text = final_agent_message(trajectory)
    root_attributes: dict[str, Any] = {
        "openinference.span.kind": "AGENT",
        "session.id": trial.trial_name,
        "input.value": instruction,
        "output.value": output_text,
        "agent.name": trial.agent,
        "llm.model_name": trial.model,
        "harbor.job.id": trial.job_id,
        "harbor.trial.name": trial.trial_name,
        "harbor.trial.id": result.get("id"),
        "harbor.task.name": trial.task_name,
        "harbor.task.digest": trial.task_digest,
        "harbor.task.source": trial.task_source,
        "harbor.agent.version": (result.get("agent_info") or {}).get("version"),
        "harbor.reward": outcome.reward if outcome.reward_valid else None,
        "harbor.reward_valid": outcome.reward_valid,
        "harbor.rewards": outcome.rewards or None,
        "harbor.error_class": outcome.error_class,
        "harbor.failure_cause": outcome.failure_cause,
        **TEST_TRAFFIC_ATTRIBUTES,
    }
    spans = [
        Span(
            trace_id=trace_id,
            span_id=root_id,
            name="harbor.trial",
            kind=Span.SPAN_KIND_INTERNAL,
            start_time_unix_nano=start,
            end_time_unix_nano=end,
            status=status,
            attributes=_attrs(root_attributes),
        )
    ]

    agent_parent = root_id
    for field_name, span_name in _PHASES:
        timing = result.get(field_name) or {}
        if not timing.get("started_at"):
            continue
        phase_id = _span_id(seed, field_name)
        phase_start = _nanos(timing.get("started_at"))
        spans.append(
            Span(
                trace_id=trace_id,
                span_id=phase_id,
                parent_span_id=root_id,
                name=span_name,
                kind=Span.SPAN_KIND_INTERNAL,
                start_time_unix_nano=phase_start,
                end_time_unix_nano=max(_nanos(timing.get("finished_at")), phase_start),
                status=Status(code=Status.STATUS_CODE_OK),
                attributes=_attrs({"openinference.span.kind": "CHAIN", "harbor.phase": field_name}),
            )
        )
        if field_name == "agent_execution":
            agent_parent = phase_id

    if trajectory:
        converted = _convert_trajectory(trajectory, seed)
        if converted is not None:
            for scope in converted.scope_spans:
                for span in scope.spans:
                    copy = Span()
                    copy.CopyFrom(span)
                    copy.trace_id = trace_id
                    if not copy.parent_span_id:
                        copy.parent_span_id = agent_parent
                    spans.append(copy)

    for span in spans:
        _redact_span(span, redactor)

    resource = Resource(
        attributes=_attrs(
            {
                "service.name": SERVICE_NAME,
                "telemetry.sdk.name": SCOPE_NAME,
                "telemetry.sdk.language": "python",
                "telemetry.sdk.version": __version__,
                **TEST_TRAFFIC_ATTRIBUTES,
                **resource_attributes,
            }
        )
    )
    return ExportTraceServiceRequest(
        resource_spans=[
            ResourceSpans(
                resource=resource,
                scope_spans=[
                    ScopeSpans(
                        scope=InstrumentationScope(name=SCOPE_NAME, version=__version__),
                        spans=spans,
                    )
                ],
            )
        ]
    )


def _redact_span(span: Span, redactor: Redactor) -> None:
    for attribute in span.attributes:
        if attribute.value.HasField("string_value"):
            attribute.value.string_value = redactor.text(attribute.value.string_value)
    if span.status.message:
        span.status.message = redactor.text(span.status.message)
    span.name = redactor.text(span.name)
