"""Read-only, network-free SDK readiness diagnostics."""

from __future__ import annotations

import importlib
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from importlib import metadata
from typing import Any, Literal
from urllib.parse import urlparse

from opentelemetry import trace as otel_trace

from .schema_v2 import (
    TELEMETRY_CONTRACT_VERSION,
    TELEMETRY_SCHEMA_SHA256,
    verify_telemetry_schema,
)
from .version import __version__

DOCTOR_FORMAT_VERSION = "neatlogs.doctor/v1"
DoctorStatus = Literal["pass", "warn", "fail", "unknown"]


@dataclass(frozen=True)
class DoctorCheck:
    """One stable diagnostic whose bounded message never contains secrets."""

    name: str
    status: DoctorStatus
    reason_code: str
    message: str
    details: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if not self.details:
            value.pop("details")
        return value


@dataclass(frozen=True)
class DoctorResult:
    """Versioned local-readiness result shared with other Neatlogs SDKs."""

    format_version: str
    sdk_version: str
    ready: bool
    checks: tuple[DoctorCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "sdk_version": self.sdk_version,
            "ready": self.ready,
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _check(
    name: str,
    status: DoctorStatus,
    reason_code: str,
    message: str,
    details: dict[str, str] | None = None,
) -> DoctorCheck:
    return DoctorCheck(name, status, reason_code, message, details or {})


def _runtime_check() -> DoctorCheck:
    version = platform.python_version()
    supported = (3, 10) <= sys.version_info[:2] < (3, 14)
    return _check(
        "runtime",
        "pass" if supported else "fail",
        "PYTHON_RUNTIME_SUPPORTED" if supported else "PYTHON_RUNTIME_UNSUPPORTED",
        "Python runtime is supported" if supported else "Python 3.10 through 3.13 is required",
        {"version": version},
    )


def _package_check() -> DoctorCheck:
    try:
        installed = metadata.version("neatlogs")
    except metadata.PackageNotFoundError:
        return _check(
            "package",
            "warn",
            "PACKAGE_METADATA_UNAVAILABLE",
            "Installed Neatlogs package metadata is unavailable",
        )
    if installed != __version__:
        return _check(
            "package",
            "fail",
            "PACKAGE_VERSION_MISMATCH",
            "Imported and installed Neatlogs versions do not match",
            {"imported_version": __version__, "installed_version": installed},
        )
    return _check(
        "package",
        "pass",
        "PACKAGE_METADATA_PRESENT",
        "Neatlogs package metadata is present",
        {"version": installed},
    )


def _schema_check() -> DoctorCheck:
    try:
        verify_telemetry_schema()
    except Exception:
        return _check(
            "schema",
            "fail",
            "SCHEMA_V2_INVALID",
            "Packaged telemetry schema or manifest failed validation",
        )
    return _check(
        "schema",
        "pass",
        "SCHEMA_V2_HASH_VALID",
        "Packaged telemetry schema v2 hash is valid",
        {
            "contract_version": TELEMETRY_CONTRACT_VERSION,
            "schema_sha256": TELEMETRY_SCHEMA_SHA256,
        },
    )


def _transport_check() -> DoctorCheck:
    return _check(
        "transport",
        "pass",
        "TRANSPORT_OTLP_HTTP_PROTOBUF",
        "SDK transport is OTLP HTTP/protobuf",
        {"path": "/v1/traces"},
    )


def _endpoint_check(endpoint: str | None) -> DoctorCheck:
    raw = (endpoint if endpoint is not None else os.getenv("NEATLOGS_ENDPOINT", "")).strip()
    raw = raw or "https://ingest.neatlogs.com"
    try:
        parsed = urlparse(raw)
        valid_origin = (
            parsed.scheme in {"http", "https"}
            and bool(parsed.netloc)
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
        )
    except (TypeError, ValueError):
        valid_origin = False
        parsed = None
    if not valid_origin or parsed is None:
        return _check(
            "endpoint",
            "fail",
            "ENDPOINT_INVALID",
            "Endpoint must be HTTP(S) without credentials, query, or fragment",
        )
    if parsed.path.rstrip("/") not in {"", "/v1/traces"}:
        return _check(
            "endpoint",
            "fail",
            "ENDPOINT_PATH_UNSUPPORTED",
            "Endpoint must be an origin or end in /v1/traces",
            {"scheme": parsed.scheme, "host": parsed.netloc},
        )
    return _check(
        "endpoint",
        "pass",
        "ENDPOINT_VALID",
        "Endpoint is valid",
        {"scheme": parsed.scheme, "host": parsed.netloc},
    )


def _sampler_check(sample_rate: float) -> DoctorCheck:
    valid = (
        not isinstance(sample_rate, bool)
        and isinstance(sample_rate, (int, float))
        and math.isfinite(float(sample_rate))
        and 0.0 <= float(sample_rate) <= 1.0
    )
    if not valid:
        return _check(
            "sampler",
            "fail",
            "SAMPLER_INVALID",
            "Sample rate must be finite and between 0 and 1",
        )
    return _check(
        "sampler",
        "pass",
        "SAMPLER_PARENT_BASED_VALID",
        "ParentBased trace sampling is valid",
        {"root_sample_rate": format(float(sample_rate), "g")},
    )


def _selected_runtime(client: Any | None) -> tuple[Any | None, Any | None, list[Any]]:
    from ._wrap_utils import get_active_client

    active = client or get_active_client()
    if active is not None:
        return (
            active.tracer_provider,
            active._span_processor,
            list(active._exporters + active._log_exporters),
        )
    state = importlib.import_module("neatlogs.init")
    if not state._initialized:
        return None, None, []
    return state._tracer_provider, state._span_processor, list(state._export_health)


def _ownership_check(provider: Any | None) -> DoctorCheck:
    if provider is None:
        return _check(
            "ownership",
            "pass",
            "OTEL_PROVIDER_PRIVATE",
            "Neatlogs creates or requires a private provider and leaves global OpenTelemetry state untouched",
        )
    if provider is otel_trace.get_tracer_provider():
        return _check(
            "ownership",
            "fail",
            "OTEL_PROVIDER_NOT_PRIVATE",
            "Selected Neatlogs provider unexpectedly matches the process-global provider",
        )
    return _check(
        "ownership",
        "pass",
        "OTEL_PROVIDER_PRIVATE",
        "Selected Neatlogs provider is private",
    )


def _queue_check(
    disable_export: bool | None, exporters: list[Any], provider: Any | None
) -> DoctorCheck:
    disabled = (
        disable_export if disable_export is not None else provider is not None and not exporters
    )
    if disabled:
        return _check(
            "queue",
            "warn",
            "EXPORT_QUEUE_DISABLED",
            "Export is disabled, so no batch queue is active",
        )
    return _check(
        "queue",
        "pass",
        "EXPORT_QUEUE_BATCHED",
        "Export uses the OpenTelemetry batch span processor",
    )


def _export_health_check(provider: Any | None, exporters: list[Any]) -> DoctorCheck:
    if provider is None:
        return _check(
            "export_health",
            "unknown",
            "EXPORT_HEALTH_UNOBSERVABLE",
            "No running Neatlogs runtime is selected",
        )
    failures = sum(int(getattr(item.health, "failures", 0)) for item in exporters)
    drops = sum(int(getattr(item.health, "drops", 0)) for item in exporters)
    details = {"dropped_spans": str(drops), "export_failures": str(failures)}
    if failures or drops:
        return _check(
            "export_health",
            "fail",
            "EXPORT_HEALTH_UNHEALTHY",
            "The selected runtime has masking drops or exporter failures",
            details,
        )
    return _check(
        "export_health",
        "pass",
        "EXPORT_HEALTHY",
        "The selected runtime has no observed export failures or drops",
        details,
    )


def _root_check(provider: Any | None, processor: Any | None) -> DoctorCheck:
    if provider is None or processor is None:
        return _check(
            "root", "unknown", "ROOT_UNOBSERVABLE", "No running Neatlogs runtime is selected"
        )
    from ._wrap_utils import _current_neatlogs_parent

    current = _current_neatlogs_parent() or otel_trace.get_current_span()
    context = current.get_span_context()
    if not context.is_valid:
        return _check(
            "root",
            "unknown",
            "ROOT_NOT_ACTIVE",
            "Context does not carry an active root owned by the selected runtime",
        )
    with processor._active_spans_lock:
        active = dict(processor._active_spans)
    span = active.get(context.span_id)
    if span is None:
        return _check(
            "root",
            "unknown",
            "ROOT_NOT_ACTIVE",
            "Context does not carry an active root owned by the selected runtime",
        )
    seen: set[int] = set()
    while span.parent is not None and span.parent.span_id in active:
        if span.context.span_id in seen:
            return _check(
                "root",
                "fail",
                "ROOT_OWNERSHIP_INVALID",
                "Owned context does not resolve to one active root",
            )
        seen.add(span.context.span_id)
        span = active[span.parent.span_id]
    root = span.get_span_context()
    if not root.is_valid:
        return _check(
            "root",
            "fail",
            "ROOT_OWNERSHIP_INVALID",
            "Owned context does not resolve to one valid root",
        )
    return _check(
        "root",
        "pass",
        "ROOT_IDS_VALID",
        "Active owned root has valid trace and span IDs",
        {"trace_id": f"{root.trace_id:032x}", "span_id": f"{root.span_id:016x}"},
    )


def doctor(
    *,
    endpoint: str | None = None,
    sample_rate: float = 1.0,
    disable_export: bool | None = None,
    client: Any | None = None,
) -> DoctorResult:
    """Inspect local configuration and observable runtime health without mutation.

    This function never initializes, flushes, shuts down, exports, or performs a
    network request. Credential values are neither accepted nor returned.
    """

    provider, processor, exporters = _selected_runtime(client)
    checks = (
        _runtime_check(),
        _package_check(),
        _schema_check(),
        _transport_check(),
        _endpoint_check(endpoint),
        _sampler_check(sample_rate),
        _ownership_check(provider),
        _queue_check(disable_export, exporters, provider),
        _export_health_check(provider, exporters),
        _root_check(provider, processor),
    )
    return DoctorResult(
        format_version=DOCTOR_FORMAT_VERSION,
        sdk_version=__version__,
        ready=not any(check.status == "fail" for check in checks),
        checks=checks,
    )
