"""
Span-attached metric capture (no OpenLLMetry changes required).

Problem:
  OpenLLMetry records OTel metrics during an operation, but those metric datapoints
  do not include trace_id/span_id, so you cannot attach them to the right span row
  in ClickHouse deterministically.

Constraint:
  We must not patch/fork OpenLLMetry.

Solution:
  Wrap the global MeterProvider so that whenever any counter/histogram is recorded
  while a span context is active, we ALSO emit a *raw metric point* to NeatlogsExporter
  with trace_id/span_id (for downstream span-join), while still recording the metric
  normally on the underlying instrument (with the original attributes, preserving
  low-cardinality time-series semantics if you later decide to export metrics).
"""

from __future__ import annotations

import time
from typing import Any, Optional

from opentelemetry import trace as trace_api


def _current_trace_span_ids() -> tuple[Optional[str], Optional[str]]:
    span = trace_api.get_current_span()
    ctx = span.get_span_context() if span is not None else None
    if ctx is None or not ctx.is_valid:
        return None, None
    return f"{ctx.trace_id:032x}", f"{ctx.span_id:016x}"


class _MetricEmitter:
    def __init__(self, exporter: Any):
        self._exporter = exporter

    def emit(
        self,
        *,
        metric_name: str,
        metric_type: str,
        value: float,
        unit: str,
        description: str,
        attributes: Optional[dict[str, Any]],
    ) -> None:
        trace_id, span_id = _current_trace_span_ids()
        if not trace_id or not span_id:
            return

        point = {
            "trace_id": trace_id,
            "span_id": span_id,
            "metric_name": metric_name,
            "metric_type": metric_type,
            "description": description or "",
            "unit": unit or "",
            "value": value,
            "attributes": dict(attributes or {}),
            "timestamp": int(time.time() * 1_000_000_000),  # unix ns
        }
        try:
            self._exporter.export_metrics([point])
        except Exception:
            # Never break application logic due to metrics emission.
            return


class _CounterProxy:
    def __init__(self, inner: Any, emitter: _MetricEmitter, *, name: str, unit: str, description: str):
        self._inner = inner
        self._emitter = emitter
        self._name = name
        self._unit = unit
        self._desc = description

    def add(self, amount: Any, attributes: Optional[dict[str, Any]] = None, context: Any = None) -> None:
        # Emit raw point correlated to the active span (for span-join).
        try:
            self._emitter.emit(
                metric_name=self._name,
                metric_type="sum",
                value=float(amount),
                unit=self._unit,
                description=self._desc,
                attributes=attributes,
            )
        except Exception:
            pass
        # Preserve standard OTel metric recording (no trace/span injected).
        return self._inner.add(amount, attributes=attributes, context=context)


class _UpDownCounterProxy:
    def __init__(self, inner: Any, emitter: _MetricEmitter, *, name: str, unit: str, description: str):
        self._inner = inner
        self._emitter = emitter
        self._name = name
        self._unit = unit
        self._desc = description

    def add(self, amount: Any, attributes: Optional[dict[str, Any]] = None, context: Any = None) -> None:
        try:
            self._emitter.emit(
                metric_name=self._name,
                metric_type="sum",
                value=float(amount),
                unit=self._unit,
                description=self._desc,
                attributes=attributes,
            )
        except Exception:
            pass
        return self._inner.add(amount, attributes=attributes, context=context)


class _HistogramProxy:
    def __init__(self, inner: Any, emitter: _MetricEmitter, *, name: str, unit: str, description: str):
        self._inner = inner
        self._emitter = emitter
        self._name = name
        self._unit = unit
        self._desc = description

    def record(self, amount: Any, attributes: Optional[dict[str, Any]] = None, context: Any = None) -> None:
        try:
            self._emitter.emit(
                metric_name=self._name,
                metric_type="histogram",
                value=float(amount),
                unit=self._unit,
                description=self._desc,
                attributes=attributes,
            )
        except Exception:
            pass
        return self._inner.record(amount, attributes=attributes, context=context)


class SpanMetricMeterProxy:
    def __init__(self, inner_meter: Any, emitter: _MetricEmitter):
        self._inner = inner_meter
        self._emitter = emitter

    def create_counter(self, name: str, unit: str = "", description: str = "", **kwargs: Any) -> Any:
        inner = self._inner.create_counter(name=name, unit=unit, description=description, **kwargs)
        return _CounterProxy(inner, self._emitter, name=name, unit=unit, description=description)

    def create_up_down_counter(self, name: str, unit: str = "", description: str = "", **kwargs: Any) -> Any:
        inner = self._inner.create_up_down_counter(name=name, unit=unit, description=description, **kwargs)
        return _UpDownCounterProxy(inner, self._emitter, name=name, unit=unit, description=description)

    def create_histogram(self, name: str, unit: str = "", description: str = "", **kwargs: Any) -> Any:
        inner = self._inner.create_histogram(name=name, unit=unit, description=description, **kwargs)
        return _HistogramProxy(inner, self._emitter, name=name, unit=unit, description=description)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class SpanMetricMeterProviderProxy:
    def __init__(self, inner_provider: Any, exporter: Any):
        self._inner = inner_provider
        self._emitter = _MetricEmitter(exporter)

    def get_meter(self, *args: Any, **kwargs: Any) -> Any:
        meter = self._inner.get_meter(*args, **kwargs)
        return SpanMetricMeterProxy(meter, self._emitter)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

