"""Output formatters for evaluation, breakdown, and forecast reports.

Each report type has three formatters (text / json / csv) and a
top-level dispatcher that auto-detects TTY for color and validates the
style argument. Text formatters take an explicit ``use_color`` flag;
the dispatcher defaults it to ``True`` when stdout is a TTY.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from typing import Any, Dict, Iterable, List, Optional, TextIO

from .breakdown import BreakdownReport, ModelCostBreakdown
from .forecast import ForecastReport
from .pricing import ModelDefinition
from .ranking import EvaluationReport, ScoredModel

# ---------------------------------------------------------------------------
# Color
# ---------------------------------------------------------------------------


_USE_COLOR = True


def _supports_color(stream: TextIO) -> bool:
    if not _USE_COLOR:
        return False
    if not hasattr(stream, "isatty"):
        return False
    return bool(stream.isatty())


def _ansi(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"


# ---------------------------------------------------------------------------
# Evaluation (ranking)
# ---------------------------------------------------------------------------


def format_evaluation_text(report: EvaluationReport, *, use_color: bool = True) -> str:
    buf = io.StringIO()
    if not report.alternatives and report.baseline is None:
        buf.write("(no candidate models in the pricing catalog)\n")
        return buf.getvalue()
    p = report.profile
    buf.write(
        f"Workload: {p.spans_with_tokens} spans across {len(p.models_used)} model(s); "
        f"prompt p50={p.prompt_stats.p50:,} p99={p.prompt_stats.p99:,} max={p.prompt_stats.max:,} "
        f"completion p50={p.completion_stats.p50:,} max={p.completion_stats.max:,}\n"
    )
    buf.write(f"  models: {sorted(p.models_used)}\n")
    if p.capabilities_inferred:
        buf.write(f"  capabilities inferred: {sorted(p.capabilities_inferred)}\n")
    constraints = report.constraints.explains()
    if constraints:
        buf.write(f"  constraints: {'; '.join(constraints)}\n")
    buf.write("\n")

    header = f"{'Model':<34} {'Prov':<12} {'Compat':>7} " f"{'Cost':>10} {'vs base':>10}"
    buf.write(header + "\n")
    buf.write("-" * len(header) + "\n")
    for sm in report.ranked():
        cost_disp = f"${sm.total_cost:.4f}" if sm.meets_min_compatibility else "incompatible"
        if sm.is_baseline:
            delta_disp = "(baseline)"
        else:
            pct = report.delta_pct_for(sm)
            delta_disp = f"{pct:+.0f}%" if pct is not None else "n/a"
        compat_disp = f"{sm.compatibility_pct * 100:.0f}%"
        line = (
            f"{sm.model_key[:34]:<34} {sm.provider[:12]:<12} "
            f"{compat_disp:>7} {cost_disp:>10} {delta_disp:>10}"
        )
        if use_color:
            if sm.is_baseline:
                line = _ansi("1;33", line)
            elif not sm.meets_min_compatibility:
                line = _ansi("31", line)
            elif report.delta_pct_for(sm) is not None and report.delta_pct_for(sm) < 0:
                line = _ansi("32", line)
        buf.write(line + "\n")
    buf.write("-" * len(header) + "\n")

    if report.baseline is not None:
        any_gap = any(sm.missing_capabilities for sm in report.alternatives)
        if any_gap:
            buf.write("\nCapability gap vs inferred workload needs:\n")
            for sm in report.alternatives:
                if sm.missing_capabilities:
                    buf.write(f"  {sm.model_key}: missing {sorted(sm.missing_capabilities)}\n")

    if p.spans_skipped:
        buf.write(f"\n({p.spans_skipped} span(s) skipped: no model or no token counts.)\n")
    return buf.getvalue()


def format_evaluation_json(report: EvaluationReport) -> str:
    def row(sm: ScoredModel) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "model": sm.model_key,
            "provider": sm.provider,
            "total_cost": round(sm.total_cost, 6),
            "compatibility_pct": round(sm.compatibility_pct * 100, 2),
            "meets_min_compatibility": sm.meets_min_compatibility,
            "missing_capabilities": sorted(sm.missing_capabilities),
            "context_window": sm.context_window,
            "is_baseline": sm.is_baseline,
        }
        if not sm.is_baseline and report.baseline is not None:
            pct = report.delta_pct_for(sm)
            if pct is not None:
                d["delta_pct_vs_baseline"] = round(pct, 2)
        return d

    p = report.profile
    return (
        json.dumps(
            {
                "currency": "USD",
                "workload": {
                    "total_spans": p.total_spans,
                    "spans_with_tokens": p.spans_with_tokens,
                    "spans_skipped": p.spans_skipped,
                    "models_used": sorted(p.models_used),
                    "capabilities_inferred": sorted(p.capabilities_inferred),
                    "prompt_stats": {
                        "p50": p.prompt_stats.p50,
                        "p90": p.prompt_stats.p90,
                        "p99": p.prompt_stats.p99,
                        "max": p.prompt_stats.max,
                        "total": p.prompt_stats.total,
                    },
                    "completion_stats": {
                        "p50": p.completion_stats.p50,
                        "p90": p.completion_stats.p90,
                        "p99": p.completion_stats.p99,
                        "max": p.completion_stats.max,
                        "total": p.completion_stats.total,
                    },
                },
                "constraints": {
                    "need_capabilities": sorted(report.constraints.need_capabilities),
                    "min_context_window": report.constraints.min_context_window,
                    "min_compatibility_pct": report.constraints.min_compatibility_pct,
                },
                "explicit_constraints": {
                    "need_capabilities": sorted(report.explicit_constraints.need_capabilities),
                    "min_context_window": report.explicit_constraints.min_context_window,
                    "min_compatibility_pct": report.explicit_constraints.min_compatibility_pct,
                },
                "baseline": row(report.baseline) if report.baseline else None,
                "alternatives": [row(sm) for sm in report.alternatives],
                "ranked": [row(sm) for sm in report.ranked()],
            },
            indent=2,
        )
        + "\n"
    )


def format_evaluation_csv(report: EvaluationReport) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(
        [
            "model",
            "provider",
            "is_baseline",
            "total_cost",
            "compatibility_pct",
            "meets_min_compatibility",
            "context_window",
            "missing_capabilities",
            "delta_pct_vs_baseline",
        ]
    )
    for sm in report.ranked():
        pct = report.delta_pct_for(sm)
        w.writerow(
            [
                sm.model_key,
                sm.provider,
                "true" if sm.is_baseline else "false",
                f"{sm.total_cost:.6f}",
                f"{sm.compatibility_pct * 100:.2f}",
                "true" if sm.meets_min_compatibility else "false",
                sm.context_window if sm.context_window is not None else "",
                ";".join(sorted(sm.missing_capabilities)),
                "" if pct is None else f"{pct:.2f}",
            ]
        )
    return buf.getvalue()


def format_evaluation(
    report: EvaluationReport,
    *,
    style: str = "text",
    stream: Optional[TextIO] = None,
) -> str:
    if style not in ("text", "json", "csv"):
        raise ValueError(f"unknown style: {style!r}")
    if style == "json":
        return format_evaluation_json(report)
    if style == "csv":
        return format_evaluation_csv(report)
    use_color = _supports_color(stream or sys.stdout)
    return format_evaluation_text(report, use_color=use_color)


# ---------------------------------------------------------------------------
# Breakdown
# ---------------------------------------------------------------------------


def format_breakdown_text(report: BreakdownReport, *, use_color: bool = True) -> str:
    buf = io.StringIO()
    if not report.models:
        buf.write("(no candidate models in the pricing catalog)\n")
        return buf.getvalue()
    p = report.profile
    buf.write(
        f"Workload: {p.spans_with_tokens} spans across {len(p.models_used)} model(s); "
        f"prompt p50={p.prompt_stats.p50:,} p99={p.prompt_stats.p99:,} max={p.prompt_stats.max:,} "
        f"completion p50={p.completion_stats.p50:,} max={p.completion_stats.max:,}\n\n"
    )

    header = (
        f"{'Model':<34} {'Input':>10} {'Output':>10} "
        f"{'CacheR':>10} {'CacheW':>10} {'Reason':>10} {'Total':>10}"
    )
    buf.write(header + "\n")
    buf.write("-" * len(header) + "\n")
    ranked = report.ranked()
    for m in ranked:
        line = (
            f"{m.model_key[:34]:<34} "
            f"${m.input_cost:>9.4f} ${m.output_cost:>9.4f} "
            f"${m.cache_read_cost:>9.4f} ${m.cache_write_cost:>9.4f} "
            f"${m.reasoning_cost:>9.4f} ${m.total_cost:>9.4f}"
        )
        if use_color and m == ranked[0] and ranked[0].total_cost > 0:
            line = _ansi("32", line)
        buf.write(line + "\n")
    buf.write("-" * len(header) + "\n")

    if p.spans_skipped:
        buf.write(f"\n({p.spans_skipped} span(s) skipped: no model or no token counts.)\n")
    return buf.getvalue()


def format_breakdown_json(report: BreakdownReport) -> str:
    def row(m: ModelCostBreakdown) -> Dict[str, Any]:
        return {
            "model": m.model_key,
            "provider": m.provider,
            "total_cost": round(m.total_cost, 6),
            "input_cost": round(m.input_cost, 6),
            "output_cost": round(m.output_cost, 6),
            "cache_read_cost": round(m.cache_read_cost, 6),
            "cache_write_cost": round(m.cache_write_cost, 6),
            "reasoning_cost": round(m.reasoning_cost, 6),
            "image_cost": round(m.image_cost, 6),
            "audio_cost": round(m.audio_cost, 6),
            "spans_with_tokens": m.spans_with_tokens,
            "spans_total": m.spans_total,
            "spans_skipped": m.spans_skipped,
        }

    p = report.profile
    return (
        json.dumps(
            {
                "currency": "USD",
                "workload": {
                    "total_spans": p.total_spans,
                    "spans_with_tokens": p.spans_with_tokens,
                    "spans_skipped": p.spans_skipped,
                    "models_used": sorted(p.models_used),
                    "prompt_stats": {
                        "p50": p.prompt_stats.p50,
                        "p90": p.prompt_stats.p90,
                        "p99": p.prompt_stats.p99,
                        "max": p.prompt_stats.max,
                        "total": p.prompt_stats.total,
                    },
                    "completion_stats": {
                        "p50": p.completion_stats.p50,
                        "p90": p.completion_stats.p90,
                        "p99": p.completion_stats.p99,
                        "max": p.completion_stats.max,
                        "total": p.completion_stats.total,
                    },
                },
                "models": [row(m) for m in report.ranked()],
            },
            indent=2,
        )
        + "\n"
    )


def format_breakdown_csv(report: BreakdownReport) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(
        [
            "model",
            "provider",
            "input_cost",
            "output_cost",
            "cache_read_cost",
            "cache_write_cost",
            "reasoning_cost",
            "image_cost",
            "audio_cost",
            "total_cost",
            "spans_with_tokens",
            "spans_total",
            "spans_skipped",
        ]
    )
    for m in report.ranked():
        w.writerow(
            [
                m.model_key,
                m.provider,
                f"{m.input_cost:.6f}",
                f"{m.output_cost:.6f}",
                f"{m.cache_read_cost:.6f}",
                f"{m.cache_write_cost:.6f}",
                f"{m.reasoning_cost:.6f}",
                f"{m.image_cost:.6f}",
                f"{m.audio_cost:.6f}",
                f"{m.total_cost:.6f}",
                m.spans_with_tokens,
                m.spans_total,
                m.spans_skipped,
            ]
        )
    return buf.getvalue()


def format_breakdown(
    report: BreakdownReport,
    *,
    style: str = "text",
    stream: Optional[TextIO] = None,
) -> str:
    if style not in ("text", "json", "csv"):
        raise ValueError(f"unknown style: {style!r}")
    if style == "json":
        return format_breakdown_json(report)
    if style == "csv":
        return format_breakdown_csv(report)
    use_color = _supports_color(stream or sys.stdout)
    return format_breakdown_text(report, use_color=use_color)


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------


def format_forecast_text(report: ForecastReport, *, use_color: bool = True) -> str:
    buf = io.StringIO()
    buf.write(f"Forecast for {report.model_key} at " f"{report.monthly_calls:,} call(s)/month\n")
    buf.write(
        f"  avg prompt: {report.avg_prompt_tokens:,} tokens, "
        f"avg completion: {report.avg_completion_tokens:,} tokens\n"
    )
    if report.cache_hit_rate:
        buf.write(f"  cache hit rate: {report.cache_hit_rate * 100:.0f}%\n")
    if report.reasoning_per_call:
        buf.write(f"  reasoning per call: {report.reasoning_per_call:,} tokens\n")
    buf.write("\n")
    header = f"{'':24} {'per call':>12} {'monthly':>14} {'annual':>14}"
    buf.write(header + "\n")
    buf.write("-" * len(header) + "\n")
    rows = [
        ("Input", report.input_cost),
        ("Output", report.output_cost),
        ("Cache", report.cache_cost),
        ("Reasoning", report.reasoning_cost),
        ("Total", report.per_call_cost),
    ]
    for label, per_call in rows:
        monthly = per_call * report.monthly_calls
        annual = monthly * 12
        line = f"{label:<24} ${per_call:>11.6f} ${monthly:>13.2f} ${annual:>13.2f}"
        if use_color and label == "Total":
            line = _ansi("1;33", line)
        buf.write(line + "\n")
    if report.notes:
        buf.write("\nNotes:\n")
        for n in report.notes:
            buf.write(f"  {n}\n")
    return buf.getvalue()


def format_forecast_json(report: ForecastReport) -> str:
    return (
        json.dumps(
            {
                "currency": "USD",
                "model": report.model_key,
                "monthly_calls": report.monthly_calls,
                "avg_prompt_tokens": report.avg_prompt_tokens,
                "avg_completion_tokens": report.avg_completion_tokens,
                "cache_hit_rate": report.cache_hit_rate,
                "reasoning_per_call": report.reasoning_per_call,
                "per_call_cost": round(report.per_call_cost, 6),
                "input_cost": round(report.input_cost, 6),
                "output_cost": round(report.output_cost, 6),
                "cache_cost": round(report.cache_cost, 6),
                "reasoning_cost": round(report.reasoning_cost, 6),
                "monthly_cost": round(report.monthly_cost, 2),
                "annual_cost": round(report.annual_cost, 2),
                "notes": list(report.notes),
            },
            indent=2,
        )
        + "\n"
    )


def format_forecast(
    report: ForecastReport,
    *,
    style: str = "text",
    stream: Optional[TextIO] = None,
) -> str:
    if style not in ("text", "json"):
        raise ValueError(f"unknown style: {style!r}")
    if style == "json":
        return format_forecast_json(report)
    use_color = _supports_color(stream or sys.stdout)
    return format_forecast_text(report, use_color=use_color)


# ---------------------------------------------------------------------------
# Pricing catalog (list + show)
# ---------------------------------------------------------------------------


def _sorted_catalog(models: Iterable[ModelDefinition]) -> List[ModelDefinition]:
    return sorted(models, key=lambda m: (m.provider, m.model_key))


def _pricing_model_row(m: ModelDefinition) -> Dict[str, Any]:
    return {
        "model": m.model_key,
        "provider": m.provider,
        "context_window": m.context_window,
        "capabilities": sorted(m.capabilities),
        "usage_types": {k: round(v, 6) for k, v in m.usage_types.items()},
        "tiers": {
            k: [{"above_tokens": t.above_tokens, "rate": round(t.rate, 6)} for t in v]
            for k, v in m.tiers.items()
        },
    }


def _pricing_list_row(m: ModelDefinition) -> Dict[str, Any]:
    return {
        "model": m.model_key,
        "provider": m.provider,
        "context_window": m.context_window,
        "capabilities": sorted(m.capabilities),
    }


def format_pricing_list_text(models: Iterable[ModelDefinition], *, use_color: bool = True) -> str:
    buf = io.StringIO()
    rows = _sorted_catalog(models)
    if not rows:
        buf.write("(no models in the pricing catalog)\n")
        return buf.getvalue()
    header = f"{'provider':<14} {'model':<44} {'context':>10} {'capabilities'}"
    buf.write(header + "\n")
    buf.write("-" * len(header) + "\n")
    for m in rows:
        ctx = f"{m.context_window:,}" if m.context_window is not None else "-"
        line = (
            f"{m.provider[:14]:<14} {m.model_key[:44]:<44} "
            f"{ctx:>10} {','.join(sorted(m.capabilities))}"
        )
        if use_color:
            line = _ansi("2", line)
        buf.write(line + "\n")
    buf.write("-" * len(header) + "\n")
    buf.write(f"{len(rows)} model(s) listed.\n")
    return buf.getvalue()


def format_pricing_list_json(models: Iterable[ModelDefinition]) -> str:
    rows = _sorted_catalog(models)
    return (
        json.dumps(
            {
                "currency": "USD",
                "model_count": len(rows),
                "models": [_pricing_list_row(m) for m in rows],
            },
            indent=2,
        )
        + "\n"
    )


def format_pricing_list_csv(models: Iterable[ModelDefinition]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["model", "provider", "context_window", "capabilities"])
    for m in _sorted_catalog(models):
        w.writerow(
            [
                m.model_key,
                m.provider,
                m.context_window if m.context_window is not None else "",
                ";".join(sorted(m.capabilities)),
            ]
        )
    return buf.getvalue()


def format_pricing_list(
    models: Iterable[ModelDefinition],
    *,
    style: str = "text",
    stream: Optional[TextIO] = None,
) -> str:
    if style not in ("text", "json", "csv"):
        raise ValueError(f"unknown style: {style!r}")
    if style == "json":
        return format_pricing_list_json(models)
    if style == "csv":
        return format_pricing_list_csv(models)
    use_color = _supports_color(stream or sys.stdout)
    return format_pricing_list_text(models, use_color=use_color)


def format_pricing_show_text(model: ModelDefinition, *, use_color: bool = True) -> str:
    buf = io.StringIO()
    buf.write(f"provider:        {model.provider}\n")
    ctx = f"{model.context_window:,}" if model.context_window is not None else "not declared"
    buf.write(f"context_window:  {ctx}\n")
    caps = sorted(model.capabilities)
    buf.write(f"capabilities:    {', '.join(caps) if caps else '(none)'}\n")
    buf.write("\n")
    if model.usage_types:
        buf.write("rates (USD per 1M tokens):\n")
        for usage_type in sorted(model.usage_types):
            buf.write(f"  {usage_type:<14} ${model.usage_types[usage_type]:>9.6f}\n")
    else:
        buf.write("(no usage rates declared)\n")
    if model.tiers:
        buf.write("\ntiers:\n")
        for usage_type in sorted(model.tiers):
            for tier in model.tiers[usage_type]:
                buf.write(
                    f"  {usage_type:<14} > {tier.above_tokens:>10,} → " f"${tier.rate:>9.6f}\n"
                )
    if use_color:
        pass
    return buf.getvalue()


def format_pricing_show_json(model: ModelDefinition) -> str:
    return json.dumps(_pricing_model_row(model), indent=2) + "\n"


def format_pricing_show(
    model: ModelDefinition,
    *,
    style: str = "text",
    stream: Optional[TextIO] = None,
) -> str:
    if style not in ("text", "json"):
        raise ValueError(f"unknown style: {style!r}")
    if style == "json":
        return format_pricing_show_json(model)
    use_color = _supports_color(stream or sys.stdout)
    return format_pricing_show_text(model, use_color=use_color)
