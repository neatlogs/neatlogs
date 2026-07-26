"""
Estimate USD cost from neatlogs span JSONL logs.

When ``NEATLOGS_LOG_SPANS=true`` or ``NEATLOGS_LOG_RAW_SPANS=true`` is set, every
LLM span carries a model name and prompt/completion token counts. This module
reads one or more log files, looks up the per-token price for each model, and
prints a per-model breakdown plus a grand total.

Pricing is read from ``neatlogs/config/pricing.json`` by default. Users with
custom rates (enterprise agreements, committed-use discounts, region-specific
billing) can override with ``--pricing-file PATH`` or
``NEATLOGS_PRICING_FILE=PATH``.

Programmatic::

    from neatlogs.cost import compute, format_report
    report = compute("spans_optimized.log")
    print(format_report(report, style="text"))

CLI::

    neatlogs-cost spans_optimized.log
    neatlogs-cost --format json --pricing-file my-rates.json spans.log
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, TextIO, Union

PathLike = Union[str, os.PathLike]


# ---------------------------------------------------------------------------
# Span reading (kept local so this module is self-contained; the same
# brace-balanced format is also read by neatlogs.replay when both land).
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SpanRecord:
    """Minimal view of one span: just the attributes dict. We don't need
    trace/parent structure for cost aggregation."""

    attributes: Dict[str, Any]
    source: str  # "processed" | "raw" | "unknown"


def _detect_source(obj: Dict[str, Any]) -> str:
    ctx = obj.get("context")
    if isinstance(ctx, dict) and "trace_id" in ctx:
        return "raw"
    if (
        "trace_id" in obj
        and "span_id" in obj
        and isinstance(obj.get("parent_span_id"), (str, type(None)))
    ):
        return "processed"
    return "unknown"


def _iter_json_objects(text: str) -> Iterator[Dict[str, Any]]:
    """Yield top-level JSON objects from a brace-balanced span-log file."""
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                chunk = text[start : i + 1]
                try:
                    yield json.loads(chunk)
                except json.JSONDecodeError:
                    pass
                start = -1
            elif depth < 0:
                depth = 0


def _read_spans(path: PathLike) -> List[SpanRecord]:
    """Read one path and return a list of SpanRecords (or [] if missing/empty)."""
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []
    out: List[SpanRecord] = []
    for obj in _iter_json_objects(text):
        if not isinstance(obj, dict):
            continue
        source = _detect_source(obj)
        attrs = obj.get("attributes")
        if not isinstance(attrs, dict):
            attrs = {}
        out.append(SpanRecord(attributes=attrs, source=source))
    return out


@dataclasses.dataclass
class ParseResult:
    spans: List[SpanRecord]
    files_read: int


def _read_paths(paths: Sequence[PathLike]) -> ParseResult:
    all_spans: List[SpanRecord] = []
    files_read = 0
    for raw in paths:
        spans = _read_spans(raw)
        if Path(raw).exists():
            files_read += 1
        all_spans.extend(spans)
    return ParseResult(spans=all_spans, files_read=files_read)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

DEFAULT_PRICING_PATH = Path(__file__).parent / "config" / "pricing.json"


@dataclasses.dataclass
class ModelPrice:
    """USD cost per 1M tokens for one model. ``input`` and ``output`` are split."""

    model: str
    provider: str
    input_per_1m: float
    output_per_1m: float


def _load_pricing(path: Optional[PathLike] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load a pricing table. Returns ``{provider: {model: {"input": x, "output": y}}}``.

    ``None`` falls back to the bundled default. The shape is intentionally
    flat-per-model; provider keys exist for human readability and to disambiguate
    when the same model name appears under multiple providers.
    """
    src = Path(path) if path is not None else DEFAULT_PRICING_PATH
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _lookup_price(
    pricing: Dict[str, Dict[str, Dict[str, float]]],
    provider: Optional[str],
    model: str,
) -> Optional[ModelPrice]:
    """Resolve a (provider, model) pair to a price. Falls back to model-only search."""
    if provider and provider in pricing and model in pricing[provider]:
        p = pricing[provider][model]
        return ModelPrice(
            model=model, provider=provider, input_per_1m=p["input"], output_per_1m=p["output"]
        )
    for prov, models in pricing.items():
        if model in models:
            p = models[model]
            return ModelPrice(
                model=model, provider=prov, input_per_1m=p["input"], output_per_1m=p["output"]
            )
    return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ModelCost:
    """Aggregated cost for a single model across all calls."""

    model: str
    provider: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    usd: float
    price: Optional[ModelPrice] = None

    @property
    def is_unknown(self) -> bool:
        return self.price is None


@dataclasses.dataclass
class CostReport:
    """Cost summary across one or more span log files."""

    per_model: List[ModelCost]
    unknown_models: List[str]
    files_read: int
    spans_with_tokens: int
    spans_missing_tokens: int

    @property
    def total_usd(self) -> float:
        return sum(m.usd for m in self.per_model)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(m.prompt_tokens for m in self.per_model)

    @property
    def total_completion_tokens(self) -> int:
        return sum(m.completion_tokens for m in self.per_model)

    @property
    def total_calls(self) -> int:
        return sum(m.calls for m in self.per_model)


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


def _span_provider(span: SpanRecord) -> Optional[str]:
    p = span.attributes.get("neatlogs.llm.provider")
    if isinstance(p, str) and p:
        return p.lower()
    s = span.attributes.get("neatlogs.llm.system")
    if isinstance(s, str) and s:
        return s.lower()
    return None


def _span_model(span: SpanRecord) -> Optional[str]:
    m = span.attributes.get("neatlogs.llm.model_name")
    return m if isinstance(m, str) and m else None


def _span_tokens(span: SpanRecord) -> tuple[Optional[int], Optional[int]]:
    """Return (prompt_tokens, completion_tokens) from a span, or (None, None)."""
    p = span.attributes.get("neatlogs.llm.token_count.prompt")
    c = span.attributes.get("neatlogs.llm.token_count.completion")
    p_int = p if isinstance(p, int) and p >= 0 else None
    c_int = c if isinstance(c, int) and c >= 0 else None
    return p_int, c_int


def compute(
    paths: Union[PathLike, Sequence[PathLike]],
    *,
    pricing: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    warn_stream: Optional[TextIO] = None,
) -> CostReport:
    """Read span log files and return a CostReport.

    Args:
        paths: One path, a list of paths. Missing files are skipped silently.
        pricing: Override the bundled pricing. ``None`` loads the default.
        warn_stream: Where to write warnings about unknown models. Defaults to
            stderr. Pass ``io.StringIO()`` to silence.
    """
    if isinstance(paths, (str, os.PathLike)):
        seq: List[PathLike] = [paths]
    else:
        seq = list(paths)
    parsed = _read_paths(seq)
    if pricing is None:
        pricing = _load_pricing()
    sink = warn_stream if warn_stream is not None else sys.stderr

    by_key: Dict[tuple[str, str], ModelCost] = {}
    unknown: set[str] = set()
    with_tokens = 0
    missing = 0

    for span in parsed.spans:
        model = _span_model(span)
        if not model:
            continue
        prompt, completion = _span_tokens(span)
        if prompt is None and completion is None:
            missing += 1
            continue
        prompt = prompt or 0
        completion = completion or 0
        with_tokens += 1

        provider = _span_provider(span) or ""
        price = _lookup_price(pricing, provider or None, model)
        if price is None:
            unknown.add(model)
            effective_provider = provider or "unknown"
            key = (effective_provider, model)
            if key not in by_key:
                by_key[key] = ModelCost(
                    model=model,
                    provider=effective_provider,
                    calls=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    usd=0.0,
                    price=None,
                )
            entry = by_key[key]
            entry.calls += 1
            entry.prompt_tokens += prompt
            entry.completion_tokens += completion
            continue

        key = (price.provider, model)
        if key not in by_key:
            by_key[key] = ModelCost(
                model=model,
                provider=price.provider,
                calls=0,
                prompt_tokens=0,
                completion_tokens=0,
                usd=0.0,
                price=price,
            )
        entry = by_key[key]
        entry.calls += 1
        entry.prompt_tokens += prompt
        entry.completion_tokens += completion
        # Cost is per 1M tokens; the price is already in those units.
        entry.usd += (prompt / 1_000_000) * price.input_per_1m
        entry.usd += (completion / 1_000_000) * price.output_per_1m

    per_model = sorted(
        by_key.values(),
        key=lambda m: (m.is_unknown, -(m.usd), m.model),
    )

    if unknown:
        for m in sorted(unknown):
            sink.write(
                f"[neatlogs-cost] warning: no pricing for model {m!r}; "
                f"spans using it contribute $0.00 to the total. "
                f"Override via --pricing-file or update "
                f"neatlogs/config/pricing.json.\n"
            )

    return CostReport(
        per_model=per_model,
        unknown_models=sorted(unknown),
        files_read=parsed.files_read,
        spans_with_tokens=with_tokens,
        spans_missing_tokens=missing,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_report(
    report: CostReport,
    *,
    style: str = "text",
    stream: Optional[TextIO] = None,
) -> str:
    """Render a CostReport as a string.

    style:
        - "text": human-readable table with a totals line.
        - "json": one JSON object with per-model breakdown and totals.
        - "csv":  CSV with columns model, provider, calls, prompt_tokens,
                  completion_tokens, usd, unknown.
    """
    if style not in ("text", "json", "csv"):
        raise ValueError(f"unknown style: {style!r}")
    if style == "json":
        return _format_json(report)
    if style == "csv":
        return _format_csv(report)
    return _format_text(report, use_color=_supports_color(stream or sys.stdout))


def _format_json(report: CostReport) -> str:
    obj = {
        "currency": "USD",
        "total_usd": round(report.total_usd, 6),
        "total_calls": report.total_calls,
        "total_prompt_tokens": report.total_prompt_tokens,
        "total_completion_tokens": report.total_completion_tokens,
        "files_read": report.files_read,
        "spans_with_tokens": report.spans_with_tokens,
        "spans_missing_tokens": report.spans_missing_tokens,
        "unknown_models": report.unknown_models,
        "per_model": [
            {
                "model": m.model,
                "provider": m.provider,
                "calls": m.calls,
                "prompt_tokens": m.prompt_tokens,
                "completion_tokens": m.completion_tokens,
                "usd": round(m.usd, 6),
                "unknown": m.is_unknown,
            }
            for m in report.per_model
        ],
    }
    return json.dumps(obj, indent=2) + "\n"


def _format_csv(report: CostReport) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(
        [
            "model",
            "provider",
            "calls",
            "prompt_tokens",
            "completion_tokens",
            "usd",
            "unknown",
        ]
    )
    for m in report.per_model:
        w.writerow(
            [
                m.model,
                m.provider,
                m.calls,
                m.prompt_tokens,
                m.completion_tokens,
                f"{m.usd:.6f}",
                "true" if m.is_unknown else "false",
            ]
        )
    return buf.getvalue()


def _format_text(report: CostReport, *, use_color: bool) -> str:
    buf = io.StringIO()
    if not report.per_model:
        buf.write("(no LLM spans with token counts found)\n")
        return buf.getvalue()
    header = (
        f"{'Model':<32} {'Provider':<14} {'Calls':>6} " f"{'Prompt':>10} {'Compl.':>10} {'USD':>12}"
    )
    buf.write(header + "\n")
    buf.write("-" * len(header) + "\n")
    for m in report.per_model:
        line = (
            f"{m.model[:32]:<32} {m.provider[:14]:<14} {m.calls:>6} "
            f"{m.prompt_tokens:>10} {m.completion_tokens:>10} "
            f"{'unknown' if m.is_unknown else f'${m.usd:.6f}':>12}"
        )
        if use_color and not m.is_unknown:
            line = _ansi("32", line)
        elif use_color and m.is_unknown:
            line = _ansi("33", line)
        buf.write(line + "\n")
    buf.write("-" * len(header) + "\n")
    has_unknown = bool(report.unknown_models)
    total_disp = (
        "unknown" if has_unknown and report.total_usd == 0.0 else f"${report.total_usd:.6f}"
    )
    total_line = (
        f"{'TOTAL':<32} {'':<14} {report.total_calls:>6} "
        f"{report.total_prompt_tokens:>10} {report.total_completion_tokens:>10} "
        f"{total_disp:>12}"
    )
    if use_color:
        total_line = _ansi("1;33", total_line)
    buf.write(total_line + "\n")
    if report.spans_missing_tokens:
        buf.write(
            f"\n({report.spans_missing_tokens} LLM span(s) skipped: "
            f"no prompt/completion token counts.)\n"
        )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Color helpers
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
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neatlogs-cost",
        description=(
            "Estimate USD cost from neatlogs span JSONL logs. "
            "Reads the same on-disk format as neatlogs-replay and applies "
            "a pricing table (bundled by default; override with --pricing-file)."
        ),
    )
    p.add_argument("paths", nargs="+", help="One or more span log file paths.")
    p.add_argument(
        "--format",
        choices=("text", "json", "csv"),
        default="text",
        help="Output format. Default: text.",
    )
    p.add_argument(
        "--pricing-file",
        default=None,
        help="Path to a JSON pricing table. Default: the bundled table.",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output even when stdout is a TTY.",
    )
    p.add_argument(
        "--color",
        action="store_true",
        help="Force ANSI color output even when stdout is not a TTY.",
    )
    return p


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    global _USE_COLOR
    if args.no_color:
        _USE_COLOR = False
    elif args.color:
        _USE_COLOR = True
    try:
        pricing = _load_pricing(args.pricing_file)
    except FileNotFoundError as exc:
        print(f"[neatlogs-cost] error: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"[neatlogs-cost] error: invalid pricing JSON: {exc}", file=sys.stderr)
        return 2
    try:
        report = compute(args.paths, pricing=pricing)
    except FileNotFoundError as exc:
        print(f"[neatlogs-cost] error: {exc}", file=sys.stderr)
        return 2
    if not report.per_model:
        print("[neatlogs-cost] no LLM spans with token counts found", file=sys.stderr)
    out = format_report(report, style=args.format)
    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
