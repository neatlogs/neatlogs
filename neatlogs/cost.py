"""
LLM cost comparison engine for neatlogs span JSONL logs.

This module powers two related things:

1. **Cost comparison** — given a span log written by ``NEATLOGS_LOG_SPANS=true`` /
   ``NEATLOGS_LOG_RAW_SPANS=true`` and a list of candidate models, compute what
   the same workload would have cost on each. The point is to answer the
   "should I switch models?" question from local data, without making new API
   calls or signing up for a SaaS dashboard.

2. **Cost forecasting** — given expected call volume and a model choice,
   project monthly / annual cost. Useful for budgeting before running a batch
   evaluation.

Pricing data comes from ``neatlogs/config/pricing.json`` (a curated catalog
covering the major OpenAI / Anthropic / Google GenAI / DeepSeek / Groq
models the SDK already supports). Users with custom rates override via
``--pricing-file`` or ``NEATLOGS_PRICING_FILE``.

The bundled pricing schema is intentionally simple and explicit:

* Per-token prices are USD per 1M tokens (matches the industry convention).
* Model keys are ``provider/model_name`` to avoid namespace collisions.
* Tiered / long-context pricing is a ``tiers`` sub-dict.
* Cache pricing is split into ``cache_read_per_1m`` and ``cache_write_per_1m``
  (Anthropic, OpenAI, DeepSeek).
* Reasoning tokens have a separate ``reasoning_output_per_1m`` rate (o-series,
  DeepSeek-R1).
* Capability flags (``supports_vision``, ``supports_tools``, ``supports_reasoning``,
  ``supports_prompt_cache``) drive the comparison report's "missing capability"
  warnings.

Programmatic::

    from neatlogs.cost import compare, forecast, format_comparison
    report = compare("spans.log", models=["gpt-4o-mini", "gpt-4o", "claude-3-5-haiku-latest"])
    print(format_comparison(report, style="text"))

CLI::

    neatlogs-compare spans.log --models gpt-4o-mini,gpt-4o,claude-3-5-haiku-latest
    neatlogs-compare spans.log --models gpt-4o-mini --current gpt-4o-mini
    neatlogs-compare spans.log --forecast --monthly-calls 50000 --avg-prompt 2000 --avg-completion 500
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, TextIO, Tuple, Union

PathLike = Union[str, os.PathLike]


# ---------------------------------------------------------------------------
# Span reading
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SpanUsage:
    """One LLM span's token usage, extracted from span attributes.

    Missing fields are ``None`` rather than 0 so the cost engine can decide
    whether to skip the span or treat the field as 0.
    """

    span_id: str
    trace_id: str
    model: str
    provider: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    cache_creation_tokens: Optional[int]
    cache_read_tokens: Optional[int]
    reasoning_tokens: Optional[int]

    @property
    def has_tokens(self) -> bool:
        # A span "has tokens" iff at least one usage field is a positive
        # integer. All-zero or all-missing means the span ran but reported
        # no usage (e.g. a failed call, or a span kind that isn't billed
        # by tokens). Either way, cost for it is $0 and there's no point
        # counting it as a real LLM call.
        return any(
            t is not None and t > 0
            for t in (
                self.prompt_tokens,
                self.completion_tokens,
                self.cache_creation_tokens,
                self.cache_read_tokens,
                self.reasoning_tokens,
            )
        )

    @property
    def uses_prompt_cache(self) -> bool:
        return (self.cache_creation_tokens or 0) > 0 or (self.cache_read_tokens or 0) > 0

    @property
    def uses_reasoning(self) -> bool:
        return (self.reasoning_tokens or 0) > 0


_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


def _int_attr(attrs: Dict[str, Any], *keys: str) -> Optional[int]:
    """First key that exists and is a non-negative int wins. Used to tolerate
    the various casing / naming conventions the SDK and providers use."""
    for k in keys:
        v = attrs.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, int) and v >= 0:
            return v
        if isinstance(v, float) and v.is_integer() and v >= 0:
            return int(v)
    return None


def _extract_usage(obj: Dict[str, Any]) -> SpanUsage:
    attrs = obj.get("attributes") or {}
    if not isinstance(attrs, dict):
        attrs = {}
    # Model: prefer neatlogs.llm.model_name, fall back to gen_ai.response.model.
    model = (
        attrs.get("neatlogs.llm.model_name")
        or attrs.get("gen_ai.response.model")
        or attrs.get("gen_ai.request.model")
        or ""
    )
    provider = (
        attrs.get("neatlogs.llm.provider")
        or attrs.get("neatlogs.llm.system")
        or attrs.get("gen_ai.system")
    )
    if isinstance(provider, str):
        provider = provider.lower() or None
    else:
        provider = None
    return SpanUsage(
        span_id=str(obj.get("span_id") or obj.get("context", {}).get("span_id") or ""),
        trace_id=str(obj.get("trace_id") or obj.get("context", {}).get("trace_id") or ""),
        model=str(model),
        provider=provider,
        prompt_tokens=_int_attr(
            attrs,
            "neatlogs.llm.token_count.prompt",
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.prompt_tokens",
        ),
        completion_tokens=_int_attr(
            attrs,
            "neatlogs.llm.token_count.completion",
            "gen_ai.usage.output_tokens",
            "gen_ai.usage.completion_tokens",
        ),
        cache_creation_tokens=_int_attr(
            attrs,
            "neatlogs.llm.token_count.cache_creation",
            "gen_ai.usage.cache_creation_input_tokens",
        ),
        cache_read_tokens=_int_attr(
            attrs,
            "neatlogs.llm.token_count.cache_read",
            "gen_ai.usage.cache_read_input_tokens",
        ),
        reasoning_tokens=_int_attr(
            attrs,
            "neatlogs.llm.token_count.reasoning",
            "gen_ai.usage.reasoning_tokens",
        ),
    )


def _detect_source(obj: Dict[str, Any]) -> str:
    if isinstance(obj.get("context"), dict) and "trace_id" in obj.get("context", {}):
        return "raw"
    if (
        "trace_id" in obj
        and "span_id" in obj
        and isinstance(obj.get("parent_span_id"), (str, type(None)))
    ):
        return "processed"
    return "unknown"


def _iter_json_objects(text: str) -> Iterator[Dict[str, Any]]:
    """Brace-balanced parser. Tolerates newlines inside string values."""
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
                try:
                    yield json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    pass
                start = -1
            elif depth < 0:
                depth = 0


def _read_usages(path: PathLike) -> Tuple[List[SpanUsage], int]:
    """Read one path; return (usages, files_read_delta). Missing → ([], 0).

    All parsed objects that look like span dicts become a ``SpanUsage``,
    including ones with no model or no token counts. The caller (compare)
    is responsible for deciding which to skip.
    """
    p = Path(path)
    if not p.exists():
        return [], 0
    text = p.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return [], 1
    out: List[SpanUsage] = []
    for obj in _iter_json_objects(text):
        if not isinstance(obj, dict):
            continue
        _detect_source(obj)  # currently unused; reserved for future raw-specific paths
        out.append(_extract_usage(obj))
    return out, 1


def _read_paths_usages(paths: Sequence[PathLike]) -> Tuple[List[SpanUsage], int]:
    usages: List[SpanUsage] = []
    files_read = 0
    for raw in paths:
        u, n = _read_usages(raw)
        usages.extend(u)
        files_read += n
    return usages, files_read


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


DEFAULT_PRICING_PATH = Path(__file__).parent / "config" / "pricing.json"


@dataclasses.dataclass
class PriceCard:
    """Pricing and capability data for one model.

    All token prices are USD per 1M tokens. ``None`` means "not priced" and
    the corresponding usage type is ignored. ``tiers`` covers long-context
    pricing: a dict where keys are ``input_above_{N}k_per_1m`` /
    ``output_above_{N}k_per_1m`` and values are the rate to use when the
    threshold is crossed. The whole span's input (or output) tokens are
    billed at the tier rate when the threshold is crossed (matches the
    Langfuse and LiteLLM convention).
    """

    model_key: str  # "provider/model"
    provider: str
    input_per_1m: float
    output_per_1m: float
    cache_read_per_1m: Optional[float] = None
    cache_write_per_1m: Optional[float] = None
    reasoning_output_per_1m: Optional[float] = None
    context_window: Optional[int] = None
    supports_vision: bool = False
    supports_tools: bool = False
    supports_reasoning: bool = False
    supports_prompt_cache: bool = False
    tiers: Dict[str, float] = dataclasses.field(default_factory=dict)

    @property
    def supports_cache(self) -> bool:
        # The catalog flag is the source of truth. The presence of cache
        # rates is what determines the actual cost; missing rates just mean
        # cache cost is $0, not that the model is incompatible.
        return self.supports_prompt_cache


@dataclasses.dataclass
class PricingCatalog:
    """Loaded pricing catalog with cheap lookup."""

    cards: Dict[str, PriceCard]
    by_provider_and_name: Dict[Tuple[str, str], str]  # (provider, model_name) -> key

    def get(self, model_key: str) -> Optional[PriceCard]:
        return self.cards.get(model_key)

    def get_by_provider_and_name(
        self, provider: Optional[str], model_name: str
    ) -> Optional[PriceCard]:
        if provider:
            key = self.by_provider_and_name.get((provider.lower(), model_name))
            if key:
                return self.cards.get(key)
            # Provider was given but no match under that provider. Do NOT
            # silently fall back to another provider — that would mis-bill
            # when the user has aliased model names across providers.
            return None
        # Provider-agnostic: any provider that has this model.
        for prov, name_key in self.by_provider_and_name.keys():
            if name_key == model_name:
                return self.cards.get(self.by_provider_and_name[(prov, name_key)])
        return None


def _load_catalog(path: Optional[PathLike] = None) -> PricingCatalog:
    src = Path(path) if path is not None else DEFAULT_PRICING_PATH
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    models = data.get("models", {})
    cards: Dict[str, PriceCard] = {}
    by_pn: Dict[Tuple[str, str], str] = {}
    for key, raw in models.items():
        if not isinstance(raw, dict):
            continue
        provider = str(raw.get("provider", "")).lower()
        if "/" not in key or not provider:
            continue
        card = PriceCard(
            model_key=key,
            provider=provider,
            input_per_1m=float(raw.get("input_per_1m", 0.0)),
            output_per_1m=float(raw.get("output_per_1m", 0.0)),
            cache_read_per_1m=_maybe_float(raw.get("cache_read_per_1m")),
            cache_write_per_1m=_maybe_float(raw.get("cache_write_per_1m")),
            reasoning_output_per_1m=_maybe_float(raw.get("reasoning_output_per_1m")),
            context_window=_maybe_int(raw.get("context_window")),
            supports_vision=bool(raw.get("supports_vision", False)),
            supports_tools=bool(raw.get("supports_tools", False)),
            supports_reasoning=bool(raw.get("supports_reasoning", False)),
            supports_prompt_cache=bool(raw.get("supports_prompt_cache", False)),
            tiers={
                k: float(v)
                for k, v in (raw.get("tiers") or {}).items()
                if isinstance(v, (int, float))
            },
        )
        cards[key] = card
        model_name = key.split("/", 1)[1]
        by_pn[(provider, model_name)] = key
    return PricingCatalog(cards=cards, by_provider_and_name=by_pn)


def _maybe_float(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) else None


def _maybe_int(v: Any) -> Optional[int]:
    return int(v) if isinstance(v, int) and not isinstance(v, bool) else None


def _model_key_for(usage: SpanUsage, catalog: PricingCatalog) -> Optional[str]:
    """Resolve a span's model+provider to a catalog key, with fallbacks."""
    if not usage.model:
        return None
    # Exact: provider/model
    if usage.provider:
        candidate = f"{usage.provider}/{usage.model}"
        if candidate in catalog.cards:
            return candidate
    # Provider-agnostic match on model name.
    for prov, name in catalog.by_provider_and_name:
        if name == usage.model:
            return catalog.by_provider_and_name[(prov, name)]
    return None


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SpanCost:
    """Cost for one span on one model."""

    span_id: str
    model_key: str
    input_cost: float
    output_cost: float
    cache_read_cost: float
    cache_write_cost: float
    reasoning_cost: float
    tier_applied: Optional[str] = None  # e.g. "input_above_200k_per_1m"
    incompatible: bool = False  # set when a span uses a feature the model lacks

    @property
    def total(self) -> float:
        if self.incompatible:
            return 0.0
        return (
            self.input_cost
            + self.output_cost
            + self.cache_read_cost
            + self.cache_write_cost
            + self.reasoning_cost
        )


def _tier_for(card: PriceCard, usage: SpanUsage) -> Tuple[Optional[str], float, float]:
    """Return (tier_label, effective_input_rate, effective_output_rate).

    Long-context tiered pricing: each side (input / output) has its own
    threshold. A side is billed at the tier rate iff its token count
    crosses the threshold. When both sides cross, the side with the higher
    *rate* (more expensive) wins, so the cost estimate is conservative.
    """
    if not card.tiers:
        return None, card.input_per_1m, card.output_per_1m
    # Parse "input_above_200k_per_1m" / "output_above_200k_per_1m".
    # Also accept litellm-style "above_200k_per_1m_input" / "above_200k_per_1m_output".
    thresholds: List[Tuple[int, str, float]] = []
    for k, rate in card.tiers.items():
        m = re.match(
            r"(?:(input|output)_above_(\d+)k_per_1m|above_(\d+)k_per_1m_(input|output))", k
        )
        if not m:
            continue
        if m.group(1):
            side, n = m.group(1), int(m.group(2))
        else:
            side, n = m.group(4), int(m.group(3))
        thresholds.append((n * 1000, side, float(rate)))
    if not thresholds:
        return None, card.input_per_1m, card.output_per_1m
    in_t = usage.prompt_tokens or 0
    out_t = usage.completion_tokens or 0
    # Per-side "crossed" check.
    in_crossed = [t for t in thresholds if t[1] == "input" and t[0] < in_t]
    out_crossed = [t for t in thresholds if t[1] == "output" and t[0] < out_t]
    if not in_crossed and not out_crossed:
        return None, card.input_per_1m, card.output_per_1m
    # Pick the more expensive crossed tier; on tie, prefer the output side
    # (matches langfuse "first match in priority order" with priority 0 =
    # default / base, and higher-priority tiers above).
    in_best = max(in_crossed, key=lambda t: t[2], default=None)
    out_best = max(out_crossed, key=lambda t: t[2], default=None)
    if in_best and out_best:
        if out_best[2] >= in_best[2]:
            chosen = out_best
        else:
            chosen = in_best
    elif out_best:
        chosen = out_best
    else:
        chosen = in_best
    assert chosen is not None
    threshold_n, side, rate = chosen
    if side == "input":
        return f"input_above_{threshold_n // 1000}k_per_1m", rate, card.output_per_1m
    return f"output_above_{threshold_n // 1000}k_per_1m", card.input_per_1m, rate


def cost_span(usage: SpanUsage, card: PriceCard) -> SpanCost:
    """Compute the cost of one span on one model. Returns a SpanCost with
    ``incompatible=True`` if the span uses a feature the model doesn't
    support (e.g. cache when the model has no cache rate)."""
    incompatible = False
    if usage.uses_prompt_cache and not card.supports_cache:
        incompatible = True
    if usage.uses_reasoning and not card.supports_reasoning:
        incompatible = True
    tier_label, in_rate, out_rate = _tier_for(card, usage)
    p = usage.prompt_tokens or 0
    c = usage.completion_tokens or 0
    cc = usage.cache_creation_tokens or 0
    cr = usage.cache_read_tokens or 0
    r = usage.reasoning_tokens or 0
    return SpanCost(
        span_id=usage.span_id,
        model_key=card.model_key,
        input_cost=(p / 1_000_000) * in_rate if not incompatible else 0.0,
        output_cost=(c / 1_000_000) * out_rate if not incompatible else 0.0,
        cache_read_cost=(
            (cr / 1_000_000) * card.cache_read_per_1m
            if (not incompatible and card.cache_read_per_1m is not None)
            else 0.0
        ),
        cache_write_cost=(
            (cc / 1_000_000) * card.cache_write_per_1m
            if (not incompatible and card.cache_write_per_1m is not None)
            else 0.0
        ),
        reasoning_cost=(
            (r / 1_000_000) * card.reasoning_output_per_1m
            if (not incompatible and card.reasoning_output_per_1m is not None)
            else 0.0
        ),
        tier_applied=tier_label,
        incompatible=incompatible,
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ModelComparison:
    """Cost results for one candidate model across all spans."""

    model_key: str
    provider: str
    total_usd: float
    input_usd: float
    output_usd: float
    cache_read_usd: float
    cache_write_usd: float
    reasoning_usd: float
    spans_total: int
    spans_incompatible: int
    missing_capabilities: List[str] = dataclasses.field(default_factory=list)
    is_baseline: bool = False

    @property
    def delta_pct(self) -> Optional[float]:
        """Percent change vs baseline. ``None`` for the baseline itself, or if
        there is no baseline."""
        return None  # filled in by ComparisonReport.delta_pct_for()


@dataclasses.dataclass
class ComparisonReport:
    baseline: Optional[ModelComparison]
    alternatives: List[ModelComparison]
    unknown_models: List[str]
    files_read: int
    spans_with_tokens: int
    spans_skipped: int  # no model, no tokens
    catalog_size: int

    def all(self) -> List[ModelComparison]:
        if self.baseline is None:
            return list(self.alternatives)
        return [self.baseline] + self.alternatives

    def delta_pct_for(self, mc: ModelComparison) -> Optional[float]:
        if self.baseline is None or mc.is_baseline:
            return None
        if self.baseline.total_usd == 0:
            return None
        return (mc.total_usd - self.baseline.total_usd) / self.baseline.total_usd * 100.0

    def find(self, model_key: str) -> Optional[ModelComparison]:
        for mc in self.all():
            if mc.model_key == model_key:
                return mc
        return None


def compare(
    paths: Union[PathLike, Sequence[PathLike]],
    *,
    models: Sequence[str],
    current_model: Optional[str] = None,
    catalog: Optional[PricingCatalog] = None,
    warn_stream: Optional[TextIO] = None,
) -> ComparisonReport:
    """Read span log files and produce a ComparisonReport.

    Args:
        paths: One path or a list of paths to span JSONL files.
        models: Candidate model keys (``provider/model``) to compare against.
            The first model in this list is treated as the baseline by
            default; override with ``current_model``.
        current_model: Optional explicit baseline. If given, the baseline
            section of the report uses this model; ``models`` lists the
            alternatives.
        catalog: Override the bundled pricing. ``None`` loads the default.
        warn_stream: Where to write unknown-model warnings. Defaults to
            stderr; pass ``io.StringIO()`` to silence.
    """
    if isinstance(paths, (str, os.PathLike)):
        seq: List[PathLike] = [paths]
    else:
        seq = list(paths)
    usages, files_read = _read_paths_usages(seq)
    if catalog is None:
        catalog = _load_catalog()
    sink = warn_stream if warn_stream is not None else sys.stderr

    # Resolve which model key is the baseline.
    baseline_key: Optional[str] = None
    if current_model:
        baseline_key = current_model
    elif models:
        baseline_key = models[0]

    # Build per-model cost results.
    results: Dict[str, ModelComparison] = {}
    unknown: set[str] = set()
    for key in models:
        if key not in catalog.cards:
            unknown.add(key)
    # Also track all distinct models the spans actually used (so we can warn
    # if a span's source model is unknown).
    source_models: set[str] = set()
    skipped = 0
    spans_with_tokens = 0

    for key in models:
        if key not in catalog.cards:
            continue
        card = catalog.cards[key]
        comp = ModelComparison(
            model_key=key,
            provider=card.provider,
            total_usd=0.0,
            input_usd=0.0,
            output_usd=0.0,
            cache_read_usd=0.0,
            cache_write_usd=0.0,
            reasoning_usd=0.0,
            spans_total=0,
            spans_incompatible=0,
            is_baseline=(key == baseline_key),
        )
        for usage in usages:
            if not usage.model or not usage.has_tokens:
                skipped += 1
                continue
            spans_with_tokens += 1
            source_models.add(f"{usage.provider}/{usage.model}" if usage.provider else usage.model)
            sc = cost_span(usage, card)
            comp.spans_total += 1
            if sc.incompatible:
                comp.spans_incompatible += 1
                continue
            comp.input_usd += sc.input_cost
            comp.output_usd += sc.output_cost
            comp.cache_read_usd += sc.cache_read_cost
            comp.cache_write_usd += sc.cache_write_cost
            comp.reasoning_usd += sc.reasoning_cost
            comp.total_usd += sc.total
        results[key] = comp

    # Capability diff vs baseline: for each alternative, list capabilities
    # the baseline supports that the alternative does not.
    baseline_card = catalog.get(baseline_key) if baseline_key else None
    baseline_caps = _capability_dict(baseline_card) if baseline_card else {}
    for key, comp in results.items():
        if comp.is_baseline:
            continue
        card = catalog.get(key)
        if card is None:
            continue
        alt_caps = _capability_dict(card)
        for cap, supported in baseline_caps.items():
            if supported and not alt_caps.get(cap, False):
                comp.missing_capabilities.append(cap)

    # Split baseline out for the report.
    baseline = results.pop(baseline_key, None) if baseline_key else None
    alternatives = [results[k] for k in models if k in results and k != baseline_key]

    # Warnings.
    for k in sorted(unknown):
        sink.write(
            f"[neatlogs-cost] warning: candidate model {k!r} not in pricing catalog; "
            f"it will be omitted from the comparison. "
            f"Override via --pricing-file or update "
            f"neatlogs/config/pricing.json.\n"
        )
    # Warn for span source models that aren't in the catalog at all.
    known_keys = set(catalog.cards.keys())
    unknown_source = sorted(m for m in source_models if m not in known_keys and m not in unknown)
    for k in unknown_source:
        sink.write(
            f"[neatlogs-cost] note: span source model {k!r} is not in the pricing "
            f"catalog. The comparison still uses the candidate models you specified, "
            f"but cost for the original model isn't reported. Override via --pricing-file.\n"
        )

    return ComparisonReport(
        baseline=baseline,
        alternatives=alternatives,
        unknown_models=sorted(unknown),
        files_read=files_read,
        spans_with_tokens=spans_with_tokens,
        spans_skipped=skipped,
        catalog_size=len(catalog.cards),
    )


def _capability_dict(card: PriceCard) -> Dict[str, bool]:
    return {
        "supports_vision": card.supports_vision,
        "supports_tools": card.supports_tools,
        "supports_reasoning": card.supports_reasoning,
        "supports_prompt_cache": card.supports_prompt_cache,
    }


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ForecastReport:
    """Cost projection for a given monthly call volume + model choice."""

    model_key: str
    monthly_calls: int
    avg_prompt_tokens: int
    avg_completion_tokens: int
    monthly_cost_usd: float
    annual_cost_usd: float
    cache_hit_rate: float = 0.0  # 0.0–1.0; only used if model supports cache
    reasoning_per_call: int = 0  # extra output tokens (for o-series etc.)
    incompatible: bool = False
    notes: List[str] = dataclasses.field(default_factory=list)


def forecast(
    *,
    model_key: str,
    monthly_calls: int,
    avg_prompt_tokens: int,
    avg_completion_tokens: int,
    catalog: Optional[PricingCatalog] = None,
    cache_hit_rate: float = 0.0,
    reasoning_per_call: int = 0,
) -> ForecastReport:
    """Project monthly and annual cost for a given traffic pattern.

    The model is treated as a single point estimate: every call is assumed
    to use the same average prompt and completion token counts. For more
    realistic estimates, vary the inputs across a range and aggregate.
    """
    if catalog is None:
        catalog = _load_catalog()
    card = catalog.get(model_key)
    if card is None:
        raise ValueError(f"unknown model: {model_key!r}")
    notes: List[str] = []
    incompatible = False

    cache_hit_rate = max(0.0, min(1.0, cache_hit_rate))
    hit_prompt = int(avg_prompt_tokens * cache_hit_rate)
    miss_prompt = avg_prompt_tokens - hit_prompt

    # Detect explicit incompatible requests before building the synthetic
    # usage, so the note points to what the user asked for rather than to
    # an inferred internal state.
    if cache_hit_rate > 0 and not card.supports_cache:
        incompatible = True
        notes.append(
            f"cache_hit_rate={cache_hit_rate} requested but model does not support prompt caching; cache cost set to 0"
        )
    if reasoning_per_call > 0 and not card.supports_reasoning:
        incompatible = True
        notes.append(
            f"reasoning_per_call={reasoning_per_call} requested but model is not a reasoning model; reasoning cost set to 0"
        )

    # Build a synthetic usage and use cost_span to keep the math in one place.
    usage = SpanUsage(
        span_id="forecast",
        trace_id="forecast",
        model=card.model_key.split("/", 1)[1],
        provider=card.provider,
        prompt_tokens=miss_prompt,
        completion_tokens=avg_completion_tokens,
        cache_creation_tokens=miss_prompt if card.supports_cache else None,
        cache_read_tokens=hit_prompt if card.supports_cache else None,
        reasoning_tokens=reasoning_per_call if card.supports_reasoning else None,
    )
    sc = cost_span(usage, card)
    per_call = sc.total if not incompatible else 0.0
    monthly = per_call * monthly_calls
    return ForecastReport(
        model_key=model_key,
        monthly_calls=monthly_calls,
        avg_prompt_tokens=avg_prompt_tokens,
        avg_completion_tokens=avg_completion_tokens,
        monthly_cost_usd=monthly,
        annual_cost_usd=monthly * 12,
        cache_hit_rate=cache_hit_rate,
        reasoning_per_call=reasoning_per_call,
        incompatible=incompatible,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_comparison(
    report: ComparisonReport,
    *,
    style: str = "text",
    stream: Optional[TextIO] = None,
) -> str:
    """Render a ComparisonReport as a string.

    style:
        - "text": human-readable table with delta vs baseline + capability diffs.
        - "json": one JSON object with all rows and totals.
        - "csv":  CSV with one row per (model, usage_type).
    """
    if style not in ("text", "json", "csv"):
        raise ValueError(f"unknown style: {style!r}")
    if style == "json":
        return _format_comparison_json(report)
    if style == "csv":
        return _format_comparison_csv(report)
    return _format_comparison_text(report, use_color=_supports_color(stream or sys.stdout))


def _format_comparison_text(report: ComparisonReport, *, use_color: bool) -> str:
    buf = io.StringIO()
    has_results = report.baseline is not None or bool(report.alternatives)
    if not has_results and report.spans_with_tokens == 0:
        if report.spans_skipped:
            buf.write(
                f"(no comparable models and no LLM spans with token counts found; "
                f"{report.spans_skipped} span(s) skipped.)\n"
            )
        else:
            buf.write(
                "(no comparable models — none of the candidates are in the pricing catalog)\n"
            )
        return buf.getvalue()
    if not has_results:
        buf.write("(no comparable models — none of the candidates are in the pricing catalog)\n")
        return buf.getvalue()
    if report.spans_with_tokens == 0:
        # Baseline selected, but no input spans; we still render the row at $0.
        pass  # fall through to the normal table
    header = (
        f"{'Model':<32} {'Provider':<12} {'Input':>10} {'Output':>10} "
        f"{'Cache':>10} {'Reason':>8} {'Total':>12} {'vs base':>9}"
    )
    buf.write(header + "\n")
    buf.write("-" * len(header) + "\n")
    rows: List[ModelComparison] = []
    if report.baseline is not None:
        rows.append(report.baseline)
    rows.extend(report.alternatives)
    for mc in rows:
        cache = mc.cache_read_usd + mc.cache_write_usd
        total_disp = f"${mc.total_usd:.4f}"
        if mc.is_baseline:
            delta_disp = "(baseline)"
        else:
            pct = report.delta_pct_for(mc)
            delta_disp = f"{pct:+.0f}%" if pct is not None else "n/a"
        line = (
            f"{mc.model_key[:32]:<32} {mc.provider[:12]:<12} "
            f"${mc.input_usd:.4f}    ${mc.output_usd:.4f}    "
            f"${cache:.4f}    ${mc.reasoning_usd:.4f}  "
            f"{total_disp:>12} {delta_disp:>9}"
        )
        if use_color:
            if mc.is_baseline:
                line = _ansi("1;33", line)
            elif report.delta_pct_for(mc) is not None and report.delta_pct_for(mc) < 0:
                line = _ansi("32", line)  # savings
            else:
                line = _ansi("31", line)  # more expensive
        buf.write(line + "\n")
    buf.write("-" * len(header) + "\n")

    # Capability diff.
    if report.baseline is not None:
        any_diff = any(a.missing_capabilities for a in report.alternatives)
        if any_diff:
            buf.write("\nCapability diff vs baseline:\n")
            for mc in report.alternatives:
                if mc.missing_capabilities:
                    buf.write(
                        f"  {_ansi('33', mc.model_key) if use_color else mc.model_key}: "
                        f"missing {', '.join(mc.missing_capabilities)}\n"
                    )
                else:
                    buf.write(f"  {mc.model_key}: all baseline capabilities supported\n")

    if report.spans_skipped:
        buf.write(f"\n({report.spans_skipped} span(s) skipped: no model or no token counts.)\n")
    buf.write(
        f"\n({report.spans_with_tokens} span(s) projected across "
        f"{len(report.all())} model(s); "
        f"catalog has {report.catalog_size} priced models.)\n"
    )
    return buf.getvalue()


def _format_comparison_json(report: ComparisonReport) -> str:
    def row(mc: ModelComparison) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "model": mc.model_key,
            "provider": mc.provider,
            "input_usd": round(mc.input_usd, 6),
            "output_usd": round(mc.output_usd, 6),
            "cache_read_usd": round(mc.cache_read_usd, 6),
            "cache_write_usd": round(mc.cache_write_usd, 6),
            "reasoning_usd": round(mc.reasoning_usd, 6),
            "total_usd": round(mc.total_usd, 6),
            "spans_total": mc.spans_total,
            "spans_incompatible": mc.spans_incompatible,
            "missing_capabilities": mc.missing_capabilities,
            "is_baseline": mc.is_baseline,
        }
        if not mc.is_baseline and report.baseline is not None:
            pct = report.delta_pct_for(mc)
            if pct is not None:
                d["delta_pct_vs_baseline"] = round(pct, 2)
        return d

    obj: Dict[str, Any] = {
        "currency": "USD",
        "files_read": report.files_read,
        "spans_with_tokens": report.spans_with_tokens,
        "spans_skipped": report.spans_skipped,
        "catalog_size": report.catalog_size,
        "unknown_models": report.unknown_models,
        "baseline": row(report.baseline) if report.baseline else None,
        "alternatives": [row(mc) for mc in report.alternatives],
    }
    return json.dumps(obj, indent=2) + "\n"


def _format_comparison_csv(report: ComparisonReport) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(
        [
            "model",
            "provider",
            "is_baseline",
            "input_usd",
            "output_usd",
            "cache_read_usd",
            "cache_write_usd",
            "reasoning_usd",
            "total_usd",
            "spans_total",
            "spans_incompatible",
            "delta_pct_vs_baseline",
        ]
    )
    for mc in [report.baseline] + report.alternatives if report.baseline else report.alternatives:
        if mc is None:
            continue
        pct = report.delta_pct_for(mc)
        w.writerow(
            [
                mc.model_key,
                mc.provider,
                "true" if mc.is_baseline else "false",
                f"{mc.input_usd:.6f}",
                f"{mc.output_usd:.6f}",
                f"{mc.cache_read_usd:.6f}",
                f"{mc.cache_write_usd:.6f}",
                f"{mc.reasoning_usd:.6f}",
                f"{mc.total_usd:.6f}",
                mc.spans_total,
                mc.spans_incompatible,
                "" if pct is None else f"{pct:.2f}",
            ]
        )
    return buf.getvalue()


def format_forecast(
    report: ForecastReport,
    *,
    style: str = "text",
    stream: Optional[TextIO] = None,
) -> str:
    if style not in ("text", "json"):
        raise ValueError(f"unknown style: {style!r}")
    if style == "json":
        return (
            json.dumps(
                {
                    "model": report.model_key,
                    "monthly_calls": report.monthly_calls,
                    "avg_prompt_tokens": report.avg_prompt_tokens,
                    "avg_completion_tokens": report.avg_completion_tokens,
                    "cache_hit_rate": report.cache_hit_rate,
                    "reasoning_per_call": report.reasoning_per_call,
                    "monthly_cost_usd": round(report.monthly_cost_usd, 4),
                    "annual_cost_usd": round(report.annual_cost_usd, 2),
                    "currency": "USD",
                    "incompatible": report.incompatible,
                    "notes": report.notes,
                },
                indent=2,
            )
            + "\n"
        )
    use_color = _supports_color(stream or sys.stdout)
    buf = io.StringIO()
    if report.incompatible:
        buf.write(
            _ansi("33", "WARNING: ") + "some requested features aren't supported by this model:\n"
        )
        for n in report.notes:
            buf.write(f"  - {n}\n")
        buf.write("\n")
    line = (
        f"Model:                {report.model_key}\n"
        f"Monthly calls:        {report.monthly_calls}\n"
        f"Avg prompt tokens:    {report.avg_prompt_tokens}\n"
        f"Avg completion:       {report.avg_completion_tokens}\n"
        f"Cache hit rate:       {report.cache_hit_rate * 100:.0f}%\n"
        f"Reasoning tokens:     {report.reasoning_per_call}\n"
        f"Monthly cost (USD):   ${report.monthly_cost_usd:.2f}\n"
        f"Annual cost (USD):    ${report.annual_cost_usd:.2f}\n"
    )
    if use_color:
        line = _ansi("1;33", line)
    buf.write(line)
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
        prog="neatlogs-compare",
        description=(
            "Compare what a span log would have cost across multiple models. "
            "Reads the same on-disk format as neatlogs-replay and applies the "
            "bundled pricing catalog (overridable via --pricing-file)."
        ),
    )
    p.add_argument("paths", nargs="+", help="One or more span log file paths.")
    p.add_argument(
        "--models",
        required=True,
        help=(
            "Comma-separated list of model keys to compare, in "
            "`provider/model` form. Example: "
            "'openai/gpt-4o-mini,anthropic/claude-3-5-haiku-latest,openai/gpt-4o'."
        ),
    )
    p.add_argument(
        "--current",
        default=None,
        help=("Optional explicit baseline model key. Defaults to the first model " "in --models."),
    )
    p.add_argument(
        "--format",
        choices=("text", "json", "csv"),
        default="text",
        help="Output format. Default: text.",
    )
    p.add_argument(
        "--pricing-file",
        default=None,
        help="Path to a JSON pricing catalog. Default: the bundled catalog.",
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
    p.add_argument(
        "--forecast",
        action="store_true",
        help=(
            "Instead of comparing past spans, project monthly/annual cost for "
            "a given traffic pattern (see --monthly-calls, --avg-prompt, "
            "--avg-completion, --cache-hit-rate, --reasoning-tokens)."
        ),
    )
    p.add_argument(
        "--monthly-calls",
        type=int,
        default=10000,
        help="Forecast: number of calls per month. Default: 10000.",
    )
    p.add_argument(
        "--avg-prompt",
        type=int,
        default=2000,
        help="Forecast: average prompt tokens per call. Default: 2000.",
    )
    p.add_argument(
        "--avg-completion",
        type=int,
        default=500,
        help="Forecast: average completion tokens per call. Default: 500.",
    )
    p.add_argument(
        "--cache-hit-rate",
        type=float,
        default=0.0,
        help="Forecast: fraction of prompt tokens served from cache (0.0–1.0). Default: 0.",
    )
    p.add_argument(
        "--reasoning-tokens",
        type=int,
        default=0,
        help="Forecast: extra reasoning tokens per call (for o-series, R1, etc.). Default: 0.",
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
        catalog = _load_catalog(args.pricing_file)
    except FileNotFoundError as exc:
        print(f"[neatlogs-cost] error: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"[neatlogs-cost] error: invalid pricing JSON: {exc}", file=sys.stderr)
        return 2

    if args.forecast:
        # Forecast mode: --models[0] is the model to project.
        model = args.models.split(",")[0].strip()
        try:
            report = forecast(
                model_key=model,
                monthly_calls=args.monthly_calls,
                avg_prompt_tokens=args.avg_prompt,
                avg_completion_tokens=args.avg_completion,
                catalog=catalog,
                cache_hit_rate=args.cache_hit_rate,
                reasoning_per_call=args.reasoning_tokens,
            )
        except ValueError as exc:
            print(f"[neatlogs-cost] error: {exc}", file=sys.stderr)
            return 2
        sys.stdout.write(format_forecast(report, style=args.format))
        return 0

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    report = compare(
        args.paths,
        models=models,
        current_model=args.current,
        catalog=catalog,
    )
    if not report.alternatives and report.baseline is None:
        print(
            "[neatlogs-cost] no comparable models: none of the candidates are in the pricing catalog",
            file=sys.stderr,
        )
        return 2
    sys.stdout.write(format_comparison(report, style=args.format))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
