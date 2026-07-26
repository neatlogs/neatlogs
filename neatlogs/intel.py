"""
LLM cost intelligence engine.

Given a historical span log and a list of candidate ``provider/model``
keys, answer the question finance and engineering teams actually ask:

    "Which of these models is the cheapest one that can still serve
    the same workload?"

The engine reads span JSONL logs, builds a workload profile (token
P50/P90/P99, capability usage, cache + reasoning totals), scores every
candidate model against that profile, and reports:

* per-model cost of replaying the workload
* capability compatibility (what fraction of spans can be served)
* the capability gap when a model can't serve the workload
* ranking: baseline first, then compatible-cheap, then incompatible

Pricing is decoupled from the engine via a ``PricingProvider`` chain,
so users can override rates per-environment, fetch a remote catalog,
or plug in LiteLLM's mirror without touching core code.

CLI::

    neatlogs-eval spans.log \\
        --candidates openai/gpt-4o-mini,openai/gpt-4o,anthropic/claude-3-5-haiku-latest \\
        --baseline openai/gpt-4o-mini

Programmatic::

    from neatlogs.intel import (
        evaluate_workload, BuiltinProvider, CustomProvider, ChainProvider,
        WorkloadConstraints, format_evaluation,
    )
    chain = ChainProvider([CustomProvider("my-rates.json"), BuiltinProvider()])
    report = evaluate_workload(
        paths=["spans.log"],
        candidates=["openai/gpt-4o-mini", "openai/gpt-4o"],
        baseline="openai/gpt-4o-mini",
        constraints=WorkloadConstraints(need_capabilities={"vision", "tools"}),
        pricing=chain,
    )
    print(format_evaluation(report, "text"))
"""

from __future__ import annotations

import abc
import argparse
import csv
import dataclasses
import io
import json
import os
import sys
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Union,
)

PathLike = Union[str, os.PathLike]


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class Capability:
    """Standard capability names. The catalog is a free-form set, so
    third-party providers can extend this without code changes —
    ``is_compatible`` only fails closed on explicit declarations."""

    VISION = "vision"
    TOOLS = "tools"
    JSON_MODE = "json_mode"
    PROMPT_CACHE = "prompt_cache"
    REASONING = "reasoning"
    AUDIO = "audio"
    IMAGE_INPUT = "image_input"
    EMBEDDING = "embedding"


class UsageType:
    """Standard usage type names. The ``usage_types`` dict on a
    ``ModelDefinition`` can contain any string key, so this list is
    documentation rather than a closed enum."""

    INPUT = "input"
    OUTPUT = "output"
    CACHE_READ = "cache_read"
    CACHE_WRITE = "cache_write"
    REASONING = "reasoning"
    IMAGE = "image"
    AUDIO = "audio"


# ---------------------------------------------------------------------------
# Span reading (lightweight — we only need token counts and model name)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SpanUsage:
    span_id: str
    trace_id: str
    model: str
    provider: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    reasoning_tokens: int

    @property
    def input_total(self) -> int:
        """Total input side: prompt + cache_read + cache_creation."""
        return self.prompt_tokens + self.cache_creation_tokens + self.cache_read_tokens

    @property
    def output_total(self) -> int:
        """Total output side: completion + reasoning."""
        return self.completion_tokens + self.reasoning_tokens

    @property
    def has_tokens(self) -> bool:
        return any(
            t > 0
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
        return self.cache_creation_tokens > 0 or self.cache_read_tokens > 0

    @property
    def uses_reasoning(self) -> bool:
        return self.reasoning_tokens > 0


def _int_attr(attrs: Dict[str, Any], *keys: str) -> int:
    for k in keys:
        v = attrs.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, int) and v >= 0:
            return v
        if isinstance(v, float) and v.is_integer() and v >= 0:
            return int(v)
    return 0


def _extract_usage(obj: Dict[str, Any]) -> SpanUsage:
    attrs = obj.get("attributes") or {}
    if not isinstance(attrs, dict):
        attrs = {}
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


def _iter_json_objects(text: str) -> Iterator[Dict[str, Any]]:
    """Brace-balanced parser; tolerates newlines inside string values."""
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


def _read_usages(path: PathLike) -> List[SpanUsage]:
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []
    return [_extract_usage(obj) for obj in _iter_json_objects(text) if isinstance(obj, dict)]


def _read_paths(paths: Sequence[PathLike]) -> List[SpanUsage]:
    out: List[SpanUsage] = []
    for raw in paths:
        out.extend(_read_usages(raw))
    return out


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Tier:
    """A single tiered rate: if the usage count is strictly above
    ``above_tokens``, use ``rate`` instead of the base rate. A model
    with multiple tiers picks the largest ``above_tokens`` that is
    crossed."""

    above_tokens: int
    rate: float


@dataclasses.dataclass
class ModelDefinition:
    """Pricing and capability data for one model.

    ``usage_types`` maps a usage type (``"input"``, ``"output"``,
    ``"cache_read"``, ``"cache_write"``, ``"reasoning"``,
    ``"image"``, ``"audio"``, ...) to USD per 1M tokens. Missing
    keys mean that usage type is not billed (or not supported).

    ``tiers`` is a per-usage-type list of ``Tier`` entries. When a
    span's token count crosses a tier threshold, that tier's rate
    is used. The largest crossed tier wins.
    """

    model_key: str  # "provider/model"
    provider: str
    context_window: Optional[int] = None
    capabilities: Set[str] = dataclasses.field(default_factory=set)
    usage_types: Dict[str, float] = dataclasses.field(default_factory=dict)
    tiers: Dict[str, List[Tier]] = dataclasses.field(default_factory=dict)

    def rate_for(self, usage_type: str) -> Optional[float]:
        """Base rate for a usage type, or None if the model doesn't
        bill for it. Does not consider tiers."""
        return self.usage_types.get(usage_type)

    def effective_rate(self, usage_type: str, count: int) -> Optional[float]:
        """Effective rate for a usage type at a given token count,
        taking tiers into account. ``None`` if not billed."""
        base = self.rate_for(usage_type)
        if base is None:
            return None
        if count <= 0 or usage_type not in self.tiers:
            return base
        crossed = [t for t in self.tiers[usage_type] if t.above_tokens < count]
        if not crossed:
            return base
        return max(crossed, key=lambda t: t.above_tokens).rate

    def has_capability(self, cap: str) -> bool:
        return cap in self.capabilities

    def has_all_capabilities(self, caps: Iterable[str]) -> bool:
        return all(c in self.capabilities for c in caps)

    def missing_capabilities(self, caps: Iterable[str]) -> Set[str]:
        return {c for c in caps if c not in self.capabilities}


class PricingProvider(abc.ABC):
    """Source of ``ModelDefinition`` for a given model key.

    Implementations include ``BuiltinProvider`` (bundled JSON),
    ``CustomProvider`` (user-supplied override file), and any future
    HTTP-backed or litellm-mirror provider. ``ChainProvider`` composes
    multiple providers in priority order.
    """

    @abc.abstractmethod
    def lookup(self, model_key: str) -> Optional[ModelDefinition]: ...

    def lookup_by_provider_and_name(
        self, provider: Optional[str], model_name: str
    ) -> Optional[ModelDefinition]:
        return None


def _build_def(key: str, raw: Dict[str, Any]) -> Optional[ModelDefinition]:
    if not isinstance(raw, dict):
        return None
    provider = str(raw.get("provider", "")).lower()
    if "/" not in key or not provider:
        return None
    caps_raw = raw.get("capabilities") or []
    if not isinstance(caps_raw, list):
        caps_raw = []
    capabilities = {str(c) for c in caps_raw if isinstance(c, str)}
    usage_types_raw = raw.get("usage_types") or {}
    if not isinstance(usage_types_raw, dict):
        usage_types_raw = {}
    usage_types = {
        str(k): float(v) for k, v in usage_types_raw.items() if isinstance(v, (int, float))
    }
    tiers_raw = raw.get("tiers") or {}
    if not isinstance(tiers_raw, dict):
        tiers_raw = {}
    tiers: Dict[str, List[Tier]] = {}
    for usage_type, entries in tiers_raw.items():
        if not isinstance(entries, list):
            continue
        out_entries: List[Tier] = []
        for e in entries:
            if (
                isinstance(e, dict)
                and isinstance(e.get("above_tokens"), int)
                and isinstance(e.get("rate"), (int, float))
            ):
                out_entries.append(Tier(above_tokens=e["above_tokens"], rate=float(e["rate"])))
        if out_entries:
            tiers[str(usage_type)] = out_entries
    cw_raw = raw.get("context_window")
    context_window = cw_raw if isinstance(cw_raw, int) and not isinstance(cw_raw, bool) else None
    return ModelDefinition(
        model_key=key,
        provider=provider,
        context_window=context_window,
        capabilities=capabilities,
        usage_types=usage_types,
        tiers=tiers,
    )


class BuiltinProvider(PricingProvider):
    """Reads ``neatlogs/config/pricing.json``."""

    def __init__(self, path: Optional[PathLike] = None):
        src = Path(path) if path is not None else Path(__file__).parent / "config" / "pricing.json"
        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._models: Dict[str, ModelDefinition] = {}
        self._by_pn: Dict[Tuple[str, str], str] = {}
        for key, raw in (data.get("models") or {}).items():
            d = _build_def(key, raw)
            if d is not None:
                self._models[key] = d
                self._by_pn[(d.provider, key.split("/", 1)[1])] = key

    def lookup(self, model_key: str) -> Optional[ModelDefinition]:
        return self._models.get(model_key)

    def lookup_by_provider_and_name(
        self, provider: Optional[str], model_name: str
    ) -> Optional[ModelDefinition]:
        if provider:
            key = self._by_pn.get((provider.lower(), model_name))
            if key:
                return self._models.get(key)
            return None
        for prov, name in self._by_pn.keys():
            if name == model_name:
                return self._models.get(self._by_pn[(prov, name)])
        return None


class CustomProvider(PricingProvider):
    """A user-supplied override file. Sits on top of the builtin catalog
    in a ``ChainProvider`` so the override wins for any model it declares
    and falls through for everything else.
    """

    def __init__(self, path: PathLike):
        self._path = Path(path)
        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._models: Dict[str, ModelDefinition] = {}
        self._by_pn: Dict[Tuple[str, str], str] = {}
        for key, raw in (data.get("models") or {}).items():
            d = _build_def(key, raw)
            if d is not None:
                self._models[key] = d
                self._by_pn[(d.provider, key.split("/", 1)[1])] = key

    def lookup(self, model_key: str) -> Optional[ModelDefinition]:
        return self._models.get(model_key)

    def lookup_by_provider_and_name(
        self, provider: Optional[str], model_name: str
    ) -> Optional[ModelDefinition]:
        if provider:
            key = self._by_pn.get((provider.lower(), model_name))
            if key:
                return self._models.get(key)
            return None
        for prov, name in self._by_pn.keys():
            if name == model_name:
                return self._models.get(self._by_pn[(prov, name)])
        return None


class ChainProvider(PricingProvider):
    """Composes a list of providers in priority order. The first one
    to return a non-None result wins. Matches the Langfuse model-
    definition resolution pattern: custom override first, then bundled
    catalog, then fallback to provider alias / regex / heuristic.
    """

    def __init__(self, providers: Sequence[PricingProvider]):
        self._providers = list(providers)

    def lookup(self, model_key: str) -> Optional[ModelDefinition]:
        for p in self._providers:
            r = p.lookup(model_key)
            if r is not None:
                return r
        return None

    def lookup_by_provider_and_name(
        self, provider: Optional[str], model_name: str
    ) -> Optional[ModelDefinition]:
        for p in self._providers:
            r = p.lookup_by_provider_and_name(provider, model_name)
            if r is not None:
                return r
        return None


def default_chain(pricing_file: Optional[PathLike] = None) -> ChainProvider:
    """Build the default chain: optional user override, then bundled."""
    providers: List[PricingProvider] = []
    if pricing_file is not None:
        providers.append(CustomProvider(pricing_file))
    elif os.environ.get("NEATLOGS_PRICING_FILE"):
        providers.append(CustomProvider(os.environ["NEATLOGS_PRICING_FILE"]))
    providers.append(BuiltinProvider())
    return ChainProvider(providers)


# ---------------------------------------------------------------------------
# Workload profile
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TokenStats:
    """Lightweight summary of a token distribution."""

    p50: int
    p90: int
    p99: int
    max: int
    total: int

    @classmethod
    def from_values(cls, values: Sequence[int]) -> "TokenStats":
        if not values:
            return cls(0, 0, 0, 0, 0)
        sorted_vals = sorted(values)
        n = len(sorted_vals)

        def pct(p: float) -> int:
            # Floor: "lower" percentile. For [1..100] p50 is 50 (sorted_vals[49]).
            idx = min(int(p * (n - 1)), n - 1)
            return sorted_vals[idx]

        return cls(
            p50=pct(0.50),
            p90=pct(0.90),
            p99=pct(0.99),
            max=sorted_vals[-1],
            total=sum(sorted_vals),
        )


@dataclasses.dataclass
class WorkloadProfile:
    """Summary of a span log, derived once and reused by the evaluator."""

    total_spans: int
    spans_with_tokens: int
    spans_skipped: int
    models_used: Set[str]
    capabilities_inferred: Set[str]
    prompt_stats: TokenStats
    completion_stats: TokenStats
    cache_read_total: int
    cache_write_total: int
    reasoning_total: int
    files_read: int

    @property
    def needs_cache(self) -> bool:
        return (self.cache_read_total + self.cache_write_total) > 0

    @property
    def needs_reasoning(self) -> bool:
        return self.reasoning_total > 0


def _resolve_definition(usage: SpanUsage, pricing: PricingProvider) -> Optional[ModelDefinition]:
    if not usage.model:
        return None
    if usage.provider:
        candidate = f"{usage.provider}/{usage.model}"
        d = pricing.lookup(candidate)
        if d is not None:
            return d
    return pricing.lookup_by_provider_and_name(usage.provider, usage.model)


def build_workload_profile(
    paths: Sequence[PathLike],
    pricing: PricingProvider,
    *,
    auto_infer_capabilities: bool = True,
) -> Tuple[WorkloadProfile, List[SpanUsage]]:
    """Read span logs, derive a profile, and return ``(profile, usages)``.

    ``auto_infer_capabilities`` (default True) infers the capability set
    the workload needs from the source models in the log. A workload
    that ran on ``gpt-4o-mini`` is assumed to need whatever
    ``gpt-4o-mini`` supports. The user can override via
    ``WorkloadConstraints.need_capabilities``.
    """
    usages = _read_paths(paths)
    files_read = sum(1 for p in paths if Path(p).exists())
    with_tokens = [u for u in usages if u.model and u.has_tokens]
    skipped = len(usages) - len(with_tokens)
    models_used: Set[str] = set()
    capabilities: Set[str] = set()
    prompt_vals: List[int] = []
    completion_vals: List[int] = []
    cache_read_total = 0
    cache_write_total = 0
    reasoning_total = 0
    for u in with_tokens:
        models_used.add(f"{u.provider}/{u.model}" if u.provider else u.model)
        prompt_vals.append(u.prompt_tokens)
        completion_vals.append(u.completion_tokens)
        cache_read_total += u.cache_read_tokens
        cache_write_total += u.cache_creation_tokens
        reasoning_total += u.reasoning_tokens
        if auto_infer_capabilities:
            d = _resolve_definition(u, pricing)
            if d is not None:
                capabilities.update(d.capabilities)
    profile = WorkloadProfile(
        total_spans=len(usages),
        spans_with_tokens=len(with_tokens),
        spans_skipped=skipped,
        models_used=models_used,
        capabilities_inferred=capabilities,
        prompt_stats=TokenStats.from_values(prompt_vals),
        completion_stats=TokenStats.from_values(completion_vals),
        cache_read_total=cache_read_total,
        cache_write_total=cache_write_total,
        reasoning_total=reasoning_total,
        files_read=files_read,
    )
    return profile, with_tokens


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class WorkloadConstraints:
    """What a candidate model must satisfy to be considered.

    ``need_capabilities``: set of capability strings the candidate must
    declare. Missing one is a hard fail.
    ``min_context_window``: candidate's ``context_window`` must be ``>=``
    this value. ``0`` (default) means no constraint.
    ``min_compatibility_pct``: minimum fraction of spans (by count) the
    candidate can serve. Default 0.95 = "you can switch 95% of the
    workload to this model".
    """

    need_capabilities: Set[str] = dataclasses.field(default_factory=set)
    min_context_window: int = 0
    min_compatibility_pct: float = 0.95

    def explains(self) -> List[str]:
        """Human-readable summary of active constraints."""
        out: List[str] = []
        if self.need_capabilities:
            out.append(f"need capabilities: {sorted(self.need_capabilities)}")
        if self.min_context_window > 0:
            out.append(f"min context: {self.min_context_window:,} tokens")
        if self.min_compatibility_pct > 0:
            out.append(f"min compatibility: {self.min_compatibility_pct * 100:.0f}%")
        return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SpanVerdict:
    """One span's compatibility with one candidate model."""

    span_id: str
    compatible: bool
    reasons: List[str] = dataclasses.field(default_factory=list)
    cost: float = 0.0


def _span_cost(usage: SpanUsage, model: ModelDefinition) -> float:
    """Cost of one span on one model. Returns 0 for incompatible spans.
    Tiered pricing is applied per usage type.
    """
    # Required usage type: at least one of input/output must be billed.
    in_rate = model.effective_rate(UsageType.INPUT, usage.prompt_tokens)
    out_rate = model.effective_rate(UsageType.OUTPUT, usage.completion_tokens)
    if in_rate is None and out_rate is None:
        return 0.0
    cost = 0.0
    if in_rate is not None:
        cost += (usage.prompt_tokens / 1_000_000) * in_rate
    if out_rate is not None:
        cost += (usage.completion_tokens / 1_000_000) * out_rate
    # Cache: model supports it AND user used it.
    if usage.cache_creation_tokens > 0:
        rate = model.effective_rate(UsageType.CACHE_WRITE, usage.cache_creation_tokens)
        if rate is not None:
            cost += (usage.cache_creation_tokens / 1_000_000) * rate
    if usage.cache_read_tokens > 0:
        rate = model.effective_rate(UsageType.CACHE_READ, usage.cache_read_tokens)
        if rate is not None:
            cost += (usage.cache_read_tokens / 1_000_000) * rate
    # Reasoning: model supports it AND user used it.
    if usage.reasoning_tokens > 0:
        rate = model.effective_rate(UsageType.REASONING, usage.reasoning_tokens)
        if rate is not None:
            cost += (usage.reasoning_tokens / 1_000_000) * rate
    return cost


def _is_span_compatible(
    usage: SpanUsage, model: ModelDefinition, constraints: WorkloadConstraints
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    compatible = True
    # Context window: prompt tokens must fit (with some headroom for the
    # response; we don't know the response size at planning time, so use
    # the request input as a conservative lower bound).
    if constraints.min_context_window > 0:
        if model.context_window is None:
            compatible = False
            reasons.append(
                f"min context {constraints.min_context_window:,} required; model has no declared context"
            )
        elif model.context_window < usage.input_total:
            compatible = False
            reasons.append(f"prompt {usage.input_total:,} > context {model.context_window:,}")
    # Capabilities the workload needs that the model lacks.
    missing = model.missing_capabilities(constraints.need_capabilities)
    if missing:
        compatible = False
        reasons.append(f"missing {sorted(missing)}")
    # Span-specific capability inference: cache used, no cache support.
    if (
        usage.uses_prompt_cache
        and UsageType.CACHE_READ not in model.usage_types
        and UsageType.CACHE_WRITE not in model.usage_types
    ):
        compatible = False
        reasons.append("used cache; model has no cache rate")
    if usage.uses_reasoning and UsageType.REASONING not in model.usage_types:
        compatible = False
        reasons.append("used reasoning; model has no reasoning rate")
    return compatible, reasons


def _score_model(
    model: ModelDefinition,
    usages: List[SpanUsage],
    constraints: WorkloadConstraints,
    is_baseline: bool,
) -> "ScoredModel":
    """Score one candidate model against the workload."""
    per_span: List[SpanVerdict] = []
    for u in usages:
        ok, reasons = _is_span_compatible(u, model, constraints)
        cost = _span_cost(u, model) if ok else 0.0
        per_span.append(SpanVerdict(span_id=u.span_id, compatible=ok, reasons=reasons, cost=cost))
    compatible_spans = sum(1 for v in per_span if v.compatible)
    total = sum(v.cost for v in per_span)
    compat_pct = compatible_spans / len(per_span) if per_span else 0.0
    meets_pct = compat_pct >= constraints.min_compatibility_pct
    return ScoredModel(
        model_key=model.model_key,
        provider=model.provider,
        total_cost=total,
        compatible_spans=compatible_spans,
        total_spans=len(per_span),
        compatibility_pct=compat_pct,
        meets_min_compatibility=meets_pct,
        missing_capabilities=model.missing_capabilities(constraints.need_capabilities),
        context_window=model.context_window,
        per_span=per_span,
        is_baseline=is_baseline,
    )


@dataclasses.dataclass
class ScoredModel:
    """One candidate model's score against the workload."""

    model_key: str
    provider: str
    total_cost: float
    compatible_spans: int
    total_spans: int
    compatibility_pct: float
    meets_min_compatibility: bool
    missing_capabilities: Set[str]
    context_window: Optional[int]
    per_span: List[SpanVerdict]
    is_baseline: bool = False

    @property
    def rank_key(self) -> Tuple[int, int, float]:
        """Lower is better. Sort: baseline first, then compatible by
        cost (cheap first), then incompatible last."""
        return (
            0 if self.is_baseline else 1,
            0 if self.meets_min_compatibility else 1,
            self.total_cost,
        )

    @property
    def delta_pct_vs_baseline(self) -> Optional[float]:
        return None  # filled in by EvaluationReport


@dataclasses.dataclass
class EvaluationReport:
    baseline: Optional[ScoredModel]
    alternatives: List[ScoredModel]
    profile: WorkloadProfile
    constraints: WorkloadConstraints
    explicit_constraints: WorkloadConstraints
    unknown_models: List[str]
    files_read: int

    def all(self) -> List[ScoredModel]:
        if self.baseline is None:
            return list(self.alternatives)
        return [self.baseline] + self.alternatives

    def delta_pct_for(self, sm: ScoredModel) -> Optional[float]:
        if self.baseline is None or sm.is_baseline:
            return None
        if self.baseline.total_cost == 0:
            return None
        return (sm.total_cost - self.baseline.total_cost) / self.baseline.total_cost * 100.0

    def ranked(self) -> List[ScoredModel]:
        """All candidates sorted: baseline first, then compatible by
        cost, then incompatible last."""
        return sorted(self.all(), key=lambda s: s.rank_key)

    def find(self, model_key: str) -> Optional[ScoredModel]:
        for sm in self.all():
            if sm.model_key == model_key:
                return sm
        return None


def evaluate_workload(
    paths: Union[PathLike, Sequence[PathLike]],
    *,
    candidates: Sequence[str],
    baseline: Optional[str] = None,
    pricing: Optional[PricingProvider] = None,
    constraints: Optional[WorkloadConstraints] = None,
    auto_infer_capabilities: bool = True,
    warn_stream: Optional[TextIO] = None,
) -> EvaluationReport:
    """Read span logs, score every candidate model, return a report.

    Args:
        paths: one path or a list of paths to span JSONL files.
        candidates: ``provider/model`` keys to evaluate.
        baseline: explicit baseline. Defaults to the first candidate.
        pricing: override the default chain. ``None`` builds the default
            (custom override from ``--pricing-file`` / ``NEATLOGS_PRICING_FILE``
            if set, then bundled catalog).
        constraints: capability / context-window filters. ``None`` = no filter.
        auto_infer_capabilities: when True (default), capability
            requirements are inferred from the source models in the
            workload. When False, only ``constraints.need_capabilities``
            applies.
        warn_stream: where to write warnings (unknown models, etc.).
            Defaults to stderr.
    """
    if isinstance(paths, (str, os.PathLike)):
        seq: List[PathLike] = [paths]
    else:
        seq = list(paths)
    if pricing is None:
        pricing = default_chain()
    if constraints is None:
        constraints = WorkloadConstraints()
    sink = warn_stream if warn_stream is not None else sys.stderr

    profile, usages = build_workload_profile(
        seq,
        pricing,
        auto_infer_capabilities=auto_infer_capabilities,
    )
    # When auto-infer is on, merge inferred capabilities with explicit ones.
    effective_caps = set(constraints.need_capabilities)
    if auto_infer_capabilities:
        effective_caps |= profile.capabilities_inferred
    effective_constraints = WorkloadConstraints(
        need_capabilities=effective_caps,
        min_context_window=constraints.min_context_window,
        min_compatibility_pct=constraints.min_compatibility_pct,
    )

    baseline_key = baseline or (candidates[0] if candidates else None)
    unknown: List[str] = []
    resolved: Dict[str, ModelDefinition] = {}
    for key in candidates:
        d = pricing.lookup(key)
        if d is None:
            unknown.append(key)
        else:
            resolved[key] = d

    # Build scored models.
    scored: Dict[str, ScoredModel] = {}
    for key, d in resolved.items():
        sm = _score_model(
            d,
            usages,
            effective_constraints,
            is_baseline=(key == baseline_key),
        )
        scored[key] = sm
    baseline_sm = scored.pop(baseline_key, None) if baseline_key else None
    alternatives = [scored[k] for k in candidates if k in scored and k != baseline_key]

    # Warnings.
    for k in sorted(unknown):
        sink.write(
            f"[neatlogs-eval] warning: candidate model {k!r} not in pricing catalog; "
            f"it will be omitted from the report. "
            f"Override via --pricing-file or update "
            f"neatlogs/config/pricing.json.\n"
        )

    return EvaluationReport(
        baseline=baseline_sm,
        alternatives=alternatives,
        profile=profile,
        constraints=effective_constraints,
        explicit_constraints=constraints,
        unknown_models=sorted(unknown),
        files_read=profile.files_read,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_evaluation_text(report: EvaluationReport, *, use_color: bool = True) -> str:
    buf = io.StringIO()
    if not report.alternatives and report.baseline is None:
        buf.write("(no candidate models in the pricing catalog)\n")
        return buf.getvalue()
    # Header: workload summary.
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

    # Per-model rows.
    header = f"{'Model':<34} {'Prov':<12} {'Compat':>7} " f"{'Cost':>10} {'vs base':>10}"
    buf.write(header + "\n")
    buf.write("-" * len(header) + "\n")
    for sm in report.ranked():
        # For an incompatible model, show $0 — the cost is meaningless
        # because the user can't actually run on it.
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

    # Capability gap.
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
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neatlogs-eval",
        description=(
            "Evaluate candidate models against a span-log workload. "
            "For each candidate, computes the cost of running the same "
            "workload on that model and reports capability compatibility. "
            "Ranks compatible + cheap first, incompatible last. This is "
            "the 'Replay + Model Optimizer' for cost: given a past run, "
            "find the cheapest model that can still serve it."
        ),
    )
    p.add_argument("paths", nargs="+", help="One or more span log file paths.")
    p.add_argument(
        "--candidates",
        required=True,
        help=(
            "Comma-separated list of `provider/model` keys to evaluate. "
            "The first one is the baseline; use --baseline to override."
        ),
    )
    p.add_argument(
        "--baseline",
        default=None,
        help="Explicit baseline model key. Default: first --candidates entry.",
    )
    p.add_argument(
        "--need",
        action="append",
        default=[],
        help=(
            "Require a capability (repeatable). Values: vision, tools, "
            "json_mode, prompt_cache, reasoning, audio, image_input, "
            "embedding. By default the engine infers capability needs "
            "from the source models in the log; --need adds to the set."
        ),
    )
    p.add_argument(
        "--min-context",
        type=int,
        default=0,
        help=(
            "Minimum context window in tokens. Models with a smaller "
            "context_window (or no context_window declared) are marked "
            "incompatible for any span that exceeds the constraint. "
            "Default: 0 (no constraint)."
        ),
    )
    p.add_argument(
        "--min-compatibility",
        type=float,
        default=0.95,
        help=(
            "Minimum fraction of spans a candidate must be able to serve "
            "(0.0–1.0). Candidates below this threshold are reported as "
            "incompatible. Default: 0.95."
        ),
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
        "--no-auto-infer",
        action="store_true",
        help=(
            "Do NOT infer capability needs from the source models. "
            "Use only --need. Use this when the source model declares "
            "capabilities the workload does not actually exercise."
        ),
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
    pricing = default_chain(args.pricing_file)
    constraints = WorkloadConstraints(
        need_capabilities=set(args.need or []),
        min_context_window=args.min_context,
        min_compatibility_pct=args.min_compatibility,
    )
    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    report = evaluate_workload(
        args.paths,
        candidates=candidates,
        baseline=args.baseline,
        pricing=pricing,
        constraints=constraints,
        auto_infer_capabilities=not args.no_auto_infer,
    )
    if not report.alternatives and report.baseline is None:
        print(
            "[neatlogs-eval] no candidate models found in the pricing catalog",
            file=sys.stderr,
        )
        return 2
    sys.stdout.write(format_evaluation(report, style=args.format))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
