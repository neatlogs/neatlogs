"""Per-model cost breakdown.

Given a span log and a list of candidate models, compute the cost of
running the same workload on each model and report a per-usage-type
split (input / output / cache_read / cache_write / reasoning /
image / audio).

Unlike the ranking evaluator, every model gets a real cost row — there
is no compatibility filter, no capability gap, no "incompatible"
state. If a model doesn't bill a usage type, that field is $0.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from typing import List, Optional, Sequence, TextIO, Union

from .pricing import ModelDefinition, PricingProvider
from .spans import PathLike, SpanUsage
from .workload import WorkloadProfile, _read_paths, build_workload_profile


@dataclasses.dataclass
class SpanCost:
    """Cost of one span on one model, broken down by usage type.

    Any field not billed by the model is left at 0. A model that does
    not declare an ``input`` rate but does declare an ``output`` rate
    will have ``input_cost == 0`` and ``output_cost > 0``.
    """

    span_id: str
    model_key: str
    input_cost: float = 0.0
    output_cost: float = 0.0
    cache_read_cost: float = 0.0
    cache_write_cost: float = 0.0
    reasoning_cost: float = 0.0
    image_cost: float = 0.0
    audio_cost: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.input_cost
            + self.output_cost
            + self.cache_read_cost
            + self.cache_write_cost
            + self.reasoning_cost
            + self.image_cost
            + self.audio_cost
        )


def cost_span(usage: SpanUsage, model: ModelDefinition) -> SpanCost:
    """Cost of one span on one model, broken down by usage type.

    Tiered pricing is applied per usage type. Returns all zeros if the
    model does not bill any of the usage types the span actually used.
    """
    from .pricing import UsageType

    sc = SpanCost(span_id=usage.span_id, model_key=model.model_key)
    in_rate = model.effective_rate(UsageType.INPUT, usage.prompt_tokens)
    out_rate = model.effective_rate(UsageType.OUTPUT, usage.completion_tokens)
    if in_rate is None and out_rate is None:
        return sc
    if in_rate is not None and usage.prompt_tokens > 0:
        sc.input_cost = (usage.prompt_tokens / 1_000_000) * in_rate
    if out_rate is not None and usage.completion_tokens > 0:
        sc.output_cost = (usage.completion_tokens / 1_000_000) * out_rate
    if usage.cache_creation_tokens > 0:
        rate = model.effective_rate(UsageType.CACHE_WRITE, usage.cache_creation_tokens)
        if rate is not None:
            sc.cache_write_cost = (usage.cache_creation_tokens / 1_000_000) * rate
    if usage.cache_read_tokens > 0:
        rate = model.effective_rate(UsageType.CACHE_READ, usage.cache_read_tokens)
        if rate is not None:
            sc.cache_read_cost = (usage.cache_read_tokens / 1_000_000) * rate
    if usage.reasoning_tokens > 0:
        rate = model.effective_rate(UsageType.REASONING, usage.reasoning_tokens)
        if rate is not None:
            sc.reasoning_cost = (usage.reasoning_tokens / 1_000_000) * rate
    return sc


@dataclasses.dataclass
class ModelCostBreakdown:
    """Aggregated cost breakdown for one model across a workload."""

    model_key: str
    provider: str
    total_cost: float
    input_cost: float
    output_cost: float
    cache_read_cost: float
    cache_write_cost: float
    reasoning_cost: float
    image_cost: float
    audio_cost: float
    spans_with_tokens: int
    spans_total: int
    spans_skipped: int


def _aggregate_costs(usages: List[SpanUsage], model: ModelDefinition) -> ModelCostBreakdown:
    sums = {
        "input_cost": 0.0,
        "output_cost": 0.0,
        "cache_read_cost": 0.0,
        "cache_write_cost": 0.0,
        "reasoning_cost": 0.0,
        "image_cost": 0.0,
        "audio_cost": 0.0,
    }
    with_tokens = 0
    for u in usages:
        if not u.has_tokens:
            continue
        with_tokens += 1
        sc = cost_span(u, model)
        for k in sums:
            sums[k] += getattr(sc, k)
    total = sum(sums.values())
    return ModelCostBreakdown(
        model_key=model.model_key,
        provider=model.provider,
        total_cost=total,
        **sums,
        spans_with_tokens=with_tokens,
        spans_total=len(usages),
        spans_skipped=len(usages) - with_tokens,
    )


@dataclasses.dataclass
class BreakdownReport:
    """A side-by-side cost breakdown of N candidate models against one
    workload."""

    profile: WorkloadProfile
    models: List[ModelCostBreakdown]
    unknown_models: List[str]
    files_read: int

    def ranked(self) -> List[ModelCostBreakdown]:
        return sorted(self.models, key=lambda m: m.total_cost)

    def find(self, model_key: str) -> Optional[ModelCostBreakdown]:
        for m in self.models:
            if m.model_key == model_key:
                return m
        return None


def breakdown_workload(
    paths: Union[PathLike, Sequence[PathLike]],
    *,
    candidates: Sequence[str],
    pricing: Optional[PricingProvider] = None,
    warn_stream: Optional[TextIO] = None,
) -> BreakdownReport:
    """Read span logs and compute a per-model cost breakdown for each
    candidate. Unlike ``evaluate_workload``, every model is reported
    with a real cost — there is no compatibility filter.
    """
    if isinstance(paths, (str, os.PathLike)):
        seq: List[PathLike] = [paths]
    else:
        seq = list(paths)
    if pricing is None:
        from .pricing import default_chain

        pricing = default_chain()
    sink = warn_stream if warn_stream is not None else sys.stderr

    profile, _ = build_workload_profile(seq, pricing, auto_infer_capabilities=False)
    all_usages = _read_paths(seq)

    unknown: List[str] = []
    resolved: dict = {}
    for key in candidates:
        d = pricing.lookup(key)
        if d is None:
            unknown.append(key)
        else:
            resolved[key] = d

    for k in sorted(unknown):
        sink.write(
            f"[neatlogs-cost] warning: candidate model {k!r} not in pricing catalog; "
            f"it will be omitted from the report. "
            f"Override via --pricing-file or update "
            f"neatlogs/config/pricing.json.\n"
        )

    breakdowns = [_aggregate_costs(all_usages, d) for d in resolved.values()]
    return BreakdownReport(
        profile=profile,
        models=breakdowns,
        unknown_models=sorted(unknown),
        files_read=profile.files_read,
    )
