"""Workload profile and constraints.

A ``WorkloadProfile`` is derived once from a span log and reused by
both the breakdown and the ranking evaluators. The same percentile
stats power the per-model breakdown and the what-if ranking tables.

``WorkloadConstraints`` declares what a candidate model must satisfy
to be considered. The constraint object is the same in both
``breakdown_workload`` and ``evaluate_workload``; the breakdown mode
ignores the capability / context-window filters (every model gets a
real cost row, no compatibility filter).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Sequence, Set, Tuple

from .pricing import PricingProvider
from .spans import PathLike, SpanUsage, _read_paths


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
    """Summary of a span log, derived once and reused by both the
    breakdown and the evaluator."""

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


def _resolve_definition(usage: SpanUsage, pricing: PricingProvider):
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
    the workload needs from the source models in the log. The user can
    override via ``WorkloadConstraints.need_capabilities``.
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
        out: List[str] = []
        if self.need_capabilities:
            out.append(f"need capabilities: {sorted(self.need_capabilities)}")
        if self.min_context_window > 0:
            out.append(f"min context: {self.min_context_window:,} tokens")
        if self.min_compatibility_pct > 0:
            out.append(f"min compatibility: {self.min_compatibility_pct * 100:.0f}%")
        return out
