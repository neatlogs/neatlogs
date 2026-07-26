"""What-if model ranking with capability scoring.

Given a span log and a list of candidate models, score every candidate
against the workload and return a report ranked by cost under
capability / context / compatibility constraints.

A span is "compatible" with a candidate when:

* the model's ``context_window`` is large enough (when constrained), AND
* the model declares every capability the workload needs, AND
* the span didn't use a feature the model doesn't bill (cache on a
  no-cache model, reasoning on a non-reasoning model).

The candidate's score is the sum of compatible-span costs and the
``compatibility_pct`` is the fraction of spans it can serve. Ranking
is: baseline first, then compatible by cost (cheap first), then
incompatible last.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from typing import Dict, List, Optional, Sequence, Set, TextIO, Tuple, Union

from .breakdown import cost_span
from .pricing import ModelDefinition, PricingProvider, UsageType
from .spans import PathLike, SpanUsage
from .workload import WorkloadConstraints, WorkloadProfile, build_workload_profile


@dataclasses.dataclass
class SpanVerdict:
    """One span's compatibility with one candidate model."""

    span_id: str
    compatible: bool
    reasons: List[str] = dataclasses.field(default_factory=list)
    cost: float = 0.0


def _is_span_compatible(
    usage: SpanUsage, model: ModelDefinition, constraints: WorkloadConstraints
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    compatible = True
    if constraints.min_context_window > 0:
        if model.context_window is None:
            compatible = False
            reasons.append(
                f"min context {constraints.min_context_window:,} required; "
                f"model has no declared context"
            )
        elif model.context_window < usage.input_total:
            compatible = False
            reasons.append(f"prompt {usage.input_total:,} > context {model.context_window:,}")
    missing = model.missing_capabilities(constraints.need_capabilities)
    if missing:
        compatible = False
        reasons.append(f"missing {sorted(missing)}")
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
    per_span: List[SpanVerdict] = []
    for u in usages:
        ok, reasons = _is_span_compatible(u, model, constraints)
        cost = cost_span(u, model).total if ok else 0.0
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

    ``candidates``: ``provider/model`` keys to evaluate.
    ``baseline``: explicit baseline. Defaults to the first candidate.
    ``pricing``: override the default chain. ``None`` builds the default.
    ``constraints``: capability / context-window filters. ``None`` = no filter.
    ``auto_infer_capabilities``: when True (default), capability
        requirements are inferred from the source models in the
        workload. When False, only ``constraints.need_capabilities``
        applies.
    """
    if isinstance(paths, (str, os.PathLike)):
        seq: List[PathLike] = [paths]
    else:
        seq = list(paths)
    if pricing is None:
        from .pricing import default_chain

        pricing = default_chain()
    if constraints is None:
        constraints = WorkloadConstraints()
    sink = warn_stream if warn_stream is not None else sys.stderr

    profile, usages = build_workload_profile(
        seq, pricing, auto_infer_capabilities=auto_infer_capabilities
    )
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

    for k in sorted(unknown):
        sink.write(
            f"[neatlogs-cost] warning: candidate model {k!r} not in pricing catalog; "
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
