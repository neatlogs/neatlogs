"""
LLM cost intelligence engine for neatlogs span JSONL logs.

Given a past span log and a list of candidate ``provider/model`` keys,
answer the questions a finance or engineering team actually asks:

* "What did this workload cost on each model, broken down by input /
  output / cache / reasoning?" (``breakdown_workload``)
* "Which candidate is the cheapest one that can still serve the same
  workload?" (``evaluate_workload``)
* "What will my monthly bill be at a given call volume?" (``forecast``)

All three share one schema and one CLI (``neatlogs-cost``). Pricing is
decoupled from the engine via a ``PricingProvider`` chain so users can
override rates per-environment, fetch a remote catalog, or plug in
LiteLLM's mirror without touching core code.

CLI::

    neatlogs-cost spans.log \\
        --candidates openai/gpt-4o-mini,openai/gpt-4o,anthropic/claude-3-5-haiku-latest

    neatlogs-cost spans.log --candidates openai/gpt-4o-mini,openai/gpt-4o --breakdown

    neatlogs-cost --forecast --model openai/gpt-4o-mini \\
        --monthly-calls 50000 --avg-prompt 2000 --avg-completion 500 --cache-hit-rate 0.5

Programmatic::

    from neatlogs.cost import (
        evaluate_workload, breakdown_workload, forecast,
        BuiltinProvider, CustomProvider, ChainProvider, WorkloadConstraints,
        format_evaluation, format_breakdown, format_forecast,
    )
"""

from .breakdown import (
    BreakdownReport,
    ModelCostBreakdown,
    SpanCost,
    breakdown_workload,
    cost_span,
)
from .forecast import ForecastReport, forecast
from .formatters import (
    format_breakdown,
    format_breakdown_csv,
    format_breakdown_json,
    format_breakdown_text,
    format_evaluation,
    format_evaluation_csv,
    format_evaluation_json,
    format_evaluation_text,
    format_forecast,
    format_forecast_json,
    format_forecast_text,
    format_pricing_list,
    format_pricing_list_csv,
    format_pricing_list_json,
    format_pricing_list_text,
    format_pricing_show,
    format_pricing_show_json,
    format_pricing_show_text,
)
from .pricing import (
    BuiltinProvider,
    Capability,
    ChainProvider,
    CustomProvider,
    ModelDefinition,
    PricingProvider,
    Tier,
    UsageType,
    default_chain,
)
from .ranking import (
    EvaluationReport,
    ScoredModel,
    SpanVerdict,
    evaluate_workload,
)
from .spans import SpanUsage
from .workload import (
    TokenStats,
    WorkloadConstraints,
    WorkloadProfile,
    build_workload_profile,
)

__all__ = [
    # Spans
    "SpanUsage",
    # Pricing
    "Capability",
    "UsageType",
    "Tier",
    "ModelDefinition",
    "PricingProvider",
    "BuiltinProvider",
    "CustomProvider",
    "ChainProvider",
    "default_chain",
    # Workload
    "TokenStats",
    "WorkloadProfile",
    "WorkloadConstraints",
    "build_workload_profile",
    # Breakdown
    "SpanCost",
    "ModelCostBreakdown",
    "BreakdownReport",
    "cost_span",
    "breakdown_workload",
    # Ranking
    "SpanVerdict",
    "ScoredModel",
    "EvaluationReport",
    "evaluate_workload",
    # Forecast
    "ForecastReport",
    "forecast",
    # Formatters
    "format_evaluation",
    "format_evaluation_text",
    "format_evaluation_json",
    "format_evaluation_csv",
    "format_breakdown",
    "format_breakdown_text",
    "format_breakdown_json",
    "format_breakdown_csv",
    "format_forecast",
    "format_forecast_text",
    "format_forecast_json",
    "format_pricing_list",
    "format_pricing_list_text",
    "format_pricing_list_json",
    "format_pricing_list_csv",
    "format_pricing_show",
    "format_pricing_show_text",
    "format_pricing_show_json",
]
