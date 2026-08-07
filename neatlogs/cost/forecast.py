"""Monthly / annual cost projection for a given traffic pattern.

The forecast treats the model as a single point estimate: every call
is assumed to use the same average prompt and completion token counts.
For more realistic estimates, vary the inputs across a range and
aggregate.

Notes are added when a requested feature (cache or reasoning) is not
supported by the model, so the user can see why a portion of the
forecast is $0.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional

from .pricing import PricingProvider, UsageType


@dataclasses.dataclass
class ForecastReport:
    """Monthly / annual cost projection for a given traffic pattern."""

    model_key: str
    monthly_calls: int
    avg_prompt_tokens: int
    avg_completion_tokens: int
    cache_hit_rate: float
    reasoning_per_call: int
    per_call_cost: float
    monthly_cost: float
    annual_cost: float
    input_cost: float
    output_cost: float
    cache_cost: float
    reasoning_cost: float
    notes: List[str] = dataclasses.field(default_factory=list)


def forecast(
    *,
    model_key: str,
    monthly_calls: int,
    avg_prompt_tokens: int = 0,
    avg_completion_tokens: int = 0,
    cache_hit_rate: float = 0.0,
    reasoning_per_call: int = 0,
    pricing: Optional[PricingProvider] = None,
) -> ForecastReport:
    """Project monthly and annual cost for a given traffic pattern.

    ``cache_hit_rate`` is the fraction of prompt tokens served from
    cache (0.0–1.0). When the model does not support prompt caching
    and ``cache_hit_rate > 0``, the cache portion contributes $0 and a
    note is added.

    ``reasoning_per_call`` is the average reasoning tokens per call.
    When the model does not support reasoning and ``reasoning_per_call > 0``,
    the reasoning portion contributes $0 and a note is added.
    """
    if pricing is None:
        from .pricing import default_chain

        pricing = default_chain()
    d = pricing.lookup(model_key)
    if d is None:
        raise ValueError(f"unknown model: {model_key!r}")
    notes: List[str] = []
    cache_hit_rate = max(0.0, min(1.0, float(cache_hit_rate)))
    hit_prompt = int(avg_prompt_tokens * cache_hit_rate)
    miss_prompt = avg_prompt_tokens - hit_prompt

    if cache_hit_rate > 0 and (
        UsageType.CACHE_READ not in d.usage_types and UsageType.CACHE_WRITE not in d.usage_types
    ):
        notes.append(
            f"cache_hit_rate={cache_hit_rate} requested but model does not support prompt caching; cache cost set to 0"
        )
    if reasoning_per_call > 0 and UsageType.REASONING not in d.usage_types:
        notes.append(
            f"reasoning_per_call={reasoning_per_call} requested but model is not a reasoning model; reasoning cost set to 0"
        )

    in_rate = d.effective_rate(UsageType.INPUT, miss_prompt)
    out_rate = d.effective_rate(UsageType.OUTPUT, avg_completion_tokens)
    input_cost = (miss_prompt / 1_000_000) * in_rate if in_rate is not None else 0.0
    output_cost = (avg_completion_tokens / 1_000_000) * out_rate if out_rate is not None else 0.0

    cache_cost = 0.0
    if hit_prompt > 0:
        cr = d.effective_rate(UsageType.CACHE_READ, hit_prompt)
        if cr is not None:
            cache_cost = (hit_prompt / 1_000_000) * cr
        else:
            cw = d.effective_rate(UsageType.CACHE_WRITE, hit_prompt)
            if cw is not None:
                cache_cost = (hit_prompt / 1_000_000) * cw

    reasoning_cost = 0.0
    if reasoning_per_call > 0:
        rr = d.effective_rate(UsageType.REASONING, reasoning_per_call)
        if rr is not None:
            reasoning_cost = (reasoning_per_call / 1_000_000) * rr

    per_call = input_cost + output_cost + cache_cost + reasoning_cost
    monthly = per_call * monthly_calls
    return ForecastReport(
        model_key=model_key,
        monthly_calls=monthly_calls,
        avg_prompt_tokens=avg_prompt_tokens,
        avg_completion_tokens=avg_completion_tokens,
        cache_hit_rate=cache_hit_rate,
        reasoning_per_call=reasoning_per_call,
        per_call_cost=per_call,
        monthly_cost=monthly,
        annual_cost=monthly * 12,
        input_cost=input_cost,
        output_cost=output_cost,
        cache_cost=cache_cost,
        reasoning_cost=reasoning_cost,
        notes=notes,
    )
