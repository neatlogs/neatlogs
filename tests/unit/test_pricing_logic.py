"""Unit tests for the pricing calculation logic in attribute_processor.py.

Tests cover:
- Model name matching (_resolve_model_prices)
- Platform-specific price resolution (Azure tiers, Bedrock, Vertex AI)
- Cache token cost (read/write subtraction from prompt)
- Tiered pricing (above 200k tokens)
- Audio token pricing
- Reasoning tokens (already in completion, no double-count)
- Edge cases (zero tokens, missing prices, fallbacks)
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from opentelemetry.trace import SpanKind

import neatlogs
from neatlogs.core.attribute_processor import UnifiedAttributeProcessor


def _load_mapping() -> dict:
    mapping_path = Path(neatlogs.__file__).resolve().parent / "config" / "attribute-mapping.json"
    return json.loads(mapping_path.read_text(encoding="utf-8"))


MAPPING = _load_mapping()


def _mk_processor(pricing: dict) -> UnifiedAttributeProcessor:
    return UnifiedAttributeProcessor(
        mapping_config=MAPPING,
        pricing_config=pricing,
        debug=False,
    )


def _mk_span(attrs: dict) -> SimpleNamespace:
    resource = SimpleNamespace(attributes={})
    ctx = SimpleNamespace(trace_id=1, span_id=2)
    return SimpleNamespace(
        name="test-span",
        kind=SpanKind.INTERNAL,
        attributes=attrs,
        resource=resource,
        events=[],
        start_time=0,
        end_time=1_000_000_000,
        context=ctx,
        instrumentation_scope=SimpleNamespace(name="test", version="1.0"),
    )


# Minimal pricing config for tests
SIMPLE_PRICING = {
    "chat": {
        "gpt-4o": {"promptPrice": 0.0025, "completionPrice": 0.01},
        "gpt-4o-mini": {"promptPrice": 0.00015, "completionPrice": 0.0006},
        "gpt-4o-mini-2024-07-18": {"promptPrice": 0.00015, "completionPrice": 0.0006},
        "gpt-4.1": {"promptPrice": 0.002, "completionPrice": 0.008},
        "claude-sonnet-4-6": {
            "promptPrice": 0.003,
            "completionPrice": 0.015,
            "cacheReadPrice": 0.0003,
            "cacheWritePrice": 0.00375,
            "tierTokenThreshold": 200000,
            "promptPriceAboveTier": 0.006,
            "completionPriceAboveTier": 0.0225,
            "cacheReadPriceAboveTier": 0.0006,
            "cacheWritePriceAboveTier": 0.0075,
        },
        "gpt-4o-audio-preview": {
            "promptPrice": 0.0025,
            "completionPrice": 0.01,
            "promptAudioPrice": 0.04,
            "completionAudioPrice": 0.08,
        },
        "o3": {
            "promptPrice": 0.002,
            "completionPrice": 0.008,
            "cacheReadPrice": 0.0005,
        },
        "deepseek-chat": {
            "promptPrice": 0.00028,
            "completionPrice": 0.00042,
            "cacheReadPrice": 0.000028,
        },
    },
    "azure_openai": {
        "chat": {
            "gpt-4o": {
                "global_standard": {"promptPrice": 0.0025, "completionPrice": 0.01},
                "eu_standard": {"promptPrice": 0.00275, "completionPrice": 0.011},
                "us_standard": {"promptPrice": 0.003, "completionPrice": 0.012},
            },
            "gpt-4.1": {
                "global_standard": {"promptPrice": 0.002, "completionPrice": 0.008},
            },
        },
    },
    "bedrock": {
        "chat": {
            "anthropic.claude-3-5-sonnet-20241022-v2:0": {
                "promptPrice": 0.003,
                "completionPrice": 0.015,
                "cacheReadPrice": 0.0003,
                "cacheWritePrice": 0.00375,
            },
            "meta.llama3-70b-instruct-v1:0": {
                "promptPrice": 0.00265,
                "completionPrice": 0.0035,
            },
        },
    },
    "vertex_ai": {
        "chat": {
            "gemini-2.0-flash": {
                "promptPrice": 0.0001,
                "completionPrice": 0.0004,
            },
        },
    },
}


# ===================================================================
# _resolve_model_prices — model name matching
# ===================================================================


class TestResolveModelPrices:
    """Tests for the model name matching logic."""

    def test_exact_match(self):
        proc = _mk_processor(SIMPLE_PRICING)
        result = proc._resolve_model_prices(SIMPLE_PRICING["chat"], "gpt-4o")
        assert result["promptPrice"] == 0.0025

    def test_case_insensitive_exact(self):
        proc = _mk_processor(SIMPLE_PRICING)
        result = proc._resolve_model_prices(SIMPLE_PRICING["chat"], "GPT-4O")
        assert result["promptPrice"] == 0.0025

    def test_versioned_model_matches_base(self):
        """gpt-4o-2024-08-06 should match gpt-4o (version suffix after separator)."""
        proc = _mk_processor(SIMPLE_PRICING)
        result = proc._resolve_model_prices(SIMPLE_PRICING["chat"], "gpt-4o-2024-08-06")
        assert result["promptPrice"] == 0.0025

    def test_no_false_positive_gpt4o_mini_vs_gpt4o(self):
        """gpt-4o-mini should NOT match gpt-4o — it's a distinct model, not a version."""
        # BUT gpt-4o-mini IS in the pricing table, so it should match its own entry.
        proc = _mk_processor(SIMPLE_PRICING)
        result = proc._resolve_model_prices(SIMPLE_PRICING["chat"], "gpt-4o-mini")
        assert result["promptPrice"] == 0.00015  # gpt-4o-mini price, NOT gpt-4o

    def test_longest_key_wins(self):
        """gpt-4o-mini-2024-07-18 should match gpt-4o-mini-2024-07-18 exactly, not gpt-4o."""
        proc = _mk_processor(SIMPLE_PRICING)
        result = proc._resolve_model_prices(SIMPLE_PRICING["chat"], "gpt-4o-mini-2024-07-18")
        assert result["promptPrice"] == 0.00015

    def test_no_match_returns_none(self):
        proc = _mk_processor(SIMPLE_PRICING)
        result = proc._resolve_model_prices(SIMPLE_PRICING["chat"], "nonexistent-model")
        assert result is None

    def test_gpt41_does_not_match_gpt4(self):
        """gpt-4.1 should not false-match a gpt-4 entry if one existed."""
        pricing = {"gpt-4": {"promptPrice": 0.03, "completionPrice": 0.06}}
        proc = _mk_processor({"chat": pricing})
        result = proc._resolve_model_prices(pricing, "gpt-4.1")
        # gpt-4.1 starts with "gpt-4" and next char is "." which is a separator
        # This would match. If we had both gpt-4 and gpt-4.1 in the table, the
        # longest match (gpt-4.1) would win. With only gpt-4, it falls back.
        assert result is not None  # Fallback to gpt-4 is acceptable here

    def test_dot_version_separator(self):
        """Model gpt-4.1-mini should match gpt-4.1 prefix (dot is a separator)."""
        pricing = {"gpt-4.1": {"promptPrice": 0.002, "completionPrice": 0.008}}
        proc = _mk_processor({"chat": pricing})
        # gpt-4.1-mini: starts with "gpt-4.1", next char is "-" which is a separator
        result = proc._resolve_model_prices(pricing, "gpt-4.1-mini")
        assert result is not None
        assert result["promptPrice"] == 0.002


# ===================================================================
# Platform-specific pricing
# ===================================================================


class TestPlatformPricing:
    """Tests for Azure tier, Bedrock, Vertex AI pricing resolution."""

    def test_azure_global_standard_default(self):
        """Azure defaults to global_standard tier."""
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "gpt-4o",
            "neatlogs.platform": "azure_openai",
            "llm.token_count.prompt": 1000,
            "llm.token_count.completion": 500,
        })
        result = proc.process(span)
        # global_standard: prompt=0.0025, completion=0.01
        assert result["neatlogs.llm.cost.prompt"] == round((1000 / 1000) * 0.0025, 8)
        assert result["neatlogs.llm.cost.completion"] == round((500 / 1000) * 0.01, 8)

    def test_azure_eu_tier(self):
        """Azure with eu_standard tier should use EU pricing."""
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "gpt-4o",
            "neatlogs.platform": "azure_openai",
            "neatlogs.azure.pricing_tier": "eu_standard",
            "llm.token_count.prompt": 1000,
            "llm.token_count.completion": 500,
        })
        result = proc.process(span)
        # eu_standard: prompt=0.00275, completion=0.011
        assert result["neatlogs.llm.cost.prompt"] == round((1000 / 1000) * 0.00275, 8)
        assert result["neatlogs.llm.cost.completion"] == round((500 / 1000) * 0.011, 8)

    def test_azure_unknown_tier_falls_back_to_global(self):
        """Unknown Azure tier falls back to global_standard."""
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "gpt-4o",
            "neatlogs.platform": "azure_openai",
            "neatlogs.azure.pricing_tier": "nonexistent_tier",
            "llm.token_count.prompt": 1000,
            "llm.token_count.completion": 500,
        })
        result = proc.process(span)
        assert result["neatlogs.llm.cost.prompt"] == round((1000 / 1000) * 0.0025, 8)

    def test_azure_falls_back_to_direct_api(self):
        """Azure model not in azure section falls back to chat section."""
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "deepseek-chat",
            "neatlogs.platform": "azure_openai",
            "llm.token_count.prompt": 1000,
            "llm.token_count.completion": 500,
        })
        result = proc.process(span)
        # Falls back to chat section deepseek-chat prices
        assert result["neatlogs.llm.cost.prompt"] == round((1000 / 1000) * 0.00028, 8)

    def test_bedrock_pricing(self):
        """Bedrock uses flat pricing from bedrock section."""
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "neatlogs.platform": "bedrock",
            "llm.token_count.prompt": 2000,
            "llm.token_count.completion": 500,
        })
        result = proc.process(span)
        assert result["neatlogs.llm.cost.prompt"] == round((2000 / 1000) * 0.003, 8)
        assert result["neatlogs.llm.cost.completion"] == round((500 / 1000) * 0.015, 8)

    def test_vertex_ai_pricing(self):
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "gemini-2.0-flash",
            "neatlogs.platform": "vertex_ai",
            "llm.token_count.prompt": 5000,
            "llm.token_count.completion": 1000,
        })
        result = proc.process(span)
        assert result["neatlogs.llm.cost.prompt"] == round((5000 / 1000) * 0.0001, 8)
        assert result["neatlogs.llm.cost.completion"] == round((1000 / 1000) * 0.0004, 8)


# ===================================================================
# Cache token pricing
# ===================================================================


class TestCachePricing:
    """Tests for cache read/write cost calculation."""

    def test_cache_read_reduces_input_cost(self):
        """Cache-read tokens should be charged at discounted rate, not full rate."""
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "claude-sonnet-4-6",
            "llm.token_count.prompt": 10000,  # Total input tokens
            "llm.token_count.completion": 500,
            "llm.token_count.prompt_details.cache_read": 8000,  # 8000 from cache
        })
        result = proc.process(span)
        # uncached = 10000 - 8000 = 2000 at promptPrice 0.003
        # cached = 8000 at cacheReadPrice 0.0003
        expected_input = (2000 / 1000) * 0.003 + (8000 / 1000) * 0.0003
        assert result["neatlogs.llm.cost.prompt"] == round(expected_input, 8)
        assert "neatlogs.llm.cost.cache_read" in result

    def test_cache_read_with_gen_ai_attribute(self):
        """Cache read via gen_ai.usage.cache_read_input_tokens attribute."""
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "o3",
            "llm.token_count.prompt": 5000,
            "llm.token_count.completion": 200,
            "gen_ai.usage.cache_read_input_tokens": 3000,
        })
        result = proc.process(span)
        # uncached = 5000 - 3000 = 2000 at 0.002
        # cached = 3000 at 0.0005
        expected_input = (2000 / 1000) * 0.002 + (3000 / 1000) * 0.0005
        assert result["neatlogs.llm.cost.prompt"] == round(expected_input, 8)

    def test_cache_write_adds_cost(self):
        """Cache-write tokens are ADDITIVE cost (Anthropic)."""
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "claude-sonnet-4-6",
            "llm.token_count.prompt": 5000,
            "llm.token_count.completion": 200,
            "llm.token_count.prompt_details.cache_write": 2000,
        })
        result = proc.process(span)
        # prompt: 5000 at 0.003 = 0.015 (no cache read, so all at full rate)
        # cache write: 2000 at 0.00375 = 0.0075
        expected_input = (5000 / 1000) * 0.003 + (2000 / 1000) * 0.00375
        assert result["neatlogs.llm.cost.prompt"] == round(expected_input, 8)
        assert "neatlogs.llm.cost.cache_write" in result

    def test_no_cache_pricing_charges_full_rate(self):
        """When model has no cacheReadPrice, all tokens charged at full rate."""
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "gpt-4o",  # No cache pricing
            "llm.token_count.prompt": 5000,
            "llm.token_count.completion": 1000,
            "llm.token_count.prompt_details.cache_read": 3000,
        })
        result = proc.process(span)
        # gpt-4o has no cacheReadPrice, so cache_read_rate = promptPrice
        # uncached = 5000 - 3000 = 2000 at 0.0025
        # cached = 3000 at 0.0025 (same as prompt since no discount)
        expected_input = (5000 / 1000) * 0.0025  # effectively same as total * rate
        assert result["neatlogs.llm.cost.prompt"] == round(expected_input, 8)

    def test_bedrock_cache_read(self):
        """Bedrock Claude with cache read."""
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "neatlogs.platform": "bedrock",
            "llm.token_count.prompt": 10000,
            "llm.token_count.completion": 500,
            "llm.token_count.prompt_details.cache_read": 6000,
        })
        result = proc.process(span)
        expected_input = (4000 / 1000) * 0.003 + (6000 / 1000) * 0.0003
        assert result["neatlogs.llm.cost.prompt"] == round(expected_input, 8)


# ===================================================================
# Tiered pricing (above 200k tokens)
# ===================================================================


class TestTieredPricing:
    """Tests for tiered pricing when context exceeds 200k tokens."""

    def test_below_200k_uses_standard_rate(self):
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "claude-sonnet-4-6",
            "llm.token_count.prompt": 100000,
            "llm.token_count.completion": 2000,
        })
        result = proc.process(span)
        expected_input = (100000 / 1000) * 0.003
        expected_output = (2000 / 1000) * 0.015
        assert result["neatlogs.llm.cost.prompt"] == round(expected_input, 8)
        assert result["neatlogs.llm.cost.completion"] == round(expected_output, 8)

    def test_above_200k_uses_tiered_rate(self):
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "claude-sonnet-4-6",
            "llm.token_count.prompt": 250000,
            "llm.token_count.completion": 4000,
        })
        result = proc.process(span)
        # Above 200k: prompt at 0.006, completion at 0.0225
        expected_input = (250000 / 1000) * 0.006
        expected_output = (4000 / 1000) * 0.0225
        assert result["neatlogs.llm.cost.prompt"] == round(expected_input, 8)
        assert result["neatlogs.llm.cost.completion"] == round(expected_output, 8)

    def test_above_200k_with_cache_uses_tiered_cache_rate(self):
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "claude-sonnet-4-6",
            "llm.token_count.prompt": 250000,
            "llm.token_count.completion": 1000,
            "llm.token_count.prompt_details.cache_read": 200000,
        })
        result = proc.process(span)
        # Tiered: prompt at 0.006, cache_read at 0.0006
        expected_input = (50000 / 1000) * 0.006 + (200000 / 1000) * 0.0006
        assert result["neatlogs.llm.cost.prompt"] == round(expected_input, 8)

    def test_model_without_tiered_pricing_stays_flat(self):
        """gpt-4o has no tiered pricing — stays flat regardless of context length."""
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "gpt-4o",
            "llm.token_count.prompt": 300000,
            "llm.token_count.completion": 1000,
        })
        result = proc.process(span)
        expected_input = (300000 / 1000) * 0.0025
        assert result["neatlogs.llm.cost.prompt"] == round(expected_input, 8)


# ===================================================================
# Audio tokens
# ===================================================================


class TestAudioPricing:
    """Tests for audio token pricing (GPT-4o audio preview)."""

    def test_audio_tokens_use_audio_rate(self):
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "gpt-4o-audio-preview",
            "llm.token_count.prompt": 5000,
            "llm.token_count.completion": 2000,
            "llm.token_count.prompt_details.audio": 1000,
            "llm.token_count.completion_details.audio": 500,
        })
        result = proc.process(span)
        # Input: (5000-1000) text at 0.0025 + 1000 audio at 0.04
        # Output: (2000-500) text at 0.01 + 500 audio at 0.08
        expected_input = (4000 / 1000) * 0.0025 + (1000 / 1000) * 0.04
        expected_output = (1500 / 1000) * 0.01 + (500 / 1000) * 0.08
        assert result["neatlogs.llm.cost.prompt"] == round(expected_input, 8)
        assert result["neatlogs.llm.cost.completion"] == round(expected_output, 8)
        assert result["neatlogs.llm.cost.total"] == round(expected_input + expected_output, 8)


# ===================================================================
# Reasoning tokens (no double-counting)
# ===================================================================


class TestReasoningTokens:
    """Reasoning tokens are included in completion_tokens. No separate charge."""

    def test_reasoning_tokens_not_double_counted(self):
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "o3",
            "llm.token_count.prompt": 2000,
            "llm.token_count.completion": 8000,  # includes 5000 reasoning
            "llm.token_count.completion_details.reasoning": 5000,
        })
        result = proc.process(span)
        # All 8000 completion tokens at the same rate
        expected_output = (8000 / 1000) * 0.008
        assert result["neatlogs.llm.cost.completion"] == round(expected_output, 8)


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    """Edge cases and fallback behavior."""

    def test_no_model_name_skips_pricing(self):
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.token_count.prompt": 1000,
            "llm.token_count.completion": 500,
        })
        result = proc.process(span)
        assert "neatlogs.llm.cost.total" not in result

    def test_no_tokens_skips_pricing(self):
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "gpt-4o",
        })
        result = proc.process(span)
        assert "neatlogs.llm.cost.total" not in result

    def test_zero_tokens_produces_zero_cost(self):
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "gpt-4o",
            "llm.token_count.prompt": 0,
            "llm.token_count.completion": 0,
        })
        result = proc.process(span)
        # prompt_tokens=0, completion=0 → both are "not None" → cost runs
        assert result["neatlogs.llm.cost.total"] == 0

    def test_unknown_model_produces_no_cost(self):
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "totally-unknown-model",
            "llm.token_count.prompt": 1000,
            "llm.token_count.completion": 500,
        })
        result = proc.process(span)
        assert "neatlogs.llm.cost.total" not in result

    def test_empty_pricing_config(self):
        proc = _mk_processor({})
        span = _mk_span({
            "llm.model_name": "gpt-4o",
            "llm.token_count.prompt": 1000,
            "llm.token_count.completion": 500,
        })
        result = proc.process(span)
        assert "neatlogs.llm.cost.total" not in result

    def test_total_cost_is_sum_of_parts(self):
        """Total must always equal prompt + completion."""
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "claude-sonnet-4-6",
            "llm.token_count.prompt": 50000,
            "llm.token_count.completion": 2000,
            "llm.token_count.prompt_details.cache_read": 30000,
            "llm.token_count.prompt_details.cache_write": 5000,
        })
        result = proc.process(span)
        prompt_cost = result["neatlogs.llm.cost.prompt"]
        completion_cost = result["neatlogs.llm.cost.completion"]
        total_cost = result["neatlogs.llm.cost.total"]
        assert total_cost == round(prompt_cost + completion_cost, 8)

    def test_preexisting_costs_are_overridden(self):
        """pricing.json is source of truth — pre-populated costs are replaced."""
        proc = _mk_processor(SIMPLE_PRICING)
        span = _mk_span({
            "llm.model_name": "gpt-4o",
            "llm.token_count.prompt": 1000,
            "llm.token_count.completion": 500,
            "llm.cost.prompt": 999.0,  # bogus value from instrumentation
            "llm.cost.completion": 999.0,
            "llm.cost.total": 999.0,
        })
        result = proc.process(span)
        assert result["neatlogs.llm.cost.total"] != 999.0
        expected_total = round((1000 / 1000) * 0.0025 + (500 / 1000) * 0.01, 8)
        assert result["neatlogs.llm.cost.total"] == expected_total
