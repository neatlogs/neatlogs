"""
Unit tests for neatlogs.cost.

The cost module is a what-if comparison engine. Given a span log written
by the SDK (NEATLOGS_LOG_SPANS=true or NEATLOGS_LOG_RAW_SPANS=true) and a
list of candidate models, it computes what each model would have cost
for the same workload, surfaces capability mismatches, and renders the
result as text / json / csv.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from neatlogs.cost import (
    ComparisonReport,
    ForecastReport,
    ModelComparison,
    PriceCard,
    PricingCatalog,
    SpanCost,
    SpanUsage,
    _capability_dict,
    _detect_source,
    _extract_usage,
    _format_comparison_csv,
    _format_comparison_json,
    _format_comparison_text,
    _int_attr,
    _iter_json_objects,
    _load_catalog,
    _maybe_float,
    _maybe_int,
    _model_key_for,
    _read_paths_usages,
    _read_usages,
    _tier_for,
    compare,
    cost_span,
    forecast,
    format_comparison,
    format_forecast,
)

# ---------------------------------------------------------------------------
# Test pricing catalog
# ---------------------------------------------------------------------------


CATALOG_DICT: Dict[str, Any] = {
    "_meta": {"currency": "USD", "units": "per_1m_tokens", "schema_version": "1.0"},
    "models": {
        "openai/gpt-4o-mini": {
            "provider": "openai",
            "input_per_1m": 0.15,
            "output_per_1m": 0.60,
            "cache_read_per_1m": 0.075,
            "supports_prompt_cache": True,
            "supports_vision": True,
            "supports_tools": True,
            "context_window": 128000,
        },
        "openai/gpt-4o": {
            "provider": "openai",
            "input_per_1m": 2.50,
            "output_per_1m": 10.00,
            "cache_read_per_1m": 1.25,
            "supports_prompt_cache": True,
            "supports_vision": True,
            "supports_tools": True,
            "context_window": 128000,
        },
        "openai/o3-mini": {
            "provider": "openai",
            "input_per_1m": 1.10,
            "output_per_1m": 4.40,
            "reasoning_output_per_1m": 4.40,
            "cache_read_per_1m": 0.55,
            "supports_prompt_cache": True,
            "supports_reasoning": True,
            "supports_tools": True,
            "context_window": 200000,
        },
        "anthropic/claude-3-5-haiku-latest": {
            "provider": "anthropic",
            "input_per_1m": 0.80,
            "output_per_1m": 4.00,
            "cache_write_per_1m": 1.00,
            "cache_read_per_1m": 0.08,
            "supports_prompt_cache": True,
            "supports_tools": True,
            "context_window": 200000,
        },
        "anthropic/claude-3-5-sonnet-latest": {
            "provider": "anthropic",
            "input_per_1m": 3.00,
            "output_per_1m": 15.00,
            "cache_write_per_1m": 3.75,
            "cache_read_per_1m": 0.30,
            "supports_prompt_cache": True,
            "supports_vision": True,
            "supports_tools": True,
            "context_window": 200000,
            "tiers": {
                "input_above_200k_per_1m": 6.00,
                "output_above_200k_per_1m": 22.50,
            },
        },
    },
}


def _catalog() -> PricingCatalog:
    return _load_catalog_from_dict(CATALOG_DICT)


def _load_catalog_from_dict(d: Dict[str, Any]) -> PricingCatalog:
    """Bypass the file system: build a PricingCatalog from an in-memory dict.

    This mirrors _load_catalog but without touching the disk, so each test
    can use a self-contained catalog without writing temp files.
    """
    from neatlogs.cost import DEFAULT_PRICING_PATH, _load_catalog

    # Write to a temp file and load it.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(d, f)
        path = f.name
    try:
        return _load_catalog(path)
    finally:
        Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Span reading + extraction
# ---------------------------------------------------------------------------


class TestIterJsonObjects:
    def test_single_object(self):
        out = list(_iter_json_objects('{"a": 1, "b": 2}'))
        assert out == [{"a": 1, "b": 2}]

    def test_multiple_objects(self):
        out = list(_iter_json_objects('{"a": 1}\n{"b": 2}\n{"c": 3}'))
        assert out == [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_nested_braces_in_strings(self):
        # The brace-balancer must NOT count braces inside string values.
        text = '{"name": "a {b} c", "value": 1}\n{"name": "x", "value": 2}'
        out = list(_iter_json_objects(text))
        assert len(out) == 2
        assert out[0]["name"] == "a {b} c"
        assert out[1]["name"] == "x"

    def test_escaped_quotes(self):
        text = '{"name": "a \\"quoted\\" word"}'
        out = list(_iter_json_objects(text))
        assert len(out) == 1
        assert out[0]["name"] == 'a "quoted" word'

    def test_malformed_object_skipped(self):
        text = '{"a": 1}\n{not valid}\n{"b": 2}'
        out = list(_iter_json_objects(text))
        assert len(out) == 2
        assert {"a": 1} in out
        assert {"b": 2} in out

    def test_empty(self):
        assert list(_iter_json_objects("")) == []


class TestIntAttr:
    def test_missing(self):
        assert _int_attr({}, "foo") is None

    def test_int(self):
        assert _int_attr({"k": 42}, "k") == 42

    def test_negative_int_ignored(self):
        assert _int_attr({"k": -5}, "k") is None

    def test_float_int(self):
        assert _int_attr({"k": 42.0}, "k") == 42

    def test_float_non_int_ignored(self):
        assert _int_attr({"k": 42.5}, "k") is None

    def test_bool_ignored(self):
        # bool is a subclass of int; we explicitly exclude True/False.
        assert _int_attr({"k": True}, "k") is None

    def test_first_key_wins(self):
        assert _int_attr({"a": 1, "b": 2}, "a", "b") == 1
        assert _int_attr({"b": 2}, "a", "b") == 2


class TestExtractUsage:
    def test_minimal(self):
        d = {
            "attributes": {
                "neatlogs.llm.model_name": "gpt-4o-mini",
                "neatlogs.llm.token_count.prompt": 100,
                "neatlogs.llm.token_count.completion": 50,
            }
        }
        u = _extract_usage(d)
        assert u.model == "gpt-4o-mini"
        assert u.provider is None
        assert u.prompt_tokens == 100
        assert u.completion_tokens == 50
        assert u.cache_creation_tokens is None
        assert u.cache_read_tokens is None
        assert u.reasoning_tokens is None

    def test_with_provider_and_cache(self):
        d = {
            "attributes": {
                "neatlogs.llm.model_name": "claude-3-5-haiku-latest",
                "neatlogs.llm.provider": "Anthropic",
                "neatlogs.llm.token_count.prompt": 1000,
                "neatlogs.llm.token_count.completion": 200,
                "neatlogs.llm.token_count.cache_creation": 500,
                "neatlogs.llm.token_count.cache_read": 700,
                "neatlogs.llm.token_count.reasoning": 0,
            }
        }
        u = _extract_usage(d)
        assert u.model == "claude-3-5-haiku-latest"
        assert u.provider == "anthropic"  # lowercased
        assert u.cache_creation_tokens == 500
        assert u.cache_read_tokens == 700
        # reasoning=0 should NOT be reported as "uses reasoning"
        assert u.uses_reasoning is False

    def test_otel_genai_fallback(self):
        d = {
            "attributes": {
                "gen_ai.system": "openai",
                "gen_ai.response.model": "gpt-4o",
                "gen_ai.usage.input_tokens": 200,
                "gen_ai.usage.output_tokens": 100,
            }
        }
        u = _extract_usage(d)
        assert u.model == "gpt-4o"
        assert u.provider == "openai"
        assert u.prompt_tokens == 200
        assert u.completion_tokens == 100

    def test_request_model_fallback(self):
        d = {
            "attributes": {
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o-mini",
            }
        }
        u = _extract_usage(d)
        assert u.model == "gpt-4o-mini"

    def test_no_attributes(self):
        u = _extract_usage({})
        assert u.model == ""
        assert u.prompt_tokens is None

    def test_attrs_not_a_dict(self):
        u = _extract_usage({"attributes": "nope"})
        assert u.model == ""

    def test_uses_cache_and_reasoning(self):
        u = SpanUsage(
            span_id="s",
            trace_id="t",
            model="m",
            provider=None,
            prompt_tokens=100,
            completion_tokens=50,
            cache_creation_tokens=10,
            cache_read_tokens=0,
            reasoning_tokens=5,
        )
        assert u.uses_prompt_cache is True
        assert u.uses_reasoning is True
        assert u.has_tokens is True

    def test_uses_cache_false_when_zero(self):
        u = SpanUsage(
            span_id="s",
            trace_id="t",
            model="m",
            provider=None,
            prompt_tokens=100,
            completion_tokens=50,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            reasoning_tokens=0,
        )
        assert u.uses_prompt_cache is False
        assert u.uses_reasoning is False


class TestDetectSource:
    def test_processed(self):
        assert (
            _detect_source({"trace_id": "a", "span_id": "b", "parent_span_id": None}) == "processed"
        )

    def test_raw(self):
        assert _detect_source({"context": {"trace_id": "a"}}) == "raw"

    def test_unknown(self):
        assert _detect_source({"foo": "bar"}) == "unknown"


# ---------------------------------------------------------------------------
# Pricing catalog
# ---------------------------------------------------------------------------


class TestLoadCatalog:
    def test_basic(self):
        c = _catalog()
        assert "openai/gpt-4o-mini" in c.cards
        card = c.cards["openai/gpt-4o-mini"]
        assert card.provider == "openai"
        assert card.input_per_1m == 0.15
        assert card.output_per_1m == 0.60
        assert card.cache_read_per_1m == 0.075
        assert card.supports_prompt_cache is True
        assert card.supports_vision is True
        assert card.supports_tools is True
        assert card.context_window == 128000

    def test_tier_parsing(self):
        c = _catalog()
        sonnet = c.cards["anthropic/claude-3-5-sonnet-latest"]
        assert "input_above_200k_per_1m" in sonnet.tiers
        assert sonnet.tiers["input_above_200k_per_1m"] == 6.00
        assert sonnet.tiers["output_above_200k_per_1m"] == 22.50

    def test_reasoning(self):
        c = _catalog()
        o3 = c.cards["openai/o3-mini"]
        assert o3.supports_reasoning is True
        assert o3.reasoning_output_per_1m == 4.40

    def test_bundled_default_loads(self):
        # The bundled file must be loadable.
        c = _load_catalog()
        assert len(c.cards) > 0
        # Headline providers present.
        assert any(card.provider == "openai" for card in c.cards.values())
        assert any(card.provider == "anthropic" for card in c.cards.values())
        assert any(card.provider == "google_genai" for card in c.cards.values())

    def test_lookup_by_provider_and_name(self):
        c = _catalog()
        # Provider-agnostic lookup by model name.
        card = c.get_by_provider_and_name(None, "gpt-4o-mini")
        assert card is not None
        assert card.model_key == "openai/gpt-4o-mini"

    def test_lookup_with_provider(self):
        c = _catalog()
        card = c.get_by_provider_and_name("anthropic", "claude-3-5-haiku-latest")
        assert card is not None
        assert card.model_key == "anthropic/claude-3-5-haiku-latest"

    def test_lookup_miss(self):
        c = _catalog()
        assert c.get_by_provider_and_name("openai", "gpt-99") is None
        assert c.get_by_provider_and_name("nonexistent", "gpt-4o-mini") is None

    def test_load_invalid_json(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("not json")
            path = f.name
        try:
            with pytest.raises(json.JSONDecodeError):
                _load_catalog(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            _load_catalog("/nonexistent/path/pricing.json")

    def test_skips_malformed_entries(self):
        d = {
            "models": {
                "openai/gpt-4o": "not a dict",  # type skip
                "no-slash-key": {"provider": "openai", "input_per_1m": 1.0},  # no slash
                "openai/no-provider": {"input_per_1m": 1.0},  # no provider
                "openai/gpt-3.5": {"provider": "openai", "input_per_1m": 1.5, "output_per_1m": 2.0},
            }
        }
        c = _load_catalog_from_dict(d)
        assert "openai/gpt-3.5" in c.cards
        assert "openai/gpt-4o" not in c.cards
        assert "no-slash-key" not in c.cards

    def test_supports_cache(self):
        c = _catalog()
        mini = c.cards["openai/gpt-4o-mini"]
        assert mini.supports_cache is True
        sonnet = c.cards["anthropic/claude-3-5-sonnet-latest"]
        assert sonnet.supports_cache is True


class TestMaybeFloatInt:
    def test_maybe_float(self):
        assert _maybe_float(1.5) == 1.5
        assert _maybe_float(1) == 1.0
        assert _maybe_float(None) is None
        assert _maybe_float("x") is None
        assert _maybe_float(True) == 1.0  # bool is int

    def test_maybe_int(self):
        assert _maybe_int(42) == 42
        assert _maybe_int(42.0) is None  # not an int
        assert _maybe_int(None) is None
        assert _maybe_int(True) is None
        assert _maybe_int("x") is None


# ---------------------------------------------------------------------------
# Tier logic
# ---------------------------------------------------------------------------


class TestTierFor:
    def test_no_tiers(self):
        card = PriceCard(
            model_key="x",
            provider="p",
            input_per_1m=1.0,
            output_per_1m=2.0,
        )
        usage = SpanUsage("s", "t", "m", None, 100000, 100000, None, None, None)
        label, in_r, out_r = _tier_for(card, usage)
        assert label is None
        assert in_r == 1.0
        assert out_r == 2.0

    def test_below_threshold(self):
        card = PriceCard(
            model_key="x",
            provider="p",
            input_per_1m=1.0,
            output_per_1m=2.0,
            tiers={"input_above_200k_per_1m": 5.0, "output_above_200k_per_1m": 10.0},
        )
        usage = SpanUsage("s", "t", "m", None, 100000, 100000, None, None, None)
        label, in_r, out_r = _tier_for(card, usage)
        assert label is None
        assert in_r == 1.0
        assert out_r == 2.0

    def test_input_crosses_threshold(self):
        card = PriceCard(
            model_key="x",
            provider="p",
            input_per_1m=1.0,
            output_per_1m=2.0,
            tiers={"input_above_200k_per_1m": 5.0, "output_above_200k_per_1m": 10.0},
        )
        usage = SpanUsage("s", "t", "m", None, 250000, 1000, None, None, None)
        label, in_r, out_r = _tier_for(card, usage)
        assert label == "input_above_200k_per_1m"
        assert in_r == 5.0
        assert out_r == 2.0  # output not tiered

    def test_output_crosses_threshold(self):
        card = PriceCard(
            model_key="x",
            provider="p",
            input_per_1m=1.0,
            output_per_1m=2.0,
            tiers={"input_above_200k_per_1m": 5.0, "output_above_200k_per_1m": 10.0},
        )
        usage = SpanUsage("s", "t", "m", None, 1000, 250000, None, None, None)
        label, in_r, out_r = _tier_for(card, usage)
        assert label == "output_above_200k_per_1m"
        assert in_r == 1.0
        assert out_r == 10.0

    def test_both_cross_picks_largest(self):
        card = PriceCard(
            model_key="x",
            provider="p",
            input_per_1m=1.0,
            output_per_1m=2.0,
            tiers={"input_above_200k_per_1m": 5.0, "output_above_200k_per_1m": 10.0},
        )
        usage = SpanUsage("s", "t", "m", None, 300000, 300000, None, None, None)
        label, in_r, out_r = _tier_for(card, usage)
        # Both crossed; we only have one threshold in this card so it picks that one.
        # Both sides crossed; output side is more expensive (10 vs 5) so it
        # dominates. With single threshold, output_above wins.
        assert label == "output_above_200k_per_1m"
        assert out_r == 10.0

    def test_unparseable_tier_ignored(self):
        card = PriceCard(
            model_key="x",
            provider="p",
            input_per_1m=1.0,
            output_per_1m=2.0,
            tiers={"weird_key": 99.0},
        )
        usage = SpanUsage("s", "t", "m", None, 300000, 1000, None, None, None)
        label, in_r, out_r = _tier_for(card, usage)
        assert label is None
        assert in_r == 1.0

    def test_zero_token_span(self):
        card = PriceCard(
            model_key="x",
            provider="p",
            input_per_1m=1.0,
            output_per_1m=2.0,
            tiers={"input_above_200k_per_1m": 5.0},
        )
        usage = SpanUsage("s", "t", "m", None, 0, 0, None, None, None)
        label, in_r, out_r = _tier_for(card, usage)
        assert label is None


# ---------------------------------------------------------------------------
# Model key resolution
# ---------------------------------------------------------------------------


class TestModelKeyFor:
    def test_exact_with_provider(self):
        c = _catalog()
        u = SpanUsage("s", "t", "gpt-4o-mini", "openai", 100, 50, None, None, None)
        assert _model_key_for(u, c) == "openai/gpt-4o-mini"

    def test_provider_agnostic_fallback(self):
        c = _catalog()
        u = SpanUsage("s", "t", "gpt-4o-mini", None, 100, 50, None, None, None)
        assert _model_key_for(u, c) == "openai/gpt-4o-mini"

    def test_no_match(self):
        c = _catalog()
        u = SpanUsage("s", "t", "gpt-99", None, 100, 50, None, None, None)
        assert _model_key_for(u, c) is None

    def test_no_model(self):
        c = _catalog()
        u = SpanUsage("s", "t", "", None, 100, 50, None, None, None)
        assert _model_key_for(u, c) is None


# ---------------------------------------------------------------------------
# cost_span
# ---------------------------------------------------------------------------


class TestCostSpan:
    def test_basic(self):
        c = _catalog()
        card = c.cards["openai/gpt-4o-mini"]
        u = SpanUsage("s", "t", "gpt-4o-mini", "openai", 1_000_000, 1_000_000, None, None, None)
        sc = cost_span(u, card)
        # input 1M * 0.15 + output 1M * 0.60
        assert sc.input_cost == pytest.approx(0.15)
        assert sc.output_cost == pytest.approx(0.60)
        assert sc.total == pytest.approx(0.75)
        assert sc.incompatible is False

    def test_cache_read(self):
        c = _catalog()
        card = c.cards["openai/gpt-4o-mini"]
        u = SpanUsage("s", "t", "gpt-4o-mini", "openai", 0, 0, None, 1_000_000, None)
        sc = cost_span(u, card)
        # 1M cache_read at 0.075
        assert sc.cache_read_cost == pytest.approx(0.075)
        assert sc.total == pytest.approx(0.075)

    def test_cache_write(self):
        c = _catalog()
        card = c.cards["anthropic/claude-3-5-haiku-latest"]
        u = SpanUsage(
            "s", "t", "claude-3-5-haiku-latest", "anthropic", 1_000_000, 0, 1_000_000, 0, None
        )
        sc = cost_span(u, card)
        # 1M cache_write at 1.00
        assert sc.cache_write_cost == pytest.approx(1.00)

    def test_reasoning(self):
        c = _catalog()
        card = c.cards["openai/o3-mini"]
        u = SpanUsage("s", "t", "o3-mini", "openai", 100_000, 100_000, None, None, 100_000)
        sc = cost_span(u, card)
        # 100k reasoning at 4.40/1M = 0.44
        assert sc.reasoning_cost == pytest.approx(0.44)

    def test_tier_applied(self):
        c = _catalog()
        card = c.cards["anthropic/claude-3-5-sonnet-latest"]
        u = SpanUsage(
            "s", "t", "claude-3-5-sonnet-latest", "anthropic", 250_000, 1000, None, None, None
        )
        sc = cost_span(u, card)
        # input 250k * 6.00/1M = 1.50; output 1k * 15.00/1M = 0.015
        assert sc.tier_applied == "input_above_200k_per_1m"
        assert sc.input_cost == pytest.approx(1.50)
        assert sc.output_cost == pytest.approx(0.015)

    def test_incompatible_cache_on_model_without_cache(self):
        # Use a model without cache support: o3-mini is the only OpenAI reasoning
        # model in CATALOG_DICT. But it has cache_read. Let me add a no-cache case
        # by using anthropic claude-3-5-haiku which has cache. Hmm, all our models
        # have cache. Let me build a no-cache card inline.
        card = PriceCard(
            model_key="openai/gpt-3.5",
            provider="openai",
            input_per_1m=1.0,
            output_per_1m=2.0,
            supports_prompt_cache=False,
        )
        u = SpanUsage("s", "t", "gpt-3.5", "openai", 100, 50, 10, 0, None)
        sc = cost_span(u, card)
        assert sc.incompatible is True
        assert sc.total == 0.0

    def test_incompatible_reasoning_on_non_reasoning_model(self):
        card = PriceCard(
            model_key="openai/gpt-4o-mini",
            provider="openai",
            input_per_1m=0.15,
            output_per_1m=0.60,
            supports_reasoning=False,
        )
        u = SpanUsage("s", "t", "gpt-4o-mini", "openai", 100, 50, None, None, 100)
        sc = cost_span(u, card)
        assert sc.incompatible is True

    def test_cache_on_model_without_cache_pricing_only(self):
        # supports_prompt_cache=True but no cache_read/cache_write rate set
        card = PriceCard(
            model_key="x",
            provider="y",
            input_per_1m=1.0,
            output_per_1m=2.0,
            supports_prompt_cache=True,
            cache_read_per_1m=None,
            cache_write_per_1m=None,
        )
        u = SpanUsage("s", "t", "m", "y", 100, 50, 5, 0, None)
        sc = cost_span(u, card)
        # model "supports" cache but has no rates, so cache cost is 0
        # Span doesn't fail compatibility because supports_cache is False
        # (no rates).
        assert sc.cache_write_cost == 0.0
        assert sc.incompatible is False  # supports_cache requires rates


# ---------------------------------------------------------------------------
# Span reading from disk
# ---------------------------------------------------------------------------


def _write_span_log(path: Path, spans: List[Dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(s) for s in spans))


def _processed_span(
    trace_id: str = "a",
    span_id: str = "1",
    name: str = "openai.chat.completions.create",
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    prompt: int = 1000,
    completion: int = 500,
    cache_creation: int = None,
    cache_read: int = None,
    reasoning: int = None,
) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {
        "neatlogs.llm.model_name": model,
        "neatlogs.llm.provider": provider,
        "neatlogs.llm.token_count.prompt": prompt,
        "neatlogs.llm.token_count.completion": completion,
    }
    if cache_creation is not None:
        attrs["neatlogs.llm.token_count.cache_creation"] = cache_creation
    if cache_read is not None:
        attrs["neatlogs.llm.token_count.cache_read"] = cache_read
    if reasoning is not None:
        attrs["neatlogs.llm.token_count.reasoning"] = reasoning
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": None,
        "name": name,
        "kind": "llm",
        "start_time": "2024-01-15T10:00:00Z",
        "end_time": "2024-01-15T10:00:01Z",
        "attributes": attrs,
        "status": {"code": "OK", "description": ""},
        "events": [],
    }


def _raw_span(**kwargs: Any) -> Dict[str, Any]:
    d = _processed_span(**kwargs)
    return {
        "name": d["name"],
        "context": {"trace_id": "0xabcd", "span_id": "0x1234"},
        "parent_id": None,
        "kind": "SpanKind.CLIENT",
        "start_time": "2024-01-15T10:00:00.000Z",
        "end_time": "2024-01-15T10:00:00.001Z",
        "status": {"status_code": "OK", "description": ""},
        "attributes": d["attributes"],
        "events": [],
        "links": [],
        "resource": {},
    }


class TestReadUsages:
    def test_processed(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=1000, completion=500)])
        usages, n = _read_usages(p)
        assert n == 1
        assert len(usages) == 1
        assert usages[0].prompt_tokens == 1000
        assert usages[0].completion_tokens == 500

    def test_raw_format(self, tmp_path: Path):
        p = tmp_path / "raw.jsonl"
        _write_span_log(p, [_raw_span(prompt=2000, completion=1000)])
        usages, n = _read_usages(p)
        assert n == 1
        assert usages[0].prompt_tokens == 2000

    def test_missing_file(self, tmp_path: Path):
        usages, n = _read_usages(tmp_path / "missing.jsonl")
        assert usages == []
        assert n == 0

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        usages, n = _read_usages(p)
        assert usages == []
        assert n == 1

    def test_blank_lines_ignored(self, tmp_path: Path):
        p = tmp_path / "blank.jsonl"
        p.write_text("\n\n\n")
        usages, n = _read_usages(p)
        assert usages == []
        assert n == 1

    def test_malformed_object_skipped(self, tmp_path: Path):
        # A brace-balanced chunk that isn't valid JSON is dropped silently.
        # The surrounding valid objects are still picked up.
        p = tmp_path / "malformed.jsonl"
        p.write_text(
            json.dumps(_processed_span(span_id="1", prompt=100))
            + "\n"
            + "{not valid}"
            + "\n"
            + json.dumps(_processed_span(span_id="2", prompt=200))
        )
        usages, n = _read_usages(p)
        assert n == 1
        assert len(usages) == 2
        assert {u.prompt_tokens for u in usages} == {100, 200}

    def test_spans_without_model_passed_through(self, tmp_path: Path):
        # The reader returns ALL parsed spans (not just LLM ones). The
        # compare() function is responsible for filtering. This keeps the
        # reader simple and the skip count in compare() accurate.
        d = _processed_span(span_id="1")
        d["attributes"] = {"foo": "bar"}  # no model
        p = tmp_path / "no_model.jsonl"
        _write_span_log(p, [d])
        usages, _ = _read_usages(p)
        assert len(usages) == 1
        assert usages[0].model == ""

    def test_brace_balanced_nested_strings(self, tmp_path: Path):
        # Real span data can have embedded newlines in event payloads.
        d = _processed_span(span_id="1")
        d["attributes"]["some_prompt"] = "line 1\nline 2 {x} line 3"
        p = tmp_path / "nested.jsonl"
        _write_span_log(p, [d])
        usages, _ = _read_usages(p)
        assert len(usages) == 1


class TestReadPathsUsages:
    def test_multi_file(self, tmp_path: Path):
        p1 = tmp_path / "a.jsonl"
        p2 = tmp_path / "b.jsonl"
        _write_span_log(p1, [_processed_span(span_id="1", prompt=100)])
        _write_span_log(p2, [_processed_span(span_id="2", prompt=200)])
        usages, n = _read_paths_usages([p1, p2])
        assert n == 2
        assert len(usages) == 2

    def test_missing_paths_counted_zero(self, tmp_path: Path):
        usages, n = _read_paths_usages([tmp_path / "missing.jsonl"])
        assert usages == []
        assert n == 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


class TestCompare:
    def test_basic_single_model(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=1_000_000, completion=1_000_000)])
        catalog = _catalog()
        buf = io.StringIO()
        report = compare(
            p,
            models=["openai/gpt-4o-mini"],
            catalog=catalog,
            warn_stream=buf,
        )
        assert report.baseline is not None
        assert report.baseline.model_key == "openai/gpt-4o-mini"
        # 1M * 0.15 + 1M * 0.60 = 0.75
        assert report.baseline.total_usd == pytest.approx(0.75)
        assert buf.getvalue() == ""

    def test_explicit_baseline_differs_from_models_order(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=1_000_000, completion=1_000_000)])
        catalog = _catalog()
        report = compare(
            p,
            models=["openai/gpt-4o-mini", "openai/gpt-4o"],
            current_model="openai/gpt-4o-mini",
            catalog=catalog,
        )
        assert report.baseline is not None
        assert report.baseline.model_key == "openai/gpt-4o-mini"
        assert len(report.alternatives) == 1
        assert report.alternatives[0].model_key == "openai/gpt-4o"
        # gpt-4o: 1M * 2.50 + 1M * 10.00 = 12.50
        assert report.alternatives[0].total_usd == pytest.approx(12.50)

    def test_savings_detection(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=1_000_000, completion=1_000_000)])
        catalog = _catalog()
        report = compare(
            p,
            models=["openai/gpt-4o", "openai/gpt-4o-mini"],
            catalog=catalog,
        )
        # gpt-4o-mini is much cheaper than gpt-4o, so the delta should be negative.
        mini = report.find("openai/gpt-4o-mini")
        assert mini is not None
        delta = report.delta_pct_for(mini)
        # (0.75 - 12.50) / 12.50 * 100 = -94%
        assert delta == pytest.approx(-94, abs=0.1)

    def test_capability_diff(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=1_000_000, completion=1_000_000)])
        catalog = _catalog()
        # Use a model without vision as the alternative vs gpt-4o-mini (has vision).
        # Build a custom catalog with a vision-less alternative.
        custom = {
            "_meta": {"schema_version": "1.0"},
            "models": {
                "openai/gpt-4o-mini": {
                    "provider": "openai",
                    "input_per_1m": 0.15,
                    "output_per_1m": 0.60,
                    "supports_vision": True,
                    "supports_tools": True,
                },
                "openai/text-only": {
                    "provider": "openai",
                    "input_per_1m": 0.10,
                    "output_per_1m": 0.40,
                    "supports_vision": False,
                    "supports_tools": True,
                },
            },
        }
        c2 = _load_catalog_from_dict(custom)
        report = compare(
            p,
            models=["openai/gpt-4o-mini", "openai/text-only"],
            catalog=c2,
        )
        alt = report.find("openai/text-only")
        assert alt is not None
        assert "supports_vision" in alt.missing_capabilities

    def test_capability_diff_both_support(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=100, completion=50)])
        catalog = _catalog()
        report = compare(
            p,
            models=["openai/gpt-4o-mini", "openai/gpt-4o"],
            catalog=catalog,
        )
        alt = report.find("openai/gpt-4o")
        assert alt is not None
        assert alt.missing_capabilities == []

    def test_unknown_model_warning(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=100, completion=50)])
        catalog = _catalog()
        buf = io.StringIO()
        report = compare(
            p,
            models=["openai/gpt-4o-mini", "nonexistent/model"],
            catalog=catalog,
            warn_stream=buf,
        )
        assert "nonexistent/model" in buf.getvalue()
        # The unknown model is omitted; only the known one is in the report.
        assert report.find("nonexistent/model") is None
        assert report.find("openai/gpt-4o-mini") is not None

    def test_unknown_source_model_note(self, tmp_path: Path):
        # Span uses a model not in the catalog. Should surface a note.
        p = tmp_path / "spans.jsonl"
        _write_span_log(
            p,
            [_processed_span(prompt=100, completion=50, model="some/future-model")],
        )
        catalog = _catalog()
        buf = io.StringIO()
        compare(
            p,
            models=["openai/gpt-4o-mini"],
            catalog=catalog,
            warn_stream=buf,
        )
        assert "some/future-model" in buf.getvalue()

    def test_spans_with_no_tokens_counted(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        d = _processed_span(span_id="1", prompt=100, completion=50)
        # Force zero tokens
        d["attributes"]["neatlogs.llm.token_count.prompt"] = 0
        d["attributes"]["neatlogs.llm.token_count.completion"] = 0
        _write_span_log(p, [d])
        catalog = _catalog()
        report = compare(
            p,
            models=["openai/gpt-4o-mini"],
            catalog=catalog,
        )
        assert report.spans_with_tokens == 0
        assert report.spans_skipped == 1

    def test_spans_with_no_model_skipped(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        d = _processed_span(span_id="1")
        d["attributes"] = {"foo": "bar"}  # no model
        _write_span_log(p, [d])
        catalog = _catalog()
        report = compare(
            p,
            models=["openai/gpt-4o-mini"],
            catalog=catalog,
        )
        assert report.spans_with_tokens == 0

    def test_empty_spans_file(self, tmp_path: Path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        catalog = _catalog()
        report = compare(
            p,
            models=["openai/gpt-4o-mini"],
            catalog=catalog,
        )
        assert report.spans_with_tokens == 0
        assert report.files_read == 1

    def test_missing_paths(self, tmp_path: Path):
        catalog = _catalog()
        report = compare(
            [tmp_path / "missing.jsonl"],
            models=["openai/gpt-4o-mini"],
            catalog=catalog,
        )
        assert report.files_read == 0
        assert report.spans_with_tokens == 0

    def test_single_path_string(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=100, completion=50)])
        catalog = _catalog()
        report = compare(
            str(p),  # str, not Path
            models=["openai/gpt-4o-mini"],
            catalog=catalog,
        )
        assert report.spans_with_tokens == 1

    def test_incompatible_span_counted(self, tmp_path: Path):
        # Use a span with cache tokens against a model that has no cache support.
        custom = {
            "_meta": {"schema_version": "1.0"},
            "models": {
                "openai/with-cache": {
                    "provider": "openai",
                    "input_per_1m": 1.0,
                    "output_per_1m": 2.0,
                    "cache_read_per_1m": 0.1,
                    "supports_prompt_cache": True,
                },
                "openai/no-cache": {
                    "provider": "openai",
                    "input_per_1m": 1.0,
                    "output_per_1m": 2.0,
                },
            },
        }
        p = tmp_path / "spans.jsonl"
        _write_span_log(
            p,
            [_processed_span(prompt=100, completion=50, cache_read=100, model="openai/with-cache")],
        )
        c2 = _load_catalog_from_dict(custom)
        report = compare(
            p,
            models=["openai/no-cache"],
            catalog=c2,
        )
        mc = report.find("openai/no-cache")
        assert mc is not None
        assert mc.spans_total == 1
        assert mc.spans_incompatible == 1
        assert mc.total_usd == 0.0

    def test_files_read_count(self, tmp_path: Path):
        p1 = tmp_path / "a.jsonl"
        p2 = tmp_path / "b.jsonl"
        _write_span_log(p1, [_processed_span(span_id="1", prompt=100, completion=50)])
        _write_span_log(p2, [_processed_span(span_id="2", prompt=100, completion=50)])
        catalog = _catalog()
        report = compare(
            [p1, p2],
            models=["openai/gpt-4o-mini"],
            catalog=catalog,
        )
        assert report.files_read == 2
        assert report.spans_with_tokens == 2

    def test_all_helper(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=100, completion=50)])
        catalog = _catalog()
        # No baseline (no models[0])
        report = compare(p, models=[], catalog=catalog)
        assert report.all() == []
        # With baseline + alternatives
        report = compare(
            p,
            models=["openai/gpt-4o-mini", "openai/gpt-4o"],
            catalog=catalog,
        )
        all_rows = report.all()
        assert len(all_rows) == 2
        assert all_rows[0].is_baseline
        assert not all_rows[1].is_baseline


# ---------------------------------------------------------------------------
# forecast
# ---------------------------------------------------------------------------


class TestForecast:
    def test_basic(self):
        c = _catalog()
        report = forecast(
            model_key="openai/gpt-4o-mini",
            monthly_calls=10_000,
            avg_prompt_tokens=2_000,
            avg_completion_tokens=500,
            catalog=c,
        )
        # Per call: 2k * 0.15/1M + 500 * 0.60/1M = 0.0003 + 0.0003 = 0.0006
        # Monthly: 10k * 0.0006 = 6.00
        # Annual: 72.00
        assert report.monthly_cost_usd == pytest.approx(6.00, abs=1e-4)
        assert report.annual_cost_usd == pytest.approx(72.00, abs=1e-2)

    def test_cache_hit_rate(self):
        c = _catalog()
        report = forecast(
            model_key="openai/gpt-4o-mini",
            monthly_calls=10_000,
            avg_prompt_tokens=2_000,
            avg_completion_tokens=500,
            catalog=c,
            cache_hit_rate=0.5,  # half the prompt tokens come from cache
        )
        # 1k miss at 0.15/1M + 1k hit at 0.075/1M + 500 completion at 0.60/1M
        # = 0.00015 + 0.000075 + 0.0003 = 0.000525 per call
        # monthly: 5.25
        assert report.monthly_cost_usd == pytest.approx(5.25, abs=1e-4)
        assert report.cache_hit_rate == 0.5

    def test_cache_hit_rate_clamped(self):
        c = _catalog()
        # Negative or >1 is clamped.
        r1 = forecast(
            model_key="openai/gpt-4o-mini",
            monthly_calls=100,
            avg_prompt_tokens=100,
            avg_completion_tokens=100,
            catalog=c,
            cache_hit_rate=-0.5,
        )
        assert r1.cache_hit_rate == 0.0
        r2 = forecast(
            model_key="openai/gpt-4o-mini",
            monthly_calls=100,
            avg_prompt_tokens=100,
            avg_completion_tokens=100,
            catalog=c,
            cache_hit_rate=2.0,
        )
        assert r2.cache_hit_rate == 1.0

    def test_reasoning_per_call(self):
        c = _catalog()
        report = forecast(
            model_key="openai/o3-mini",
            monthly_calls=10_000,
            avg_prompt_tokens=2_000,
            avg_completion_tokens=500,
            catalog=c,
            reasoning_per_call=2_000,
        )
        # Per call: 2k input at 1.10/1M + 500 output at 4.40/1M + 2k reasoning at 4.40/1M
        # = 0.0022 + 0.0022 + 0.0088 = 0.0132
        # monthly: 132.00
        assert report.monthly_cost_usd == pytest.approx(132.00, abs=0.01)

    def test_unknown_model_raises(self):
        c = _catalog()
        with pytest.raises(ValueError):
            forecast(
                model_key="openai/nonexistent",
                monthly_calls=100,
                avg_prompt_tokens=100,
                avg_completion_tokens=100,
                catalog=c,
            )

    def test_cache_requested_on_no_cache_model(self):
        # Model has no cache support, but we ask for cache_hit_rate=0.5.
        # Should produce a note and treat the cache as 0.
        custom = {
            "_meta": {"schema_version": "1.0"},
            "models": {
                "openai/no-cache": {
                    "provider": "openai",
                    "input_per_1m": 1.0,
                    "output_per_1m": 2.0,
                },
            },
        }
        c = _load_catalog_from_dict(custom)
        report = forecast(
            model_key="openai/no-cache",
            monthly_calls=100,
            avg_prompt_tokens=1000,
            avg_completion_tokens=100,
            catalog=c,
            cache_hit_rate=0.5,
        )
        assert report.incompatible is True
        assert any("cache" in n.lower() for n in report.notes)

    def test_reasoning_on_non_reasoning_model(self):
        custom = {
            "_meta": {"schema_version": "1.0"},
            "models": {
                "openai/no-reasoning": {
                    "provider": "openai",
                    "input_per_1m": 1.0,
                    "output_per_1m": 2.0,
                    "supports_reasoning": False,
                },
            },
        }
        c = _load_catalog_from_dict(custom)
        report = forecast(
            model_key="openai/no-reasoning",
            monthly_calls=100,
            avg_prompt_tokens=1000,
            avg_completion_tokens=100,
            catalog=c,
            reasoning_per_call=500,
        )
        assert report.incompatible is True
        assert any("reasoning" in n.lower() for n in report.notes)

    def test_monthly_to_annual_multiplier(self):
        c = _catalog()
        report = forecast(
            model_key="openai/gpt-4o-mini",
            monthly_calls=10_000,
            avg_prompt_tokens=1000,
            avg_completion_tokens=100,
            catalog=c,
        )
        assert report.annual_cost_usd == pytest.approx(report.monthly_cost_usd * 12)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


class TestFormatComparison:
    def test_text_renders_baseline_and_alternatives(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=1_000_000, completion=1_000_000)])
        catalog = _catalog()
        report = compare(
            p,
            models=["openai/gpt-4o-mini", "openai/gpt-4o"],
            catalog=catalog,
        )
        out = format_comparison(report, style="text")
        assert "openai/gpt-4o-mini" in out
        assert "openai/gpt-4o" in out
        assert "(baseline)" in out
        assert "vs base" in out
        assert "$" in out  # currency marker

    def test_text_renders_capability_diff(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=100, completion=50)])
        custom = {
            "_meta": {"schema_version": "1.0"},
            "models": {
                "openai/gpt-4o-mini": {
                    "provider": "openai",
                    "input_per_1m": 0.15,
                    "output_per_1m": 0.60,
                    "supports_vision": True,
                    "supports_tools": True,
                },
                "openai/text-only": {
                    "provider": "openai",
                    "input_per_1m": 0.10,
                    "output_per_1m": 0.40,
                    "supports_vision": False,
                    "supports_tools": True,
                },
            },
        }
        c = _load_catalog_from_dict(custom)
        report = compare(
            p,
            models=["openai/gpt-4o-mini", "openai/text-only"],
            catalog=c,
        )
        out = format_comparison(report, style="text")
        assert "Capability diff" in out
        assert "supports_vision" in out

    def test_text_empty_report(self):
        report = ComparisonReport(
            baseline=None,
            alternatives=[],
            unknown_models=[],
            files_read=0,
            spans_with_tokens=0,
            spans_skipped=0,
            catalog_size=0,
        )
        out = format_comparison(report, style="text")
        assert "no comparable models" in out

    def test_text_no_spans_with_tokens(self):
        # With a baseline configured but no spans processed, we still print
        # the row showing $0 — the user wants to know "what would I have
        # spent?" even when there's no data to project.
        c = _catalog()
        report = ComparisonReport(
            baseline=ModelComparison(
                model_key="openai/gpt-4o-mini",
                provider="openai",
                total_usd=0,
                input_usd=0,
                output_usd=0,
                cache_read_usd=0,
                cache_write_usd=0,
                reasoning_usd=0,
                spans_total=0,
                spans_incompatible=0,
                is_baseline=True,
            ),
            alternatives=[],
            unknown_models=[],
            files_read=1,
            spans_with_tokens=0,
            spans_skipped=0,
            catalog_size=42,
        )
        out = format_comparison(report, style="text")
        assert "openai/gpt-4o-mini" in out
        assert "$0.0000" in out

    def test_text_no_spans_no_baseline(self):
        # No baseline and no alternatives and no spans → tell the user.
        report = ComparisonReport(
            baseline=None,
            alternatives=[],
            unknown_models=[],
            files_read=1,
            spans_with_tokens=0,
            spans_skipped=2,
            catalog_size=42,
        )
        out = format_comparison(report, style="text")
        assert "no LLM spans" in out
        assert "2 span(s) skipped" in out

    def test_text_skipped_count(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        d = _processed_span()
        d["attributes"] = {"foo": "bar"}  # no model, will be skipped
        _write_span_log(p, [d])
        c = _catalog()
        report = compare(p, models=["openai/gpt-4o-mini"], catalog=c)
        out = format_comparison(report, style="text")
        assert report.spans_skipped == 1
        assert "1 span(s) skipped" in out

    def test_json_shape(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=1_000_000, completion=1_000_000)])
        c = _catalog()
        report = compare(
            p,
            models=["openai/gpt-4o-mini", "openai/gpt-4o"],
            catalog=c,
        )
        out = format_comparison(report, style="json")
        obj = json.loads(out)
        assert obj["currency"] == "USD"
        assert obj["baseline"]["model"] == "openai/gpt-4o-mini"
        assert obj["baseline"]["is_baseline"] is True
        assert len(obj["alternatives"]) == 1
        # delta is present for the alternative
        assert "delta_pct_vs_baseline" in obj["alternatives"][0]

    def test_json_no_baseline(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=100, completion=50)])
        c = _catalog()
        report = compare(
            p,
            models=["openai/gpt-4o-mini"],
            current_model="nonexistent/baseline",
            catalog=c,
        )
        out = format_comparison(report, style="json")
        obj = json.loads(out)
        assert obj["baseline"] is None

    def test_csv_shape(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=1_000_000, completion=1_000_000)])
        c = _catalog()
        report = compare(
            p,
            models=["openai/gpt-4o-mini", "openai/gpt-4o"],
            catalog=c,
        )
        out = format_comparison(report, style="csv")
        lines = out.strip().split("\n")
        assert lines[0] == (
            "model,provider,is_baseline,input_usd,output_usd,cache_read_usd,"
            "cache_write_usd,reasoning_usd,total_usd,spans_total,spans_incompatible,"
            "delta_pct_vs_baseline"
        )
        # Two data rows.
        assert len(lines) == 3
        # First row is the baseline; no delta_pct value.
        baseline_cols = lines[1].split(",")
        assert baseline_cols[0] == "openai/gpt-4o-mini"
        assert baseline_cols[2] == "true"
        assert baseline_cols[-1] == ""  # no delta for baseline
        # Second row has a delta.
        alt_cols = lines[2].split(",")
        assert alt_cols[0] == "openai/gpt-4o"
        assert alt_cols[2] == "false"
        assert alt_cols[-1] != ""  # has a delta

    def test_csv_no_baseline(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=100, completion=50)])
        c = _catalog()
        report = compare(
            p,
            models=["openai/gpt-4o-mini"],
            current_model="nonexistent",
            catalog=c,
        )
        out = format_comparison(report, style="csv")
        lines = out.strip().split("\n")
        # Header + 1 alternative row
        assert len(lines) == 2

    def test_unknown_style_raises(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=100, completion=50)])
        c = _catalog()
        report = compare(p, models=["openai/gpt-4o-mini"], catalog=c)
        with pytest.raises(ValueError):
            format_comparison(report, style="bogus")


class TestFormatComparisonTextHelpers:
    def test_internal_text_helper_no_baseline(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=100, completion=50)])
        c = _catalog()
        report = compare(
            p,
            models=["openai/gpt-4o-mini"],
            current_model="nonexistent",
            catalog=c,
        )
        out = _format_comparison_text(report, use_color=False)
        # No baseline marker
        assert "(baseline)" not in out
        assert "openai/gpt-4o-mini" in out

    def test_internal_json_helper(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=100, completion=50)])
        c = _catalog()
        report = compare(p, models=["openai/gpt-4o-mini"], catalog=c)
        out = _format_comparison_json(report)
        obj = json.loads(out)
        assert obj["currency"] == "USD"

    def test_internal_csv_helper_lf_only(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=100, completion=50)])
        c = _catalog()
        report = compare(p, models=["openai/gpt-4o-mini"], catalog=c)
        out = _format_comparison_csv(report)
        # No \r line endings.
        assert "\r\n" not in out
        # Always ends with newline.
        assert out.endswith("\n")


class TestFormatForecast:
    def test_text_shape(self):
        c = _catalog()
        report = forecast(
            model_key="openai/gpt-4o-mini",
            monthly_calls=10_000,
            avg_prompt_tokens=2000,
            avg_completion_tokens=500,
            catalog=c,
        )
        out = format_forecast(report, style="text")
        assert "openai/gpt-4o-mini" in out
        assert "Monthly calls:" in out
        assert "Monthly cost" in out
        assert "Annual cost" in out
        assert "$" in out

    def test_text_warns_on_incompatible(self):
        custom = {
            "_meta": {"schema_version": "1.0"},
            "models": {
                "openai/no-cache": {
                    "provider": "openai",
                    "input_per_1m": 1.0,
                    "output_per_1m": 2.0,
                },
            },
        }
        c = _load_catalog_from_dict(custom)
        report = forecast(
            model_key="openai/no-cache",
            monthly_calls=100,
            avg_prompt_tokens=1000,
            avg_completion_tokens=100,
            catalog=c,
            cache_hit_rate=0.5,
        )
        out = format_forecast(report, style="text")
        assert "WARNING" in out
        assert "cache" in out.lower()

    def test_json_shape(self):
        c = _catalog()
        report = forecast(
            model_key="openai/gpt-4o-mini",
            monthly_calls=10_000,
            avg_prompt_tokens=2000,
            avg_completion_tokens=500,
            catalog=c,
        )
        out = format_forecast(report, style="json")
        obj = json.loads(out)
        assert obj["model"] == "openai/gpt-4o-mini"
        assert obj["currency"] == "USD"
        assert obj["monthly_cost_usd"] == pytest.approx(6.0, abs=1e-3)
        assert obj["annual_cost_usd"] == pytest.approx(72.0, abs=1e-2)

    def test_unknown_style_raises(self):
        c = _catalog()
        report = forecast(
            model_key="openai/gpt-4o-mini",
            monthly_calls=100,
            avg_prompt_tokens=100,
            avg_completion_tokens=100,
            catalog=c,
        )
        with pytest.raises(ValueError):
            format_forecast(report, style="bogus")


# ---------------------------------------------------------------------------
# Capability dict
# ---------------------------------------------------------------------------


class TestCapabilityDict:
    def test_all_caps(self):
        c = PriceCard(
            model_key="x",
            provider="y",
            input_per_1m=1,
            output_per_1m=2,
            supports_vision=True,
            supports_tools=True,
            supports_reasoning=True,
            supports_prompt_cache=True,
        )
        caps = _capability_dict(c)
        assert caps == {
            "supports_vision": True,
            "supports_tools": True,
            "supports_reasoning": True,
            "supports_prompt_cache": True,
        }

    def test_no_caps(self):
        c = PriceCard(model_key="x", provider="y", input_per_1m=1, output_per_1m=2)
        caps = _capability_dict(c)
        assert all(v is False for v in caps.values())


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


class TestCLI:
    def test_compare_cli(self, tmp_path: Path, capsys):
        from neatlogs.cost import _cli

        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=1_000_000, completion=1_000_000)])
        rc = _cli(
            [
                str(p),
                "--models",
                "openai/gpt-4o-mini,openai/gpt-4o",
                "--no-color",
                "--format",
                "text",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "openai/gpt-4o-mini" in out
        assert "openai/gpt-4o" in out

    def test_compare_cli_json(self, tmp_path: Path, capsys):
        from neatlogs.cost import _cli

        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=100, completion=50)])
        rc = _cli([str(p), "--models", "openai/gpt-4o-mini", "--format", "json"])
        assert rc == 0
        out = capsys.readouterr().out
        obj = json.loads(out)
        assert obj["currency"] == "USD"

    def test_compare_cli_csv(self, tmp_path: Path, capsys):
        from neatlogs.cost import _cli

        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span(prompt=100, completion=50)])
        rc = _cli([str(p), "--models", "openai/gpt-4o-mini", "--format", "csv"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "model,provider,is_baseline" in out

    def test_forecast_cli(self, tmp_path: Path, capsys):
        from neatlogs.cost import _cli

        p = tmp_path / "ignored.jsonl"  # forecast mode doesn't read this
        _write_span_log(p, [_processed_span()])
        rc = _cli(
            [
                str(p),
                "--models",
                "openai/gpt-4o-mini",
                "--forecast",
                "--monthly-calls",
                "1000",
                "--avg-prompt",
                "1000",
                "--avg-completion",
                "100",
                "--no-color",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Monthly cost" in out
        assert "openai/gpt-4o-mini" in out

    def test_missing_pricing_file(self, tmp_path: Path, capsys):
        from neatlogs.cost import _cli

        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span()])
        rc = _cli(
            [
                str(p),
                "--models",
                "openai/gpt-4o-mini",
                "--pricing-file",
                str(tmp_path / "does-not-exist.json"),
            ]
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "error" in err

    def test_invalid_pricing_json(self, tmp_path: Path, capsys):
        from neatlogs.cost import _cli

        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span()])
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        rc = _cli(
            [
                str(p),
                "--models",
                "openai/gpt-4o-mini",
                "--pricing-file",
                str(bad),
            ]
        )
        assert rc == 2

    def test_no_comparable_models(self, tmp_path: Path, capsys):
        from neatlogs.cost import _cli

        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span()])
        rc = _cli([str(p), "--models", "totally/nonexistent"])
        assert rc == 2
        err = capsys.readouterr().err
        assert "no comparable models" in err

    def test_forecast_unknown_model(self, tmp_path: Path, capsys):
        from neatlogs.cost import _cli

        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span()])
        rc = _cli([str(p), "--models", "openai/nonexistent", "--forecast"])
        assert rc == 2

    def test_comma_separated_models_parsed(self, tmp_path: Path, capsys):
        from neatlogs.cost import _cli

        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_processed_span()])
        rc = _cli(
            [
                str(p),
                "--models",
                "openai/gpt-4o-mini, openai/gpt-4o, anthropic/claude-3-5-haiku-latest",
                "--no-color",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        # All three should appear.
        assert "openai/gpt-4o-mini" in out
        assert "openai/gpt-4o" in out
        assert "anthropic/claude-3-5-haiku-latest" in out
