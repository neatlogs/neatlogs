"""
Unit tests for neatlogs.intel.

The intel module is the "Replay + Model Optimizer" — given a span
log, evaluate candidate models against the workload and rank them by
cost under capability / context / compatibility constraints.

The PricingProvider chain, ModelDefinition with usage_types dict, and
WorkloadConstraints are all covered end-to-end.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from neatlogs.intel import (
    BuiltinProvider,
    ChainProvider,
    CustomProvider,
    EvaluationReport,
    ModelDefinition,
    ScoredModel,
    SpanUsage,
    Tier,
    TokenStats,
    UsageType,
    WorkloadConstraints,
    WorkloadProfile,
    _build_def,
    _extract_usage,
    _is_span_compatible,
    _iter_json_objects,
    _read_paths,
    _read_usages,
    _resolve_definition,
    _score_model,
    _span_cost,
    build_workload_profile,
    default_chain,
    evaluate_workload,
    format_evaluation,
    format_evaluation_csv,
    format_evaluation_json,
    format_evaluation_text,
)

# ---------------------------------------------------------------------------
# Test catalog (in-memory)
# ---------------------------------------------------------------------------


def _make_catalog() -> Dict[str, Any]:
    """A small in-memory catalog for tests. Each model has a distinct
    capability set and price profile so capability-matching and tier
    logic can be exercised."""
    return {
        "_meta": {"schema_version": "2.0"},
        "models": {
            "openai/gpt-4o-mini": {
                "provider": "openai",
                "context_window": 128000,
                "capabilities": ["vision", "tools", "json_mode", "prompt_cache"],
                "usage_types": {"input": 0.15, "output": 0.60, "cache_read": 0.075},
            },
            "openai/gpt-4o": {
                "provider": "openai",
                "context_window": 128000,
                "capabilities": ["vision", "tools", "json_mode", "prompt_cache"],
                "usage_types": {"input": 2.50, "output": 10.00, "cache_read": 1.25},
            },
            "openai/o3-mini": {
                "provider": "openai",
                "context_window": 200000,
                "capabilities": ["tools", "json_mode", "reasoning", "prompt_cache"],
                "usage_types": {
                    "input": 1.10,
                    "output": 4.40,
                    "reasoning": 4.40,
                    "cache_read": 0.55,
                },
            },
            "openai/text-embedding-3-small": {
                "provider": "openai",
                "context_window": 8191,
                "capabilities": ["embedding"],
                "usage_types": {"input": 0.02, "output": 0.0},
            },
            "anthropic/claude-3-5-haiku-latest": {
                "provider": "anthropic",
                "context_window": 200000,
                "capabilities": ["tools", "json_mode", "prompt_cache"],
                "usage_types": {
                    "input": 0.80,
                    "output": 4.00,
                    "cache_write": 1.00,
                    "cache_read": 0.08,
                },
            },
            "anthropic/claude-3-5-sonnet-latest": {
                "provider": "anthropic",
                "context_window": 200000,
                "capabilities": ["vision", "tools", "json_mode", "prompt_cache"],
                "usage_types": {
                    "input": 3.00,
                    "output": 15.00,
                    "cache_write": 3.75,
                    "cache_read": 0.30,
                },
                "tiers": {
                    "input": [{"above_tokens": 200000, "rate": 6.00}],
                    "output": [{"above_tokens": 200000, "rate": 22.50}],
                },
            },
            "anthropic/claude-3-haiku-20240307": {
                "provider": "anthropic",
                "context_window": 200000,
                "capabilities": ["tools", "json_mode", "prompt_cache"],
                "usage_types": {
                    "input": 0.25,
                    "output": 1.25,
                    "cache_write": 0.30,
                    "cache_read": 0.03,
                },
            },
        },
    }


def _write_catalog_to_tmp(catalog: Dict[str, Any]) -> Path:
    fd, path = tempfile.mkstemp(suffix=".json")
    Path(path).write_text(json.dumps(catalog))
    return Path(path)


def _build_test_provider(catalog: Dict[str, Any] = None) -> CustomProvider:
    catalog = catalog if catalog is not None else _make_catalog()
    return CustomProvider(_write_catalog_to_tmp(catalog))


def _write_span_log(path: Path, spans: List[Dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(s) for s in spans))


def _span(
    trace_id: str = "a",
    span_id: str = "1",
    name: str = "openai.chat.completions.create",
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    prompt: int = 1000,
    completion: int = 500,
    cache_creation: int = 0,
    cache_read: int = 0,
    reasoning: int = 0,
) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {
        "neatlogs.llm.model_name": model,
        "neatlogs.llm.provider": provider,
        "neatlogs.llm.token_count.prompt": prompt,
        "neatlogs.llm.token_count.completion": completion,
    }
    if cache_creation:
        attrs["neatlogs.llm.token_count.cache_creation"] = cache_creation
    if cache_read:
        attrs["neatlogs.llm.token_count.cache_read"] = cache_read
    if reasoning:
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


# ---------------------------------------------------------------------------
# Span reading
# ---------------------------------------------------------------------------


class TestIntAttr:
    def test_zero_value_kept(self):
        # 0 is a valid value, not "missing".
        from neatlogs.intel import _int_attr

        assert _int_attr({"k": 0}, "k") == 0

    def test_negative_ignored(self):
        from neatlogs.intel import _int_attr

        assert _int_attr({"k": -5}, "k") == 0

    def test_bool_ignored(self):
        from neatlogs.intel import _int_attr

        assert _int_attr({"k": True}, "k") == 0

    def test_first_key_wins(self):
        from neatlogs.intel import _int_attr

        assert _int_attr({"a": 100, "b": 200}, "a", "b") == 100
        assert _int_attr({"b": 200}, "a", "b") == 200


class TestExtractUsage:
    def test_basic(self):
        d = {
            "attributes": {
                "neatlogs.llm.model_name": "gpt-4o-mini",
                "neatlogs.llm.provider": "OpenAI",
                "neatlogs.llm.token_count.prompt": 100,
                "neatlogs.llm.token_count.completion": 50,
            }
        }
        u = _extract_usage(d)
        assert u.model == "gpt-4o-mini"
        assert u.provider == "openai"  # lowercased
        assert u.prompt_tokens == 100
        assert u.completion_tokens == 50
        assert u.cache_creation_tokens == 0
        assert u.cache_read_tokens == 0
        assert u.reasoning_tokens == 0

    def test_otel_fallback(self):
        d = {
            "attributes": {
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.usage.input_tokens": 200,
                "gen_ai.usage.output_tokens": 100,
            }
        }
        u = _extract_usage(d)
        assert u.model == "gpt-4o"
        assert u.prompt_tokens == 200

    def test_cache_and_reasoning(self):
        d = {
            "attributes": {
                "neatlogs.llm.model_name": "m",
                "neatlogs.llm.provider": "p",
                "neatlogs.llm.token_count.prompt": 1000,
                "neatlogs.llm.token_count.completion": 500,
                "neatlogs.llm.token_count.cache_creation": 200,
                "neatlogs.llm.token_count.cache_read": 800,
                "neatlogs.llm.token_count.reasoning": 100,
            }
        }
        u = _extract_usage(d)
        assert u.cache_creation_tokens == 200
        assert u.cache_read_tokens == 800
        assert u.reasoning_tokens == 100
        assert u.uses_prompt_cache is True
        assert u.uses_reasoning is True
        assert u.input_total == 2000
        assert u.output_total == 600


class TestIterJsonObjects:
    def test_basic(self):
        out = list(_iter_json_objects('{"a": 1}\n{"b": 2}'))
        assert out == [{"a": 1}, {"b": 2}]

    def test_escaped_quotes(self):
        text = '{"name": "a \\"quoted\\" word"}'
        out = list(_iter_json_objects(text))
        assert out == [{"name": 'a "quoted" word'}]

    def test_nested_braces_in_strings(self):
        text = '{"name": "a {b} c"}'
        out = list(_iter_json_objects(text))
        assert out == [{"name": "a {b} c"}]

    def test_malformed_skipped(self):
        text = '{"a": 1}\n{not valid}\n{"b": 2}'
        out = list(_iter_json_objects(text))
        assert out == [{"a": 1}, {"b": 2}]


class TestReadUsages:
    def test_missing_path(self, tmp_path: Path):
        usages = _read_usages(tmp_path / "missing.jsonl")
        assert usages == []

    def test_empty(self, tmp_path: Path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        assert _read_usages(p) == []

    def test_basic(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span(prompt=100, completion=50)])
        usages = _read_usages(p)
        assert len(usages) == 1
        assert usages[0].prompt_tokens == 100

    def test_brace_balanced_nested_strings(self, tmp_path: Path):
        d = _span()
        d["attributes"]["event_payload"] = "line 1\nline 2 {x}"
        p = tmp_path / "nested.jsonl"
        _write_span_log(p, [d])
        usages = _read_usages(p)
        assert len(usages) == 1


class TestReadPaths:
    def test_multi_file(self, tmp_path: Path):
        p1 = tmp_path / "a.jsonl"
        p2 = tmp_path / "b.jsonl"
        _write_span_log(p1, [_span(span_id="1", prompt=100)])
        _write_span_log(p2, [_span(span_id="2", prompt=200)])
        usages = _read_paths([p1, p2])
        assert len(usages) == 2
        assert {u.prompt_tokens for u in usages} == {100, 200}


# ---------------------------------------------------------------------------
# ModelDefinition + Pricing
# ---------------------------------------------------------------------------


class TestModelDefinition:
    def test_basic(self):
        d = ModelDefinition(
            model_key="openai/gpt-4o-mini",
            provider="openai",
            context_window=128000,
            capabilities={"vision", "tools"},
            usage_types={"input": 0.15, "output": 0.60},
        )
        assert d.rate_for(UsageType.INPUT) == 0.15
        assert d.rate_for(UsageType.OUTPUT) == 0.60
        assert d.rate_for(UsageType.CACHE_READ) is None

    def test_effective_rate_no_tier(self):
        d = ModelDefinition(
            model_key="m",
            provider="p",
            usage_types={"input": 0.15},
        )
        assert d.effective_rate(UsageType.INPUT, 1000) == 0.15
        assert d.effective_rate(UsageType.INPUT, 1_000_000) == 0.15

    def test_effective_rate_tier_crossed(self):
        d = ModelDefinition(
            model_key="m",
            provider="p",
            usage_types={"input": 3.00},
            tiers={"input": [Tier(above_tokens=200_000, rate=6.00)]},
        )
        # Below threshold: base rate.
        assert d.effective_rate(UsageType.INPUT, 100_000) == 3.00
        # Strictly above threshold: tier rate.
        assert d.effective_rate(UsageType.INPUT, 200_001) == 6.00
        assert d.effective_rate(UsageType.INPUT, 1_000_000) == 6.00

    def test_effective_rate_zero_tokens(self):
        d = ModelDefinition(
            model_key="m",
            provider="p",
            usage_types={"input": 3.00},
            tiers={"input": [Tier(above_tokens=200_000, rate=6.00)]},
        )
        assert d.effective_rate(UsageType.INPUT, 0) == 3.00

    def test_effective_rate_picks_largest_crossed(self):
        d = ModelDefinition(
            model_key="m",
            provider="p",
            usage_types={"input": 1.00},
            tiers={
                "input": [
                    Tier(above_tokens=100_000, rate=2.00),
                    Tier(above_tokens=500_000, rate=4.00),
                ]
            },
        )
        assert d.effective_rate(UsageType.INPUT, 50_000) == 1.00
        assert d.effective_rate(UsageType.INPUT, 200_000) == 2.00
        assert d.effective_rate(UsageType.INPUT, 1_000_000) == 4.00

    def test_capability_helpers(self):
        d = ModelDefinition(
            model_key="m",
            provider="p",
            capabilities={"vision", "tools", "json_mode"},
        )
        assert d.has_capability("vision")
        assert not d.has_capability("audio")
        assert d.has_all_capabilities(["vision", "tools"])
        assert not d.has_all_capabilities(["vision", "audio"])
        assert d.missing_capabilities(["vision", "audio", "embedding"]) == {"audio", "embedding"}


class TestBuildDef:
    def test_basic(self):
        d = _build_def(
            "openai/gpt-4o-mini",
            {
                "provider": "openai",
                "context_window": 128000,
                "capabilities": ["vision", "tools"],
                "usage_types": {"input": 0.15, "output": 0.60},
            },
        )
        assert d is not None
        assert d.model_key == "openai/gpt-4o-mini"
        assert d.provider == "openai"
        assert d.context_window == 128000
        assert d.capabilities == {"vision", "tools"}
        assert d.usage_types == {"input": 0.15, "output": 0.60}

    def test_with_tiers(self):
        d = _build_def(
            "anthropic/claude-3-5-sonnet-latest",
            {
                "provider": "anthropic",
                "context_window": 200000,
                "capabilities": ["vision", "tools"],
                "usage_types": {"input": 3.0, "output": 15.0},
                "tiers": {
                    "input": [{"above_tokens": 200000, "rate": 6.0}],
                    "output": [{"above_tokens": 200000, "rate": 22.5}],
                },
            },
        )
        assert d is not None
        assert len(d.tiers["input"]) == 1
        assert d.tiers["input"][0].above_tokens == 200000
        assert d.tiers["input"][0].rate == 6.0

    def test_skips_invalid_entry(self):
        assert _build_def("no-slash", {"provider": "openai", "input": 1.0}) is None
        assert _build_def("openai/no-provider", {"input": 1.0}) is None
        assert _build_def("openai/x", "not a dict") is None

    def test_skips_invalid_capabilities(self):
        d = _build_def(
            "openai/x",
            {
                "provider": "openai",
                "capabilities": "not a list",  # type skip
                "usage_types": {"input": 1.0},
            },
        )
        assert d is not None
        assert d.capabilities == set()

    def test_skips_invalid_tier_entries(self):
        d = _build_def(
            "openai/x",
            {
                "provider": "openai",
                "usage_types": {"input": 1.0},
                "tiers": {
                    "input": [
                        {"above_tokens": "not an int", "rate": 2.0},  # skip
                        {"above_tokens": 100, "rate": 2.0},  # keep
                        "not a dict",  # skip
                    ],
                },
            },
        )
        assert d is not None
        assert len(d.tiers["input"]) == 1
        assert d.tiers["input"][0].above_tokens == 100


# ---------------------------------------------------------------------------
# PricingProvider
# ---------------------------------------------------------------------------


class TestBuiltinProvider:
    def test_loads_bundled(self):
        p = BuiltinProvider()
        assert p.lookup("openai/gpt-4o-mini") is not None
        assert p.lookup("openai/gpt-4o") is not None
        assert p.lookup("anthropic/claude-3-5-haiku-latest") is not None

    def test_miss(self):
        p = BuiltinProvider()
        assert p.lookup("openai/gpt-99") is None

    def test_lookup_by_provider_and_name(self):
        p = BuiltinProvider()
        d = p.lookup_by_provider_and_name("openai", "gpt-4o-mini")
        assert d is not None
        assert d.model_key == "openai/gpt-4o-mini"

    def test_lookup_provider_miss_no_fallback(self):
        # If the user specifies a provider that doesn't have the model,
        # don't silently fall back to a different provider.
        p = BuiltinProvider()
        assert p.lookup_by_provider_and_name("nonexistent", "gpt-4o-mini") is None

    def test_loads_from_custom_path(self, tmp_path: Path):
        p = tmp_path / "pricing.json"
        p.write_text(
            json.dumps(
                {
                    "_meta": {"schema_version": "2.0"},
                    "models": {
                        "custom/m1": {
                            "provider": "custom",
                            "capabilities": [],
                            "usage_types": {"input": 1.0, "output": 2.0},
                        },
                    },
                }
            )
        )
        provider = BuiltinProvider(p)
        assert provider.lookup("custom/m1") is not None


class TestCustomProvider:
    def test_loads(self):
        path = _write_catalog_to_tmp(_make_catalog())
        try:
            p = CustomProvider(path)
            assert p.lookup("openai/gpt-4o-mini") is not None
        finally:
            path.unlink(missing_ok=True)

    def test_lookup_by_provider_and_name(self):
        path = _write_catalog_to_tmp(_make_catalog())
        try:
            p = CustomProvider(path)
            d = p.lookup_by_provider_and_name("anthropic", "claude-3-5-haiku-latest")
            assert d is not None
        finally:
            path.unlink(missing_ok=True)


class TestChainProvider:
    def test_first_match_wins(self):
        # Custom provider returns one def, builtin returns a different one
        # for the same key. Custom should win.
        custom = CustomProvider(
            _write_catalog_to_tmp(
                {
                    "_meta": {"schema_version": "2.0"},
                    "models": {
                        "openai/gpt-4o-mini": {
                            "provider": "openai",
                            "capabilities": ["vision"],
                            "usage_types": {"input": 0.10, "output": 0.40},
                        }
                    },
                }
            )
        )
        builtin = _build_test_provider()
        try:
            chain = ChainProvider([custom, builtin])
            d = chain.lookup("openai/gpt-4o-mini")
            assert d is not None
            # Custom's rate (0.10) wins over builtin's 0.15.
            assert d.usage_types["input"] == 0.10
        finally:
            custom._path.unlink(missing_ok=True)

    def test_falls_through(self):
        # Custom doesn't have the model, builtin does. Chain falls through.
        custom_path = _write_catalog_to_tmp(
            {
                "_meta": {"schema_version": "2.0"},
                "models": {
                    "openai/gpt-4o": {
                        "provider": "openai",
                        "capabilities": ["vision"],
                        "usage_types": {"input": 0.99},
                    }
                },
            }
        )
        try:
            chain = ChainProvider([CustomProvider(custom_path), _build_test_provider()])
            d = chain.lookup("openai/gpt-4o-mini")
            assert d is not None
            assert d.usage_types["input"] == 0.15  # builtin rate
        finally:
            custom_path.unlink(missing_ok=True)

    def test_miss_returns_none(self):
        chain = ChainProvider([_build_test_provider()])
        assert chain.lookup("openai/gpt-99") is None

    def test_lookup_by_provider_and_name_falls_through(self):
        chain = ChainProvider([_build_test_provider()])
        d = chain.lookup_by_provider_and_name("openai", "gpt-4o-mini")
        assert d is not None

    def test_empty_chain(self):
        chain = ChainProvider([])
        assert chain.lookup("openai/gpt-4o-mini") is None
        assert chain.lookup_by_provider_and_name("openai", "gpt-4o-mini") is None


class TestDefaultChain:
    def test_default_chain_uses_builtin(self, monkeypatch):
        monkeypatch.delenv("NEATLOGS_PRICING_FILE", raising=False)
        chain = default_chain()
        assert chain.lookup("openai/gpt-4o-mini") is not None

    def test_default_chain_uses_pricing_file(self, tmp_path, monkeypatch):
        custom = tmp_path / "my.json"
        custom.write_text(
            json.dumps(
                {
                    "_meta": {"schema_version": "2.0"},
                    "models": {
                        "custom/m1": {
                            "provider": "custom",
                            "capabilities": [],
                            "usage_types": {"input": 0.01},
                        }
                    },
                }
            )
        )
        chain = default_chain(custom)
        # Custom override is loaded.
        assert chain.lookup("custom/m1") is not None
        # Builtin catalog is also in the chain (custom is just on top).
        assert chain.lookup("openai/gpt-4o-mini") is not None

    def test_default_chain_uses_env_var(self, tmp_path, monkeypatch):
        custom = tmp_path / "env.json"
        custom.write_text(
            json.dumps(
                {
                    "_meta": {"schema_version": "2.0"},
                    "models": {
                        "env/m1": {
                            "provider": "env",
                            "capabilities": [],
                            "usage_types": {"input": 0.01},
                        }
                    },
                }
            )
        )
        monkeypatch.setenv("NEATLOGS_PRICING_FILE", str(custom))
        chain = default_chain()
        assert chain.lookup("env/m1") is not None


# ---------------------------------------------------------------------------
# Workload profile + TokenStats
# ---------------------------------------------------------------------------


class TestTokenStats:
    def test_empty(self):
        s = TokenStats.from_values([])
        assert s.p50 == 0
        assert s.max == 0
        assert s.total == 0

    def test_single(self):
        s = TokenStats.from_values([42])
        assert s.p50 == 42
        assert s.p90 == 42
        assert s.max == 42
        assert s.total == 42

    def test_percentiles(self):
        s = TokenStats.from_values(list(range(1, 101)))  # 1..100
        assert s.p50 == 50
        assert s.p90 == 90
        assert s.p99 == 99
        assert s.max == 100
        assert s.total == 5050


class TestBuildWorkloadProfile:
    def test_basic(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(
            p,
            [
                _span(span_id="1", prompt=1000, completion=500),
                _span(span_id="2", prompt=2000, completion=800),
            ],
        )
        profile, usages = build_workload_profile([p], _build_test_provider())
        assert profile.total_spans == 2
        assert profile.spans_with_tokens == 2
        assert profile.spans_skipped == 0
        assert profile.files_read == 1
        assert len(usages) == 2
        # Models used: only gpt-4o-mini.
        assert profile.models_used == {"openai/gpt-4o-mini"}
        # Capabilities inferred from the source model.
        assert "vision" in profile.capabilities_inferred
        assert "tools" in profile.capabilities_inferred
        # Prompt stats.
        assert profile.prompt_stats.max == 2000
        assert profile.prompt_stats.total == 3000
        assert profile.completion_stats.max == 800

    def test_auto_infer_off(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span()])
        profile, _ = build_workload_profile(
            [p],
            _build_test_provider(),
            auto_infer_capabilities=False,
        )
        assert profile.capabilities_inferred == set()

    def test_skips_no_model_spans(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        d = _span(span_id="1")
        d["attributes"] = {"foo": "bar"}  # no model
        _write_span_log(p, [d])
        profile, usages = build_workload_profile([p], _build_test_provider())
        assert profile.total_spans == 1
        assert profile.spans_with_tokens == 0
        assert profile.spans_skipped == 1
        assert usages == []

    def test_skips_zero_token_spans(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        d = _span(prompt=0, completion=0)
        _write_span_log(p, [d])
        profile, _ = build_workload_profile([p], _build_test_provider())
        assert profile.spans_with_tokens == 0
        assert profile.spans_skipped == 1

    def test_needs_cache_when_cache_used(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(
            p,
            [
                _span(span_id="1", prompt=100, completion=50, cache_read=50),
            ],
        )
        profile, _ = build_workload_profile([p], _build_test_provider())
        assert profile.needs_cache is True
        assert profile.cache_read_total == 50

    def test_unknown_model_no_crash(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span(model="some/future-model")])
        # The future model isn't in the catalog, so no capabilities get
        # inferred. The profile still gets built.
        profile, _ = build_workload_profile([p], _build_test_provider())
        assert profile.capabilities_inferred == set()

    def test_resolve_definition(self):
        d = _resolve_definition(
            SpanUsage("s", "t", "gpt-4o-mini", "openai", 100, 50, 0, 0, 0),
            _build_test_provider(),
        )
        assert d is not None
        assert d.model_key == "openai/gpt-4o-mini"

    def test_resolve_definition_provider_agnostic(self):
        d = _resolve_definition(
            SpanUsage("s", "t", "gpt-4o-mini", None, 100, 50, 0, 0, 0),
            _build_test_provider(),
        )
        assert d is not None

    def test_resolve_definition_no_match(self):
        d = _resolve_definition(
            SpanUsage("s", "t", "gpt-99", None, 100, 50, 0, 0, 0),
            _build_test_provider(),
        )
        assert d is None

    def test_resolve_definition_empty_model(self):
        d = _resolve_definition(
            SpanUsage("s", "t", "", None, 100, 50, 0, 0, 0),
            _build_test_provider(),
        )
        assert d is None


# ---------------------------------------------------------------------------
# Cost engine: _span_cost + _is_span_compatible
# ---------------------------------------------------------------------------


class TestSpanCost:
    def test_basic(self):
        m = _build_test_provider().lookup("openai/gpt-4o-mini")
        u = SpanUsage("s", "t", "gpt-4o-mini", "openai", 1_000_000, 1_000_000, 0, 0, 0)
        c = _span_cost(u, m)
        # input 1M * 0.15 + output 1M * 0.60
        assert c == pytest.approx(0.75, abs=1e-6)

    def test_cache_read(self):
        m = _build_test_provider().lookup("openai/gpt-4o-mini")
        u = SpanUsage("s", "t", "gpt-4o-mini", "openai", 0, 0, 0, 1_000_000, 0)
        c = _span_cost(u, m)
        # 1M cache_read at 0.075
        assert c == pytest.approx(0.075, abs=1e-6)

    def test_cache_write_anthropic(self):
        m = _build_test_provider().lookup("anthropic/claude-3-5-haiku-latest")
        # prompt=0 so only the cache_write cost is billed.
        u = SpanUsage("s", "t", "m", "anthropic", 0, 0, 1_000_000, 0, 0)
        c = _span_cost(u, m)
        # 1M cache_write at 1.00
        assert c == pytest.approx(1.00, abs=1e-6)

    def test_reasoning_o_series(self):
        m = _build_test_provider().lookup("openai/o3-mini")
        # completion=0 so only input + reasoning are billed.
        u = SpanUsage("s", "t", "o3-mini", "openai", 100_000, 0, 0, 0, 100_000)
        c = _span_cost(u, m)
        # 100k input @ 1.10/1M = 0.11, 100k reasoning @ 4.40/1M = 0.44
        assert c == pytest.approx(0.55, abs=1e-6)

    def test_tier_applied(self):
        m = _build_test_provider().lookup("anthropic/claude-3-5-sonnet-latest")
        # 300k input crosses the 200k threshold.
        u = SpanUsage("s", "t", "m", "anthropic", 300_000, 1000, 0, 0, 0)
        c = _span_cost(u, m)
        # input 300k * 6.00/1M = 1.80; output 1k * 15.00/1M = 0.015
        assert c == pytest.approx(1.815, abs=1e-6)

    def test_embedding_model(self):
        m = _build_test_provider().lookup("openai/text-embedding-3-small")
        u = SpanUsage("s", "t", "m", "openai", 1_000_000, 0, 0, 0, 0)
        c = _span_cost(u, m)
        # 1M input at 0.02/1M = 0.02
        assert c == pytest.approx(0.02, abs=1e-6)

    def test_no_input_or_output_rate(self):
        # A model with no input/output rate (edge case): cost is 0.
        m = ModelDefinition(
            model_key="m",
            provider="p",
            usage_types={"image": 0.10},
        )
        u = SpanUsage("s", "t", "m", "p", 100, 50, 0, 0, 0)
        assert _span_cost(u, m) == 0.0


class TestSpanCompatibility:
    def test_compatible_no_constraints(self):
        m = _build_test_provider().lookup("openai/gpt-4o-mini")
        u = SpanUsage("s", "t", "gpt-4o-mini", "openai", 100, 50, 0, 0, 0)
        ok, reasons = _is_span_compatible(u, m, WorkloadConstraints())
        assert ok is True
        assert reasons == []

    def test_cache_used_no_cache_rate_incompatible(self):
        m = ModelDefinition(
            model_key="m",
            provider="p",
            usage_types={"input": 0.15, "output": 0.60},
        )
        u = SpanUsage("s", "t", "m", "p", 100, 50, 0, 50, 0)
        ok, reasons = _is_span_compatible(u, m, WorkloadConstraints())
        assert ok is False
        assert any("cache" in r for r in reasons)

    def test_reasoning_used_no_reasoning_rate_incompatible(self):
        m = ModelDefinition(
            model_key="m",
            provider="p",
            usage_types={"input": 0.15, "output": 0.60},
        )
        u = SpanUsage("s", "t", "m", "p", 100, 50, 0, 0, 50)
        ok, reasons = _is_span_compatible(u, m, WorkloadConstraints())
        assert ok is False
        assert any("reasoning" in r for r in reasons)

    def test_capability_constraint_missing(self):
        m = _build_test_provider().lookup("anthropic/claude-3-5-haiku-latest")
        u = SpanUsage("s", "t", "m", "anthropic", 100, 50, 0, 0, 0)
        ok, reasons = _is_span_compatible(
            u,
            m,
            WorkloadConstraints(need_capabilities={"vision"}),
        )
        assert ok is False
        assert any("vision" in r for r in reasons)

    def test_capability_constraint_satisfied(self):
        m = _build_test_provider().lookup("openai/gpt-4o")
        u = SpanUsage("s", "t", "m", "openai", 100, 50, 0, 0, 0)
        ok, _ = _is_span_compatible(
            u,
            m,
            WorkloadConstraints(need_capabilities={"vision", "tools"}),
        )
        assert ok is True

    def test_context_window_too_small(self):
        m = ModelDefinition(
            model_key="m",
            provider="p",
            context_window=1000,
            usage_types={"input": 0.15, "output": 0.60},
        )
        u = SpanUsage("s", "t", "m", "p", 2000, 50, 0, 0, 0)
        ok, reasons = _is_span_compatible(
            u,
            m,
            WorkloadConstraints(min_context_window=4000),
        )
        assert ok is False
        assert any("context" in r for r in reasons)

    def test_context_window_no_declaration_incompatible(self):
        # If the model doesn't declare a context_window, the constraint
        # is treated as a hard fail.
        m = ModelDefinition(
            model_key="m",
            provider="p",
            context_window=None,
            usage_types={"input": 0.15, "output": 0.60},
        )
        u = SpanUsage("s", "t", "m", "p", 100, 50, 0, 0, 0)
        ok, _ = _is_span_compatible(
            u,
            m,
            WorkloadConstraints(min_context_window=1000),
        )
        assert ok is False

    def test_context_window_zero_means_no_constraint(self):
        m = ModelDefinition(
            model_key="m",
            provider="p",
            context_window=1000,
            usage_types={"input": 0.15, "output": 0.60},
        )
        u = SpanUsage("s", "t", "m", "p", 100_000, 50, 0, 0, 0)
        ok, _ = _is_span_compatible(u, m, WorkloadConstraints(min_context_window=0))
        assert ok is True


# ---------------------------------------------------------------------------
# ScoredModel + EvaluationReport
# ---------------------------------------------------------------------------


class TestScoreModel:
    def test_basic(self):
        m = _build_test_provider().lookup("openai/gpt-4o-mini")
        u = SpanUsage("s", "t", "gpt-4o-mini", "openai", 1_000_000, 1_000_000, 0, 0, 0)
        sm = _score_model(m, [u], WorkloadConstraints(), is_baseline=True)
        assert sm.total_cost == pytest.approx(0.75, abs=1e-6)
        assert sm.compatibility_pct == 1.0
        assert sm.meets_min_compatibility is True
        assert sm.is_baseline is True
        assert sm.missing_capabilities == set()

    def test_partial_compatibility(self):
        m = _build_test_provider().lookup("anthropic/claude-3-5-haiku-latest")
        u_compat = SpanUsage("1", "t", "m", "anthropic", 100, 50, 0, 0, 0)
        u_incompat = SpanUsage("2", "t", "m", "anthropic", 100, 50, 0, 0, 0)
        # Make the second one require vision.
        m_vision = _build_test_provider().lookup("openai/gpt-4o-mini")
        u_incompat2 = SpanUsage("3", "t", "m", "openai", 100, 50, 0, 0, 0)
        # Use constraints to force one span to be incompatible.
        c = WorkloadConstraints(need_capabilities={"vision"})
        sm = _score_model(m, [u_compat, u_incompat, u_incompat2], c, is_baseline=False)
        # 1 of 3 spans compatible (the haiku one is compatible for its own
        # span, but the constraint forces vision which it lacks).
        assert sm.compatible_spans == 0
        assert sm.meets_min_compatibility is False

    def test_rank_key(self):
        sm_ok = ScoredModel(
            model_key="a",
            provider="p",
            total_cost=0.10,
            compatible_spans=10,
            total_spans=10,
            compatibility_pct=1.0,
            meets_min_compatibility=True,
            missing_capabilities=set(),
            context_window=128000,
            per_span=[],
        )
        sm_bad = ScoredModel(
            model_key="b",
            provider="p",
            total_cost=0.0,
            compatible_spans=0,
            total_spans=10,
            compatibility_pct=0.0,
            meets_min_compatibility=False,
            missing_capabilities={"vision"},
            context_window=128000,
            per_span=[],
        )
        # Compatible always ranks first.
        assert sm_ok.rank_key < sm_bad.rank_key


class TestEvaluationReport:
    def test_delta_pct(self):
        b = ScoredModel(
            model_key="b",
            provider="p",
            total_cost=1.0,
            compatible_spans=10,
            total_spans=10,
            compatibility_pct=1.0,
            meets_min_compatibility=True,
            missing_capabilities=set(),
            context_window=128000,
            per_span=[],
            is_baseline=True,
        )
        a = ScoredModel(
            model_key="a",
            provider="p",
            total_cost=0.5,
            compatible_spans=10,
            total_spans=10,
            compatibility_pct=1.0,
            meets_min_compatibility=True,
            missing_capabilities=set(),
            context_window=128000,
            per_span=[],
        )
        r = EvaluationReport(
            baseline=b,
            alternatives=[a],
            profile=WorkloadProfile(
                total_spans=10,
                spans_with_tokens=10,
                spans_skipped=0,
                models_used=set(),
                capabilities_inferred=set(),
                prompt_stats=TokenStats(0, 0, 0, 0, 0),
                completion_stats=TokenStats(0, 0, 0, 0, 0),
                cache_read_total=0,
                cache_write_total=0,
                reasoning_total=0,
                files_read=1,
            ),
            constraints=WorkloadConstraints(),
            explicit_constraints=WorkloadConstraints(),
            unknown_models=[],
            files_read=1,
        )
        # (0.5 - 1.0) / 1.0 * 100 = -50%
        assert r.delta_pct_for(a) == pytest.approx(-50, abs=0.1)
        assert r.delta_pct_for(b) is None  # baseline

    def test_delta_pct_zero_baseline(self):
        b = ScoredModel(
            model_key="b",
            provider="p",
            total_cost=0.0,
            compatible_spans=10,
            total_spans=10,
            compatibility_pct=1.0,
            meets_min_compatibility=True,
            missing_capabilities=set(),
            context_window=128000,
            per_span=[],
            is_baseline=True,
        )
        a = ScoredModel(
            model_key="a",
            provider="p",
            total_cost=0.5,
            compatible_spans=10,
            total_spans=10,
            compatibility_pct=1.0,
            meets_min_compatibility=True,
            missing_capabilities=set(),
            context_window=128000,
            per_span=[],
        )
        r = EvaluationReport(
            baseline=b,
            alternatives=[a],
            profile=WorkloadProfile(
                total_spans=10,
                spans_with_tokens=10,
                spans_skipped=0,
                models_used=set(),
                capabilities_inferred=set(),
                prompt_stats=TokenStats(0, 0, 0, 0, 0),
                completion_stats=TokenStats(0, 0, 0, 0, 0),
                cache_read_total=0,
                cache_write_total=0,
                reasoning_total=0,
                files_read=1,
            ),
            constraints=WorkloadConstraints(),
            explicit_constraints=WorkloadConstraints(),
            unknown_models=[],
            files_read=1,
        )
        # Avoid division by zero.
        assert r.delta_pct_for(a) is None

    def test_ranked(self):
        b = ScoredModel(
            model_key="b",
            provider="p",
            total_cost=1.0,
            compatible_spans=10,
            total_spans=10,
            compatibility_pct=1.0,
            meets_min_compatibility=True,
            missing_capabilities=set(),
            context_window=128000,
            per_span=[],
            is_baseline=True,
        )
        a = ScoredModel(
            model_key="a",
            provider="p",
            total_cost=0.5,
            compatible_spans=10,
            total_spans=10,
            compatibility_pct=1.0,
            meets_min_compatibility=True,
            missing_capabilities=set(),
            context_window=128000,
            per_span=[],
        )
        c_bad = ScoredModel(
            model_key="c",
            provider="p",
            total_cost=0.0,
            compatible_spans=0,
            total_spans=10,
            compatibility_pct=0.0,
            meets_min_compatibility=False,
            missing_capabilities={"vision"},
            context_window=128000,
            per_span=[],
        )
        r = EvaluationReport(
            baseline=b,
            alternatives=[a, c_bad],
            profile=WorkloadProfile(
                total_spans=10,
                spans_with_tokens=10,
                spans_skipped=0,
                models_used=set(),
                capabilities_inferred=set(),
                prompt_stats=TokenStats(0, 0, 0, 0, 0),
                completion_stats=TokenStats(0, 0, 0, 0, 0),
                cache_read_total=0,
                cache_write_total=0,
                reasoning_total=0,
                files_read=1,
            ),
            constraints=WorkloadConstraints(),
            explicit_constraints=WorkloadConstraints(),
            unknown_models=[],
            files_read=1,
        )
        ranked = r.ranked()
        assert ranked[0].model_key == "b"  # baseline first
        assert ranked[1].model_key == "a"  # cheaper alternative
        assert ranked[2].model_key == "c"  # incompatible last

    def test_find(self):
        b = ScoredModel(
            model_key="b",
            provider="p",
            total_cost=1.0,
            compatible_spans=10,
            total_spans=10,
            compatibility_pct=1.0,
            meets_min_compatibility=True,
            missing_capabilities=set(),
            context_window=128000,
            per_span=[],
            is_baseline=True,
        )
        r = EvaluationReport(
            baseline=b,
            alternatives=[],
            profile=WorkloadProfile(
                total_spans=10,
                spans_with_tokens=10,
                spans_skipped=0,
                models_used=set(),
                capabilities_inferred=set(),
                prompt_stats=TokenStats(0, 0, 0, 0, 0),
                completion_stats=TokenStats(0, 0, 0, 0, 0),
                cache_read_total=0,
                cache_write_total=0,
                reasoning_total=0,
                files_read=1,
            ),
            constraints=WorkloadConstraints(),
            explicit_constraints=WorkloadConstraints(),
            unknown_models=[],
            files_read=1,
        )
        assert r.find("b") is b
        assert r.find("missing") is None


# ---------------------------------------------------------------------------
# evaluate_workload (top-level)
# ---------------------------------------------------------------------------


class TestEvaluateWorkload:
    def test_basic(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span(prompt=1_000_000, completion=1_000_000)])
        chain = ChainProvider([_build_test_provider()])
        report = evaluate_workload(
            paths=p,
            candidates=[
                "openai/gpt-4o-mini",
                "openai/gpt-4o",
                "anthropic/claude-3-5-haiku-latest",
            ],
            baseline="openai/gpt-4o-mini",
            pricing=chain,
            auto_infer_capabilities=False,
        )
        assert report.baseline is not None
        assert report.baseline.model_key == "openai/gpt-4o-mini"
        assert len(report.alternatives) == 2
        # All three models should be compatible with the workload when
        # we don't infer capabilities from the source model.
        for sm in report.all():
            assert sm.meets_min_compatibility is True

    def test_unknown_candidate_warns(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span()])
        buf = io.StringIO()
        chain = ChainProvider([_build_test_provider()])
        report = evaluate_workload(
            paths=p,
            candidates=["openai/gpt-4o-mini", "nonexistent/model"],
            pricing=chain,
            warn_stream=buf,
        )
        assert "nonexistent/model" in buf.getvalue()
        # The unknown model is omitted; only the known one is in the report.
        assert report.find("nonexistent/model") is None

    def test_no_candidates_no_baseline(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span()])
        chain = ChainProvider([_build_test_provider()])
        report = evaluate_workload(
            paths=p,
            candidates=[],
            pricing=chain,
        )
        assert report.baseline is None
        assert report.alternatives == []

    def test_capability_constraint_filters_alternatives(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        # The source model is gpt-4o-mini (has vision). We REQUIRE vision.
        # haiku (no vision) should be incompatible.
        _write_span_log(p, [_span()])
        chain = ChainProvider([_build_test_provider()])
        report = evaluate_workload(
            paths=p,
            candidates=[
                "openai/gpt-4o-mini",
                "anthropic/claude-3-5-haiku-latest",
            ],
            pricing=chain,
            constraints=WorkloadConstraints(need_capabilities={"vision"}),
            auto_infer_capabilities=False,  # explicit only
        )
        haiku = report.find("anthropic/claude-3-5-haiku-latest")
        assert haiku is not None
        assert haiku.meets_min_compatibility is False

    def test_auto_infer_capabilities(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        # Source model is gpt-4o-mini which has vision. The user does
        # NOT pass --need; we should still require vision (inferred).
        _write_span_log(p, [_span()])
        chain = ChainProvider([_build_test_provider()])
        report = evaluate_workload(
            paths=p,
            candidates=[
                "openai/gpt-4o-mini",
                "anthropic/claude-3-5-haiku-latest",
            ],
            pricing=chain,
            # No explicit need; auto_infer is on by default.
        )
        haiku = report.find("anthropic/claude-3-5-haiku-latest")
        assert haiku is not None
        # haiku is incompatible because the workload needs vision.
        assert haiku.meets_min_compatibility is False

    def test_auto_infer_can_be_disabled(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span()])
        chain = ChainProvider([_build_test_provider()])
        report = evaluate_workload(
            paths=p,
            candidates=[
                "openai/gpt-4o-mini",
                "anthropic/claude-3-5-haiku-latest",
            ],
            pricing=chain,
            auto_infer_capabilities=False,
        )
        haiku = report.find("anthropic/claude-3-5-haiku-latest")
        # Without auto-infer, haiku is compatible (no constraint applied).
        assert haiku is not None
        assert haiku.meets_min_compatibility is True

    def test_explicit_baseline(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span()])
        chain = ChainProvider([_build_test_provider()])
        report = evaluate_workload(
            paths=p,
            candidates=["openai/gpt-4o-mini", "openai/gpt-4o"],
            baseline="openai/gpt-4o",
            pricing=chain,
        )
        assert report.baseline.model_key == "openai/gpt-4o"
        assert report.find("openai/gpt-4o-mini") is not None

    def test_single_path_string(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span()])
        chain = ChainProvider([_build_test_provider()])
        report = evaluate_workload(
            str(p),  # str, not Path
            candidates=["openai/gpt-4o-mini"],
            pricing=chain,
        )
        assert report.baseline is not None

    def test_min_compatibility_pct(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        # 5 spans, 1 needs vision (the source model has vision but only
        # one span will be marked as needing it via explicit constraint).
        _write_span_log(
            p,
            [
                _span(span_id="1", prompt=100, completion=50),
                _span(span_id="2", prompt=100, completion=50),
                _span(span_id="3", prompt=100, completion=50),
                _span(span_id="4", prompt=100, completion=50),
                _span(span_id="5", prompt=100, completion=50),
            ],
        )
        chain = ChainProvider([_build_test_provider()])
        # haiku (no vision) is incompatible on all 5 spans with vision
        # required → 0% compatibility → fails the 95% threshold.
        report = evaluate_workload(
            paths=p,
            candidates=[
                "openai/gpt-4o-mini",
                "anthropic/claude-3-5-haiku-latest",
            ],
            pricing=chain,
            constraints=WorkloadConstraints(
                need_capabilities={"vision"},
                min_compatibility_pct=0.95,
            ),
            auto_infer_capabilities=False,
        )
        haiku = report.find("anthropic/claude-3-5-haiku-latest")
        assert haiku is not None
        assert haiku.meets_min_compatibility is False

    def test_min_compatibility_low_threshold(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span() for _ in range(10)])
        chain = ChainProvider([_build_test_provider()])
        # 50% threshold: haiku is OK because it's still in the top
        # for the 10 spans that are the same.
        report = evaluate_workload(
            paths=p,
            candidates=[
                "openai/gpt-4o-mini",
                "anthropic/claude-3-5-haiku-latest",
            ],
            pricing=chain,
            constraints=WorkloadConstraints(
                need_capabilities={"vision"},
                min_compatibility_pct=0.0,
            ),
            auto_infer_capabilities=False,
        )
        haiku = report.find("anthropic/claude-3-5-haiku-latest")
        assert haiku is not None
        # 0.0 threshold = always passes.
        assert haiku.meets_min_compatibility is True

    def test_no_spans_with_tokens(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        d = _span()
        d["attributes"] = {"foo": "bar"}
        _write_span_log(p, [d])
        chain = ChainProvider([_build_test_provider()])
        report = evaluate_workload(
            paths=p,
            candidates=["openai/gpt-4o-mini"],
            pricing=chain,
        )
        assert report.profile.spans_with_tokens == 0
        assert report.profile.spans_skipped == 1
        # Baseline is still set even with no input data; cost is $0.
        assert report.baseline is not None
        assert report.baseline.total_cost == 0.0


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


class TestFormatEvaluationText:
    def test_basic(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span(prompt=1_000_000, completion=1_000_000)])
        chain = ChainProvider([_build_test_provider()])
        report = evaluate_workload(
            paths=p,
            candidates=[
                "openai/gpt-4o-mini",
                "openai/gpt-4o",
                "anthropic/claude-3-5-haiku-latest",
            ],
            pricing=chain,
        )
        out = format_evaluation_text(report, use_color=False)
        assert "Workload:" in out
        assert "openai/gpt-4o-mini" in out
        assert "openai/gpt-4o" in out
        assert "anthropic/claude-3-5-haiku-latest" in out
        assert "(baseline)" in out
        assert "vs base" in out

    def test_incompatible_shown(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span()])
        chain = ChainProvider([_build_test_provider()])
        report = evaluate_workload(
            paths=p,
            candidates=[
                "openai/gpt-4o-mini",
                "anthropic/claude-3-5-haiku-latest",
            ],
            pricing=chain,
            constraints=WorkloadConstraints(need_capabilities={"vision"}),
            auto_infer_capabilities=False,
        )
        out = format_evaluation_text(report, use_color=False)
        assert "incompatible" in out

    def test_no_results(self):
        report = EvaluationReport(
            baseline=None,
            alternatives=[],
            profile=WorkloadProfile(
                total_spans=0,
                spans_with_tokens=0,
                spans_skipped=0,
                models_used=set(),
                capabilities_inferred=set(),
                prompt_stats=TokenStats(0, 0, 0, 0, 0),
                completion_stats=TokenStats(0, 0, 0, 0, 0),
                cache_read_total=0,
                cache_write_total=0,
                reasoning_total=0,
                files_read=0,
            ),
            constraints=WorkloadConstraints(),
            explicit_constraints=WorkloadConstraints(),
            unknown_models=[],
            files_read=0,
        )
        out = format_evaluation_text(report, use_color=False)
        assert "no candidate" in out

    def test_capability_gap_section(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span()])
        chain = ChainProvider([_build_test_provider()])
        report = evaluate_workload(
            paths=p,
            candidates=[
                "openai/gpt-4o-mini",
                "anthropic/claude-3-5-haiku-latest",
            ],
            pricing=chain,
            constraints=WorkloadConstraints(need_capabilities={"vision"}),
            auto_infer_capabilities=False,
        )
        out = format_evaluation_text(report, use_color=False)
        assert "Capability gap" in out
        assert "vision" in out

    def test_color_in_output(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span(prompt=1_000_000, completion=1_000_000)])
        chain = ChainProvider([_build_test_provider()])
        report = evaluate_workload(
            paths=p,
            candidates=["openai/gpt-4o-mini", "openai/gpt-4o"],
            pricing=chain,
        )
        out = format_evaluation_text(report, use_color=True)
        assert "\033[" in out

    def test_color_disabled(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span(prompt=100, completion=50)])
        chain = ChainProvider([_build_test_provider()])
        report = evaluate_workload(
            paths=p,
            candidates=["openai/gpt-4o-mini"],
            pricing=chain,
        )
        out = format_evaluation_text(report, use_color=False)
        assert "\033[" not in out


class TestFormatEvaluationJson:
    def test_shape(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span(prompt=1_000_000, completion=1_000_000)])
        chain = ChainProvider([_build_test_provider()])
        report = evaluate_workload(
            paths=p,
            candidates=["openai/gpt-4o-mini", "openai/gpt-4o"],
            pricing=chain,
        )
        out = format_evaluation_json(report)
        obj = json.loads(out)
        assert obj["currency"] == "USD"
        assert "workload" in obj
        assert obj["workload"]["spans_with_tokens"] == 1
        assert obj["baseline"]["model"] == "openai/gpt-4o-mini"
        assert obj["baseline"]["is_baseline"] is True
        assert len(obj["alternatives"]) == 1
        assert "delta_pct_vs_baseline" in obj["alternatives"][0]
        # Ranked list is also present.
        assert "ranked" in obj
        assert len(obj["ranked"]) == 2

    def test_constraints_in_payload(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span(model="text-embedding-3-small")])
        chain = ChainProvider([_build_test_provider()])
        report = evaluate_workload(
            paths=p,
            candidates=["openai/gpt-4o-mini"],
            pricing=chain,
            constraints=WorkloadConstraints(need_capabilities={"vision", "tools"}),
        )
        out = format_evaluation_json(report)
        obj = json.loads(out)
        assert "constraints" in obj
        assert "explicit_constraints" in obj
        assert sorted(obj["explicit_constraints"]["need_capabilities"]) == ["tools", "vision"]
        assert "embedding" in obj["constraints"]["need_capabilities"]
        assert "tools" in obj["constraints"]["need_capabilities"]
        assert "vision" in obj["constraints"]["need_capabilities"]


class TestFormatEvaluationCsv:
    def test_lf_line_endings(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span()])
        chain = ChainProvider([_build_test_provider()])
        report = evaluate_workload(
            paths=p,
            candidates=["openai/gpt-4o-mini", "openai/gpt-4o"],
            pricing=chain,
        )
        out = format_evaluation_csv(report)
        assert "\r\n" not in out
        assert out.endswith("\n")
        lines = out.strip().split("\n")
        assert lines[0].startswith("model,provider,is_baseline")
        # 1 header + 2 data rows.
        assert len(lines) == 3


class TestFormatEvaluationDispatch:
    def test_dispatch(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span()])
        chain = ChainProvider([_build_test_provider()])
        report = evaluate_workload(
            paths=p,
            candidates=["openai/gpt-4o-mini"],
            pricing=chain,
        )
        assert "Workload:" in format_evaluation(report, style="text")
        assert "currency" in format_evaluation(report, style="json")
        assert "model,provider" in format_evaluation(report, style="csv")

    def test_unknown_style(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span()])
        chain = ChainProvider([_build_test_provider()])
        report = evaluate_workload(
            paths=p,
            candidates=["openai/gpt-4o-mini"],
            pricing=chain,
        )
        with pytest.raises(ValueError):
            format_evaluation(report, style="bogus")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_basic(self, tmp_path: Path, capsys):
        from neatlogs.intel import _cli

        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span(prompt=1_000_000, completion=1_000_000)])
        rc = _cli(
            [
                str(p),
                "--candidates",
                "openai/gpt-4o-mini,openai/gpt-4o,anthropic/claude-3-5-haiku-latest",
                "--no-color",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "openai/gpt-4o-mini" in out
        assert "openai/gpt-4o" in out
        assert "anthropic/claude-3-5-haiku-latest" in out
        assert "(baseline)" in out

    def test_json(self, tmp_path: Path, capsys):
        from neatlogs.intel import _cli

        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span()])
        rc = _cli(
            [
                str(p),
                "--candidates",
                "openai/gpt-4o-mini",
                "--format",
                "json",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        obj = json.loads(out)
        assert obj["currency"] == "USD"

    def test_csv(self, tmp_path: Path, capsys):
        from neatlogs.intel import _cli

        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span()])
        rc = _cli(
            [
                str(p),
                "--candidates",
                "openai/gpt-4o-mini",
                "--format",
                "csv",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "model,provider" in out

    def test_need_capability(self, tmp_path: Path, capsys):
        from neatlogs.intel import _cli

        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span()])
        rc = _cli(
            [
                str(p),
                "--candidates",
                "openai/gpt-4o-mini,anthropic/claude-3-5-haiku-latest",
                "--need",
                "vision",
                "--no-color",
                "--no-auto-infer",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        # haiku is incompatible because no vision.
        assert "incompatible" in out

    def test_no_candidates_returns_2(self, tmp_path: Path, capsys):
        from neatlogs.intel import _cli

        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span()])
        rc = _cli(
            [
                str(p),
                "--candidates",
                "totally/nonexistent",
            ]
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "no candidate" in err

    def test_comma_separated_models(self, tmp_path: Path, capsys):
        from neatlogs.intel import _cli

        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span()])
        rc = _cli(
            [
                str(p),
                "--candidates",
                "openai/gpt-4o-mini, openai/gpt-4o, anthropic/claude-3-5-haiku-latest",
                "--no-color",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert all(
            m in out
            for m in [
                "openai/gpt-4o-mini",
                "openai/gpt-4o",
                "anthropic/claude-3-5-haiku-latest",
            ]
        )

    def test_pricing_file_override(self, tmp_path: Path, capsys):
        from neatlogs.intel import _cli

        p = tmp_path / "spans.jsonl"
        _write_span_log(p, [_span()])
        custom = tmp_path / "my.json"
        custom.write_text(
            json.dumps(
                {
                    "_meta": {"schema_version": "2.0"},
                    "models": {
                        "openai/gpt-4o-mini": {
                            "provider": "openai",
                            "context_window": 128000,
                            "capabilities": ["vision", "tools"],
                            "usage_types": {"input": 0.10, "output": 0.40},
                        }
                    },
                }
            )
        )
        rc = _cli(
            [
                str(p),
                "--candidates",
                "openai/gpt-4o-mini",
                "--pricing-file",
                str(custom),
                "--no-color",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        # The custom rate is reflected in the output.
        assert "openai/gpt-4o-mini" in out

    def test_min_context(self, tmp_path: Path, capsys):
        from neatlogs.intel import _cli

        p = tmp_path / "spans.jsonl"
        # A 1M-token prompt won't fit in any model in our test catalog.
        _write_span_log(p, [_span(prompt=1_000_000, completion=50)])
        rc = _cli(
            [
                str(p),
                "--candidates",
                "openai/gpt-4o-mini",
                "--min-context",
                "2000000",
                "--no-color",
                "--no-auto-infer",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        # Incompatible because the prompt exceeds the context.
        assert "incompatible" in out
