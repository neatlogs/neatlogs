"""Unit coverage for provider/model-aware invocation-parameter defaults."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from opentelemetry.trace import SpanKind

import neatlogs
from neatlogs.config.defaults_enricher import (
    DefaultsEnricher,
    enrich_invocation_parameters,
    resolve_default_operations,
)
from neatlogs.core.attribute_processor import UnifiedAttributeProcessor

_DEFAULT_FILE = Path(neatlogs.__file__).resolve().parent / "config" / "model_defaults.json"
_FULL_OPENAI_DEFAULTS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "max_tokens": None,
    "stream": False,
    "n": 1,
}
_MINI_OPENAI_DEFAULTS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}


def _params(attributes: dict) -> dict:
    return json.loads(attributes["llm.invocation_parameters"])


class TestGetDefaults:
    def test_exact_model_match_wins(self):
        defaults = DefaultsEnricher.get_defaults("openai", "chat.completions", "gpt-4o-mini")
        assert defaults == _MINI_OPENAI_DEFAULTS

    def test_exact_match_is_case_insensitive(self):
        defaults = DefaultsEnricher.get_defaults("openai", "chat.completions", "GPT-4O-MINI")
        assert defaults == _MINI_OPENAI_DEFAULTS

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gpt-5-mini-2026-01-01", _MINI_OPENAI_DEFAULTS),
            ("gpt-5-nano-2026-01-01", _MINI_OPENAI_DEFAULTS),
            ("gpt-4o-mini-2026-01-01", _MINI_OPENAI_DEFAULTS),
            ("gpt-4.1-mini-2026-01-01", _MINI_OPENAI_DEFAULTS),
        ],
    )
    def test_longest_prefix_wins_for_versioned_mini_models(self, model, expected):
        """A short family key must not shadow a more specific configured key."""
        assert DefaultsEnricher.get_defaults("openai", "chat.completions", model) == expected

    def test_full_family_still_matches_when_no_longer_key_applies(self):
        assert (
            DefaultsEnricher.get_defaults("openai", "chat.completions", "gpt-5-2026-01-01")
            == _FULL_OPENAI_DEFAULTS
        )

    def test_longest_prefix_wins_even_when_short_key_is_listed_first(self, monkeypatch):
        # The shipped flash and flash-lite entries happen to be identical, so use
        # distinct synthetic values to prove which configured key was selected.
        monkeypatch.setattr(
            DefaultsEnricher,
            "_defaults_data",
            {
                "google": {
                    "generateContent": {
                        "gemini-2.0-flash": {"family": "flash"},
                        "gemini-2.0-flash-lite": {"family": "flash-lite"},
                    }
                }
            },
        )
        defaults = DefaultsEnricher.get_defaults(
            "google", "generateContent", "gemini-2.0-flash-lite-001"
        )
        assert defaults == {"family": "flash-lite"}

    def test_unknown_model_falls_back_to_default(self):
        defaults = DefaultsEnricher.get_defaults("openai", "chat.completions", "gpt-9-ultra")
        expected = json.loads(_DEFAULT_FILE.read_text())["openai"]["chat.completions"]["_default"]
        assert defaults == expected

    def test_missing_provider_or_operation_returns_empty(self):
        assert DefaultsEnricher.get_defaults("does-not-exist", "chat.completions", "gpt-5") == {}
        assert DefaultsEnricher.get_defaults("openai", "nope", "gpt-5") == {}

    def test_blank_model_returns_empty(self):
        assert DefaultsEnricher.get_defaults("openai", "chat.completions", "   ") == {}

    def test_returned_mapping_is_a_copy(self):
        first = DefaultsEnricher.get_defaults("openai", "chat.completions", "gpt-5")
        first["temperature"] = -1
        second = DefaultsEnricher.get_defaults("openai", "chat.completions", "gpt-5")
        assert second["temperature"] == 1.0


class TestResolveDefaultOperations:
    @pytest.mark.parametrize(
        "provider,expected",
        [
            ("openai", "chat.completions"),
            ("azure_openai", "chat.completions"),
            ("groq", "chat.completions"),
            ("anthropic", "messages"),
            ("google", "generateContent"),
            ("vertex_ai", "generateContent"),
            ("cohere", "chat"),
        ],
    )
    def test_provider_conventional_operation(self, provider, expected):
        assert resolve_default_operations(provider, "LLM", {})[-1] == expected

    def test_embedding_span_uses_embedding_operation(self):
        assert resolve_default_operations("openai", "EMBEDDING", {}) == ["embeddings"]

    def test_explicit_operation_is_preferred(self):
        ops = resolve_default_operations("cohere", "LLM", {"gen_ai.operation.name": "generate"})
        assert ops[0] == "generate"

    def test_misleading_request_type_does_not_suppress_provider_fallback(self):
        ops = resolve_default_operations("anthropic", "LLM", {"llm.request.type": "chat"})
        assert ops[-1] == "messages"

    def test_non_string_operation_is_ignored(self):
        assert resolve_default_operations("google", "LLM", {"llm.request.type": 123}) == [
            "generateContent"
        ]


class TestEnrichInvocationParameters:
    @pytest.mark.parametrize(
        "provider,model,expected_key",
        [
            ("openai", "gpt-5", "temperature"),
            ("anthropic", "claude-sonnet-4-5-20250929", "temperature"),
            ("google", "gemini-2.5-pro", "top_k"),
            ("vertex_ai", "gemini-2.5-flash", "top_k"),
            ("cohere", "command-r-plus-08-2024", "temperature"),
        ],
    )
    def test_all_providers_are_enriched(self, provider, model, expected_key):
        attrs = {
            "openinference.span.kind": "LLM",
            "neatlogs.provider": provider,
            "llm.model_name": model,
            "llm.invocation_parameters": "{}",
        }
        enrich_invocation_parameters(attrs)
        assert expected_key in _params(attrs)

    def test_google_defaults_are_the_generate_content_defaults(self):
        attrs = {
            "openinference.span.kind": "LLM",
            "neatlogs.provider": "google",
            "llm.model_name": "gemini-2.5-pro",
            "llm.invocation_parameters": "{}",
        }
        enrich_invocation_parameters(attrs)
        assert _params(attrs) == DefaultsEnricher.get_defaults(
            "google", "generateContent", "gemini-2.5-pro"
        )

    def test_explicit_parameters_take_precedence(self):
        attrs = {
            "openinference.span.kind": "LLM",
            "neatlogs.provider": "google",
            "llm.model_name": "gemini-2.5-pro",
            "llm.invocation_parameters": json.dumps({"temperature": 0.1, "custom": "kept"}),
        }
        enrich_invocation_parameters(attrs)
        params = _params(attrs)
        assert params["temperature"] == 0.1
        assert params["custom"] == "kept"
        assert params["top_k"] == 40

    def test_versioned_mini_model_gets_mini_defaults(self):
        attrs = {
            "openinference.span.kind": "LLM",
            "neatlogs.provider": "openai",
            "llm.model_name": "gpt-5-mini-2026-01-01",
            "llm.invocation_parameters": "{}",
        }
        enrich_invocation_parameters(attrs)
        params = _params(attrs)
        assert params == _MINI_OPENAI_DEFAULTS
        assert "max_tokens" not in params

    def test_explicit_generate_operation_is_honored_for_cohere(self):
        attrs = {
            "openinference.span.kind": "LLM",
            "neatlogs.provider": "cohere",
            "llm.model_name": "command-r-plus-08-2024",
            "gen_ai.operation.name": "generate",
            "llm.invocation_parameters": "{}",
        }
        enrich_invocation_parameters(attrs)
        expected = json.loads(_DEFAULT_FILE.read_text())["cohere"]["generate"]["_default"]
        assert _params(attrs) == expected

    def test_embedding_span_uses_embedding_model_name(self):
        attrs = {
            "openinference.span.kind": "EMBEDDING",
            "neatlogs.provider": "openai",
            "embedding.model_name": "text-embedding-3-small",
            "llm.invocation_parameters": "{}",
        }
        enrich_invocation_parameters(attrs)
        assert _params(attrs)

    @pytest.mark.parametrize("bad_value", ["not json", "[1, 2, 3]", "42", "null", None])
    def test_malformed_existing_parameters_are_replaced(self, bad_value):
        attrs = {
            "openinference.span.kind": "LLM",
            "neatlogs.provider": "google",
            "llm.model_name": "gemini-2.5-pro",
            "llm.invocation_parameters": bad_value,
        }
        enrich_invocation_parameters(attrs)
        assert "top_k" in _params(attrs)

    def test_non_serializable_existing_parameters_leave_value_unchanged(self):
        original = {"unserializable": object()}
        attrs = {
            "openinference.span.kind": "LLM",
            "neatlogs.provider": "google",
            "llm.model_name": "gemini-2.5-pro",
            "llm.invocation_parameters": original,
        }
        enrich_invocation_parameters(attrs)
        assert attrs["llm.invocation_parameters"] is original

    def test_enrichment_can_be_disabled(self):
        attrs = {
            "openinference.span.kind": "LLM",
            "neatlogs.provider": "google",
            "llm.model_name": "gemini-2.5-pro",
            "llm.invocation_parameters": "{}",
        }
        enrich_invocation_parameters(attrs, enable_enrichment=False)
        assert attrs["llm.invocation_parameters"] == "{}"

    @pytest.mark.parametrize("span_kind", ["TOOL", "CHAIN", "WORKFLOW", ""])
    def test_non_llm_spans_are_untouched(self, span_kind):
        attrs = {
            "openinference.span.kind": span_kind,
            "neatlogs.provider": "google",
            "llm.model_name": "gemini-2.5-pro",
            "llm.invocation_parameters": "{}",
        }
        enrich_invocation_parameters(attrs)
        assert attrs["llm.invocation_parameters"] == "{}"

    def test_unknown_model_still_gets_provider_default(self):
        attrs = {
            "openinference.span.kind": "LLM",
            "neatlogs.provider": "openai",
            "llm.model_name": "gpt-9-ultra",
            "llm.invocation_parameters": "{}",
        }
        enrich_invocation_parameters(attrs)
        expected = json.loads(_DEFAULT_FILE.read_text())["openai"]["chat.completions"]["_default"]
        assert _params(attrs) == expected


def _processor_span(attributes: dict) -> SimpleNamespace:
    resource = SimpleNamespace(attributes={})
    ctx = SimpleNamespace(trace_id=1, span_id=2)
    return SimpleNamespace(
        kind=SpanKind.INTERNAL,
        name="llm",
        attributes=attributes,
        resource=resource,
        instrumentation_scope=None,
        events=[],
        links=[],
        start_time=0,
        end_time=1_000_000,
        context=ctx,
        status=SimpleNamespace(status_code=None, description=None),
    )


def test_attribute_processor_enriches_google_llm_parameters_end_to_end():
    mapping_path = Path(neatlogs.__file__).resolve().parent / "config" / "attribute-mapping.json"
    processor = UnifiedAttributeProcessor(
        mapping_config=json.loads(mapping_path.read_text()), debug=False
    )

    unified = processor.process(
        _processor_span(
            {
                "openinference.span.kind": "LLM",
                "llm.model_name": "gemini-2.5-pro",
                "llm.invocation_parameters": "{}",
                "gen_ai.system": "google",
            }
        )
    )

    params = json.loads(unified["neatlogs.llm.invocation_parameters"])
    assert params["top_k"] == 40
    assert params["temperature"] == 1.0
