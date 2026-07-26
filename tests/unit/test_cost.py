"""
Unit tests for neatlogs.cost.

The cost module reads JSONL span logs (same format as neatlogs.replay), pulls
the model name and prompt/completion token counts from each LLM span, looks up
the per-token price, and emits a per-model breakdown plus totals. These tests
exercise compute + format_report end-to-end without spinning up an OTel
pipeline.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List

import pytest

from neatlogs.cost import (
    CostReport,
    ModelCost,
    ModelPrice,
    SpanRecord,
    compute,
    format_report,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Minimal pricing table for tests. Kept in the test file (not on disk) so the
# tests don't depend on the bundled JSON for their own assertions.
PRICING: Dict[str, Dict[str, Dict[str, float]]] = {
    "openai": {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
    },
    "anthropic": {
        "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00},
    },
}


def _processed(
    trace_id: str = "a",
    span_id: str = "1",
    parent: str = None,
    name: str = "openai.chat.completions.create",
    kind: str = "llm",
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    prompt_tokens: int = 1000,
    completion_tokens: int = 500,
) -> SpanRecord:
    """Build a SpanRecord for tests. The local SpanRecord in cost.py is a
    minimal view (attributes + source) so the constructor differs from the
    full one in neatlogs.replay."""
    attrs: Dict[str, object] = {}
    if model is not None:
        attrs["neatlogs.llm.model_name"] = model
    if provider is not None:
        attrs["neatlogs.llm.provider"] = provider
    if prompt_tokens is not None:
        attrs["neatlogs.llm.token_count.prompt"] = prompt_tokens
    if completion_tokens is not None:
        attrs["neatlogs.llm.token_count.completion"] = completion_tokens
    return SpanRecord(attributes=attrs, source="processed")


def _to_json_dict(s: SpanRecord) -> dict:
    return {
        "trace_id": "0xabcd",
        "span_id": "0x1234",
        "parent_span_id": None,
        "name": "openai.chat.completions.create",
        "kind": "llm",
        "start_time": "2024-01-15T10:00:00.000Z",
        "end_time": "2024-01-15T10:00:00.001Z",
        "duration_ns": 1_000_000,
        "attributes": s.attributes,
        "resource": {"attributes": {}},
        "status": {"code": "OK", "description": ""},
        "events": [],
    }


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------


class TestCompute:
    def test_single_model(self, tmp_path: Path):
        spans = [
            _processed(
                "a",
                "1",
                name="openai.chat.completions.create",
                model="gpt-4o-mini",
                provider="openai",
                prompt_tokens=1_000_000,
                completion_tokens=1_000_000,
            ),
        ]
        p = tmp_path / "spans.jsonl"
        p.write_text("\n".join(json.dumps(_to_json_dict(s)) for s in spans))
        report = compute(p, pricing=PRICING, warn_stream=io.StringIO())
        assert len(report.per_model) == 1
        m = report.per_model[0]
        assert m.model == "gpt-4o-mini"
        assert m.provider == "openai"
        assert m.calls == 1
        assert m.prompt_tokens == 1_000_000
        assert m.completion_tokens == 1_000_000
        # input: 1M * 0.15 = 0.15; output: 1M * 0.60 = 0.60 → total 0.75
        assert m.usd == pytest.approx(0.75, abs=1e-6)
        assert report.total_usd == pytest.approx(0.75, abs=1e-6)
        assert report.unknown_models == []

    def test_aggregates_across_calls(self, tmp_path: Path):
        spans = [
            _processed("a", str(i), prompt_tokens=100_000, completion_tokens=50_000)
            for i in range(3)
        ]
        p = tmp_path / "spans.jsonl"
        p.write_text("\n".join(json.dumps(_to_json_dict(s)) for s in spans))
        report = compute(p, pricing=PRICING, warn_stream=io.StringIO())
        assert len(report.per_model) == 1
        m = report.per_model[0]
        assert m.calls == 3
        assert m.prompt_tokens == 300_000
        assert m.completion_tokens == 150_000
        # (300k * 0.15 + 150k * 0.60) / 1M = 0.045 + 0.09 = 0.135
        assert m.usd == pytest.approx(0.135, abs=1e-6)

    def test_multiple_models(self, tmp_path: Path):
        spans = [
            _processed(
                "a",
                "1",
                model="gpt-4o-mini",
                provider="openai",
                prompt_tokens=1_000_000,
                completion_tokens=0,
            ),
            _processed(
                "a",
                "2",
                model="gpt-4o",
                provider="openai",
                prompt_tokens=1_000_000,
                completion_tokens=0,
            ),
        ]
        p = tmp_path / "spans.jsonl"
        p.write_text("\n".join(json.dumps(_to_json_dict(s)) for s in spans))
        report = compute(p, pricing=PRICING, warn_stream=io.StringIO())
        assert len(report.per_model) == 2
        # Both rows are known; sort is by descending cost → gpt-4o first.
        assert report.per_model[0].model == "gpt-4o"
        assert report.per_model[1].model == "gpt-4o-mini"
        # gpt-4o 1M input @ $2.50 = 2.50
        assert report.per_model[0].usd == pytest.approx(2.50, abs=1e-6)
        # gpt-4o-mini 1M input @ $0.15 = 0.15
        assert report.per_model[1].usd == pytest.approx(0.15, abs=1e-6)
        assert report.total_usd == pytest.approx(2.65, abs=1e-6)

    def test_unknown_model_warns(self, tmp_path: Path):
        spans = [
            _processed(
                "a",
                "1",
                model="some-future-model",
                provider="openai",
                prompt_tokens=1000,
                completion_tokens=500,
            ),
        ]
        p = tmp_path / "spans.jsonl"
        p.write_text("\n".join(json.dumps(_to_json_dict(s)) for s in spans))
        buf = io.StringIO()
        report = compute(p, pricing=PRICING, warn_stream=buf)
        assert report.unknown_models == ["some-future-model"]
        assert "some-future-model" in buf.getvalue()
        # Unknown model still appears in the breakdown with $0 cost.
        assert len(report.per_model) == 1
        m = report.per_model[0]
        assert m.usd == 0.0
        assert m.is_unknown
        assert m.calls == 1
        assert m.prompt_tokens == 1000

    def test_provider_fallback_when_provider_unknown(self, tmp_path: Path):
        # Provider attribute is missing, but model is known under a different
        # provider. The lookup should still resolve via the model-only fallback.
        spans = [
            _processed(
                "a",
                "1",
                model="gpt-4o-mini",
                provider=None,
                prompt_tokens=1_000_000,
                completion_tokens=1_000_000,
            ),
        ]
        p = tmp_path / "spans.jsonl"
        p.write_text("\n".join(json.dumps(_to_json_dict(s)) for s in spans))
        report = compute(p, pricing=PRICING, warn_stream=io.StringIO())
        assert report.unknown_models == []
        assert len(report.per_model) == 1
        # Falls back to the first provider in the table that has this model.
        assert report.per_model[0].provider == "openai"
        assert report.per_model[0].usd == pytest.approx(0.75, abs=1e-6)

    def test_spans_without_tokens_are_skipped(self, tmp_path: Path):
        spans = [
            _processed("a", "1", prompt_tokens=None, completion_tokens=None),
            _processed("a", "2", prompt_tokens=1000, completion_tokens=500),
        ]
        p = tmp_path / "spans.jsonl"
        p.write_text("\n".join(json.dumps(_to_json_dict(s)) for s in spans))
        report = compute(p, pricing=PRICING, warn_stream=io.StringIO())
        assert report.spans_missing_tokens == 1
        assert report.spans_with_tokens == 1
        # The first span had no model (because no tokens), so only one row.
        assert len(report.per_model) == 1
        assert report.per_model[0].calls == 1

    def test_spans_without_model_are_skipped(self, tmp_path: Path):
        spans = [
            _processed(
                "a", "1", model=None, provider="openai", prompt_tokens=1000, completion_tokens=500
            ),
        ]
        p = tmp_path / "spans.jsonl"
        p.write_text("\n".join(json.dumps(_to_json_dict(s)) for s in spans))
        report = compute(p, pricing=PRICING, warn_stream=io.StringIO())
        assert report.per_model == []
        assert report.spans_with_tokens == 0

    def test_multi_file(self, tmp_path: Path):
        p1 = tmp_path / "a.jsonl"
        p2 = tmp_path / "b.jsonl"
        spans_a = [
            _processed(
                "a",
                "1",
                model="gpt-4o-mini",
                provider="openai",
                prompt_tokens=1_000_000,
                completion_tokens=0,
            )
        ]
        spans_b = [
            _processed(
                "b",
                "2",
                model="gpt-4o-mini",
                provider="openai",
                prompt_tokens=1_000_000,
                completion_tokens=0,
            )
        ]
        p1.write_text("\n".join(json.dumps(_to_json_dict(s)) for s in spans_a))
        p2.write_text("\n".join(json.dumps(_to_json_dict(s)) for s in spans_b))
        report = compute([p1, p2], pricing=PRICING, warn_stream=io.StringIO())
        assert report.files_read == 2
        assert report.per_model[0].calls == 2
        # 2M input @ 0.15 = 0.30
        assert report.per_model[0].usd == pytest.approx(0.30, abs=1e-6)

    def test_missing_path_skipped(self, tmp_path: Path):
        report = compute(
            [tmp_path / "does_not_exist.jsonl"], pricing=PRICING, warn_stream=io.StringIO()
        )
        assert report.files_read == 0
        assert report.per_model == []

    def test_raw_span_format(self, tmp_path: Path):
        # Raw ReadableSpan.to_json() shape, just like the replay tests cover.
        d = {
            "name": "openai.chat.completions.create",
            "context": {"trace_id": "0xabcd", "span_id": "0x1234"},
            "parent_id": None,
            "kind": "SpanKind.CLIENT",
            "start_time": "2024-01-15T10:00:00.000Z",
            "end_time": "2024-01-15T10:00:00.001Z",
            "status": {"status_code": "OK", "description": ""},
            "attributes": {
                "neatlogs.llm.model_name": "gpt-4o-mini",
                "neatlogs.llm.provider": "openai",
                "neatlogs.llm.token_count.prompt": 1_000_000,
                "neatlogs.llm.token_count.completion": 1_000_000,
            },
            "events": [],
            "links": [],
            "resource": {},
        }
        p = tmp_path / "raw.jsonl"
        p.write_text(json.dumps(d))
        report = compute(p, pricing=PRICING, warn_stream=io.StringIO())
        assert len(report.per_model) == 1
        m = report.per_model[0]
        assert m.model == "gpt-4o-mini"
        assert m.usd == pytest.approx(0.75, abs=1e-6)


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------


class TestFormatText:
    def test_basic(self):
        spans = [
            _processed(
                "a",
                "1",
                model="gpt-4o-mini",
                provider="openai",
                prompt_tokens=1_000_000,
                completion_tokens=0,
            ),
        ]
        import os
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for s in spans:
                f.write(json.dumps(_to_json_dict(s)) + "\n")
            path = f.name
        try:
            report = compute(path, pricing=PRICING, warn_stream=io.StringIO())
        finally:
            os.unlink(path)
        out = format_report(report, style="text")
        assert "gpt-4o-mini" in out
        assert "openai" in out
        assert "1" in out  # calls
        assert "1000000" in out  # prompt tokens
        assert "$0.150000" in out
        assert "TOTAL" in out

    def test_unknown_model_appears(self):
        spans = [
            _processed(
                "a",
                "1",
                model="mystery-model",
                provider="openai",
                prompt_tokens=1000,
                completion_tokens=500,
            ),
        ]
        import os
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for s in spans:
                f.write(json.dumps(_to_json_dict(s)) + "\n")
            path = f.name
        try:
            report = compute(path, pricing=PRICING, warn_stream=io.StringIO())
        finally:
            os.unlink(path)
        out = format_report(report, style="text")
        assert "mystery-model" in out
        assert "unknown" in out

    def test_empty_report(self):
        report = CostReport(
            per_model=[],
            unknown_models=[],
            files_read=0,
            spans_with_tokens=0,
            spans_missing_tokens=0,
        )
        out = format_report(report, style="text")
        assert "no LLM spans" in out


class TestFormatJson:
    def test_json_shape(self):
        spans = [
            _processed(
                "a",
                "1",
                model="gpt-4o-mini",
                provider="openai",
                prompt_tokens=1_000_000,
                completion_tokens=0,
            ),
        ]
        import os
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for s in spans:
                f.write(json.dumps(_to_json_dict(s)) + "\n")
            path = f.name
        try:
            report = compute(path, pricing=PRICING, warn_stream=io.StringIO())
        finally:
            os.unlink(path)
        out = format_report(report, style="json")
        obj = json.loads(out)
        assert obj["currency"] == "USD"
        assert obj["total_calls"] == 1
        assert obj["total_prompt_tokens"] == 1_000_000
        assert obj["unknown_models"] == []
        assert len(obj["per_model"]) == 1
        m = obj["per_model"][0]
        assert m["model"] == "gpt-4o-mini"
        assert m["provider"] == "openai"
        assert m["usd"] == pytest.approx(0.15, abs=1e-6)
        assert m["unknown"] is False

    def test_unknown_flag_in_json(self):
        spans = [
            _processed(
                "a",
                "1",
                model="mystery-model",
                provider="openai",
                prompt_tokens=1000,
                completion_tokens=500,
            ),
        ]
        import os
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for s in spans:
                f.write(json.dumps(_to_json_dict(s)) + "\n")
            path = f.name
        try:
            report = compute(path, pricing=PRICING, warn_stream=io.StringIO())
        finally:
            os.unlink(path)
        obj = json.loads(format_report(report, style="json"))
        assert obj["per_model"][0]["unknown"] is True
        assert obj["per_model"][0]["usd"] == 0.0
        assert obj["unknown_models"] == ["mystery-model"]


class TestFormatCsv:
    def test_csv_columns(self):
        spans = [
            _processed(
                "a",
                "1",
                model="gpt-4o-mini",
                provider="openai",
                prompt_tokens=1_000_000,
                completion_tokens=0,
            ),
        ]
        import os
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for s in spans:
                f.write(json.dumps(_to_json_dict(s)) + "\n")
            path = f.name
        try:
            report = compute(path, pricing=PRICING, warn_stream=io.StringIO())
        finally:
            os.unlink(path)
        out = format_report(report, style="csv")
        lines = out.strip().split("\n")
        assert lines[0] == "model,provider,calls,prompt_tokens,completion_tokens,usd,unknown"
        cols = lines[1].split(",")
        assert cols[0] == "gpt-4o-mini"
        assert cols[1] == "openai"
        assert cols[2] == "1"
        assert cols[3] == "1000000"
        assert cols[6] == "false"

    def test_unknown_style_raises(self):
        report = CostReport(
            per_model=[],
            unknown_models=[],
            files_read=0,
            spans_with_tokens=0,
            spans_missing_tokens=0,
        )
        with pytest.raises(ValueError):
            format_report(report, style="bogus")


# ---------------------------------------------------------------------------
# Custom pricing override
# ---------------------------------------------------------------------------


class TestPricingOverride:
    def test_custom_pricing_changes_cost(self, tmp_path: Path):
        # Use a custom price that differs from the bundled one.
        custom = {
            "openai": {
                "gpt-4o-mini": {"input": 1.00, "output": 2.00},  # 4x higher
            }
        }
        spans = [
            _processed(
                "a",
                "1",
                model="gpt-4o-mini",
                provider="openai",
                prompt_tokens=1_000_000,
                completion_tokens=1_000_000,
            ),
        ]
        p = tmp_path / "spans.jsonl"
        p.write_text("\n".join(json.dumps(_to_json_dict(s)) for s in spans))
        report = compute(p, pricing=custom, warn_stream=io.StringIO())
        # 1M * 1.00 + 1M * 2.00 = 3.00
        assert report.total_usd == pytest.approx(3.00, abs=1e-6)
        assert report.per_model[0].usd == pytest.approx(3.00, abs=1e-6)

    def test_pricing_file_override_via_path(self, tmp_path: Path):
        custom_path = tmp_path / "my-pricing.json"
        custom_path.write_text(
            json.dumps(
                {
                    "openai": {
                        "gpt-4o-mini": {"input": 0.10, "output": 0.40},
                    }
                }
            )
        )
        spans = [
            _processed(
                "a",
                "1",
                model="gpt-4o-mini",
                provider="openai",
                prompt_tokens=1_000_000,
                completion_tokens=1_000_000,
            ),
        ]
        p = tmp_path / "spans.jsonl"
        p.write_text("\n".join(json.dumps(_to_json_dict(s)) for s in spans))
        from neatlogs.cost import _load_pricing

        pricing = _load_pricing(custom_path)
        report = compute(p, pricing=pricing, warn_stream=io.StringIO())
        # 1M * 0.10 + 1M * 0.40 = 0.50
        assert report.total_usd == pytest.approx(0.50, abs=1e-6)


# ---------------------------------------------------------------------------
# Bundled pricing file
# ---------------------------------------------------------------------------


class TestBundledPricing:
    def test_bundled_file_loads(self):
        from neatlogs.cost import DEFAULT_PRICING_PATH, _load_pricing

        assert DEFAULT_PRICING_PATH.exists()
        pricing = _load_pricing()
        # Sanity: the bundled table has the headline providers.
        assert "openai" in pricing
        assert "anthropic" in pricing
        assert "google_genai" in pricing
        # Sanity: a couple of well-known models are present.
        assert "gpt-4o" in pricing["openai"]
        assert "claude-3-5-sonnet-latest" in pricing["anthropic"]
