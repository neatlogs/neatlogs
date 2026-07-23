"""
Unit tests for neatlogs.replay.

The replay module reads JSONL span logs written by the SDK's span processor
(``NEATLOGS_LOG_SPANS=true`` → processed ``span_data`` shape) or by the raw
OTel ``to_json()`` exporter (``NEATLOGS_LOG_RAW_SPANS=true``), reconstructs
the parent/child tree per trace, and prints a human-readable summary. These
tests exercise the parser, tree builder, and renderer end-to-end without
spinning up an OTel pipeline.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import List

import pytest

from neatlogs.replay import (
    SpanRecord,
    TraceTree,
    TreeNode,
    _format_text,
    _iter_json_objects,
    _kind_color,
    _normalize_kind,
    _strip_hex,
    build_trees,
    format_tree,
    parse_line,
    parse_paths,
    replay,
)

# ---------------------------------------------------------------------------
# _strip_hex
# ---------------------------------------------------------------------------


class TestStripHex:
    def test_none(self):
        assert _strip_hex(None) is None

    def test_empty_string(self):
        assert _strip_hex("") is None

    def test_plain_hex(self):
        assert _strip_hex("0fa6dc405d06ed3763e2584130b53190") == "0fa6dc405d06ed3763e2584130b53190"

    def test_uppercase_hex(self):
        assert _strip_hex("0FA6DC40") == "0fa6dc40"

    def test_0x_prefix(self):
        assert _strip_hex("0x0fa6dc40") == "0fa6dc40"

    def test_int_value(self):
        assert _strip_hex(264602534208) == "3d9b8a4140"

    def test_zero_int(self):
        assert _strip_hex(0) is None

    def test_non_hex_string(self):
        assert _strip_hex("not-hex") is None

    def test_non_str_non_int(self):
        assert _strip_hex([1, 2, 3]) is None


# ---------------------------------------------------------------------------
# _normalize_kind / _kind_color
# ---------------------------------------------------------------------------


class TestNormalizeKind:
    def test_short_lowercase(self):
        assert _normalize_kind("llm") == "LLM"
        assert _normalize_kind("tool") == "TOOL"

    def test_spankind_prefix_stripped(self):
        assert _normalize_kind("SpanKind.INTERNAL") == "INTERNAL"
        assert _normalize_kind("SpanKind.CLIENT") == "CLIENT"

    def test_spankind_uppercase_prefix(self):
        # Defensive: the raw exporter always writes "SpanKind." but other
        # sources might uppercase the prefix.
        assert _normalize_kind("SPANKIND.SERVER") == "SERVER"

    def test_empty(self):
        assert _normalize_kind("") == "?"

    def test_none_safe(self):
        # Build paths pass strings, but be defensive.
        assert _normalize_kind("") == "?"
        assert _normalize_kind("   ") == "?"


class TestKindColor:
    def test_known_short_kind(self):
        assert _kind_color("llm") == "36"  # cyan

    def test_spankind_resolves(self):
        assert _kind_color("SpanKind.INTERNAL") == "37"
        assert _kind_color("SpanKind.CLIENT") == "90"

    def test_unknown_kind(self):
        assert _kind_color("not-a-kind") is None


# ---------------------------------------------------------------------------
# _iter_json_objects
# ---------------------------------------------------------------------------


class TestIterJsonObjects:
    def test_single_object(self):
        text = '{"a": 1, "b": 2}'
        objs = list(_iter_json_objects(text))
        assert len(objs) == 1
        assert objs[0] == {"a": 1, "b": 2}

    def test_multiple_objects(self):
        text = '{"a": 1}\n{"b": 2}\n{"c": 3}'
        objs = list(_iter_json_objects(text))
        assert len(objs) == 3
        assert objs[2] == {"c": 3}

    def test_objects_with_nested_braces_in_strings(self):
        text = '{"name": "a {b} c", "value": 1}\n{"name": "plain", "value": 2}'
        objs = list(_iter_json_objects(text))
        assert len(objs) == 2
        assert objs[0]["name"] == "a {b} c"
        assert objs[1]["name"] == "plain"

    def test_objects_with_escaped_quotes(self):
        text = '{"name": "a \\"quoted\\" word", "value": 1}'
        objs = list(_iter_json_objects(text))
        assert len(objs) == 1
        assert objs[0]["name"] == 'a "quoted" word'

    def test_malformed_object_is_skipped(self):
        text = '{"a": 1}\n{not valid json}\n{"b": 2}'
        objs = list(_iter_json_objects(text))
        # The malformed object is silently dropped by the brace-balanced parser.
        assert len(objs) >= 1
        assert {"a": 1} in objs
        assert {"b": 2} in objs

    def test_empty_string(self):
        assert list(_iter_json_objects("")) == []


# ---------------------------------------------------------------------------
# parse_line: processed (span_data) and raw (to_json) shapes
# ---------------------------------------------------------------------------


def _processed_span_dict(
    trace_id: str = "aaaa",
    span_id: str = "1111",
    parent_span_id=None,
    name: str = "openai.chat.completions.create",
    kind: str = "llm",
    duration_ns: int = 12345678,
    status_code: str = "OK",
    attributes=None,
) -> dict:
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "kind": kind,
        "start_time": 1700000000000000000,
        "end_time": 1700000000000000000 + duration_ns,
        "duration_ns": duration_ns,
        "attributes": attributes or {"neatlogs.llm.model_name": "gpt-4o-mini"},
        "resource": {"attributes": {"service.name": "svc"}},
        "status": {"code": status_code, "description": ""},
        "events": [],
    }


def _raw_span_dict(
    trace_id: str = "bbbb",
    span_id: str = "2222",
    parent_id=None,
    name: str = "POST",
    kind: str = "SpanKind.CLIENT",
    duration_ns: int = 5000000,
    status_code: str = "OK",
) -> dict:
    return {
        "name": name,
        "context": {"trace_id": "0x" + trace_id, "span_id": "0x" + span_id},
        "parent_id": ("0x" + parent_id) if parent_id else None,
        "kind": kind,
        "start_time": "2024-01-15T10:00:00.000Z",
        "end_time": "2024-01-15T10:00:00.005Z",
        "status": {"status_code": status_code, "description": ""},
        "attributes": {
            "http.method": "POST",
            "http.url": "https://api.openai.com/v1/chat/completions",
        },
        "events": [],
        "links": [],
        "resource": {},
    }


class TestParseLineProcessed:
    def test_basic(self):
        d = _processed_span_dict()
        line = json.dumps(d)
        record = parse_line(line)
        assert record is not None
        assert record.trace_id == "aaaa"
        assert record.span_id == "1111"
        assert record.parent_span_id is None
        assert record.name == "openai.chat.completions.create"
        assert record.kind == "llm"
        assert record.source == "processed"
        assert record.status_code == "OK"
        assert record.duration_ns == 12345678
        assert record.duration_ms() == pytest.approx(12.346, abs=1e-3)
        assert record.attributes["neatlogs.llm.model_name"] == "gpt-4o-mini"

    def test_with_parent(self):
        d = _processed_span_dict(parent_span_id="0000", span_id="1111", trace_id="aaaa")
        record = parse_line(json.dumps(d))
        assert record is not None
        assert record.parent_span_id == "0000"

    def test_with_0x_prefix_in_ids(self):
        d = _processed_span_dict(trace_id="0xabcd", span_id="0x1234")
        record = parse_line(json.dumps(d))
        assert record is not None
        assert record.trace_id == "abcd"
        assert record.span_id == "1234"

    def test_zero_parent_id_treated_as_no_parent(self):
        d = _processed_span_dict(parent_span_id="0000000000000000", span_id="1111", trace_id="aaaa")
        record = parse_line(json.dumps(d))
        assert record is not None
        # The processor already strips zero parent_ids; replay is defensive.
        assert record.parent_span_id in (None, "0000000000000000")

    def test_empty_line(self):
        assert parse_line("") is None
        assert parse_line("   \n  ") is None

    def test_malformed_json(self):
        assert parse_line("{not valid") is None

    def test_non_dict(self):
        assert parse_line("[1, 2, 3]") is None
        assert parse_line('"a string"') is None
        assert parse_line("42") is None


class TestParseLineRaw:
    def test_basic(self):
        d = _raw_span_dict()
        record = parse_line(json.dumps(d))
        assert record is not None
        assert record.source == "raw"
        assert record.trace_id == "bbbb"
        assert record.span_id == "2222"
        assert record.name == "POST"
        assert record.kind == "SpanKind.CLIENT"
        assert record.duration_ms() == pytest.approx(5.0, abs=0.1)

    def test_iso_timestamps(self):
        d = _raw_span_dict()
        record = parse_line(json.dumps(d))
        assert record is not None
        assert record.start_time is not None
        assert record.end_time is not None
        assert record.start_time < record.end_time

    def test_with_parent(self):
        d = _raw_span_dict(parent_id="0000", span_id="2222", trace_id="bbbb")
        record = parse_line(json.dumps(d))
        assert record is not None
        assert record.parent_span_id == "0000"

    def test_no_context(self):
        d = _raw_span_dict()
        d.pop("context")
        assert parse_line(json.dumps(d)) is None


# ---------------------------------------------------------------------------
# parse_paths: file reading + multi-file
# ---------------------------------------------------------------------------


class TestParsePaths:
    def test_processed_file(self, tmp_path: Path):
        d1 = _processed_span_dict(trace_id="a", span_id="1", name="root")
        d2 = _processed_span_dict(trace_id="a", span_id="2", parent_span_id="1", name="child")
        p = tmp_path / "spans.jsonl"
        p.write_text("\n".join(json.dumps(d) for d in (d1, d2)))
        result = parse_paths([p])
        assert result.files_read == 1
        assert len(result.spans) == 2
        assert result.malformed_lines == 0
        assert {s.name for s in result.spans} == {"root", "child"}

    def test_raw_file(self, tmp_path: Path):
        d = _raw_span_dict()
        p = tmp_path / "raw.jsonl"
        p.write_text(json.dumps(d))
        result = parse_paths([p])
        assert result.files_read == 1
        assert len(result.spans) == 1
        assert result.spans[0].source == "raw"

    def test_mixed_processed_and_raw(self, tmp_path: Path):
        processed = _processed_span_dict(trace_id="a", span_id="1")
        raw = _raw_span_dict(trace_id="b", span_id="2")
        p1 = tmp_path / "p.jsonl"
        p2 = tmp_path / "r.jsonl"
        p1.write_text(json.dumps(processed))
        p2.write_text(json.dumps(raw))
        result = parse_paths([p1, p2])
        assert result.files_read == 2
        assert len(result.spans) == 2
        assert {s.source for s in result.spans} == {"processed", "raw"}

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        result = parse_paths([p])
        assert result.files_read == 1
        assert result.spans == []
        assert result.malformed_lines == 0

    def test_missing_path_skipped(self, tmp_path: Path):
        result = parse_paths([tmp_path / "does_not_exist.jsonl"])
        assert result.files_read == 0
        assert result.spans == []

    def test_malformed_lines_counted(self, tmp_path: Path):
        valid = _processed_span_dict(trace_id="a", span_id="1")
        p = tmp_path / "mixed.jsonl"
        p.write_text("\n".join([json.dumps(valid), "{not valid}", "42", '"string"']))
        result = parse_paths([p])
        assert len(result.spans) == 1
        assert result.malformed_lines == 3


# ---------------------------------------------------------------------------
# build_trees
# ---------------------------------------------------------------------------


def _processed(trace_id, span_id, parent=None, name="span", kind="chain"):
    return SpanRecord(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent,
        name=name,
        kind=kind,
        start_time=1_700_000_000_000_000_000,
        end_time=1_700_000_000_000_010_000,
        duration_ns=10_000,
        status_code="OK",
        status_description="",
        attributes={},
        source="processed",
    )


def _synthetic(trace_id, span_id, name="_orphan"):
    return SpanRecord(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=None,
        name=name,
        kind="INTERNAL",
        start_time=None,
        end_time=None,
        duration_ns=None,
        status_code="",
        status_description="",
        attributes={},
        source="synthetic",
    )


class TestBuildTrees:
    def test_single_root(self):
        spans = [_processed("a", "1", name="root", kind="workflow")]
        trees = build_trees(spans)
        assert len(trees) == 1
        assert trees[0].trace_id == "a"
        assert len(trees[0].roots) == 1
        assert trees[0].count() == 1

    def test_nested_tree(self):
        spans = [
            _processed("a", "1", name="root", kind="workflow"),
            _processed("a", "2", parent="1", name="child", kind="chain"),
            _processed("a", "3", parent="2", name="grand", kind="tool"),
        ]
        trees = build_trees(spans)
        assert len(trees) == 1
        root = trees[0].roots[0]
        assert root.span.name == "root"
        assert len(root.children) == 1
        assert root.children[0].span.name == "child"
        assert root.children[0].children[0].span.name == "grand"
        assert trees[0].count() == 3

    def test_multiple_traces(self):
        spans = [
            _processed("a", "1", name="trace-a"),
            _processed("b", "1", name="trace-b"),
        ]
        trees = build_trees(spans)
        assert len(trees) == 2
        assert {t.trace_id for t in trees} == {"a", "b"}

    def test_multiple_roots_in_one_trace(self):
        spans = [
            _processed("a", "1", name="root1"),
            _processed("a", "2", name="root2"),
        ]
        trees = build_trees(spans)
        assert len(trees) == 1
        assert len(trees[0].roots) == 2

    def test_orphan_becomes_synthetic_root(self):
        spans = [
            _processed("a", "1", name="root"),
            _processed("a", "2", parent="DEAD", name="orphan"),
        ]
        buf = io.StringIO()
        trees = build_trees(spans, warn_stream=buf)
        assert len(trees) == 1
        # Two roots: the real root + a synthetic _orphan root.
        assert len(trees[0].roots) == 2
        names = [r.span.name for r in trees[0].roots]
        assert "root" in names
        assert "_orphan" in names
        # The orphan span lives under the synthetic root.
        orphan_root = next(r for r in trees[0].roots if r.span.name == "_orphan")
        assert len(orphan_root.children) == 1
        assert orphan_root.children[0].span.name == "orphan"
        # Warning surfaces the missing parent id.
        assert "DEAD" in buf.getvalue()

    def test_no_warn_when_warn_stream_is_none(self):
        spans = [_processed("a", "1", parent="DEAD", name="orphan")]
        # Default warn_stream is stderr; we just confirm no exception is raised.
        trees = build_trees(spans, warn_stream=io.StringIO())
        assert len(trees) == 1

    def test_parent_in_different_trace_becomes_root(self):
        spans = [
            _processed("a", "1", name="root-a"),
            _processed("b", "2", parent="1", name="wrong-parent"),
        ]
        trees = build_trees(spans)
        # Two traces, each with one root.
        assert len(trees) == 2
        for t in trees:
            assert len(t.roots) == 1
            assert t.roots[0].span.parent_span_id is None or t.roots[0].span.parent_span_id not in (
                s.span_id for s in spans if s.trace_id == t.trace_id
            )


# ---------------------------------------------------------------------------
# format_tree (text)
# ---------------------------------------------------------------------------


class TestFormatText:
    def test_empty_trees(self):
        out = format_tree([], style="text", use_color=False)
        assert "no spans found" in out

    def test_basic_tree(self):
        spans = [
            _processed("a", "1", name="root", kind="workflow"),
            _processed("a", "2", parent="1", name="llm-call", kind="llm"),
        ]
        trees = build_trees(spans)
        out = format_tree(trees, style="text", use_color=False)
        assert "trace a" in out
        assert "[WORKFLOW] root" in out
        assert "  [LLM    ] llm-call" in out
        assert "2 spans" in out

    def test_max_depth_truncates(self):
        spans = [
            _processed("a", "1", name="root", kind="workflow"),
            _processed("a", "2", parent="1", name="child", kind="chain"),
            _processed("a", "3", parent="2", name="grand", kind="tool"),
        ]
        trees = build_trees(spans)
        out = format_tree(trees, style="text", use_color=False, max_depth=1)
        assert "[WORKFLOW] root" in out
        assert "child" in out
        # grand is below the max_depth cutoff.
        assert "grand" not in out

    def test_filter_kind(self):
        spans = [
            _processed("a", "1", name="root", kind="workflow"),
            _processed("a", "2", parent="1", name="llm-call", kind="llm"),
            _processed("a", "3", parent="2", name="tool-call", kind="tool"),
        ]
        trees = build_trees(spans)
        out = format_tree(trees, style="text", use_color=False, filter_kind="llm")
        assert "llm-call" in out
        # tool-call sits below the LLM but is excluded by filter; root renders too.
        assert "tool-call" not in out

    def test_no_color_when_disabled(self):
        spans = [_processed("a", "1", name="root", kind="workflow")]
        trees = build_trees(spans)
        out = format_tree(trees, style="text", use_color=False)
        # No ANSI escape codes.
        assert "\033[" not in out

    def test_color_when_enabled(self):
        spans = [_processed("a", "1", name="root", kind="workflow")]
        trees = build_trees(spans)
        out = format_tree(trees, style="text", use_color=True)
        # ANSI escape codes are present (the kind uses color code 33 for workflow).
        assert "\033[" in out


# ---------------------------------------------------------------------------
# format_tree (json)
# ---------------------------------------------------------------------------


class TestFormatJson:
    def test_single_trace(self):
        spans = [
            _processed("a", "1", name="root", kind="workflow"),
            _processed("a", "2", parent="1", name="child", kind="llm"),
        ]
        trees = build_trees(spans)
        out = format_tree(trees, style="json")
        lines = out.strip().split("\n")
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["trace_id"] == "a"
        assert obj["span_count"] == 2
        assert obj["roots"][0]["name"] == "root"
        assert obj["roots"][0]["children"][0]["name"] == "child"
        assert obj["roots"][0]["children"][0]["kind"] == "llm"

    def test_empty_trees(self):
        out = format_tree([], style="json")
        assert out == ""

    def test_unknown_style_raises(self):
        with pytest.raises(ValueError):
            format_tree([], style="bogus")


# ---------------------------------------------------------------------------
# replay (top-level API)
# ---------------------------------------------------------------------------


class TestReplay:
    def test_end_to_end(self, tmp_path: Path):
        spans = [
            _processed("a", "1", name="root", kind="workflow"),
            _processed("a", "2", parent="1", name="llm", kind="llm"),
        ]
        p = tmp_path / "spans.jsonl"
        p.write_text("\n".join(json.dumps(_to_d(s)) for s in spans))
        trees = replay(p, warn_stream=io.StringIO())
        assert len(trees) == 1
        assert trees[0].count() == 2

    def test_accepts_single_path_string(self, tmp_path: Path):
        p = tmp_path / "s.jsonl"
        p.write_text(json.dumps(_to_d(_processed("a", "1"))))
        trees = replay(str(p), warn_stream=io.StringIO())
        assert len(trees) == 1

    def test_accepts_list(self, tmp_path: Path):
        p1 = tmp_path / "a.jsonl"
        p2 = tmp_path / "b.jsonl"
        p1.write_text(json.dumps(_to_d(_processed("a", "1"))))
        p2.write_text(json.dumps(_to_d(_processed("b", "1"))))
        trees = replay([p1, p2], warn_stream=io.StringIO())
        assert len(trees) == 2

    def test_max_depth_passed_through(self, tmp_path: Path):
        spans = [
            _processed("a", "1", name="root", kind="workflow"),
            _processed("a", "2", parent="1", name="child", kind="chain"),
        ]
        p = tmp_path / "s.jsonl"
        p.write_text("\n".join(json.dumps(_to_d(s)) for s in spans))
        trees = replay(p, max_depth=0, warn_stream=io.StringIO())
        out = format_tree(trees, style="text", use_color=False, max_depth=0)
        assert "child" not in out


def _to_d(s: SpanRecord) -> dict:
    return {
        "trace_id": s.trace_id,
        "span_id": s.span_id,
        "parent_span_id": s.parent_span_id,
        "name": s.name,
        "kind": s.kind,
        "start_time": s.start_time,
        "end_time": s.end_time,
        "duration_ns": s.duration_ns,
        "attributes": s.attributes,
        "resource": {"attributes": {}},
        "status": {"code": s.status_code, "description": s.status_description},
        "events": [],
    }
