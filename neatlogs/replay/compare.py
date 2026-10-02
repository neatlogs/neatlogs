"""Structured comparison of an original trace vs its counterfactual replay.

Operates on span attribute dicts (``ReadableSpan``, SDK ``Span``, or
``ReplayedSpan`` snapshots) — never on raw JSON dumps — and reports *what
changed* per span plus aggregate latency/token/cost deltas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from .extract import _decode, _indexed, span_kind
from .types import ReplayOverrides, SpanFieldDiff, TraceComparison, span_id, trace_id


def _attributes(span: Any) -> Dict[str, Any]:
    if hasattr(span, "attributes") and not isinstance(getattr(span, "attributes"), dict):
        try:
            return dict(span.attributes)
        except Exception:
            return {}
    attrs = getattr(span, "attributes", None)
    return dict(attrs) if isinstance(attrs, Mapping) else {}


def _name(span: Any) -> str:
    return str(getattr(span, "name", None) or "unknown")


def _replay_of(span: Any) -> Optional[str]:
    return _attributes(span).get("neatlogs.replay.of_span_id")


def _float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _llm_output_text(attrs: Mapping[str, Any]) -> str:
    records = _indexed(attrs, "neatlogs.llm.output_messages")
    parts = []
    for index in sorted(records):
        content = records[index].get("content")
        if content:
            parts.append(str(content))
    if parts:
        return "\n".join(parts)
    return str(attrs.get("output.value") or "")


def _llm_input_text(attrs: Mapping[str, Any]) -> str:
    records = _indexed(attrs, "neatlogs.llm.input_messages")
    parts = []
    for index in sorted(records):
        role = records[index].get("role", "")
        content = records[index].get("content", "")
        parts.append(f"{role}: {content}")
    if parts:
        return "\n".join(parts)
    return str(attrs.get("input.value") or "")


def _input_text(span: Any) -> str:
    attrs = _attributes(span)
    if span_kind(span) == "LLM":
        return _llm_input_text(attrs)
    return str(attrs.get("input.value") or "")


def _tool_calls(attrs: Mapping[str, Any]) -> List[Dict[str, Any]]:
    records = _indexed(attrs, "neatlogs.llm.tool_calls")
    calls = []
    for index in sorted(records):
        record = records[index]
        calls.append(
            {
                "name": str(record.get("name") or ""),
                "arguments": _decode(record.get("arguments")),
            }
        )
    return calls


def _delta(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None:
        return None
    return new - old


def _sum_tokens(spans: List[Any], key: str) -> Optional[int]:
    total = 0
    found = False
    for span in spans:
        value = _int(_attributes(span).get(key))
        if value is not None:
            total += value
            found = True
    return total if found else None


def _sum_float(spans: List[Any], key: str) -> Optional[float]:
    total = 0.0
    found = False
    for span in spans:
        value = _float(_attributes(span).get(key))
        if value is not None:
            total += value
            found = True
    return total if found else None


def compare_spans(original: Any, replay: Any) -> Optional[SpanFieldDiff]:
    """Diff one matched span pair. Returns None when nothing material changed."""
    orig_attrs = _attributes(original)
    replay_attrs = _attributes(replay)
    kind = span_kind(original) or span_kind(replay) or "UNKNOWN"
    changed: List[str] = []
    orig_view: Dict[str, Any] = {}
    replay_view: Dict[str, Any] = {}

    def check(field: str, old: Any, new: Any) -> None:
        if old != new:
            changed.append(field)
            orig_view[field] = old
            replay_view[field] = new

    if kind == "LLM":
        check(
            "model",
            orig_attrs.get("neatlogs.llm.model_name"),
            replay_attrs.get("neatlogs.llm.model_name"),
        )
        check("input", _llm_input_text(orig_attrs), _llm_input_text(replay_attrs))
        check("output", _llm_output_text(orig_attrs), _llm_output_text(replay_attrs))
        check("tool_calls", _tool_calls(orig_attrs), _tool_calls(replay_attrs))
        check(
            "finish_reason",
            orig_attrs.get("neatlogs.llm.finish_reason"),
            replay_attrs.get("neatlogs.llm.finish_reason"),
        )
    else:
        check(
            "input",
            str(orig_attrs.get("input.value") or ""),
            str(replay_attrs.get("input.value") or ""),
        )
        check(
            "output",
            str(orig_attrs.get("output.value") or ""),
            str(replay_attrs.get("output.value") or ""),
        )

    if not changed:
        return None
    return SpanFieldDiff(
        span_name=_name(original),
        span_kind=kind,
        original_span_id=span_id(original),
        replay_span_id=span_id(replay),
        changed_fields=changed,
        original=orig_view,
        replay=replay_view,
    )


def _match_spans(
    original: List[Any], replay: List[Any]
) -> Tuple[List[Tuple[Any, Any]], List[Any], List[Any]]:
    """Match replay spans to originals via replay linkage, else (name, kind)."""
    orig_by_id = {span_id(span): span for span in original}
    matched: List[Tuple[Any, Any]] = []
    added: List[Any] = []
    used_orig_ids = set()
    for span in replay:
        target = _replay_of(span)
        if target is not None and target in orig_by_id:
            matched.append((orig_by_id[target], span))
            used_orig_ids.add(target)
            continue
        candidates = [
            s
            for s in original
            if span_id(s) not in used_orig_ids
            and _name(s) == _name(span)
            and span_kind(s) == span_kind(span)
        ]
        if candidates:
            matched.append((candidates[0], span))
            used_orig_ids.add(span_id(candidates[0]))
        else:
            added.append(span)
    removed = [s for s in original if span_id(s) not in used_orig_ids]
    return matched, added, removed


def compare_traces(
    original_spans: List[Any],
    replay_spans: List[Any],
    overrides: Union[ReplayOverrides, Dict[str, Any], None] = None,
) -> TraceComparison:
    """Build a structured original-vs-replay comparison."""
    if isinstance(overrides, dict):
        overrides = ReplayOverrides.from_dict(overrides)
    resolved = overrides or ReplayOverrides()
    original = list(original_spans)
    replay = list(replay_spans)
    matched, added, removed = _match_spans(original, replay)

    changed_spans: List[SpanFieldDiff] = []
    output_changes: List[Dict[str, Any]] = []
    tool_call_changes: List[Dict[str, Any]] = []
    for orig_span, replay_span in matched:
        diff = compare_spans(orig_span, replay_span)
        if diff is None:
            continue
        changed_spans.append(diff)
        if "output" in diff.changed_fields:
            output_changes.append(
                {
                    "span": diff.span_name,
                    "original_span_id": diff.original_span_id,
                    "replay_span_id": diff.replay_span_id,
                    "original_input": _input_text(orig_span),
                    "original_output": diff.original.get("output"),
                    "replay_output": diff.replay.get("output"),
                }
            )
        if "tool_calls" in diff.changed_fields:
            tool_call_changes.append(
                {
                    "span": diff.span_name,
                    "original_span_id": diff.original_span_id,
                    "replay_span_id": diff.replay_span_id,
                    "original_tool_calls": diff.original.get("tool_calls"),
                    "replay_tool_calls": diff.replay.get("tool_calls"),
                }
            )

    orig_trace = trace_id(original[0]) if original else ""
    replay_trace = trace_id(replay[0]) if replay else ""
    return TraceComparison(
        original_trace_id=orig_trace,
        replay_trace_id=replay_trace,
        overrides=resolved,
        changed_spans=changed_spans,
        added_spans=[span_id(s) for s in added],
        removed_spans=[span_id(s) for s in removed],
        output_changes=output_changes,
        tool_call_changes=tool_call_changes,
        latency_delta_ms=_delta(
            _sum_float(replay, "neatlogs.llm.metrics.duration_ms"),
            _sum_float(original, "neatlogs.llm.metrics.duration_ms"),
        ),
        prompt_tokens_delta=_delta(
            _sum_tokens(replay, "neatlogs.llm.token_count.prompt"),
            _sum_tokens(original, "neatlogs.llm.token_count.prompt"),
        ),
        completion_tokens_delta=_delta(
            _sum_tokens(replay, "neatlogs.llm.token_count.completion"),
            _sum_tokens(original, "neatlogs.llm.token_count.completion"),
        ),
        total_tokens_delta=_delta(
            _sum_tokens(replay, "neatlogs.llm.token_count.total"),
            _sum_tokens(original, "neatlogs.llm.token_count.total"),
        ),
        cost_delta_usd=_delta(
            _sum_float(replay, "neatlogs.llm.cost_usd"),
            _sum_float(original, "neatlogs.llm.cost_usd"),
        ),
    )
