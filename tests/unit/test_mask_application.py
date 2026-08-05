"""
Regression tests for client-side mask application in NeatlogsSpanProcessor.

Guards two bugs fixed in span_processor.on_end (§7b/§7c/§8):
  1. A per-span mask (@span(mask=)/trace(mask=), stamped as neatlogs.mask_id)
     must reach the EXPORTED span, not just the file log. Previously the mask
     sites gated on self.mask (global only), so a per-span-only mask was a no-op
     on export.
  2. The mask must be applied EXACTLY ONCE per span. Previously §7b + §7c (+ §8)
     applied it 2-3x, which silently double-transforms non-idempotent masks
     (e.g. translation).

These tests drive the processor directly with a fake ReadableSpan so they run
with no exporter/backend and no LLM calls.
"""

from types import SimpleNamespace

import pytest

from neatlogs.core.mask import register_mask
from neatlogs.core.span_processor import NeatlogsSpanProcessor


class _MutableAttrs(dict):
    """Stand-in for OTel BoundedAttributes: a mutable mapping."""

    _immutable = False


def _make_span(attrs, name="child", trace_id=0x1, span_id=0x2, parent_id=None):
    """Minimal ReadableSpan-like object the processor's on_end reads from."""
    ctx = SimpleNamespace(trace_id=trace_id, span_id=span_id)
    parent = SimpleNamespace(span_id=parent_id) if parent_id is not None else None
    status = SimpleNamespace(status_code=SimpleNamespace(name="OK"), description=None)
    scope = SimpleNamespace(name="neatlogs.test")
    a = _MutableAttrs(attrs)
    return SimpleNamespace(
        name=name,
        attributes=a,
        _attributes=a,
        context=ctx,
        parent=parent,
        start_time=1,
        end_time=2,
        status=status,
        events=[],
        resource=SimpleNamespace(attributes={}),
        instrumentation_scope=scope,
        kind=None,
    )


def _counting_mask(counter, tag):
    def mask(span):
        counter["n"] += 1
        attrs = span.get("attributes", {}) or {}
        for k in list(attrs):
            v = attrs[k]
            if isinstance(v, str) and any(h in k.lower() for h in ("input", "output", "value")):
                attrs[k] = f"[{tag}#{counter['n']}]" + v
        return span

    return mask


def test_global_mask_applied_once_and_exported():
    counter = {"n": 0}
    proc = NeatlogsSpanProcessor(mask=_counting_mask(counter, "G"))
    span = _make_span(
        {
            "openinference.span.kind": "CHAIN",
            "input.value": "IN",
            "output.value": "OUT",
        }
    )
    proc.on_end(span)

    assert counter["n"] == 1, f"global mask applied {counter['n']}x, expected 1"
    # Value on the EXPORTED span (span._attributes) is masked exactly once.
    assert span._attributes["input.value"] == "[G#1]IN"
    assert span._attributes["output.value"] == "[G#1]OUT"


def test_per_span_mask_without_global_reaches_export():
    """The bug that mattered most: per-span mask, NO global mask."""
    counter = {"n": 0}
    proc = NeatlogsSpanProcessor(mask=None)  # no global mask
    mask_id = register_mask(_counting_mask(counter, "S"))
    span = _make_span(
        {
            "openinference.span.kind": "CHAIN",
            "neatlogs.mask_id": mask_id,
            "input.value": "IN",
            "output.value": "OUT",
        }
    )
    proc.on_end(span)

    assert counter["n"] == 1, f"per-span mask applied {counter['n']}x, expected 1"
    assert (
        span._attributes["input.value"] == "[S#1]IN"
    ), "per-span mask did not reach the exported span"
    assert span._attributes["output.value"] == "[S#1]OUT"


def test_per_span_mask_takes_precedence_over_global():
    gcount, scount = {"n": 0}, {"n": 0}
    proc = NeatlogsSpanProcessor(mask=_counting_mask(gcount, "G"))
    mask_id = register_mask(_counting_mask(scount, "S"))
    span = _make_span(
        {
            "openinference.span.kind": "CHAIN",
            "neatlogs.mask_id": mask_id,
            "input.value": "IN",
        }
    )
    proc.on_end(span)

    assert scount["n"] == 1, "per-span mask should run exactly once"
    assert gcount["n"] == 0, "global mask must NOT run when a per-span mask exists"
    assert span._attributes["input.value"] == "[S#1]IN"


def test_no_mask_leaves_values_untouched():
    proc = NeatlogsSpanProcessor(mask=None)
    span = _make_span(
        {
            "openinference.span.kind": "CHAIN",
            "input.value": "IN",
        }
    )
    proc.on_end(span)
    assert span._attributes["input.value"] == "IN"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-p", "no:logfire"]))
