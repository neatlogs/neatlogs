"""
E2E demo for the 4 new Trace Doctor dimensions and 2 bug regressions.

Covers retry-loop, unbalanced-llm-usage, empty-trace,
context-propagation-broken, and the missing-io doc_url /
related_codes fixes.

Captured output is mirrored to doctor_e2e_new_checks_output.txt.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/Users/harshkashyap/Projects/Open Source/Neatlogs")
from neatlogs.doctor import diagnose

PASS = "✅"
FAIL = "❌"

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, note: str = "") -> None:
    results.append((name, ok, note))
    icon = PASS if ok else FAIL
    print(f"  {icon} {name}{(' — ' + note) if note else ''}")


def write_jsonl(path: Path, spans: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(s) for s in spans) + "\n")


def make_span(
    *,
    trace_id: str = "t",
    span_id: str = "wf",
    parent_span_id: str | None = None,
    name: str = "wf",
    kind: str = "workflow",
    attributes: dict | None = None,
    start_time: int = 100,
    end_time: int = 200,
    duration_ns: int = 100_000_000,
    status: dict | None = None,
    events: list | None = None,
) -> dict:
    if attributes is None:
        attributes = {"neatlogs.span.kind": kind}
    if status is None:
        status = {"code": "OK"}
    if events is None:
        events = []
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "kind": kind,
        "start_time": start_time,
        "end_time": end_time,
        "duration_ns": duration_ns,
        "status": status,
        "attributes": attributes,
        "events": events,
        "instrumentation_scope": {"name": "neatlogs.core.context"},
    }


# E. retry-loop
print("E. retry-loop")
spans = [
    make_span(span_id="wf", name="wf", kind="workflow",
              start_time=100, end_time=1000, duration_ns=900_000_000,
              attributes={"neatlogs.span.kind": "workflow"}),
]
for i in range(5):
    spans.append(
        make_span(span_id=f"r{i}", parent_span_id="wf", name="call_api",
                  kind="tool", start_time=200 + i * 10, end_time=210 + i * 10,
                  duration_ns=10_000_000,
                  attributes={"neatlogs.span.kind": "tool"})
    )
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
    write_jsonl(Path(f.name), spans)
    path = f.name
report = diagnose(path)
codes = [f.code for f in report.findings]
record("retry-loop fires on 5 same-name spans", "retry-loop" in codes, f"codes={codes}")
f = next((f for f in report.findings if f.code == "retry-loop"), None)
record("severity is info", f is not None and f.severity == "info")
record("evidence has count + parent",
       f is not None and "5 consecutive" in f.evidence and "wf" in f.evidence)
record("fix_class is instrumentation",
       f is not None and f.fix_class == "instrumentation")
Path(path).unlink()

# 3 same-name spans should NOT fire
spans3 = spans[:4]
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
    write_jsonl(Path(f.name), spans3)
    path = f.name
report = diagnose(path)
record("retry-loop does NOT fire on 3 same-name spans",
       not any(f.code == "retry-loop" for f in report.findings))
Path(path).unlink()


# F. unbalanced-llm-usage
print("\nF. unbalanced-llm-usage")
spans = [
    make_span(span_id="wf", name="wf", kind="workflow"),
    make_span(span_id="llm", parent_span_id="wf", name="openai", kind="llm",
              attributes={"neatlogs.span.kind": "llm",
                         "gen_ai.operation.name": "chat",
                         "gen_ai.usage.input_tokens": 100}),
]
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
    write_jsonl(Path(f.name), spans)
    path = f.name
report = diagnose(path)
f = next((f for f in report.findings if f.code == "unbalanced-llm-usage"), None)
record("fires on input-only", f is not None)
record("severity is warning", f is not None and f.severity == "warning")
record("fix_class is data_integrity", f is not None and f.fix_class == "data_integrity")
record("evidence mentions input_tokens + output_tokens",
       f is not None and "input_tokens" in f.evidence and "output_tokens" in f.evidence)
Path(path).unlink()

spans[1]["attributes"] = {"neatlogs.span.kind": "llm",
                          "gen_ai.operation.name": "chat",
                          "gen_ai.usage.output_tokens": 50}
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
    write_jsonl(Path(f.name), spans)
    path = f.name
report = diagnose(path)
record("fires on output-only", any(f.code == "unbalanced-llm-usage" for f in report.findings))
Path(path).unlink()

spans[1]["attributes"] = {"neatlogs.span.kind": "llm",
                          "gen_ai.operation.name": "chat",
                          "gen_ai.usage.input_tokens": 100,
                          "gen_ai.usage.output_tokens": 50}
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
    write_jsonl(Path(f.name), spans)
    path = f.name
report = diagnose(path)
record("does NOT fire when both are set",
       not any(f.code == "unbalanced-llm-usage" for f in report.findings))
Path(path).unlink()


# G. empty-trace
print("\nG. empty-trace")
spans = [make_span(span_id="wf", name="root_workflow", kind="workflow")]
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
    write_jsonl(Path(f.name), spans)
    path = f.name
report = diagnose(path)
f = next((f for f in report.findings if f.code == "empty-trace"), None)
record("fires on 1-span trace", f is not None)
record("severity is info", f is not None and f.severity == "info")
record("evidence has the span name", f is not None and "root_workflow" in f.evidence)
Path(path).unlink()

spans = [make_span(span_id="wf", name="root_workflow", kind="workflow"),
         make_span(span_id="child", parent_span_id="wf", name="child_op", kind="tool")]
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
    write_jsonl(Path(f.name), spans)
    path = f.name
report = diagnose(path)
record("does NOT fire on 2-span trace",
       not any(f.code == "empty-trace" for f in report.findings))
Path(path).unlink()


# H. context-propagation-broken
print("\nH. context-propagation-broken")
spans = [
    make_span(trace_id="t_parent", span_id="p", name="parent_wf", kind="workflow",
              start_time=100, end_time=1000, duration_ns=900_000_000,
              attributes={"neatlogs.span.kind": "workflow", "session.id": "run1"}),
    make_span(trace_id="t_child", span_id="c", parent_span_id="p",
              name="lost_child", kind="tool", start_time=200, end_time=300,
              duration_ns=100_000_000,
              attributes={"neatlogs.span.kind": "tool", "session.id": "run1"}),
]
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
    write_jsonl(Path(f.name), spans)
    path = f.name
report = diagnose(path)
f = next((f for f in report.findings if f.code == "context-propagation-broken"), None)
record("fires on cross-trace parent", f is not None)
record("severity is error", f is not None and f.severity == "error")
record("fix_class is hierarchy", f is not None and f.fix_class == "hierarchy")
record("evidence names the broken span", f is not None and "lost_child" in f.evidence)
Path(path).unlink()

spans = [make_span(trace_id="t", span_id="child", parent_span_id="missing_parent",
                  name="child_op", kind="tool",
                  attributes={"neatlogs.span.kind": "tool", "session.id": "run1"})]
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
    write_jsonl(Path(f.name), spans)
    path = f.name
report = diagnose(path)
codes = [f.code for f in report.findings]
record("does NOT fire on orphan-parent", "context-propagation-broken" not in codes)
record("orphan-parent DOES fire for missing parent", "orphan-parent" in codes)
Path(path).unlink()

spans = [
    make_span(trace_id="t", span_id="wf", name="wf", kind="workflow",
              start_time=100, end_time=1000, duration_ns=900_000_000,
              attributes={"neatlogs.span.kind": "workflow", "session.id": "run1"}),
    make_span(trace_id="t", span_id="child", parent_span_id="wf",
              name="child_op", kind="tool", start_time=200, end_time=300,
              duration_ns=100_000_000,
              attributes={"neatlogs.span.kind": "tool", "session.id": "run1"}),
]
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
    write_jsonl(Path(f.name), spans)
    path = f.name
report = diagnose(path)
record("does NOT fire on normal parent",
       not any(f.code == "context-propagation-broken" for f in report.findings))
Path(path).unlink()


# I. Bug regressions
print("\nI. Bug regressions")
spans = [
    make_span(span_id="wf", name="wf", kind="workflow"),
    make_span(span_id="llm", parent_span_id="wf", name="openai", kind="llm",
              attributes={"neatlogs.span.kind": "llm"}),
]
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
    write_jsonl(Path(f.name), spans)
    path = f.name
report = diagnose(path)
f = next((f for f in report.findings if f.code == "llm-missing-io"), None)
record("llm-missing-io fires on the regression shape", f is not None)
record(
    "doc_url is in-repo, NOT 404 docs.neatlogs.com",
    f is not None and f.doc_url is not None
    and "docs.neatlogs.com" not in f.doc_url
    and f.doc_url.startswith("skills/"),
    f"doc_url={f.doc_url if f else None}",
)
record(
    "related_codes points to a valid finding code",
    f is not None
    and all(ref in {
        "init-after-client", "missing-span-kind", "zero-duration-span",
        "error-status-no-event", "latency-mismatch", "otel-genai-missing",
        "otel-genai-inconsistent", "oversized-prompt", "repeated-system-prompt",
        "unused-tool-definition", "retry-loop", "unbalanced-llm-usage",
        "empty-trace", "context-propagation-broken", "foreign-instrumentation-detected",
        "missing-root-kind", "rootless-http-only", "orphan-parent",
        "self-parent", "duplicate-span-id", "multiple-roots", "cycle",
        "agent-without-llm", "multi-run-log", "scope-not-preserved",
    } for ref in f.related_codes),
    f"related_codes={f.related_codes if f else None}",
)
Path(path).unlink()


print("\n" + "=" * 72)
print("E2E SUMMARY")
print("=" * 72)
total = len(results)
passed = sum(1 for _, p, _ in results if p)
print(f"  Total assertions: {total}")
print(f"  Passed:           {passed} ({100 * passed // total}%)")
print(f"  Failed:           {total - passed}")
if passed != total:
    print("\n  FAILED:")
    for name, ok, note in results:
        if not ok:
            print(f"    - {name}{(' — ' + note) if note else ''}")
    raise SystemExit(1)
print("\n  ✅ ALL E2E TESTS PASSED")
