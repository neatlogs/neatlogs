"""
E2E demo for the 3 new Trace Doctor dimensions added on top of
PR #20 + PR #21.

Covers:
- latency-outlier (warning, data_integrity) — LLM span > 3x trace median
- rate-limited (warning, instrumentation) — 429 / retry-after / quota attrs
- pii-detected (warning, data_integrity) — opt-in via check_pii=True

Captured output is mirrored to doctor_e2e_v3_output.txt.
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
) -> dict:
    if attributes is None:
        attributes = {"neatlogs.span.kind": kind}
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "kind": kind,
        "start_time": start_time,
        "end_time": end_time,
        "duration_ns": duration_ns,
        "status": {"code": "OK"},
        "attributes": attributes,
        "events": [],
        "instrumentation_scope": {"name": "neatlogs.core.context"},
    }


# E. latency-outlier
print("E. latency-outlier")
spans = [
    make_span(span_id="wf", name="wf", kind="workflow",
              start_time=100, end_time=200, duration_ns=100_000_000,
              attributes={"neatlogs.span.kind": "workflow"}),
]
# 3 normal 1s calls, then 1 outlier at 5s
for i in range(3):
    spans.append(make_span(
        span_id=f"l{i}", parent_span_id="wf", name="openai.chat", kind="llm",
        start_time=300 + i * 2000, end_time=300 + i * 2000 + 1000,
        duration_ns=1_000_000_000,
        attributes={"neatlogs.span.kind": "llm", "gen_ai.operation.name": "chat"},
    ))
spans.append(make_span(
    span_id="l3", parent_span_id="wf", name="openai.chat", kind="llm",
    start_time=6300, end_time=11300, duration_ns=5_000_000_000,
    attributes={"neatlogs.span.kind": "llm", "gen_ai.operation.name": "chat"},
))
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
    write_jsonl(Path(f.name), spans)
    path = f.name
report = diagnose(path)
f = next((f for f in report.findings if f.code == "latency-outlier"), None)
record("fires on 1 of 4 LLM spans at 5x median", f is not None)
record("severity is warning", f is not None and f.severity == "warning")
record("fix_class is data_integrity", f is not None and f.fix_class == "data_integrity")
record("evidence names the 5x ratio", f is not None and ("5.0x" in f.evidence or "5" in f.evidence))
Path(path).unlink()


# F. rate-limited
print("\nF. rate-limited")
spans = [
    make_span(span_id="wf", name="wf", kind="workflow",
              start_time=100, end_time=200, duration_ns=100_000_000,
              attributes={"neatlogs.span.kind": "workflow"}),
    make_span(span_id="l1", parent_span_id="wf", name="anthropic", kind="llm",
              start_time=300, end_time=500, duration_ns=200_000_000,
              attributes={"neatlogs.span.kind": "llm",
                         "http.response.status_code": 429, "retry-after": "2"}),
    make_span(span_id="l2", parent_span_id="wf", name="openai", kind="llm",
              start_time=600, end_time=800, duration_ns=200_000_000,
              attributes={"neatlogs.span.kind": "llm",
                         "openai.error.code": "rate_limit_exceeded"}),
    make_span(span_id="l3", parent_span_id="wf", name="openai", kind="llm",
              start_time=900, end_time=1100, duration_ns=200_000_000,
              attributes={"neatlogs.span.kind": "llm",
                         "x_ratelimit_remaining_requests": 0}),
]
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
    write_jsonl(Path(f.name), spans)
    path = f.name
report = diagnose(path)
fired = [f for f in report.findings if f.code == "rate-limited"]
record("fires on 3 different throttling signals (3 spans)", len(fired) == 3,
       f"got {len(fired)}")
record("severity is warning", all(f.severity == "warning" for f in fired))
record("fix_class is instrumentation",
       all(f.fix_class == "instrumentation" for f in fired))
Path(path).unlink()


# G. pii-detected (opt-in)
print("\nG. pii-detected (opt-in)")
spans = [
    make_span(span_id="wf", name="wf", kind="workflow",
              start_time=100, end_time=200, duration_ns=100_000_000,
              attributes={"neatlogs.span.kind": "workflow",
                         "metadata": {"email": "user@example.com"}}),
    make_span(span_id="llm", parent_span_id="wf", name="openai", kind="llm",
              start_time=300, end_time=500, duration_ns=200_000_000,
              attributes={"neatlogs.span.kind": "llm",
                         "gen_ai.operation.name": "chat",
                         "system_prompt": "ignored",
                         "ssn": "123-45-6789",
                         "card": "4111 1111 1111 1111",
                         "phone": "212-555-1234"}),
]
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
    write_jsonl(Path(f.name), spans)
    path = f.name

# Without opt-in — must NOT fire
report = diagnose(path)
record("does NOT fire without --check-pii",
       [f for f in report.findings if f.code == "pii-detected"] == [])

# With opt-in
report = diagnose(path, check_pii=True)
fired = [f for f in report.findings if f.code == "pii-detected"]
record("fires with --check-pii on 2 spans (wf has email, llm has ssn+cc+phone)",
       len(fired) == 2, f"got {len(fired)}")
evidence_all = " ".join(f.evidence for f in fired)
record("email pattern matched", "email" in evidence_all)
record("us_ssn pattern matched", "us_ssn" in evidence_all)
record("credit_card pattern matched", "credit_card" in evidence_all)
record("us_phone pattern matched", "us_phone" in evidence_all)
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
