"""
End-to-end demo for the 3 PR #21 features added to the Trace Doctor:

  A. OTel GenAI semantic-convention validation
       - otel-genai-missing (warning)
       - otel-genai-inconsistent (info)

  C. Token-waste pattern detection
       - oversized-prompt (warning)
       - repeated-system-prompt (info, --read-prompt-content opt-in)
       - unused-tool-definition (info)

  D. Manual-fix snippet output
       - --emit-fix <code> CLI flag
       - 4 registered snippets

Each scenario writes a synthetic trace to a temp JSONL file, runs the
doctor, and asserts the expected findings + output. End-to-end in the
sense that it exercises the public Python API (diagnose, format_report,
main) the same way the CLI would.

Captured output is mirrored to docs/pr20-evidence/doctor_e2e_pr21_output.txt
for the maintainer to inspect.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

# Make the local in-repo doctor the one we test (not any installed copy).
sys.path.insert(0, "/Users/harshkashyap/Projects/Open Source/Neatlogs")
from neatlogs.doctor import (
    DoctorReport,
    diagnose,
    format_report,
    main,
    render_fix_snippet,
)

PASS = "✅"
FAIL = "❌"

results: list[tuple[str, bool, str]] = []  # (name, passed, note)


def record(name: str, passed: bool, note: str = "") -> None:
    marker = PASS if passed else FAIL
    results.append((name, passed, note))
    print(f"  {marker} {name}{(' — ' + note) if note else ''}")


def write_jsonl(path: Path, spans: list[dict]) -> None:
    with open(path, "w") as f:
        for s in spans:
            f.write(json.dumps(s) + "\n")


# ---------------------------------------------------------------------------
# Scenario A.1 — otel-genai-missing fires on a trace with neatlogs-only LLM spans
# ---------------------------------------------------------------------------

print("\n=== A.1 otel-genai-missing (neatlogs-only trace) ===")
with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
    path = Path(f.name)
write_jsonl(
    path,
    [
        {
            "trace_id": "t", "span_id": "wf", "parent_span_id": None,
            "name": "wf", "kind": "workflow", "start_time": 100,
            "end_time": 200, "duration_ns": 100_000_000,
            "status": {"code": "OK"},
            "attributes": {"neatlogs.span.kind": "workflow"},
            "events": [], "instrumentation_scope": {"name": "neatlogs.core.context"},
        },
        {
            "trace_id": "t", "span_id": "llm", "parent_span_id": "wf",
            "name": "openai.chat", "kind": "llm", "start_time": 100,
            "end_time": 200, "duration_ns": 100_000_000,
            "status": {"code": "OK"},
            "attributes": {
                "neatlogs.span.kind": "llm",
                "neatlogs.llm.input_messages.0.role": "user",
                "neatlogs.llm.input_messages.0.content": "hi",
                "neatlogs.llm.output_messages.0.role": "assistant",
                "neatlogs.llm.output_messages.0.content": "hello",
            },
            "events": [], "instrumentation_scope": {"name": "neatlogs.core.context"},
        },
    ],
)
report = diagnose(path)
codes = [f.code for f in report.findings]
record(
    "A.1 otel-genai-missing fires on neatlogs-only LLM span",
    "otel-genai-missing" in codes,
    f"codes={codes}",
)
record(
    "A.1 severity is warning",
    next((f.severity for f in report.findings if f.code == "otel-genai-missing"), None) == "warning",
)
record(
    "A.1 fix_class is config",
    next((f.fix_class for f in report.findings if f.code == "otel-genai-missing"), None) == "config",
)
path.unlink()


# ---------------------------------------------------------------------------
# Scenario A.2 — otel-genai-missing does NOT fire when gen_ai.* attrs are set
# ---------------------------------------------------------------------------

print("\n=== A.2 otel-genai-missing does NOT fire (clean OTel trace) ===")
with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
    path = Path(f.name)
write_jsonl(
    path,
    [
        {
            "trace_id": "t", "span_id": "wf", "parent_span_id": None,
            "name": "wf", "kind": "workflow", "start_time": 100,
            "end_time": 200, "duration_ns": 100_000_000,
            "status": {"code": "OK"},
            "attributes": {"neatlogs.span.kind": "workflow"},
            "events": [], "instrumentation_scope": {"name": "neatlogs.core.context"},
        },
        {
            "trace_id": "t", "span_id": "llm", "parent_span_id": "wf",
            "name": "openai.chat", "kind": "llm", "start_time": 100,
            "end_time": 200, "duration_ns": 100_000_000,
            "status": {"code": "OK"},
            "attributes": {
                "neatlogs.span.kind": "llm",
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": "openai",
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 50,
                "neatlogs.llm.input_messages.0.role": "user",
                "neatlogs.llm.input_messages.0.content": "hi",
                "neatlogs.llm.output_messages.0.role": "assistant",
                "neatlogs.llm.output_messages.0.content": "hello",
            },
            "events": [], "instrumentation_scope": {"name": "neatlogs.core.context"},
        },
    ],
)
report = diagnose(path)
codes = [f.code for f in report.findings]
record(
    "A.2 otel-genai-missing does NOT fire when gen_ai.* attrs present",
    "otel-genai-missing" not in codes,
    f"codes={codes}",
)
path.unlink()


# ---------------------------------------------------------------------------
# Scenario A.3 — otel-genai-inconsistent fires when kinds disagree
# ---------------------------------------------------------------------------

print("\n=== A.3 otel-genai-inconsistent (neatlogs=llm, OTel=embeddings) ===")
with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
    path = Path(f.name)
write_jsonl(
    path,
    [
        {
            "trace_id": "t", "span_id": "wf", "parent_span_id": None,
            "name": "wf", "kind": "workflow", "start_time": 100,
            "end_time": 200, "duration_ns": 100_000_000,
            "status": {"code": "OK"},
            "attributes": {"neatlogs.span.kind": "workflow"},
            "events": [], "instrumentation_scope": {"name": "neatlogs.core.context"},
        },
        {
            "trace_id": "t", "span_id": "x", "parent_span_id": "wf",
            "name": "embed", "kind": "llm", "start_time": 100,
            "end_time": 200, "duration_ns": 100_000_000,
            "status": {"code": "OK"},
            "attributes": {
                "neatlogs.span.kind": "llm",
                "gen_ai.operation.name": "embeddings",
                "gen_ai.provider.name": "openai",
                "gen_ai.request.model": "text-embedding-3-small",
            },
            "events": [], "instrumentation_scope": {"name": "neatlogs.core.context"},
        },
    ],
)
report = diagnose(path)
f = next((f for f in report.findings if f.code == "otel-genai-inconsistent"), None)
record(
    "A.3 otel-genai-inconsistent fires when kinds disagree",
    f is not None,
)
record("A.3 severity is info", f.severity == "info" if f else False)
path.unlink()


# ---------------------------------------------------------------------------
# Scenario C.1 — oversized-prompt fires on a >50K-char LLM span
# ---------------------------------------------------------------------------

print("\n=== C.1 oversized-prompt (>50K chars in one LLM span) ===")
with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
    path = Path(f.name)
write_jsonl(
    path,
    [
        {
            "trace_id": "t", "span_id": "wf", "parent_span_id": None,
            "name": "wf", "kind": "workflow", "start_time": 100,
            "end_time": 200, "duration_ns": 100_000_000,
            "status": {"code": "OK"},
            "attributes": {"neatlogs.span.kind": "workflow"},
            "events": [], "instrumentation_scope": {"name": "neatlogs.core.context"},
        },
        {
            "trace_id": "t", "span_id": "llm", "parent_span_id": "wf",
            "name": "leak.chat", "kind": "llm", "start_time": 100,
            "end_time": 200, "duration_ns": 100_000_000,
            "status": {"code": "OK"},
            "attributes": {
                "neatlogs.span.kind": "llm",
                "neatlogs.llm.system": "x" * 60_000,  # 60K chars — bug
                "neatlogs.llm.input_messages.0.role": "user",
                "neatlogs.llm.input_messages.0.content": "hi",
            },
            "events": [], "instrumentation_scope": {"name": "neatlogs.core.context"},
        },
    ],
)
report = diagnose(path)
f = next((f for f in report.findings if f.code == "oversized-prompt"), None)
record("C.1 oversized-prompt fires on 60K-char prompt", f is not None)
record("C.1 severity is warning", f.severity == "warning" if f else False)
record("C.1 fix_class is config", f.fix_class == "config" if f else False)
path.unlink()


# ---------------------------------------------------------------------------
# Scenario C.2 — repeated-system-prompt: OFF by default, ON with opt-in
# ---------------------------------------------------------------------------

print("\n=== C.2 repeated-system-prompt (PII opt-in) ===")
# Build 12 spans sharing the same system prompt.
spans = [
    {
        "trace_id": "t", "span_id": "wf", "parent_span_id": None,
        "name": "wf", "kind": "workflow", "start_time": 100,
        "end_time": 200, "duration_ns": 100_000_000,
        "status": {"code": "OK"},
        "attributes": {"neatlogs.span.kind": "workflow"},
        "events": [], "instrumentation_scope": {"name": "neatlogs.core.context"},
    }
]
for i in range(12):
    spans.append(
        {
            "trace_id": "t", "span_id": f"llm-{i}", "parent_span_id": "wf",
            "name": f"openai.chat-{i}", "kind": "llm", "start_time": 100,
            "end_time": 200, "duration_ns": 100_000_000,
            "status": {"code": "OK"},
            "attributes": {
                "neatlogs.span.kind": "llm",
                "neatlogs.llm.system": "You are a customer-support agent for Acme.",
            },
            "events": [], "instrumentation_scope": {"name": "neatlogs.core.context"},
        }
    )

with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
    path = Path(f.name)
write_jsonl(path, spans)

# Default: off (PII-safe)
report = diagnose(path)
codes = [f.code for f in report.findings]
record(
    "C.2 repeated-system-prompt does NOT fire by default (PII-safe)",
    "repeated-system-prompt" not in codes,
    f"codes={codes}",
)

# Opt-in: on
report = diagnose(path, read_prompt_content=True)
codes = [f.code for f in report.findings]
record(
    "C.2 repeated-system-prompt fires when --read-prompt-content=True",
    "repeated-system-prompt" in codes,
    f"codes={codes}",
)
f = next((f for f in report.findings if f.code == "repeated-system-prompt"), None)
record("C.2 fix_class is config", f.fix_class == "config" if f else False)
record(
    "C.2 suggestion mentions prompt caching",
    "caching" in f.suggestion.lower() if f else False,
)
path.unlink()


# ---------------------------------------------------------------------------
# Scenario C.3 — unused-tool-definition fires when a defined tool is never called
# ---------------------------------------------------------------------------

print("\n=== C.3 unused-tool-definition ===")
with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
    path = Path(f.name)
write_jsonl(
    path,
    [
        {
            "trace_id": "t", "span_id": "wf", "parent_span_id": None,
            "name": "wf", "kind": "workflow", "start_time": 100,
            "end_time": 200, "duration_ns": 100_000_000,
            "status": {"code": "OK"},
            "attributes": {"neatlogs.span.kind": "workflow"},
            "events": [], "instrumentation_scope": {"name": "neatlogs.core.context"},
        },
        {
            "trace_id": "t", "span_id": "llm", "parent_span_id": "wf",
            "name": "openai.chat", "kind": "llm", "start_time": 100,
            "end_time": 200, "duration_ns": 100_000_000,
            "status": {"code": "OK"},
            "attributes": {
                "neatlogs.span.kind": "llm",
                "neatlogs.llm.tools": json.dumps(
                    [{"function": {"name": "search_web"}},
                     {"function": {"name": "get_weather"}},
                     {"function": {"name": "send_email"}}]
                ),
                # No tool_calls at all — the LLM never invoked any tool.
            },
            "events": [], "instrumentation_scope": {"name": "neatlogs.core.context"},
        },
    ],
)
report = diagnose(path)
f = next((f for f in report.findings if f.code == "unused-tool-definition"), None)
record("C.3 unused-tool-definition fires", f is not None)
if f:
    record("C.3 evidence names all 3 unused tools",
        "search_web" in f.evidence and "get_weather" in f.evidence and "send_email" in f.evidence)
    record("C.3 fix_class is config", f.fix_class == "config")
path.unlink()


# ---------------------------------------------------------------------------
# Scenario D.1 — --emit-fix <code> prints a BEFORE/AFTER snippet
# ---------------------------------------------------------------------------

print("\n=== D.1 --emit-fix <code> (manual-fix snippet) ===")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = main(["--emit-fix", "init-after-client"])
out = buf.getvalue()
record("D.1 --emit-fix init-after-client exits 0", rc == 0)
record("D.1 output contains '# Finding: init-after-client'",
    "# Finding: init-after-client" in out)
record("D.1 output contains '# BEFORE:'", "# BEFORE:" in out)
record("D.1 output contains '# AFTER:'", "# AFTER:" in out)
record("D.1 BEFORE block has 'from openai import OpenAI'",
    "from openai import OpenAI" in out)
# The AFTER block must have 'import neatlogs' BEFORE 'from openai' to fix
# the init-order issue (the whole point of the snippet).
after_marker = out.find("# AFTER:")
if after_marker >= 0 and "import neatlogs" in out[after_marker:]:
    after_section = out[after_marker:]
    record("D.1 AFTER block has 'import neatlogs' before 'from openai'",
        after_section.find("import neatlogs") < after_section.find("from openai"))


# ---------------------------------------------------------------------------
# Scenario D.2 — --emit-fix with unknown code returns exit 2 + stderr message
# ---------------------------------------------------------------------------

print("\n=== D.2 --emit-fix unknown code ===")
buf_out, buf_err = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
    rc = main(["--emit-fix", "totally-bogus-code"])
record("D.2 --emit-fix unknown code exits 2", rc == 2)
record("D.2 stderr says 'Unknown finding code'",
    "Unknown finding code" in buf_err.getvalue())
record("D.2 stdout is empty", buf_out.getvalue() == "")


# ---------------------------------------------------------------------------
# Scenario D.3 — --emit-fix works for all 4 registered snippets
# ---------------------------------------------------------------------------

print("\n=== D.3 --emit-fix for all 4 registered snippets ===")
import neatlogs.doctor as _doc
for code in _doc._FIX_SNIPPETS:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["--emit-fix", code])
    out = buf.getvalue()
    desc, before, after = _doc._FIX_SNIPPETS[code]
    passed = (
        rc == 0
        and f"# Finding: {code}" in out
        and "# BEFORE:" in out
        and "# AFTER:" in out
        and before.strip() in out
        and after.strip() in out
    )
    record(f"D.3 --emit-fix {code} renders full snippet", passed)


# ---------------------------------------------------------------------------
# Scenario D.4 — --emit-fix ignores the path argument
# ---------------------------------------------------------------------------

print("\n=== D.4 --emit-fix ignores path ===")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = main([
        "--emit-fix", "missing-span-kind",
        "/nonexistent/path/that/should/not/be/read.log",
    ])
out = buf.getvalue()
record("D.4 --emit-fix ignores nonexistent path", rc == 0 and "missing-span-kind" in out)


# ---------------------------------------------------------------------------
# Scenario E — real CLI: full workflow with --json
# ---------------------------------------------------------------------------

print("\n=== E. real CLI --json end-to-end ===")
with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
    path = Path(f.name)
write_jsonl(
    path,
    [
        {
            "trace_id": "demo", "span_id": "wf", "parent_span_id": None,
            "name": "wf", "kind": "workflow", "start_time": 100,
            "end_time": 200, "duration_ns": 100_000_000,
            "status": {"code": "OK"},
            "attributes": {"neatlogs.span.kind": "workflow"},
            "events": [], "instrumentation_scope": {"name": "neatlogs.core.context"},
        },
        {
            "trace_id": "demo", "span_id": "llm", "parent_span_id": "wf",
            "name": "openai.chat", "kind": "llm", "start_time": 100,
            "end_time": 200, "duration_ns": 100_000_000,
            "status": {"code": "OK"},
            "attributes": {
                "neatlogs.span.kind": "llm",
                "neatlogs.llm.input_messages.0.role": "user",
                "neatlogs.llm.input_messages.0.content": "hi",
                "neatlogs.llm.output_messages.0.role": "assistant",
                "neatlogs.llm.output_messages.0.content": "hello",
                "neatlogs.llm.tools": json.dumps(
                    [{"function": {"name": "dead_tool"}}]
                ),
            },
            "events": [], "instrumentation_scope": {"name": "neatlogs.core.context"},
        },
    ],
)
buf_out, buf_err = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
    rc = main([str(path), "--json"])
data = json.loads(buf_out.getvalue())
record("E. real CLI --json returns valid JSON", isinstance(data, dict))
record("E. real CLI --json has 'findings' list", isinstance(data.get("findings"), list))
codes = [f["code"] for f in data["findings"]]
record("E. real CLI --json includes 'otel-genai-missing'", "otel-genai-missing" in codes,
    f"codes={codes}")
record("E. real CLI --json includes 'unused-tool-definition'",
    "unused-tool-definition" in codes, f"codes={codes}")
# exit code 0 since no error-severity findings
record("E. real CLI --json exit code is 0 (no error-severity findings)", rc == 0)
path.unlink()


# ---------------------------------------------------------------------------
# Scenario F — real CLI: --read-prompt-content + --json
# ---------------------------------------------------------------------------

print("\n=== F. real CLI --read-prompt-content --json ===")
# Build 12 spans sharing the same system prompt.
spans = [
    {
        "trace_id": "demo", "span_id": "wf", "parent_span_id": None,
        "name": "wf", "kind": "workflow", "start_time": 100,
        "end_time": 200, "duration_ns": 100_000_000,
        "status": {"code": "OK"},
        "attributes": {"neatlogs.span.kind": "workflow"},
        "events": [], "instrumentation_scope": {"name": "neatlogs.core.context"},
    }
]
for i in range(12):
    spans.append({
        "trace_id": "demo", "span_id": f"llm-{i}", "parent_span_id": "wf",
        "name": f"openai.chat-{i}", "kind": "llm", "start_time": 100,
        "end_time": 200, "duration_ns": 100_000_000,
        "status": {"code": "OK"},
        "attributes": {
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.system": "static prompt repeated 12 times",
        },
        "events": [], "instrumentation_scope": {"name": "neatlogs.core.context"},
    })
with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
    path = Path(f.name)
write_jsonl(path, spans)

buf_out = io.StringIO()
with contextlib.redirect_stdout(buf_out):
    rc = main([str(path), "--read-prompt-content", "--json"])
data = json.loads(buf_out.getvalue())
codes = [f["code"] for f in data["findings"]]
record(
    "F. real CLI --read-prompt-content surfaces 'repeated-system-prompt'",
    "repeated-system-prompt" in codes,
    f"codes={codes}",
)
path.unlink()


# ---------------------------------------------------------------------------
# Scenario G — real SDK roundtrip: a real neatlogs span emission flows to the
# doctor and the 3 new findings fire correctly.
# ---------------------------------------------------------------------------

print("\n=== G. real SDK roundtrip ===")
# Reuse the proven pattern from doctor_e2e_realsdk.py: configure a file
# log via env vars, use @neatlogs decorators, flush+shutdown, then diagnose.
# The new thing this test verifies: the doctor (PR #21) runs correctly
# against real-SDK output and produces no new false positives.
import logging
import os
import neatlogs

log_path = Path(tempfile.mkstemp(suffix=".log")[1])
os.environ["NEATLOGS_API_KEY"] = "test-key-noop"
os.environ["NEATLOGS_LOG_SPANS"] = "true"
os.environ["NEATLOGS_LOG_SPANS_FILE"] = str(log_path)
os.environ["NEATLOGS_TELEMETRY_ENABLED"] = "false"
os.environ["NEATLOGS_LOGGING_LEVEL"] = "ERROR"

neatlogs.init(
    api_key="test-noop",
    workflow_name="real_e2e_roundtrip",
    disable_export=True,
    tracer_provider=None,
)


@neatlogs.trace(name="root_workflow")
def main():
    @neatlogs.span("CHAIN", name="setup_step")
    def setup():
        return "ready"

    @neatlogs.span("CHAIN", name="process_step")
    def process():
        return "done"

    setup()
    process()
    return "ok"


main()
neatlogs.flush()
neatlogs.shutdown()

# Read the JSONL log the SDK wrote.
spans = []
with open(log_path) as f:
    for line in f:
        try:
            spans.append(json.loads(line))
        except json.JSONDecodeError:
            pass

record(
    "G. real SDK roundtrip: SDK wrote a JSONL log",
    len(spans) >= 2,
    f"spans_read={len(spans)}",
)
record(
    "G. real SDK roundtrip: every span has instrumentation_scope",
    all("instrumentation_scope" in s for s in spans),
)

# Run the doctor on the real log.
report = diagnose(log_path)
codes = [f.code for f in report.findings]
record(
    "G. real SDK roundtrip: doctor runs without error",
    isinstance(report, DoctorReport),
)
record(
    "G. real SDK roundtrip: no false foreign-scope finding",
    not any(f.code == "foreign-instrumentation-detected" for f in report.findings),
)
record(
    "G. real SDK roundtrip: no false 'otel-genai-missing' on CHAIN-only trace",
    "otel-genai-missing" not in codes,
    f"codes={codes}",
    # The CHAIN spans are not LLM-kind, so otel-genai-missing MUST NOT fire.
)
log_path.unlink()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

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
