"""
Neatlogs trace doctor — local linter for processed span JSONL logs.

Reads a span log written by ``neatlogs-doctor`` (or the log-file exporter
emitting ``neatlogs.trace.complete`` events) and surfaces the most common
instrumentation problems:

- hierarchy pathologies (orphan parent, multiple roots, cycles, duplicate
  span_id, self-parent)
- empty / missing input or output on LLM, tool, and retriever spans
- agent spans whose subtree has no LLM child
- foreign instrumentation polluting the trace
- instrumentation that was requested but never reached its patched entry point
- multi-run log files where the user might mistake cross-run pollution for a bug

The doctor is read-only and offline. It never talks to the backend; it only
reads the local JSONL.

Usage (CLI):

    neatlogs-doctor ./spans.log
    neatlogs-doctor ./spans.log --json
    neatlogs-doctor ./spans.log --run-id abc123      # one run only
    neatlogs-doctor ./spans.log --foreign-only       # only foreign-instrumentation findings

Usage (programmatic):

    from neatlogs.doctor import diagnose, format_report
    report = diagnose("./spans.log")
    print(format_report(report))
    if report.has_errors:
        sys.exit(1)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

# --- Constants --------------------------------------------------------------

#: Span kinds that count as "orchestration roots" (must have at least one).
ROOT_KINDS = {"workflow", "chain", "agent", "mcp_tool"}

#: Span kinds where input + output attributes are expected.
IO_KINDS = {"llm", "tool", "retriever", "embedding"}

#: Foreign-instrumentation scope names that should be flagged as pollution.
#: Anything other than ``"neatlogs"`` (or namespaced under ``"neatlogs."``)
#: is treated as foreign.
NEATLOGS_SCOPE_PREFIX = "neatlogs"

#: Default value for missing session.id when grouping by run.
DEFAULT_SESSION_ID = "<no-session>"

#: Evidence-string truncation for span names / role / content.
MAX_EVIDENCE_LEN = 200

#: Expected default duration for an instant span (ns). Anything below this is
#: treated as a "zero duration" finding. Default: 1ms = 1_000_000 ns.
MIN_REASONABLE_DURATION_NS = 1_000_000

#: OTel attribute key for the service name (per OTel semantic conventions).
OTEL_SERVICE_NAME_KEY = "service.name"

#: OTel attribute key for the service version.
OTEL_SERVICE_VERSION_KEY = "service.version"

#: Required attributes every emitted span should have. Used by the
#: attribute-completeness check to detect missing or stripped attributes.
REQUIRED_SPAN_ATTRIBUTES = ("neatlogs.span.kind",)

#: Attribute keys the SDK checks before claiming init succeeded. If a span is
#: present but none of these are set, the most likely cause is a wrapper that
#: was created BEFORE neatlogs.init() — the SDK was loaded but the wrapper
#: was already monkey-patched.
INIT_MARKER_KEYS = (
    "neatlogs.instrumentation.name",
    "neatlogs.span.kind",
    "neatlogs.workflow_name",
)


# --- Result types -----------------------------------------------------------


@dataclass(frozen=True)
class DoctorFinding:
    """A single diagnostic finding emitted by the doctor.

    Findings are immutable so the report can be hashed / diffed in tests.

    The optional ``fix_class``, ``automated_fix_available``, and ``doc_url``
    fields make findings self-describing for coding agents and LLMs: a
    remediation bot can read the ``fix_class`` to know which kind of fix
    to attempt, and ``automated_fix_available`` to know whether to apply
    it or hand off to a human.
    """

    severity: str
    code: str
    title: str
    evidence: str
    suggestion: str
    trace_id: Optional[str] = None
    run_id: Optional[str] = None
    # --- LLM-actionability metadata (Option B scope, see PR #20) -----------
    # fix_class: which category of fix is needed. One of:
    #   "init_order"        - the user must reorder init() and client creation
    #   "attribute"         - a required attribute is missing or malformed
    #   "capture"           - the wrapper is not capturing input/output/latency
    #   "config"            - an env var or config is wrong
    #   "pipeline"          - a stage of the SDK pipeline is broken
    #   "hierarchy"         - the parent/child span structure is wrong
    #   "instrumentation"   - a wrapper is missing or wrongly registered
    #   "data_integrity"    - a span's own data is corrupted (zero duration, etc.)
    #   "none"              - finding is informational; no fix is needed
    fix_class: Optional[str] = None
    # automated_fix_available: True if a tool/agent could fix this without a human.
    automated_fix_available: bool = False
    # doc_url: pointer to the human-readable doc that explains the finding.
    doc_url: Optional[str] = None
    # related_codes: cross-references to other finding codes (for cluster diagnostics).
    related_codes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "title": self.title,
            "evidence": self.evidence,
            "suggestion": self.suggestion,
        }
        if self.trace_id:
            data["trace_id"] = self.trace_id
        if self.run_id:
            data["run_id"] = self.run_id
        if self.fix_class:
            data["fix_class"] = self.fix_class
        if self.automated_fix_available:
            data["automated_fix_available"] = self.automated_fix_available
        if self.doc_url:
            data["doc_url"] = self.doc_url
        if self.related_codes:
            data["related_codes"] = list(self.related_codes)
        return data


@dataclass(frozen=True)
class DoctorReport:
    """Full diagnostic report for one span log file."""

    path: str
    spans_read: int
    trace_count: int
    run_count: int
    invalid_lines: list[int] = field(default_factory=list)
    findings: tuple[DoctorFinding, ...] = field(default_factory=tuple)

    @property
    def has_errors(self) -> bool:
        return any(f.severity == "error" for f in self.findings)

    def findings_by_fix_class(self) -> dict[str, list[DoctorFinding]]:
        """Group findings by ``fix_class`` for LLM/coding-agent consumption.

        Findings without a ``fix_class`` are dropped from the result. The
        returned dict is keyed by fix_class string; each value is a tuple
        of findings sharing that class.
        """
        out: dict[str, list[DoctorFinding]] = {}
        for f in self.findings:
            if f.fix_class is None:
                continue
            out.setdefault(f.fix_class, []).append(f)
        return out

    def findings_by_pipeline_stage(self) -> dict[str, list[DoctorFinding]]:
        """Group findings by the inferred SDK pipeline stage they relate to.

        Stages (in execution order):
          - ``init``        : SDK was not initialized properly
          - ``instrument``  : wrappers did not register or did not capture
          - ``span``        : span creation/recording is broken
          - ``export``      : data is not reaching the backend or log file
          - ``hierarchy``   : parent/child relationships are wrong
        Returns a dict keyed by stage. Findings without a fix_class in the
        known stages are dropped.
        """
        stage_map = {
            "init_order": "init",
            "config": "init",
            "pipeline": "init",
            "instrumentation": "instrument",
            "capture": "instrument",
            "data_integrity": "span",
            "attribute": "span",
            "hierarchy": "hierarchy",
        }
        out: dict[str, list[DoctorFinding]] = {}
        for f in self.findings:
            stage = stage_map.get(f.fix_class or "")
            if stage is None:
                continue
            out.setdefault(stage, []).append(f)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "spans_read": self.spans_read,
            "trace_count": self.trace_count,
            "run_count": self.run_count,
            "invalid_lines": list(self.invalid_lines),
            "findings": [f.to_dict() for f in self.findings],
        }


# --- Entry points -----------------------------------------------------------


def diagnose(
    path: str | Path,
    *,
    run_id: Optional[str] = None,
    foreign_only: bool = False,
) -> DoctorReport:
    """Diagnose a processed span JSONL file.

    Args:
        path: Path to a span log written by the neatlogs log exporter.
        run_id: If set, only analyze spans whose ``session.id`` (or trace_id
            fallback) matches. Useful when a single log file contains many runs.
        foreign_only: If True, only return foreign-instrumentation findings.
            Used by the ``--foreign-only`` CLI flag.

    Returns:
        A :class:`DoctorReport` with the findings sorted by severity then code.
    """
    path_obj = Path(path)
    findings: list[DoctorFinding] = []
    spans, invalid_lines = _read_spans(path_obj, findings)
    runs = _group_by_run(spans)

    if invalid_lines:
        severity = "error" if not spans else "warning"
        findings.append(
            DoctorFinding(
                severity=severity,
                code="invalid-jsonl",
                title="Span log contains invalid JSON lines",
                evidence=f"Invalid line numbers: {', '.join(str(i) for i in invalid_lines[:5])}",
                suggestion="Use a processed span log written by NEATLOGS_LOG_SPANS_FILE.",
            )
        )

    if not spans and not any(f.code == "file-not-found" for f in findings):
        findings.append(
            DoctorFinding(
                severity="error",
                code="no-spans",
                title="No spans found",
                evidence=f"{path_obj} did not contain any processed span records.",
                suggestion=(
                    "Set NEATLOGS_LOG_SPANS=true, run the app again, then call "
                    "neatlogs.flush() and neatlogs.shutdown() before the process exits."
                ),
            )
        )

    # Filter to the requested run if specified.
    if run_id is not None:
        if run_id in runs:
            runs = {run_id: runs[run_id]}
        else:
            findings.append(
                DoctorFinding(
                    severity="error",
                    code="run-id-not-found",
                    title="Requested run id not present in log",
                    evidence=f"run_id={run_id!r} but log has runs: {sorted(runs.keys())[:5]}",
                    suggestion="Omit --run-id to analyze all runs, or pick one from the list.",
                )
            )
            runs = {}

    # Detect multi-run logs and warn.
    if len(runs) > 1 and run_id is None:
        findings.append(
            DoctorFinding(
                severity="warning",
                code="multi-run-log",
                title="Log file contains spans from multiple runs",
                evidence=(
                    f"{len(runs)} runs detected, {len(spans)} spans total. "
                    f"Pass --run-id <id> to scope the report to one run."
                ),
                suggestion=("Rotate NEATLOGS_LOG_SPANS_FILE between runs, or use --run-id."),
            )
        )

    # Analyze each run independently. A run can contain multiple trace_ids.
    any_scope_seen = False  # tracked for the report-level "scope not preserved" finding
    for rid, run_spans in runs.items():
        # Group by trace within the run.
        traces = _group_by_trace(run_spans)
        for tid, trace_spans in traces.items():
            findings.extend(_diagnose_trace(tid, trace_spans, run_id=rid))

        # Cross-trace: detect foreign instrumentation in this run.
        scope_findings, scope_seen_here = _foreign_instrumentation_findings(run_spans, run_id=rid)
        any_scope_seen = any_scope_seen or scope_seen_here
        findings.extend(scope_findings)

    # Report-level: if NO run had instrumentation_scope, emit a single info
    # finding (not N). This avoids noise when a log file has many runs.
    if not any_scope_seen and spans:
        findings.append(
            DoctorFinding(
                severity="info",
                code="scope-not-preserved",
                title="instrumentation_scope not in the log — foreign detection unavailable",
                evidence=(
                    f"All {len(spans)} span(s) lack instrumentation_scope. "
                    "Foreign-instrumentation detection cannot run."
                ),
                suggestion=(
                    "Update neatlogs to a version that preserves "
                    "instrumentation_scope in the span log (see neatlogs/core/span_processor.py)."
                ),
            )
        )

    # Optional: filter to only foreign-instrumentation findings.
    if foreign_only:
        findings = [f for f in findings if f.code.startswith("foreign-instrumentation")]

    # Run-level pipeline-stage summary: when most findings cluster at one
    # stage of the SDK pipeline, surface that to the user so they know
    # where to start fixing. Skipped under --foreign-only since foreign
    # findings don't represent a pipeline failure on the user's side.
    if not foreign_only:
        pipeline_summary = _pipeline_stage_run_finding(findings)
        if pipeline_summary is not None:
            findings.append(pipeline_summary)

    # Stable sort: errors first, then warnings, then info; alphabetical by code.
    severity_rank = {"error": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: (severity_rank.get(f.severity, 99), f.code))

    return DoctorReport(
        path=str(path_obj),
        spans_read=len(spans),
        trace_count=sum(1 for _ in _iter_traces(runs)),
        run_count=len(runs),
        invalid_lines=invalid_lines,
        findings=tuple(findings),
    )


def format_report(report: DoctorReport) -> str:
    """Render a :class:`DoctorReport` as a human-readable text block."""
    lines = [
        "Trace Doctor",
        f"File: {report.path}",
        f"Spans: {report.spans_read}",
        f"Traces: {report.trace_count}",
        f"Runs: {report.run_count}",
    ]

    if not report.findings:
        lines.append("")
        lines.append("No problems found.")
        return "\n".join(lines)

    lines.append("")
    lines.append("Findings:")
    for idx, finding in enumerate(report.findings, start=1):
        loc_parts = []
        if finding.trace_id:
            loc_parts.append(f"trace={finding.trace_id}")
        if finding.run_id and finding.run_id != DEFAULT_SESSION_ID:
            loc_parts.append(f"run={finding.run_id}")
        loc = f" {' '.join(loc_parts)}" if loc_parts else ""
        lines.append(f"{idx}. [{finding.severity}] {finding.title}{loc}")
        lines.append(f"   Evidence: {finding.evidence}")
        lines.append(f"   Fix: {finding.suggestion}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``neatlogs-doctor``."""
    parser = argparse.ArgumentParser(
        prog="neatlogs-doctor",
        description="Diagnose local Neatlogs processed span logs.",
    )
    parser.add_argument("path", help="Path to a processed span JSONL file.")
    parser.add_argument("--json", action="store_true", help="Print a JSON report instead of text.")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Only analyze spans belonging to this run (session.id or trace_id).",
    )
    parser.add_argument(
        "--foreign-only",
        action="store_true",
        help="Only show foreign-instrumentation findings.",
    )
    args = parser.parse_args(argv)

    report = diagnose(args.path, run_id=args.run_id, foreign_only=args.foreign_only)
    if args.json:
        json.dump(report.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(format_report(report) + "\n")
    return 1 if report.has_errors else 0


# --- I/O ---------------------------------------------------------------------


def _read_spans(
    path: Path, findings: list[DoctorFinding]
) -> tuple[list[dict[str, Any]], list[int]]:
    """Read a JSONL span log into a list of span dicts.

    Returns ``(spans, invalid_line_numbers)``. If the file is missing, emits
    a ``file-not-found`` finding and returns ``([], [])``.
    """
    if not path.exists():
        findings.append(
            DoctorFinding(
                severity="error",
                code="file-not-found",
                title="Span log file not found",
                evidence=str(path),
                suggestion="Pass the processed span log path from NEATLOGS_LOG_SPANS_FILE.",
            )
        )
        return [], []

    spans: list[dict[str, Any]] = []
    invalid_lines: list[int] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                invalid_lines.append(line_number)
                continue
            if isinstance(value, dict):
                spans.append(value)
            else:
                # Non-dict JSON line — record but don't fail.
                invalid_lines.append(line_number)
    return spans, invalid_lines


# --- Grouping ----------------------------------------------------------------


def _group_by_run(spans: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group spans by run (session.id when present, else the trace_id fallback).

    A "run" represents one execution of the user's app. The user can run the app
    multiple times in the same process (e.g. a script that runs 3 separate
    requests) or the file may accumulate runs across process restarts. The
    doctor treats each run independently so cross-run pollution does not look
    like a real bug.
    """
    runs: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        attrs = span.get("attributes") or {}
        # Prefer the user-supplied session.id, fall back to trace_id, fall back
        # to a sentinel so a single trace that lacks both still gets a run key.
        session = attrs.get("session.id")
        if isinstance(session, str) and session:
            run_key: str = session
        else:
            run_key = str(span.get("trace_id") or DEFAULT_SESSION_ID)
        runs.setdefault(run_key, []).append(span)
    return runs


def _group_by_trace(spans: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group spans by trace_id within a single run."""
    traces: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        trace_id = str(span.get("trace_id") or "unknown")
        traces.setdefault(trace_id, []).append(span)
    return traces


def _iter_traces(runs: dict[str, list[dict[str, Any]]]) -> Iterable[tuple[str, str]]:
    """Yield ``(trace_id, run_id)`` for every trace in every run."""
    for rid, run_spans in runs.items():
        seen: set[str] = set()
        for span in run_spans:
            tid = str(span.get("trace_id") or "unknown")
            if tid not in seen:
                seen.add(tid)
                yield tid, rid


# --- Per-trace diagnosis -----------------------------------------------------


def _diagnose_trace(
    trace_id: str, spans: list[dict[str, Any]], *, run_id: str
) -> list[DoctorFinding]:
    """Run all per-trace checks and return the resulting findings.

    Visibility rule: spans with ``neatlogs.internal`` attribute set, or whose
    name is exactly ``neatlogs.trace.complete``, are internal and excluded
    from the trace-level checks. They are still counted in ``spans_read``.
    """
    findings: list[DoctorFinding] = []
    visible = [s for s in spans if not _is_internal(s)]
    if not visible:
        return findings

    roots = [s for s in visible if not s.get("parent_span_id")]
    root_kinds = {_kind(s) for s in roots}

    # --- 0. Build child map (used by hierarchy + agent-LLM checks). ---------
    child_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in visible:
        pid = s.get("parent_span_id")
        if pid:
            child_map[pid].append(s)

    span_ids = {s.get("span_id") for s in visible if s.get("span_id")}
    seen_span_ids: set[str] = set()
    duplicate_span_ids: list[str] = []
    for s in visible:
        sid = s.get("span_id")
        if sid is None:
            continue
        if sid in seen_span_ids:
            duplicate_span_ids.append(sid)
        else:
            seen_span_ids.add(sid)

    # --- Bug #2: rootless HTTP-only trace. ------------------------------------
    if _is_rootless_http_only(visible):
        findings.append(
            DoctorFinding(
                severity="warning",
                code="rootless-http-only",
                title="Trace only contains rootless HTTP spans",
                evidence=f"{len(visible)} HTTP span(s) have no traced parent.",
                suggestion=(
                    "Wrap the request, job, or script entry point in "
                    '@span(kind="WORKFLOW") so HTTP calls belong to an application trace.'
                ),
                trace_id=trace_id,
                run_id=run_id,
            )
        )
        return findings

    # --- Bug #2 / Enh #5: missing orchestration root. -------------------------
    if not root_kinds.intersection(ROOT_KINDS):
        findings.append(
            DoctorFinding(
                severity="warning",
                code="missing-root-kind",
                title="Trace has no workflow, chain, agent, or MCP tool root",
                evidence=f"Root span kinds: {', '.join(sorted(root_kinds)) or 'none'}",
                suggestion=(
                    'Add @span(kind="WORKFLOW") to the entry point, or use a supported '
                    "provider wrapper that creates an automatic root span."
                ),
                trace_id=trace_id,
                run_id=run_id,
            )
        )

    # --- Bug #3 / Enh #5: hierarchy pathologies. -----------------------------
    findings.extend(_orphan_parent_findings(visible, span_ids, trace_id, run_id))
    findings.extend(_self_parent_findings(visible, trace_id, run_id))
    findings.extend(_duplicate_span_id_findings(duplicate_span_ids, visible, trace_id, run_id))
    findings.extend(_multiple_roots_findings(roots, trace_id, run_id))
    findings.extend(_cycle_findings(visible, child_map, trace_id, run_id))

    # --- Bug #2 (refined): agent-without-llm is now subtree-based. -----------
    findings.extend(_agent_without_llm_findings(visible, child_map, trace_id, run_id))

    # --- Bug #1 / I/O check: missing input or output on LLM/tool/retriever. --
    findings.extend(_missing_io_findings(visible, trace_id, run_id))

    # --- Pre-launch reliability: 3 new diagnostic dimensions. ----------------
    # (1) init order: wrappers created before neatlogs.init()
    findings.extend(_init_order_findings(visible, trace_id, run_id))
    # (2) attribute completeness: neatlogs.span.kind set on every span
    findings.extend(_attribute_completeness_findings(visible, trace_id, run_id))
    # (3) data integrity: zero-duration, error-no-event, latency-mismatch
    findings.extend(_data_integrity_findings(visible, trace_id, run_id))

    return findings


# --- Specific finding helpers -----------------------------------------------


def _missing_io_findings(
    spans: list[dict[str, Any]], trace_id: str, run_id: str
) -> list[DoctorFinding]:
    """Per-span check: LLM / tool / retriever spans that lack input or output."""
    missing_by_kind: dict[str, list[str]] = {}
    for span in spans:
        kind = _kind(span)
        if kind not in IO_KINDS:
            continue
        attrs = span.get("attributes") or {}
        if not (_has_input(kind, attrs) and _has_output(kind, attrs)):
            missing_by_kind.setdefault(kind, []).append(str(span.get("name") or "<unnamed>"))

    findings: list[DoctorFinding] = []
    for kind, names in sorted(missing_by_kind.items()):
        shown = ", ".join(names[:3])
        suffix = f" and {len(names) - 3} more" if len(names) > 3 else ""
        findings.append(
            DoctorFinding(
                severity="warning",
                code=f"{kind}-missing-io",
                title=f"{kind.upper()} spans are missing input or output",
                evidence=f"{len(names)} span(s): {shown}{suffix}",
                suggestion=(
                    "Check that the SDK call completed, capture_input/capture_output is enabled, "
                    "and the provider integration supports this operation."
                ),
                trace_id=trace_id,
                run_id=run_id,
                fix_class="capture",
                doc_url="https://docs.neatlogs.com/troubleshooting/missing-io",
                related_codes=("instrumentation-missing",),
            )
        )
    return findings


def _orphan_parent_findings(
    spans: list[dict[str, Any]],
    span_ids: set[Optional[str]],
    trace_id: str,
    run_id: str,
) -> list[DoctorFinding]:
    """Span whose ``parent_span_id`` does not resolve to any visible span."""
    findings: list[DoctorFinding] = []
    for s in spans:
        pid = s.get("parent_span_id")
        if pid and pid not in span_ids:
            findings.append(
                DoctorFinding(
                    severity="warning",
                    code="orphan-parent",
                    title="Span has a parent_span_id that does not exist in this trace",
                    evidence=(
                        f"span '{_truncate(s.get('name') or '<unnamed>')}' "
                        f"has parent_span_id={pid!r} but no span with that id was found"
                    ),
                    suggestion=(
                        "This usually means a wrapper ended a span twice, or two wrappers "
                        "produced overlapping traces. Inspect the call site for the named span."
                    ),
                    trace_id=trace_id,
                    run_id=run_id,
                )
            )
    return findings


def _self_parent_findings(
    spans: list[dict[str, Any]], trace_id: str, run_id: str
) -> list[DoctorFinding]:
    """Span whose ``parent_span_id`` is its own ``span_id``."""
    findings: list[DoctorFinding] = []
    for s in spans:
        sid = s.get("span_id")
        pid = s.get("parent_span_id")
        if sid and pid and sid == pid:
            findings.append(
                DoctorFinding(
                    severity="error",
                    code="self-parent",
                    title="Span has parent_span_id equal to its own span_id",
                    evidence=f"span '{_truncate(s.get('name') or '<unnamed>')}' self-cycles",
                    suggestion=(
                        "This is a serious instrumentation bug — the wrapper is using "
                        "the wrong field or initializing twice. Open an issue on the "
                        "neatlogs repo with this trace_id."
                    ),
                    trace_id=trace_id,
                    run_id=run_id,
                )
            )
    return findings


def _duplicate_span_id_findings(
    duplicate_span_ids: list[str],
    spans: list[dict[str, Any]],
    trace_id: str,
    run_id: str,
) -> list[DoctorFinding]:
    """Two or more spans share the same span_id."""
    if not duplicate_span_ids:
        return []
    return [
        DoctorFinding(
            severity="error",
            code="duplicate-span-id",
            title="Two or more spans share the same span_id",
            evidence=(
                f"span_id(s) appearing more than once: "
                f"{', '.join(sorted(set(duplicate_span_ids))[:5])}"
            ),
            suggestion=(
                "Indicates a duplicate export or a wrapper that emits a new span "
                "without a unique id. The hierarchy check is unreliable for this trace."
            ),
            trace_id=trace_id,
            run_id=run_id,
        )
    ]


def _multiple_roots_findings(
    roots: list[dict[str, Any]], trace_id: str, run_id: str
) -> list[DoctorFinding]:
    """A single trace with more than one root span."""
    if len(roots) <= 1:
        return []
    return [
        DoctorFinding(
            severity="warning",
            code="multiple-roots",
            title="Trace has more than one root span",
            evidence=(
                f"{len(roots)} root spans: "
                f"{', '.join(_truncate(s.get('name') or '<unnamed>') for s in roots[:3])}"
            ),
            suggestion=(
                "Either two entry points ran in parallel, or the trace_id is being "
                "shared across processes. Add a single @span(kind='WORKFLOW') at the "
                "top level, or generate a unique trace_id per execution."
            ),
            trace_id=trace_id,
            run_id=run_id,
        )
    ]


def _cycle_findings(
    spans: list[dict[str, Any]],
    child_map: dict[str, list[dict[str, Any]]],
    trace_id: str,
    run_id: str,
) -> list[DoctorFinding]:
    """Detect cycles in the parent → child tree (excluding self-parent, which
    is reported separately by :func:`_self_parent_findings`).

    Self-parent spans are filtered out of both the spans list and the
    child_map before the DFS starts, so a self-parent span never appears
    as a back-edge in the cycle walk. This avoids the duplicate "self-1
    → self-1" finding the cycle walker would otherwise emit.

    Algorithm: O(V + E) iterative DFS from every unvisited node. We track
    two sets:
    - ``in_path``: nodes currently on the DFS stack (a back-edge to a node
      in this set is a cycle)
    - ``done``: nodes whose entire subtree has been fully explored (a
      back-edge to a node in this set is not a cycle, just a cross-edge)

    For a forest with n nodes and no cycles, the algorithm visits each
    node exactly once = O(n). With cycles, each node is still visited at
    most once, so it stays O(V + E).

    Note: this replaces an earlier "walk-up-from-every-span" loop that
    was O(n²) in the no-cycle case (which is the common case).
    """
    # Filter out self-parent spans (sid == pid). These get their own
    # `self-parent` finding elsewhere; reporting them as cycles too would
    # be a noisy duplicate.
    spans = [s for s in spans if s.get("span_id") != s.get("parent_span_id")]

    findings: list[DoctorFinding] = []
    in_path: set[str] = set()
    done: set[str] = set()
    name_of: dict[str, str] = {
        s["span_id"]: s.get("name") or "<unnamed>" for s in spans if s.get("span_id")
    }

    def _report_cycle(back_to: str, path: list[str]) -> None:
        """Emit a cycle finding. ``path`` is the current DFS path; ``back_to``
        is the node the back-edge points to (must be in ``path``)."""
        if back_to in path:
            idx = path.index(back_to)
            cycle = path[idx:] + [back_to]
        else:
            cycle = path + [back_to]
        findings.append(
            DoctorFinding(
                severity="error",
                code="cycle",
                title="Span hierarchy contains a cycle",
                evidence=(
                    f"span '{_truncate(name_of.get(back_to, '<unnamed>'))}' "
                    f"is in a cycle: {' → '.join(cycle[:6])}"
                    f"{' → ...' if len(cycle) > 6 else ''}"
                ),
                suggestion=(
                    "Wrap a function that re-enters itself with a guard, "
                    "or fix the wrapper that is producing the cycle."
                ),
                trace_id=trace_id,
                run_id=run_id,
            )
        )

    def _walk(start: str) -> None:
        """Iterative DFS from ``start``. Pushes onto ``in_path``; back-edges
        to ``in_path`` are reported as cycles; nodes are popped and added
        to ``done`` on exit.
        """
        if start in done:
            return
        path: list[str] = [start]
        in_path.add(start)
        # Use a stack of "iterators" — each frame is a child index to visit
        # next. When all children are visited, pop the frame.
        # frame = (node_id, list_of_child_ids, next_index)
        children_of: dict[str, list[str]] = {
            parent: [c["span_id"] for c in kids if c.get("span_id")]
            for parent, kids in child_map.items()
        }
        frames: list[tuple[str, list[str], int]] = [(start, children_of.get(start, []), 0)]
        while frames:
            node, kids, i = frames[-1]
            if i >= len(kids):
                # Done with this node's children — pop.
                frames.pop()
                path.pop()
                in_path.discard(node)
                done.add(node)
                continue
            # Advance the index first so the frame state is consistent if
            # we recurse / backtrack below.
            frames[-1] = (node, kids, i + 1)
            cid = kids[i]
            if cid in done:
                continue
            if cid in in_path:
                # Back-edge: a cycle. Report and skip — don't recurse.
                _report_cycle(cid, path)
                continue
            # Recurse: push a new frame for cid.
            path.append(cid)
            in_path.add(cid)
            frames.append((cid, children_of.get(cid, []), 0))

    # Start from every node that hasn't been visited. This is necessary
    # because in a cycle, every node has a parent — there is no "root" to
    # start from. Starting from any unvisited node works because if the
    # node is in a tree (not a cycle), the DFS will reach all descendants
    # in O(V+E). If it's in a cycle, the back-edge detection will fire
    # on the first revisit.
    for s in spans:
        sid = s.get("span_id")
        if not sid:
            continue
        if sid not in done:
            _walk(sid)
    return findings


def _agent_without_llm_findings(
    spans: list[dict[str, Any]],
    child_map: dict[str, list[dict[str, Any]]],
    trace_id: str,
    run_id: str,
) -> list[DoctorFinding]:
    """Bug #2 fix: walk each agent's subtree and check for any LLM descendant.

    The previous global ``"agent" in kinds and "llm" not in kinds`` check
    produced false negatives when one agent span had an LLM and another did
    not. This is the per-subtree version.
    """
    findings: list[DoctorFinding] = []
    for span in spans:
        if _kind(span) != "agent":
            continue
        sid = span.get("span_id")
        if not sid:
            continue
        if not _has_llm_descendant(sid, child_map):
            findings.append(
                DoctorFinding(
                    severity="warning",
                    code="agent-without-llm",
                    title="Agent span ended without any LLM call in its subtree",
                    evidence=(
                        f"agent '{_truncate(span.get('name') or '<unnamed>')}' "
                        f"has no LLM descendant"
                    ),
                    suggestion=(
                        "Check import order: the LLM client must be created AFTER "
                        "neatlogs.init(), or call neatlogs.init(force_reload=True). "
                        "Also verify the LLM library is in the `instrumentations=[...]` list."
                    ),
                    trace_id=trace_id,
                    run_id=run_id,
                )
            )
    return findings


def _foreign_instrumentation_findings(
    spans: list[dict[str, Any]], *, run_id: str
) -> tuple[list[DoctorFinding], bool]:
    """Enhancement #1: group spans by ``instrumentation_scope`` and flag any
    scope that isn't ``neatlogs`` (or a ``neatlogs.*`` sub-scope).

    The scope is read from the top-level ``instrumentation_scope.name`` field
    added by the log exporter (Enhancement #4). When the field is absent
    across the whole run, the report-level caller emits a single
    ``scope-not-preserved`` info finding (we just return ``scopes_seen=False``
    here so the caller can dedupe across multiple runs).

    Returns:
        A 2-tuple ``(findings, scopes_seen)``. ``scopes_seen`` is True iff
        at least one span in this run had an ``instrumentation_scope`` field
        with a ``name``.
    """
    findings: list[DoctorFinding] = []
    scope_counts: Counter[str] = Counter()
    scopes_seen: bool = False
    for s in spans:
        scope_obj = s.get("instrumentation_scope")
        if isinstance(scope_obj, dict) and "name" in scope_obj:
            scopes_seen = True
            name = str(scope_obj["name"])
            scope_counts[name] += 1

    if not scopes_seen:
        # No scope info in this run. Caller emits a single report-level
        # info finding if NO run had it.
        return findings, False

    foreign = {name: n for name, n in scope_counts.items() if not _is_neatlogs_scope(name)}
    if not foreign:
        return findings, True

    parts = [f"{n} spans from '{name}'" for name, n in foreign.items()]
    findings.append(
        DoctorFinding(
            severity="warning",
            code="foreign-instrumentation-detected",
            title="Foreign instrumentation is polluting the neatlogs trace",
            evidence=(
                f"{len(spans)} total spans: {', '.join(parts)} "
                f"(+ {scope_counts.get(_neatlogs_scope_name(), 0)} neatlogs spans)"
            ),
            suggestion=(
                "Either disable the foreign instrumentations in this process, "
                "or set NEATLOGS_FILTER_SCOPE=neatlogs to scope the dashboard filter."
            ),
            run_id=run_id,
        )
    )
    return findings, True


# --- I/O attribute checks (Bug #1 fix) ---------------------------------------


def _has_input(kind: str, attrs: dict[str, Any]) -> bool:
    """True if the span has a meaningful input attribute.

    Bug #1 fix: for LLM spans, role alone is metadata; the doctor must require
    at least one non-empty ``content`` attribute. The same rule applies to
    ``system_prompt``.
    """
    if not isinstance(attrs, dict):
        return False
    if kind == "llm":
        return _llm_has_meaningful_input(attrs)
    if kind == "embedding":
        # Embedding spans carry their text on `neatlogs.embedding.text`.
        v = attrs.get("neatlogs.embedding.text")
        if isinstance(v, str) and v.strip():
            return True
        return False
    if kind == "tool":
        # Tool spans use either `neatlogs.tool.input` or `neatlogs.tool.parameters`.
        for key in ("neatlogs.tool.input", "neatlogs.tool.parameters"):
            v = attrs.get(key)
            if v not in (None, "", [], {}):
                return True
        return False
    if kind == "retriever":
        for key in ("neatlogs.retriever.input", "neatlogs.retriever.query"):
            v = attrs.get(key)
            if isinstance(v, str) and v.strip():
                return True
        return False
    return False


def _has_output(kind: str, attrs: dict[str, Any]) -> bool:
    if not isinstance(attrs, dict):
        return False
    if kind == "llm":
        # An output_messages.N.content of any non-empty string counts.
        for key, value in attrs.items():
            if key.startswith("neatlogs.llm.output_messages.") and key.endswith(".content"):
                if isinstance(value, str) and value.strip():
                    return True
                if value not in (None, "", [], {}):
                    # Non-string truthy values also count (e.g. tool call JSON).
                    return True
        # Or a single output block.
        v = attrs.get("neatlogs.llm.output")
        return v not in (None, "", [], {})
    if kind == "tool":
        v = attrs.get("neatlogs.tool.output")
        return v not in (None, "", [], {})
    if kind == "retriever":
        # Either a single output or one or more documents.
        v = attrs.get("neatlogs.retriever.output")
        if v not in (None, "", [], {}):
            return True
        for key, value in attrs.items():
            if key.startswith("neatlogs.retriever.documents.") and value not in (None, "", [], {}):
                return True
        return False
    if kind == "embedding":
        # The output of an embedding span is the dimensions / count attribute.
        if attrs.get("neatlogs.embedding.dimensions") is not None:
            return True
        if attrs.get("neatlogs.embedding.count") is not None:
            return True
        return False
    return False


def _llm_has_meaningful_input(attrs: dict[str, Any]) -> bool:
    """At least one ``neatlogs.llm.input_messages.N.content`` with a non-empty
    string, OR a non-empty ``neatlogs.llm.system_prompt`` / ``neatlogs.llm.input``.

    Role alone (``...input_messages.N.role = "user"``) does NOT count, per
    Bug #1 — that's metadata, not input.
    """
    for key, value in attrs.items():
        if key.startswith("neatlogs.llm.input_messages.") and key.endswith(".content"):
            if isinstance(value, str) and value.strip():
                return True
            if value not in (None, "", [], {}) and not isinstance(value, bool):
                # Structured content (list / dict) also counts.
                return True
    # Fall back to single-input / system-prompt keys.
    for key in ("neatlogs.llm.input", "neatlogs.llm.system_prompt"):
        v = attrs.get(key)
        if isinstance(v, str) and v.strip():
            return True
        if v not in (None, "", [], {}) and not isinstance(v, bool):
            return True
    return False


# --- Tree helpers ------------------------------------------------------------


def _has_llm_descendant(
    span_id: str,
    child_map: dict[str, list[dict[str, Any]]],
    visited: Optional[set[str]] = None,
) -> bool:
    """True if any descendant of ``span_id`` has kind ``llm``.

    Iterative DFS with a visited set, so a deep tree or a tree with cross-links
    cannot infinite-loop. Cross-references in the child_map are tolerated
    because we only walk down, never back up.
    """
    if visited is None:
        visited = set()
    stack = [span_id]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        for child in child_map.get(cur, []):
            if _kind(child) == "llm":
                return True
            cid = child.get("span_id")
            if cid:
                stack.append(cid)
    return False


# --- Misc helpers ------------------------------------------------------------


def _is_rootless_http_only(spans: list[dict[str, Any]]) -> bool:
    """All visible spans are rootless HTTP — almost certainly auto-instrumented
    requests without a parent workflow. Different from missing-root-kind
    because this is the specific "HTTP without a parent" pattern.
    """
    return bool(spans) and all(_kind(s) == "http" and not s.get("parent_span_id") for s in spans)


def _is_internal(span: dict[str, Any]) -> bool:
    """A span emitted internally by neatlogs (not user-facing)."""
    attrs = span.get("attributes") or {}
    return bool(attrs.get("neatlogs.internal")) or span.get("name") == "neatlogs.trace.complete"


def _kind(span: dict[str, Any]) -> str:
    """Read the kind from either the top-level ``kind`` field or
    ``attributes.neatlogs.span.kind``. Returned lowercase + stripped.
    """
    attrs = span.get("attributes") or {}
    value = span.get("kind") or attrs.get("neatlogs.span.kind") or ""
    return str(value).strip().lower()


def _is_neatlogs_scope(scope: str) -> bool:
    return scope == NEATLOGS_SCOPE_PREFIX or scope.startswith(NEATLOGS_SCOPE_PREFIX + ".")


def _neatlogs_scope_name() -> str:
    return NEATLOGS_SCOPE_PREFIX


def _truncate(s: Any, max_len: int = MAX_EVIDENCE_LEN) -> str:
    """Truncate a string for evidence fields; non-strings are coerced."""
    text = str(s)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


# --- Diagnostic dimensions: init order, attributes, data integrity, stage ---


def _init_order_findings(
    spans: list[dict[str, Any]], trace_id: str, run_id: str
) -> list[DoctorFinding]:
    """Detect wrappers created BEFORE neatlogs.init().

    The OTel SDK is loaded (we get a span out) but our attribute processor
    never ran, so none of the INIT_MARKER_KEYS are set on the span. This
    is the signature of a wrapper that was monkey-patched before init.
    Marked automated_fix_available=True because the fix is mechanical
    (re-order init to the top of the entry point).
    """
    findings: list[DoctorFinding] = []
    for span in spans:
        attrs = span.get("attributes") or {}
        if any(k in attrs for k in INIT_MARKER_KEYS):
            continue
        # Span is present but no init marker. This is the wrapper-before-init
        # signature: the OTel SDK is loaded (we got a span out) but our
        # attribute processor was never applied.
        findings.append(
            DoctorFinding(
                severity="error",
                code="init-after-client",
                title="Span has no Neatlogs init markers — wrapper likely created before neatlogs.init()",
                evidence=(
                    f"span '{_truncate(span.get('name') or '<unnamed>')}' "
                    f"has none of {list(INIT_MARKER_KEYS)}"
                ),
                suggestion=(
                    "Move neatlogs.init() to the very top of your entry point, "
                    "BEFORE constructing any LLM client (openai.Anthropic(), "
                    "ChatOpenAI(), genai.Client(), etc.). If you cannot reorder, "
                    "call neatlogs.init(force_reload=True) after construction."
                ),
                trace_id=trace_id,
                run_id=run_id,
                fix_class="init_order",
                automated_fix_available=True,
                doc_url="https://docs.neatlogs.com/getting-started/init-order",
                related_codes=("no-spans", "missing-root-kind"),
            )
        )
        # Only emit one per trace — the rest of the spans are downstream of
        # the same root cause.
        break
    return findings


def _attribute_completeness_findings(
    spans: list[dict[str, Any]], trace_id: str, run_id: str
) -> list[DoctorFinding]:
    """Find spans that lack ``neatlogs.span.kind``.

    The doctor and dashboard use this attribute to classify spans. A span
    without it shows up as a generic chain in the dashboard.
    """
    findings: list[DoctorFinding] = []
    missing_kind_count = 0
    examples: list[str] = []
    for span in spans:
        attrs = span.get("attributes") or {}
        if "neatlogs.span.kind" not in attrs:
            missing_kind_count += 1
            if len(examples) < 3:
                examples.append(str(span.get("name") or "<unnamed>"))

    if missing_kind_count == len(spans) and len(spans) > 0:
        # All spans missing the kind attribute — this is the init-order symptom
        # in milder form (wrapper is loaded but not fully wired). Skip the
        # finding; init-order check handles it.
        return findings

    if missing_kind_count > 0:
        findings.append(
            DoctorFinding(
                severity="warning",
                code="missing-span-kind",
                title="Some spans lack neatlogs.span.kind — dashboard will mis-categorize them",
                evidence=(
                    f"{missing_kind_count} of {len(spans)} span(s) missing "
                    f"neatlogs.span.kind: {', '.join(examples)}"
                    f"{' ...' if missing_kind_count > 3 else ''}"
                ),
                suggestion=(
                    "Set neatlogs.span.kind on every emitted span. "
                    "@neatlogs.span(kind=...) populates it automatically; "
                    "if you wrap a third-party client, the wrapper should set it."
                ),
                trace_id=trace_id,
                run_id=run_id,
                fix_class="attribute",
                doc_url="https://docs.neatlogs.com/api/span-attributes",
            )
        )
    return findings


def _data_integrity_findings(
    spans: list[dict[str, Any]], trace_id: str, run_id: str
) -> list[DoctorFinding]:
    """Three checks on the captured data:
    a) zero-duration span (start == end) — wrapper crashed before end(),
       or an async wrapper didn't await.
    b) error status without an exception event — no stack trace recorded.
    c) end < start — clock issue.
    """
    findings: list[DoctorFinding] = []
    zero_dur: list[str] = []
    error_no_event: list[str] = []
    latency_mismatch: list[str] = []
    for span in spans:
        name = str(span.get("name") or "<unnamed>")
        start = span.get("start_time")
        end = span.get("end_time")
        duration = span.get("duration_ns")
        status = span.get("status") or {}
        events = span.get("events") or []

        # a) zero duration: duration_ns is 0 OR start == end (in nanoseconds).
        if (isinstance(duration, (int, float)) and duration == 0) or (
            isinstance(start, (int, float)) and isinstance(end, (int, float)) and end == start
        ):
            zero_dur.append(name)

        # c) latency mismatch: end < start
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end < start:
            latency_mismatch.append(name)

        # b) error status without exception event
        if isinstance(status, dict) and status.get("code") in ("ERROR", "error"):
            has_exception = any(
                isinstance(e, dict) and e.get("name") == "exception" for e in events
            )
            if not has_exception:
                error_no_event.append(name)

    if zero_dur:
        findings.append(
            DoctorFinding(
                severity="warning",
                code="zero-duration-span",
                title="Some spans ended instantly (duration_ns == 0)",
                evidence=(
                    f"{len(zero_dur)} span(s) with zero duration: "
                    f"{', '.join(zero_dur[:3])}"
                    f"{' ...' if len(zero_dur) > 3 else ''}"
                ),
                suggestion=(
                    "Wrapper likely crashed before span.end(), or an async wrapper "
                    "did not await. Check the wrapper's exception path and register "
                    "it with @contextlib.asynccontextmanager if the client is async."
                ),
                trace_id=trace_id,
                run_id=run_id,
                fix_class="data_integrity",
                related_codes=("error-status-no-event",),
            )
        )
    if error_no_event:
        findings.append(
            DoctorFinding(
                severity="warning",
                code="error-status-no-event",
                title="Spans marked ERROR but no exception event recorded",
                evidence=(
                    f"{len(error_no_event)} span(s): {', '.join(error_no_event[:3])}"
                    f"{' ...' if len(error_no_event) > 3 else ''}"
                ),
                suggestion=(
                    "Attach an exception event with stack trace when marking a span "
                    "ERROR. Without it, the dashboard's error view shows the span "
                    "as red but offers no detail. Use opentelemetry's "
                    "record_exception() inside the wrapper's except block."
                ),
                trace_id=trace_id,
                run_id=run_id,
                fix_class="data_integrity",
                related_codes=("zero-duration-span",),
            )
        )
    if latency_mismatch:
        findings.append(
            DoctorFinding(
                severity="error",
                code="latency-mismatch",
                title="Span end_time is before start_time",
                evidence=(
                    f"{len(latency_mismatch)} span(s) with end < start: "
                    f"{', '.join(latency_mismatch[:3])}"
                ),
                suggestion=(
                    "Clock issue: the wrapper captured start_time and end_time from "
                    "different clocks. Call time.time_ns() (or perf_counter_ns()) "
                    "once per phase and use that one source for both."
                ),
                trace_id=trace_id,
                run_id=run_id,
                fix_class="data_integrity",
            )
        )
    return findings


def _pipeline_stage_summary(
    findings: list[DoctorFinding],
) -> dict[str, int]:
    """Helper: count findings per pipeline stage for the run-level summary.

    Used by the ``pipeline-stage-summary`` finding (below) and exposed via
    ``DoctorReport.findings_by_pipeline_stage()``.
    """
    stage_map = {
        "init_order": "init",
        "config": "init",
        "pipeline": "init",
        "instrumentation": "instrument",
        "capture": "instrument",
        "data_integrity": "span",
        "attribute": "span",
        "hierarchy": "hierarchy",
    }
    out: dict[str, int] = {"init": 0, "instrument": 0, "span": 0, "hierarchy": 0}
    for f in findings:
        stage = stage_map.get(f.fix_class or "")
        if stage is not None:
            out[stage] += 1
    return out


def _pipeline_stage_run_finding(
    findings: list[DoctorFinding],
) -> Optional[DoctorFinding]:
    """Build the run-level pipeline-stage summary finding, or None.

    Returns None when no findings cluster at a single stage.
    """
    if not findings:
        return None
    counts = _pipeline_stage_summary(findings)
    # Find the dominant stage if any single stage has more findings than
    # all the others combined.
    total = sum(counts.values())
    if total == 0:
        return None
    dominant = max(counts, key=counts.get)  # type: ignore[arg-type]
    if counts[dominant] * 2 <= total:
        # No single stage dominates.
        return None
    stage_suggestion = _STAGE_SUGGESTIONS.get(
        dominant, f"Fix the {dominant} stage first; re-run the doctor."
    )
    return DoctorFinding(
        severity="info",
        code="pipeline-stage-summary",
        title=f"Most findings cluster at the {dominant} stage of the SDK pipeline",
        evidence=(
            f"stage breakdown: init={counts['init']}, "
            f"instrument={counts['instrument']}, span={counts['span']}, "
            f"hierarchy={counts['hierarchy']}"
        ),
        suggestion=stage_suggestion,
        fix_class="pipeline",
        related_codes=tuple(
            f.code for f in findings if f.fix_class in ("init_order", "config", "pipeline")
        ),
    )


# Stage-specific suggestions for the pipeline-stage-summary finding. Keyed
# by the dominant stage so the message is accurate (not a hardcoded "init").
_STAGE_SUGGESTIONS = {
    "init": (
        "Move neatlogs.init() to the top of the entry point (before any "
        "client is constructed), then re-run the doctor — the rest usually "
        "resolves once init is right."
    ),
    "instrument": (
        "Most findings are about wrappers not capturing or being reached. "
        "Verify the LLM client is constructed after neatlogs.init() and that "
        "the wrapper registered for the framework is actually installed."
    ),
    "span": (
        "Most findings are about the captured span data itself. Check the "
        "wrapper's end() and exception-recording paths; a crashed wrapper "
        "leaves spans with zero duration and no events."
    ),
    "hierarchy": (
        "Most findings are about parent/child relationships. Verify that "
        "each wrapper sets parent_span_id correctly; duplicates and orphan "
        "parents usually mean a wrapper is creating spans outside the "
        "active context."
    ),
}


__all__ = [
    "DoctorFinding",
    "DoctorReport",
    "diagnose",
    "format_report",
    "main",
    "ROOT_KINDS",
    "IO_KINDS",
    "NEATLOGS_SCOPE_PREFIX",
    "DEFAULT_SESSION_ID",
]
