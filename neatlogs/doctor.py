from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT_KINDS = {"workflow", "chain", "agent", "mcp_tool"}
IO_KINDS = {"llm", "tool", "retriever"}


@dataclass(frozen=True)
class DoctorFinding:
    severity: str
    code: str
    title: str
    evidence: str
    suggestion: str
    trace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "severity": self.severity,
            "code": self.code,
            "title": self.title,
            "evidence": self.evidence,
            "suggestion": self.suggestion,
        }
        if self.trace_id:
            data["trace_id"] = self.trace_id
        return data


@dataclass(frozen=True)
class DoctorReport:
    path: str
    spans_read: int
    trace_count: int
    invalid_lines: list[int] = field(default_factory=list)
    findings: list[DoctorFinding] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(f.severity == "error" for f in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "spans_read": self.spans_read,
            "trace_count": self.trace_count,
            "invalid_lines": self.invalid_lines,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def diagnose(path: str | Path) -> DoctorReport:
    path_obj = Path(path)
    findings: list[DoctorFinding] = []
    spans, invalid_lines = _read_spans(path_obj, findings)
    traces = _group_by_trace(spans)

    if invalid_lines:
        severity = "error" if not spans else "warning"
        findings.append(
            DoctorFinding(
                severity=severity,
                code="invalid-jsonl",
                title="Span log contains invalid JSON lines",
                evidence=f"Invalid line numbers: {', '.join(str(i) for i in invalid_lines[:5])}",
                suggestion="Use a processed span log written by NEATLOGS_LOG_SPANS.",
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

    for trace_id, trace_spans in traces.items():
        findings.extend(_diagnose_trace(trace_id, trace_spans))

    return DoctorReport(
        path=str(path_obj),
        spans_read=len(spans),
        trace_count=len(traces),
        invalid_lines=invalid_lines,
        findings=findings,
    )


def format_report(report: DoctorReport) -> str:
    lines = [
        "Trace Doctor",
        f"File: {report.path}",
        f"Spans: {report.spans_read}",
        f"Traces: {report.trace_count}",
    ]

    if not report.findings:
        lines.append("")
        lines.append("No problems found.")
        return "\n".join(lines)

    lines.append("")
    lines.append("Findings:")
    for idx, finding in enumerate(report.findings, start=1):
        trace = f" trace={finding.trace_id}" if finding.trace_id else ""
        lines.append(f"{idx}. [{finding.severity}] {finding.title}{trace}")
        lines.append(f"   Evidence: {finding.evidence}")
        lines.append(f"   Fix: {finding.suggestion}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="neatlogs-doctor",
        description="Diagnose local Neatlogs processed span logs.",
    )
    parser.add_argument("path", help="Path to a processed span JSONL file.")
    parser.add_argument("--json", action="store_true", help="Print a JSON report.")
    args = parser.parse_args(argv)

    report = diagnose(args.path)
    if args.json:
        json.dump(report.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(format_report(report) + "\n")
    return 1 if report.has_errors else 0


def _read_spans(
    path: Path, findings: list[DoctorFinding]
) -> tuple[list[dict[str, Any]], list[int]]:
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
    with path.open("r", encoding="utf-8") as handle:
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
                invalid_lines.append(line_number)
    return spans, invalid_lines


def _group_by_trace(spans: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    traces: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        trace_id = str(span.get("trace_id") or "unknown")
        traces.setdefault(trace_id, []).append(span)
    return traces


def _diagnose_trace(trace_id: str, spans: list[dict[str, Any]]) -> list[DoctorFinding]:
    findings: list[DoctorFinding] = []
    visible_spans = [span for span in spans if not _is_internal(span)]
    if not visible_spans:
        return findings

    kinds = [_kind(span) for span in visible_spans]
    roots = [span for span in visible_spans if not span.get("parent_span_id")]
    root_kinds = {_kind(span) for span in roots}

    if _is_rootless_http_only(visible_spans):
        findings.append(
            DoctorFinding(
                severity="warning",
                code="rootless-http-only",
                title="Trace only contains rootless HTTP spans",
                evidence=f"{len(visible_spans)} HTTP span(s) have no traced parent.",
                suggestion=(
                    "Wrap the request, job, or script entry point in "
                    '@span(kind="WORKFLOW") so HTTP calls belong to an application trace.'
                ),
                trace_id=trace_id,
            )
        )
        return findings

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
            )
        )

    if "agent" in kinds and "llm" not in kinds:
        findings.append(
            DoctorFinding(
                severity="warning",
                code="agent-without-llm",
                title="Agent trace has no LLM child span",
                evidence="An agent span ended, but no LLM span appeared in the same trace.",
                suggestion=(
                    "Check import order and include the model provider key in instrumentations, "
                    'for example ["crewai", "openai"] or ["langchain", "anthropic"].'
                ),
                trace_id=trace_id,
            )
        )

    findings.extend(_missing_io_findings(trace_id, visible_spans))
    return findings


def _missing_io_findings(trace_id: str, spans: list[dict[str, Any]]) -> list[DoctorFinding]:
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
            )
        )
    return findings


def _is_rootless_http_only(spans: list[dict[str, Any]]) -> bool:
    return all(_kind(span) == "http" and not span.get("parent_span_id") for span in spans)


def _is_internal(span: dict[str, Any]) -> bool:
    attrs = span.get("attributes") or {}
    return bool(attrs.get("neatlogs.internal")) or span.get("name") == "neatlogs.trace.complete"


def _kind(span: dict[str, Any]) -> str:
    attrs = span.get("attributes") or {}
    value = span.get("kind") or attrs.get("neatlogs.span.kind") or ""
    return str(value).strip().lower()


def _has_input(kind: str, attrs: dict[str, Any]) -> bool:
    exact, prefixes = {
        "llm": (
            ("neatlogs.llm.input", "neatlogs.llm.system_prompt"),
            ("neatlogs.llm.input_messages.",),
        ),
        "tool": (("neatlogs.tool.input", "neatlogs.tool.parameters"), ()),
        "retriever": (("neatlogs.retriever.input", "neatlogs.retriever.query"), ()),
    }[kind]
    return _has_value(attrs, exact, prefixes)


def _has_output(kind: str, attrs: dict[str, Any]) -> bool:
    exact, prefixes = {
        "llm": (("neatlogs.llm.output",), ("neatlogs.llm.output_messages.",)),
        "tool": (("neatlogs.tool.output",), ()),
        "retriever": (("neatlogs.retriever.output",), ("neatlogs.retriever.documents.",)),
    }[kind]
    return _has_value(attrs, exact, prefixes)


def _has_value(attrs: dict[str, Any], exact: tuple[str, ...], prefixes: tuple[str, ...]) -> bool:
    for key, value in attrs.items():
        if (key in exact or key.startswith(prefixes)) and value not in (None, "", [], {}):
            return True
    return False


__all__ = ["DoctorFinding", "DoctorReport", "diagnose", "format_report", "main"]
