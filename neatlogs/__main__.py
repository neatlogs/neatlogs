"""Small, dependency-free Neatlogs command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
import time

from .doctor_v2 import (
    clear_doctor_capture,
    doctor_captured_local_v2,
    doctor_probe_v2,
)
from .version import __version__


class _UsageError(Exception):
    pass


class _DoctorParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _UsageError(message)


def _print_v2(result: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"Neatlogs Doctor: {result['status'].upper()}")
    for check in result.get("checks", []):
        print(f"[{check['status'].upper()}] {check['reason_code']}: {check['message']}")
    diagnostics = next(
        (
            check.get("details")
            for check in result.get("checks", [])
            if isinstance(check.get("details"), dict) and check["details"].get("current_stage")
        ),
        None,
    )
    if diagnostics:
        failed = (
            f"; failed: {diagnostics['failed_stage']}" if diagnostics.get("failed_stage") else ""
        )
        print(
            f"Ingestion: {diagnostics.get('ingestion_state')} at "
            f"{diagnostics['current_stage']}{failed}"
        )
    if result.get("first_failure"):
        print(f"First failure: {result['first_failure']}")


def _standalone_local() -> dict:
    """Exercise an isolated generated pipeline without backend access."""

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    from .core.masking_exporter import MaskingSpanExporter
    from .core.span_processor import NeatlogsSpanProcessor

    class _MemorySink(SpanExporter):
        def export(self, spans):
            return SpanExportResult.SUCCESS

        def shutdown(self):
            return None

    clear_doctor_capture()
    provider = TracerProvider()
    provider.add_span_processor(
        NeatlogsSpanProcessor(emit_completion_markers=False, own_all_spans=True)
    )
    provider.add_span_processor(
        SimpleSpanProcessor(
            MaskingSpanExporter(_MemorySink(), lambda snapshot: snapshot, doctor_capture=True)
        )
    )
    tracer = provider.get_tracer("neatlogs.doctor", __version__)

    def mark(span, span_type: str) -> None:
        span.set_attributes(
            {
                "neatlogs.doctor": True,
                "neatlogs.doctor.version": "v1",
                "service.name": "neatlogs.doctor.v2",
                "telemetry.sdk.language": "python",
                "telemetry.sdk.version": __version__,
                "neatlogs.span.kind": span_type.lower(),
            }
        )

    started = time.monotonic()
    with tracer.start_as_current_span("doctor.probe.root") as root:
        mark(root, "WORKFLOW")
        trace_id = f"{root.get_span_context().trace_id:032x}"
        root.set_attribute("neatlogs.span.kind", "workflow")
        root.set_attribute("input.value", '{"prompt":"generated diagnostic input"}')
        with tracer.start_as_current_span("doctor.probe.agent") as agent:
            mark(agent, "AGENT")
            agent.set_attribute("neatlogs.span.kind", "agent")
            agent.set_attribute("input.value", '{"prompt":"generated diagnostic input"}')
            with tracer.start_as_current_span("doctor.probe.llm") as llm:
                mark(llm, "LLM")
                llm.set_attribute("neatlogs.span.kind", "llm")
                llm.set_attribute(
                    "input.value",
                    '{"messages":[{"role":"user","content":"generated diagnostic input"}]}',
                )
                llm.set_attribute("output.value", '{"text":"generated diagnostic output"}')
                llm.set_attribute("neatlogs.llm.token_count.prompt", 11)
                llm.set_attribute("neatlogs.llm.token_count.completion", 7)
                llm.set_attribute("neatlogs.llm.token_count.total", 18)
            agent.set_attribute("output.value", '{"text":"generated diagnostic output"}')
        with tracer.start_as_current_span("doctor.probe.tool") as tool:
            mark(tool, "TOOL")
            tool.set_attribute("neatlogs.span.kind", "tool")
            tool.set_attribute("neatlogs.tool.name", "diagnostic_tool")
            tool.set_attribute("input.value", '{"value":1}')
            tool.set_attribute("output.value", '{"value":2}')
        root.set_attribute("output.value", '{"result":{"value":2}}')
    flushed = provider.force_flush(timeout_millis=5_000)
    duration_ms = max(0, round((time.monotonic() - started) * 1_000))
    result = doctor_captured_local_v2(
        trace_id,
        flush_outcome="success" if flushed else "timeout",
        flush_duration_ms=duration_ms,
        expected_probe_fixture=True,
    )
    provider.shutdown()
    clear_doctor_capture()
    if result is None:
        raise RuntimeError("isolated Doctor capture was unavailable")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = _DoctorParser(prog="neatlogs")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor_parser = commands.add_parser(
        "doctor", help="run read-only, network-free SDK diagnostics"
    )
    modes = doctor_parser.add_mutually_exclusive_group()
    modes.add_argument("--local", action="store_true")
    modes.add_argument("--probe", action="store_true")
    doctor_parser.add_argument("--endpoint")
    doctor_parser.add_argument("--json", action="store_true")
    try:
        args = parser.parse_args(argv)
    except _UsageError as exc:
        print(f"neatlogs: {exc}", file=sys.stderr)
        print("Usage: neatlogs doctor (--local | --probe) [--json]", file=sys.stderr)
        return 4

    result = doctor_probe_v2(endpoint=args.endpoint) if args.probe else _standalone_local()
    _print_v2(result, args.json)
    if result["status"] == "pass":
        return 0
    if result["status"] == "warn":
        return 1
    return 3 if args.probe else 2


def doctor_main(argv: list[str] | None = None) -> int:
    """Entry point for the dedicated ``neatlogs-doctor`` executable."""

    return main(["doctor", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    sys.exit(main())
