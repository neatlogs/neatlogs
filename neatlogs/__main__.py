"""Small, dependency-free Neatlogs command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
import time

from .doctor import doctor
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
        SimpleSpanProcessor(MaskingSpanExporter(_MemorySink(), lambda snapshot: snapshot))
    )
    tracer = provider.get_tracer("neatlogs.doctor", __version__)
    started = time.monotonic()
    with tracer.start_as_current_span("doctor.workflow") as root:
        trace_id = f"{root.get_span_context().trace_id:032x}"
        root.set_attribute("neatlogs.span.kind", "WORKFLOW")
        root.set_attribute("input.value", '{"prompt":"generated diagnostic input"}')
        root.set_attribute("output.value", '{"result":"generated diagnostic output"}')
        root.set_attribute("neatlogs.llm.tool_calls.0.id", "doctor_call_1")
        root.set_attribute("neatlogs.llm.tool_calls.0.name", "diagnostic_tool")
        with tracer.start_as_current_span("doctor.tool") as tool:
            tool.set_attribute("neatlogs.span.kind", "TOOL")
            tool.set_attribute("neatlogs.tool.call_id", "doctor_call_1")
            tool.set_attribute("neatlogs.tool.name", "diagnostic_tool")
            tool.set_attribute("input.value", '{"value":1}')
            tool.set_attribute("output.value", '{"value":2}')
    flushed = provider.force_flush(timeout_millis=5_000)
    duration_ms = max(0, round((time.monotonic() - started) * 1_000))
    result = doctor_captured_local_v2(
        trace_id,
        flush_outcome="success" if flushed else "timeout",
        flush_duration_ms=duration_ms,
    )
    provider.shutdown()
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
    doctor_parser.add_argument("--sample-rate", type=float, default=1.0)
    doctor_parser.add_argument("--disable-export", action="store_true", default=None)
    doctor_parser.add_argument("--json", action="store_true")
    try:
        args = parser.parse_args(argv)
    except _UsageError as exc:
        print(f"neatlogs: {exc}", file=sys.stderr)
        print("Usage: neatlogs doctor (--local | --probe) [--json]", file=sys.stderr)
        return 4

    if args.local or args.probe:
        if args.sample_rate != 1.0 or args.disable_export:
            print(
                "neatlogs: --sample-rate and --disable-export belong to legacy Doctor mode",
                file=sys.stderr,
            )
            return 4
        result = doctor_probe_v2(endpoint=args.endpoint) if args.probe else _standalone_local()
        _print_v2(result, args.json)
        if result["status"] == "pass":
            return 0
        if result["status"] == "warn":
            return 1
        return 3 if args.probe else 2

    result = doctor(
        endpoint=args.endpoint,
        sample_rate=args.sample_rate,
        disable_export=args.disable_export,
    )
    if args.json:
        print(result.to_json())
    else:
        print(f"Neatlogs doctor: {'PASS' if result.ready else 'FAIL'}")
        for check in result.checks:
            print(f"[{check.status.upper()}] {check.reason_code}: {check.message}")
    return 0 if result.ready else 1


def doctor_main(argv: list[str] | None = None) -> int:
    """Entry point for the dedicated ``neatlogs-doctor`` executable."""

    return main(["doctor", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    sys.exit(main())
