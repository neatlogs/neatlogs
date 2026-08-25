"""Small, dependency-free Neatlogs command line entry point."""

from __future__ import annotations

import argparse
import sys

from .doctor import doctor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="neatlogs")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor_parser = commands.add_parser(
        "doctor", help="run read-only, network-free SDK diagnostics"
    )
    doctor_parser.add_argument("--endpoint")
    doctor_parser.add_argument("--sample-rate", type=float, default=1.0)
    doctor_parser.add_argument("--disable-export", action="store_true", default=None)
    doctor_parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

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
