from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from ..init import init
from .sync import PromptSyncResult, discover_templates, push_templates


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="neatlogs-prompts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    push_parser = subparsers.add_parser("push", help="Push code-defined prompt templates")
    push_parser.add_argument("module_path", help="Python file containing prompt templates")
    push_parser.add_argument("--label", required=True, help="Managed prompt label to apply")
    push_parser.add_argument("--prefix", help="Optional prefix for generated prompt names")
    push_parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    push_parser.add_argument("--commit-message", help="Commit message for created prompt versions")

    args = parser.parse_args(argv)
    if args.command == "push":
        return _push(args)

    parser.print_help()
    return 1


def _push(args: argparse.Namespace) -> int:
    module_path = Path(args.module_path).expanduser()
    if args.dry_run:
        prompts = discover_templates(module_path, prefix=args.prefix)
        results = [
            PromptSyncResult(prompt.name, "discovered", prompt.type, f"from {prompt.attr_name}")
            for prompt in prompts
        ]
        _print_results(results)
        return 0

    api_key = os.getenv("NEATLOGS_API_KEY", "").strip()
    if not api_key:
        print(
            "NEATLOGS_API_KEY is required for prompt sync. Use --dry-run to preview.",
            file=sys.stderr,
        )
        return 2

    init(
        api_key=api_key,
        endpoint=os.getenv("NEATLOGS_ENDPOINT", "https://ingest.neatlogs.com"),
        disable_export=True,
    )

    results = push_templates(
        module_path,
        label=args.label,
        prefix=args.prefix,
        dry_run=False,
        commit_message=args.commit_message,
    )
    _print_results(results)
    return 1 if any(result.action == "failed" for result in results) else 0


def _print_results(results: Sequence[PromptSyncResult]) -> None:
    if not results:
        print("No prompt templates found.")
        return

    print(f"{'ACTION':<12} {'TYPE':<6} NAME")
    print(f"{'-' * 12} {'-' * 6} {'-' * 40}")
    for result in results:
        detail = f" ({result.detail})" if result.detail else ""
        print(f"{result.action:<12} {result.type:<6} {result.name}{detail}")


if __name__ == "__main__":
    raise SystemExit(main())
