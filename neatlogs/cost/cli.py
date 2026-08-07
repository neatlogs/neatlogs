"""``neatlogs-cost`` CLI.

Three modes share one CLI:

* default: what-if ranking with compatibility scoring
* ``--breakdown``: per-model cost breakdown
* ``--forecast``: monthly / annual cost projection
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .breakdown import breakdown_workload
from .forecast import forecast
from .formatters import (
    _USE_COLOR,
    format_breakdown,
    format_evaluation,
    format_forecast,
)
from .pricing import default_chain
from .ranking import evaluate_workload
from .workload import WorkloadConstraints


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neatlogs-cost",
        description=(
            "LLM cost intelligence engine: per-model breakdown, what-if "
            "model ranking, and monthly cost forecasting. Given a span log "
            "and a list of `provider/model` keys, find the cheapest model "
            "that can still serve the same workload."
        ),
    )
    p.add_argument(
        "paths",
        nargs="*",
        help="One or more span log file paths. Ignored in --forecast mode.",
    )
    p.add_argument(
        "--candidates",
        default=None,
        help=(
            "Comma-separated list of `provider/model` keys to evaluate. "
            "In ranking / breakdown mode, the first one is the baseline; "
            "use --baseline to override. In --forecast mode, defaults to "
            "--model if --candidates is not given."
        ),
    )
    p.add_argument(
        "--baseline",
        default=None,
        help="Explicit baseline model key. Default: first --candidates entry.",
    )
    p.add_argument(
        "--breakdown",
        action="store_true",
        help=(
            "Show per-model cost breakdown (input / output / cache / "
            "reasoning) instead of the what-if ranking."
        ),
    )
    p.add_argument(
        "--need",
        action="append",
        default=[],
        help=(
            "Require a capability (repeatable). Values: vision, tools, "
            "json_mode, prompt_cache, reasoning, audio, image_input, "
            "embedding. By default the engine infers capability needs "
            "from the source models in the log; --need adds to the set."
        ),
    )
    p.add_argument(
        "--min-context",
        type=int,
        default=0,
        help=(
            "Minimum context window in tokens. Models with a smaller "
            "context_window (or no context_window declared) are marked "
            "incompatible for any span that exceeds the constraint. "
            "Default: 0 (no constraint)."
        ),
    )
    p.add_argument(
        "--min-compatibility",
        type=float,
        default=0.95,
        help=(
            "Minimum fraction of spans a candidate must be able to serve "
            "(0.0–1.0). Default: 0.95."
        ),
    )
    p.add_argument(
        "--format",
        choices=("text", "json", "csv"),
        default="text",
        help="Output format. Default: text.",
    )
    p.add_argument(
        "--pricing-file",
        default=None,
        help="Path to a JSON pricing catalog. Default: the bundled catalog.",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output even when stdout is a TTY.",
    )
    p.add_argument(
        "--color",
        action="store_true",
        help="Force ANSI color output even when stdout is not a TTY.",
    )
    p.add_argument(
        "--no-auto-infer",
        action="store_true",
        help=(
            "Do NOT infer capability needs from the source models. "
            "Use only --need. Use this when the source model declares "
            "capabilities the workload does not actually exercise."
        ),
    )

    fc = p.add_argument_group("forecast options (use with --forecast)")
    fc.add_argument(
        "--forecast",
        action="store_true",
        help=(
            "Switch to forecast mode: project monthly / annual cost for a "
            "given traffic pattern. Required: --monthly-calls. Useful: "
            "--avg-prompt, --avg-completion, --cache-hit-rate, "
            "--reasoning-per-call."
        ),
    )
    fc.add_argument(
        "--model",
        default=None,
        help=(
            "Model key for --forecast mode. Defaults to the first "
            "--candidates entry if --candidates is given."
        ),
    )
    fc.add_argument(
        "--monthly-calls",
        type=int,
        default=0,
        help="Expected number of calls per month.",
    )
    fc.add_argument(
        "--avg-prompt",
        type=int,
        default=0,
        help="Average prompt tokens per call.",
    )
    fc.add_argument(
        "--avg-completion",
        type=int,
        default=0,
        help="Average completion tokens per call.",
    )
    fc.add_argument(
        "--cache-hit-rate",
        type=float,
        default=0.0,
        help="Fraction of prompt tokens served from cache (0.0–1.0).",
    )
    fc.add_argument(
        "--reasoning-per-call",
        type=int,
        default=0,
        help="Average reasoning tokens per call (o-series, R1, etc.).",
    )
    return p


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    global _USE_COLOR
    if args.no_color:
        _USE_COLOR = False
    elif args.color:
        _USE_COLOR = True
    pricing = default_chain(args.pricing_file)

    if args.forecast:
        model_key = args.model
        if model_key is None and args.candidates:
            model_key = args.candidates.split(",")[0].strip()
        if model_key is None:
            print(
                "[neatlogs-cost] --forecast requires --model or --candidates",
                file=sys.stderr,
            )
            return 2
        if args.monthly_calls <= 0:
            print(
                "[neatlogs-cost] --forecast requires --monthly-calls > 0",
                file=sys.stderr,
            )
            return 2
        try:
            fc_report = forecast(
                model_key=model_key,
                monthly_calls=args.monthly_calls,
                avg_prompt_tokens=args.avg_prompt,
                avg_completion_tokens=args.avg_completion,
                cache_hit_rate=args.cache_hit_rate,
                reasoning_per_call=args.reasoning_per_call,
                pricing=pricing,
            )
        except ValueError as e:
            print(f"[neatlogs-cost] {e}", file=sys.stderr)
            return 2
        sys.stdout.write(format_forecast(fc_report, style=args.format))
        return 0

    if not args.candidates:
        print(
            "[neatlogs-cost] --candidates is required in ranking / breakdown mode",
            file=sys.stderr,
        )
        return 2
    if not args.paths:
        print(
            "[neatlogs-cost] at least one path is required in ranking / breakdown mode",
            file=sys.stderr,
        )
        return 2

    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    constraints = WorkloadConstraints(
        need_capabilities=set(args.need or []),
        min_context_window=args.min_context,
        min_compatibility_pct=args.min_compatibility,
    )

    if args.breakdown:
        bd_report = breakdown_workload(
            args.paths,
            candidates=candidates,
            pricing=pricing,
        )
        if not bd_report.models:
            print(
                "[neatlogs-cost] no candidate models found in the pricing catalog",
                file=sys.stderr,
            )
            return 2
        sys.stdout.write(format_breakdown(bd_report, style=args.format))
        return 0

    report = evaluate_workload(
        args.paths,
        candidates=candidates,
        baseline=args.baseline,
        pricing=pricing,
        constraints=constraints,
        auto_infer_capabilities=not args.no_auto_infer,
    )
    if not report.alternatives and report.baseline is None:
        print(
            "[neatlogs-cost] no candidate models found in the pricing catalog",
            file=sys.stderr,
        )
        return 2
    sys.stdout.write(format_evaluation(report, style=args.format))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
