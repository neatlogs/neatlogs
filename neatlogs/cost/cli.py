"""``neatlogs-cost`` CLI.

Three modes share one CLI:

* default: what-if ranking with compatibility scoring
* ``--breakdown``: per-model cost breakdown
* ``--forecast``: monthly / annual cost projection
* ``pricing`` subcommand: catalog discovery (``list`` and ``show``)
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
    format_pricing_list,
    format_pricing_show,
)
from .pricing import default_chain
from .ranking import evaluate_workload
from .workload import WorkloadConstraints


def _build_pricing_parser() -> argparse.ArgumentParser:
    """Parser for the ``pricing`` subcommand group (``list`` and ``show``).

    Kept separate from the main parser because argparse's subparsers
    don't play well with a top-level ``paths`` positional that uses
    ``nargs="*"``. Pre-parse dispatch (``_dispatch``) picks this parser
    when argv starts with ``pricing``.
    """
    p = argparse.ArgumentParser(
        prog="neatlogs-cost pricing",
        description=(
            "Query the pricing catalog directly (no span log required). "
            "Two subcommands: `list` for an overview of every known model, "
            "`show` for the full breakdown of one model (rates, tiered "
            "rates, capabilities, context window)."
        ),
    )
    sub = p.add_subparsers(dest="pricing_action", required=True)

    list_p = sub.add_parser(
        "list",
        help="List every model in the pricing catalog.",
        description="List every model in the pricing catalog with provider, "
        "context window, and capabilities.",
    )
    list_p.add_argument(
        "--format",
        choices=("text", "json", "csv"),
        default="text",
        help="Output format. Default: text.",
    )
    list_p.add_argument(
        "--pricing-file",
        default=None,
        help="Path to a JSON pricing catalog override. Default: the bundled catalog.",
    )
    list_p.add_argument("--no-color", action="store_true")
    list_p.add_argument("--color", action="store_true")

    show_p = sub.add_parser(
        "show",
        help="Show full pricing breakdown for one model.",
        description="Show the full pricing breakdown (rates, tiered rates, "
        "capabilities, context window) for one model.",
    )
    show_p.add_argument(
        "model_key",
        help="Provider/model key, e.g. openai/gpt-4o-mini or anthropic/claude-3-5-sonnet-latest.",
    )
    show_p.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format. Default: text.",
    )
    show_p.add_argument(
        "--pricing-file",
        default=None,
        help="Path to a JSON pricing catalog override. Default: the bundled catalog.",
    )
    show_p.add_argument("--no-color", action="store_true")
    show_p.add_argument("--color", action="store_true")

    return p


def _build_main_parser() -> argparse.ArgumentParser:
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


def _apply_color_flags(args) -> None:
    global _USE_COLOR
    if getattr(args, "no_color", False):
        _USE_COLOR = False
    elif getattr(args, "color", False):
        _USE_COLOR = True


def _handle_pricing_list(args) -> int:
    pricing = default_chain(args.pricing_file)
    models = pricing.catalog()
    sys.stdout.write(format_pricing_list(models, style=args.format, stream=sys.stdout))
    return 0


def _handle_pricing_show(args) -> int:
    pricing = default_chain(args.pricing_file)
    model = pricing.lookup(args.model_key)
    if model is None:
        print(
            f"[neatlogs-cost] unknown model: {args.model_key!r}. "
            f"Use `neatlogs-cost pricing list` to see available models, or "
            f"override with --pricing-file.",
            file=sys.stderr,
        )
        return 2
    sys.stdout.write(format_pricing_show(model, style=args.format, stream=sys.stdout))
    return 0


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    if argv and argv[0] == "pricing":
        # Strip the "pricing" prefix; the pricing parser expects the
        # subcommand (list | show) as the first positional.
        parser = _build_pricing_parser()
        args = parser.parse_args(argv[1:])
        _apply_color_flags(args)
        if args.pricing_action == "list":
            return _handle_pricing_list(args)
        if args.pricing_action == "show":
            return _handle_pricing_show(args)
        return 2

    parser = _build_main_parser()
    args = parser.parse_args(argv)
    _apply_color_flags(args)

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
