"""``neatlogs harbor import <jobs dir>`` and ``neatlogs harbor sync <job dir>``.

Both read finished Harbor job folders and write the same records, with the
same ids, as the live plugin. Running either twice, or after the plugin, does
not create duplicates.

* ``import`` takes Harbor's jobs folder (or one job folder) and sends every job
  in it, oldest first.
* ``sync`` takes one job folder and sends what the plugin could not, for
  example after a network outage. It also moves the comparison baseline to
  this job's runs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .options import Options
from .records import parse_time, plan_job
from .summary import print_summary
from .syncer import JobFiles, Syncer, load_job_dir
from .version import __version__


def _job_dirs(path: Path) -> list[JobFiles]:
    single = load_job_dir(path)
    if single is not None:
        return [single]
    jobs = [
        job for child in sorted(path.iterdir()) if child.is_dir() and (job := load_job_dir(child))
    ]
    floor = parse_time("0001-01-01T00:00:00Z")
    return sorted(jobs, key=lambda job: (parse_time(job.started_at) or floor, job.job_dir.name))


def _options(args: argparse.Namespace) -> Options:
    kwargs = {
        "project_id": args.project_id,
        "host": args.host,
        "endpoint": args.endpoint,
        "form_version_id": args.form_version_id,
        "evaluator_ids": args.evaluator_ids,
        "name": args.name,
        "baseline": args.baseline,
        "redact": args.redact,
    }
    if args.no_traces:
        kwargs["traces"] = False
    # Import and sync run after the job: register and fill runs in one pass.
    kwargs["stream"] = False
    return Options.from_kwargs({k: v for k, v in kwargs.items() if v is not None})


def _send(job: JobFiles, options: Options, *, move_baseline: bool, dry_run: bool) -> bool:
    plan = plan_job(job_id=job.job_id, job_config=job.job_config, job_lock=job.job_lock)
    if dry_run:
        print(f"{job.job_dir}  (job {job.job_id})")
        for dataset in plan.datasets.values():
            print(f"  dataset {dataset.name}: {len(dataset.tasks)} task(s), {dataset.version_key}")
        for key in plan.runs:
            print(
                f"  run {key.display_name} on {key.dataset}: "
                f"{len(plan.run_tasks[key])} task(s) × {plan.run_attempts[key]} attempt(s)"
            )
        return True
    syncer = Syncer.create(
        options=options,
        job_dir=job.job_dir,
        plan=plan,
        move_baseline_tag=move_baseline,
        check_existing_traces=True,
    )
    try:
        syncer.prepare()
        summary = syncer.finish()
    finally:
        syncer.client.close()
    print(f"{job.job_dir}  (job {job.job_id})")
    print_summary(summary)
    return summary.pending_trials == 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neatlogs harbor",
        description="Send Harbor job results to Neatlogs.",
    )
    parser.add_argument("--version", action="version", version=f"neatlogs-harbor {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, target, help_text in (
        ("import", "jobs_dir", "send every finished job in a Harbor jobs folder"),
        ("sync", "job_dir", "send what one job folder still has pending"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument(target, type=Path)
        command.add_argument("--project-id", help="Neatlogs project (env NEATLOGS_PROJECT_ID)")
        command.add_argument("--host", help="Neatlogs app origin (env NEATLOGS_HOST)")
        command.add_argument("--endpoint", help="OTLP ingest URL (env NEATLOGS_ENDPOINT)")
        command.add_argument("--form-version-id", help="grading form version for runs")
        command.add_argument("--evaluator-ids", help="comma-separated evaluator ids for runs")
        command.add_argument("--name", help="run name (default: agent · model)")
        command.add_argument("--baseline", help="run:<id> or tag:<name> to compare with")
        command.add_argument("--redact", help="extra regexes to redact, separated by ;;")
        command.add_argument("--no-traces", action="store_true", help="skip OTLP traces")
        command.add_argument(
            "--dry-run", action="store_true", help="show what would be sent and exit"
        )
        command.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    target: Path = args.jobs_dir if args.command == "import" else args.job_dir
    if not target.is_dir():
        print(f"neatlogs harbor: {target} is not a folder", file=sys.stderr)
        return 2
    jobs = _job_dirs(target) if args.command == "import" else []
    if args.command == "sync":
        job = load_job_dir(target)
        if job is None:
            print(
                f"neatlogs harbor: {target} is not a Harbor job folder "
                "(it needs result.json and config.json)",
                file=sys.stderr,
            )
            return 2
        jobs = [job]
    if not jobs:
        print(f"neatlogs harbor: no Harbor jobs found in {target}", file=sys.stderr)
        return 2
    options = _options(args)
    complete = True
    for job in jobs:
        complete &= _send(job, options, move_baseline=args.command == "sync", dry_run=args.dry_run)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
