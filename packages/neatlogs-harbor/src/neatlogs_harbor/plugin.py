"""Harbor job plugin: ``harbor run ... --plugin neatlogs``.

The plugin listens to three Harbor events and never fails the Harbor run:

* ``on_job_start``: sync the job's tasks into a Neatlogs dataset version and
  register one run per (agent, model, dataset);
* each trial end: send the trial's trace and its output (rewards, tokens,
  cost, timings, error class) as soon as it finishes;
* ``on_job_end``: complete the runs, compare each with the previous run on the
  same dataset version, and print a short summary with a link.

Anything that cannot be sent is recorded in ``jobs/<job>/neatlogs-sync.json``
and can be sent later with ``neatlogs harbor sync jobs/<job>``.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harbor.models.job.plugin import BaseJobPlugin
from harbor.trial.hooks import TrialHookEvent

from .options import Options
from .records import JobPlan, TrialRecord, load_trial_dir, parse_time, plan_job, trial_record
from .summary import print_summary
from .syncer import Syncer

if TYPE_CHECKING:
    from harbor.job import Job
    from harbor.models.job.result import JobResult

logger = logging.getLogger("neatlogs_harbor")


class NeatlogsPlugin(BaseJobPlugin):
    """Streams a Harbor job to Neatlogs. Options mirror ``--pk key=value``."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.options = Options.from_kwargs(kwargs)
        self._syncer: Syncer | None = None
        self._plan: JobPlan | None = None
        self._job_dir: Path | None = None
        # Start times of every trial seen so far, for attempt numbering.
        self._started: dict[str, TrialRecord] = {}
        self._lock = asyncio.Lock()
        self._retry: Any = None

    # ------------------------------------------------------------- job start

    async def on_job_start(self, job: Job) -> None:
        try:
            await asyncio.to_thread(self._start, job)
        except Exception as exc:  # never fail the Harbor run
            logger.warning("Neatlogs: could not start syncing this job: %s", exc)
            self._syncer = None
            return
        job.on_trial_started(self._on_trial_started)
        job.on_trial_ended(self._on_trial_ended)

    def _start(self, job: Job) -> None:
        self._job_dir = Path(job.job_dir)
        self._retry = getattr(job.config, "retry", None)
        if not self.options.api_key:
            logger.warning(
                "Neatlogs: NEATLOGS_API_KEY is not set; results will wait in %s",
                self._job_dir / "neatlogs-sync.json",
            )
        job_config = job.config.model_dump(mode="json")
        job_lock = _build_job_lock(job)
        self._plan = plan_job(job_id=str(job.id), job_config=job_config, job_lock=job_lock)
        self._syncer = Syncer.create(options=self.options, job_dir=self._job_dir, plan=self._plan)
        # A resumed job already finished some trials in an earlier process.
        for trial in self._syncer.finished_trials_on_disk():
            self._started[trial.trial_name] = trial
        self._syncer.prepare()

    # ------------------------------------------------------------- trials

    async def _on_trial_started(self, event: TrialHookEvent) -> None:
        try:
            record = self._record_from_event(event)
            async with self._lock:
                self._started[record.trial_name] = record
        except Exception as exc:
            logger.debug("Neatlogs: ignoring trial start event: %s", exc)

    async def _on_trial_ended(self, event: TrialHookEvent) -> None:
        if self._syncer is None or self._plan is None:
            return
        try:
            # Harbor writes result.json just before this event; reading the file
            # makes the plugin send exactly what the importer would.
            record = self._record_from_disk(event) or self._record_from_event(event)
            if self._may_be_retried(record):
                # Harbor reruns it under the same trial name; send the final
                # attempt from the job folder at job end instead.
                self._syncer.defer_trial(record, "Harbor may retry this trial")
                return
            async with self._lock:
                self._started[record.trial_name] = record
                attempt = self._attempt_for(record)
            await asyncio.to_thread(self._syncer.send_trial, record, attempt)
        except Exception as exc:  # never fail the Harbor run
            logger.warning("Neatlogs: could not send trial %s: %s", event.trial_name, exc)

    def _may_be_retried(self, record: TrialRecord) -> bool:
        """Mirror of Harbor's retry decision for a trial that ended with an exception."""

        retry = self._retry
        exception = record.result.get("exception_info") or {}
        if retry is None or not getattr(retry, "max_retries", 0) or not exception:
            return False
        name = str(exception.get("exception_type"))
        if retry.exclude_exceptions and name in retry.exclude_exceptions:
            return False
        if retry.include_exceptions and name not in retry.include_exceptions:
            return False
        return True

    def _record_from_disk(self, event: TrialHookEvent) -> TrialRecord | None:
        if self._job_dir is None:
            return None
        record = load_trial_dir(self._job_dir / event.config.trial_name, str(event.config.job_id))
        if record is None or record.result.get("id") != str(event.result.id):
            return None
        return record

    def _record_from_event(self, event: TrialHookEvent) -> TrialRecord:
        result = event.result.model_dump(mode="json")
        lock = event.lock.model_dump(mode="json")
        job_dir = self._job_dir or Path(event.config.trials_dir)
        return trial_record(
            job_id=str(event.config.job_id),
            result=result,
            lock=lock,
            trial_dir=job_dir / event.config.trial_name,
        )

    def _attempt_for(self, record: TrialRecord) -> int:
        """Rank by start time among attempts of the same task in the same run.

        Every attempt that started before this one has already fired its start
        event, so the rank is final when the trial ends.
        """

        assert self._plan is not None
        key = (record.run_key(self._plan), record.task_name)
        floor = parse_time("0001-01-01T00:00:00Z")
        mine = (record.started_at or floor, record.trial_name)
        earlier = 0
        for other in self._started.values():
            if other.trial_name == record.trial_name:
                continue
            if (other.run_key(self._plan), other.task_name) != key:
                continue
            if (other.started_at or floor, other.trial_name) < mine:
                earlier += 1
        return earlier + 1

    # ------------------------------------------------------------- job end

    async def on_job_end(self, job_result: JobResult) -> None:
        if self._syncer is None:
            return
        try:
            summary = await asyncio.to_thread(self._syncer.finish)
            print_summary(summary)
        except Exception as exc:  # never fail the Harbor run
            logger.warning("Neatlogs: could not finish the job: %s", exc)


def _build_job_lock(job: Job) -> dict[str, Any]:
    """The job's ``lock.json`` before Harbor writes it.

    Harbor writes ``lock.json`` when the job starts running, after plugins
    start. Building it here with Harbor's own function gives the same task
    digests the importer later reads from disk. A resumed job already has it.
    """

    lock_path = Path(job.job_dir) / "lock.json"
    if lock_path.exists():
        import json

        return json.loads(lock_path.read_text())

    from harbor.models.job.lock import build_job_lock

    lock = build_job_lock(
        config=job.config,
        trial_configs=job._trial_configs,
        task_download_results=job._task_download_results,
        source_trial_dirs=getattr(job, "_hub_source_trial_dirs", None),
    )
    return lock.model_dump(mode="json")
