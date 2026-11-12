"""Sends one Harbor job to Neatlogs. Shared by the plugin and the CLI.

Order of work for a job:

1. ``prepare``: publish each dataset version (one per distinct task-digest
   set) and, when streaming, register one run per (agent, model, dataset).
2. ``send_trial`` for each finished trial: the trace over OTLP, then the run
   output with rewards, tokens, cost, timings and the error class.
3. ``finish``: send anything still pending from the job folder, complete the
   runs, compare each with the previous Harbor run on the same dataset
   version, and move the baseline tag forward.

No method raises for a Neatlogs failure: what could not be sent is marked
pending in ``neatlogs-sync.json`` for ``neatlogs harbor sync``.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import ids
from .client import NeatlogsApiError, NeatlogsClient, idempotency_key
from .ledger import PENDING, SENT, SKIPPED, Ledger
from .options import Options
from .records import (
    DatasetPlan,
    JobPlan,
    RunKey,
    TrialRecord,
    dataset_body,
    load_trial_dir,
    outcome_of,
    output_body,
    parse_time,
    read_instruction,
    read_json,
    read_trajectory,
    verifier_attachments,
)
from .redact import Redactor
from .summary import RunSummary, Summary
from .traces import build_trial_trace

logger = logging.getLogger("neatlogs_harbor")

BASELINE_TAG = "harbor-latest"
RUN_DEADLINE_SECONDS = 3600
MAX_RUN_CASES = 100
MAX_RUN_TRIALS = 10
_ALREADY_FILLED = frozenset({"OUTPUT_CONFLICT", "IDEMPOTENCY_KEY_CONFLICT"})


@dataclass
class JobFiles:
    job_dir: Path
    job_id: str
    job_config: dict[str, Any]
    job_lock: dict[str, Any]
    started_at: str | None = None


def load_job_dir(job_dir: Path) -> JobFiles | None:
    """A Harbor job folder: ``result.json`` (job id), ``config.json`` and ``lock.json``."""

    result = read_json(job_dir / "result.json")
    config = read_json(job_dir / "config.json")
    lock = read_json(job_dir / "lock.json")
    if not isinstance(result, dict) or not result.get("id") or not isinstance(config, dict):
        return None
    if not isinstance(lock, dict):
        # Older jobs without a job-level lock: rebuild the trial list from trial folders.
        trials = [read_json(d / "lock.json") for d in sorted(job_dir.iterdir()) if d.is_dir()]
        lock = {"trials": [t for t in trials if isinstance(t, dict)]}
    return JobFiles(
        job_dir=job_dir,
        job_id=str(result["id"]),
        job_config=config,
        job_lock=lock,
        started_at=result.get("started_at"),
    )


@dataclass
class _RunHandle:
    key: RunKey
    uuid: str
    lock: threading.Lock = field(default_factory=threading.Lock)


class Syncer:
    def __init__(
        self,
        *,
        options: Options,
        job_dir: Path,
        plan: JobPlan,
        client: NeatlogsClient,
        redactor: Redactor,
        ledger: Ledger,
        move_baseline_tag: bool = True,
        check_existing_traces: bool = False,
    ) -> None:
        self.options = options
        self.job_dir = job_dir
        self.plan = plan
        self.client = client
        self.redactor = redactor
        self.ledger = ledger
        self.move_baseline_tag = move_baseline_tag
        # Import and sync may run where the plugin's ledger is missing: ask the
        # API before re-sending a trace. The live plugin knows it has not sent it.
        self.check_existing_traces = check_existing_traces
        self._runs = {key: _RunHandle(key, uuid) for key, uuid in plan.runs.items()}
        self._dataset_lock = threading.Lock()
        self._instructions: dict[tuple[str, str], str | None] = {}
        self._project_checked = False

    @classmethod
    def create(
        cls,
        *,
        options: Options,
        job_dir: Path,
        plan: JobPlan,
        client: NeatlogsClient | None = None,
        move_baseline_tag: bool = True,
        check_existing_traces: bool = False,
    ) -> Syncer:
        client = client or NeatlogsClient(
            endpoint=options.host,
            otlp_endpoint=options.otlp_url,
            token=options.token,
            project_id=options.project_id,
            ingest_key=options.api_key,
            timeout=options.timeout,
            retries=options.retries,
        )
        redactor = Redactor.from_environment(
            options.redact, extra_secrets=[s for s in (options.api_key, options.token) if s]
        )
        ledger = Ledger.for_job_dir(job_dir, plan.job_id)
        return cls(
            options=options,
            job_dir=job_dir,
            plan=plan,
            client=client,
            redactor=redactor,
            ledger=ledger,
            move_baseline_tag=move_baseline_tag,
            check_existing_traces=check_existing_traces,
        )

    # ================================================================ prepare

    def prepare(self) -> None:
        """Publish dataset versions and, when streaming, register the runs."""

        if not self._public_api_ready():
            return
        for dataset in self.plan.datasets.values():
            self._ensure_dataset(dataset)
        if self.options.stream:
            for handle in self._runs.values():
                self._ensure_run(handle)

    def _public_api_ready(self) -> bool:
        if not self.client.can_use_public_api:
            return False
        if not self._project_checked:
            self._project_checked = True
            try:
                self.client.resolve_project_id()
            except NeatlogsApiError as exc:
                self.ledger.record_error("project", exc)
        if not self.client.project_id:
            self.ledger.record_error(
                "project", "Set NEATLOGS_PROJECT_ID: the token can see more than one project"
            )
            return False
        return True

    def _instruction(self, dataset: DatasetPlan, task_name: str) -> str | None:
        key = (dataset.name, task_name)
        if key not in self._instructions:
            text = read_instruction(dataset.tasks[task_name])
            self._instructions[key] = self.redactor.text(text) if text is not None else None
        return self._instructions[key]

    def _ensure_dataset(self, dataset: DatasetPlan) -> dict[str, Any] | None:
        with self._dataset_lock:
            state = self.ledger.dataset(dataset.name)
            if state.get("version_key") == dataset.version_key and state.get("cases"):
                return state
            try:
                state = self._publish_dataset(dataset)
            except NeatlogsApiError as exc:
                self.ledger.update_dataset(
                    dataset.name, version_key=dataset.version_key, status=PENDING, error=str(exc)
                )
                logger.warning("Neatlogs: dataset %s not synced: %s", dataset.name, exc)
                return None
            self.ledger.update_dataset(dataset.name, **state, status=SENT, error=None)
            return state

    def _publish_dataset(self, dataset: DatasetPlan) -> dict[str, Any]:
        instructions = {name: self._instruction(dataset, name) for name in dataset.tasks}
        body = dataset_body(dataset, instructions)
        try:
            created = self.client.create_dataset(
                body, idempotency_key("dataset", dataset.name, dataset.version_key)
            )
        except NeatlogsApiError as exc:
            # Same version published from another machine with different
            # instruction text (for example the task folder is gone here).
            if not exc.is_conflict:
                raise
            created = self._find_dataset(dataset)
            if created is None:
                raise
        dataset_id, version_id = str(created["datasetId"]), str(created["versionId"])
        cases = self._case_ids(dataset_id, version_id, sorted(dataset.tasks))
        return {
            "dataset_id": dataset_id,
            "version_id": version_id,
            "version_key": dataset.version_key,
            "cases": cases,
        }

    def _find_dataset(self, dataset: DatasetPlan) -> dict[str, Any] | None:
        for item in self.client.list_datasets():
            if item.get("name") == dataset.name[:255] and dataset.version_key in str(
                item.get("description") or ""
            ):
                return {"datasetId": item["id"], "versionId": item["latestVersionId"]}
        return None

    def _case_ids(self, dataset_id: str, version_id: str, task_names: list[str]) -> dict[str, str]:
        cases = sorted(self.client.list_cases(dataset_id, version_id), key=lambda c: c["position"])
        if len(cases) != len(task_names):
            raise NeatlogsApiError(
                409,
                "DATASET_MISMATCH",
                f"dataset {dataset_id} has {len(cases)} cases, expected {len(task_names)}",
            )
        # Cases were published in task-name order, so positions map back to tasks.
        return {name: str(case["caseId"]) for name, case in zip(task_names, cases)}

    # ================================================================ runs

    def _run_label(self, key: RunKey) -> str:
        if not self.options.name:
            return key.display_name[:120]
        if len(self._runs) == 1:
            return str(self.options.name)[:120]
        return f"{self.options.name} · {key.display_name}"[:120]

    def _ensure_run(self, handle: _RunHandle) -> str | None:
        """The server id of the run, registering it on first use."""

        with handle.lock:
            state = self.ledger.run(handle.uuid)
            if state.get("server_id"):
                return str(state["server_id"])
            blocker = self._run_blocker(handle.key)
            if blocker:
                status, reason = blocker
                self.ledger.update_run(
                    handle.uuid,
                    name=self._run_label(handle.key),
                    dataset=handle.key.dataset,
                    status=status,
                    reason=reason,
                )
                return None
            dataset = self._ensure_dataset(self.plan.datasets[handle.key.dataset])
            if dataset is None:
                return None
            tasks = self.plan.run_tasks[handle.key]
            body = {
                "datasetVersionId": dataset["version_id"],
                "label": self._run_label(handle.key),
                "caseIds": [dataset["cases"][task] for task in tasks],
                "trials": min(self.plan.run_attempts[handle.key], MAX_RUN_TRIALS),
                "grading": {
                    "formVersionId": self.options.form_version_id,
                    "evaluatorIds": list(self.options.evaluator_ids),
                },
                "deadlineSeconds": RUN_DEADLINE_SECONDS,
                "params": self._run_params(handle.key, dataset),
            }
            try:
                receipt = self.client.register_run(
                    body, idempotency_key("run", handle.uuid, dataset["version_id"])
                )
            except NeatlogsApiError as exc:
                self.ledger.update_run(
                    handle.uuid,
                    name=body["label"],
                    dataset=handle.key.dataset,
                    status=PENDING,
                    error=str(exc),
                )
                logger.warning("Neatlogs: run %s not registered: %s", body["label"], exc)
                return None
            self.ledger.update_run(
                handle.uuid,
                server_id=receipt["runId"],
                name=body["label"],
                dataset=handle.key.dataset,
                deadline_at=receipt.get("deadlineAt"),
                trials=body["trials"],
                status=SENT,
                error=None,
                reason=None,
            )
            return str(receipt["runId"])

    def _run_blocker(self, key: RunKey) -> tuple[str, str] | None:
        """Why a run cannot be registered: (PENDING, why) can be fixed by a later sync."""

        if len(self.plan.run_tasks[key]) > MAX_RUN_CASES:
            return SKIPPED, f"runs hold at most {MAX_RUN_CASES} tasks"
        if not self.client.can_use_public_api:
            return PENDING, "runs need NEATLOGS_TOKEN (a service-account token)"
        if not self.options.grading_configured:
            return PENDING, (
                "runs need a grading form: set NEATLOGS_HARBOR_FORM_VERSION_ID and "
                "NEATLOGS_HARBOR_EVALUATOR_IDS"
            )
        if not self.client.project_id:
            return PENDING, "runs need NEATLOGS_PROJECT_ID"
        return None

    def _run_params(self, key: RunKey, dataset: dict[str, Any]) -> dict[str, Any]:
        return {
            "source": "harbor",
            "harborJobId": self.plan.job_id,
            "harborJobName": self.plan.job_name[:2000],
            "harborAgent": key.agent[:2000],
            "harborModel": key.model,
            "harborDataset": key.dataset[:2000],
            "harborDatasetVersion": dataset["version_key"],
            "harborVersion": self.plan.harbor_version,
            "attempts": self.plan.run_attempts[key],
        }

    # ================================================================ trials

    def finished_trials_on_disk(self) -> list[TrialRecord]:
        trials = []
        for child in sorted(self.job_dir.iterdir()) if self.job_dir.exists() else []:
            if child.is_dir():
                trial = load_trial_dir(child, self.plan.job_id)
                if trial is not None:
                    trials.append(trial)
        return trials

    def defer_trial(self, trial: TrialRecord, reason: str) -> None:
        """Leave a trial for :meth:`finish`, which reads its final result from disk."""

        self.ledger.update_trial(
            trial.trial_name,
            trial_id=trial.trial_uuid,
            task=trial.task_name,
            run=self.plan.runs.get(trial.run_key(self.plan)),
            output=PENDING,
            trace=PENDING,
            reason=reason,
        )

    def send_trial(self, trial: TrialRecord, attempt: int) -> None:
        """Send one finished trial: its trace, then its run output."""

        handle = self._runs.get(trial.run_key(self.plan))
        previous = self.ledger.trial(trial.trial_name)
        if previous.get("output") == SENT:
            attempt = int(previous.get("attempt") or attempt)
        self.ledger.update_trial(
            trial.trial_name,
            trial_id=trial.trial_uuid,
            task=trial.task_name,
            attempt=attempt,
            run=handle.uuid if handle else None,
        )
        dataset = self.plan.datasets.get(trial.run_key(self.plan).dataset)
        outcome = outcome_of(trial.result, trial.trial_dir)
        trajectory = read_trajectory(trial.trial_dir)
        trace_id = ids.trace_id_hex(trial.trial_uuid)

        if previous.get("trace") != SENT:
            self._send_trace(trial, outcome, trajectory, dataset, trace_id)

        if previous.get("output") == SENT:
            return
        if handle is None or dataset is None:
            self.ledger.update_trial(trial.trial_name, output=SKIPPED, reason="not in the job plan")
            return
        if not self.options.stream and not self.ledger.run(handle.uuid).get("server_id"):
            # Deferred mode: the job end registers the run and sends the outputs.
            self.ledger.update_trial(trial.trial_name, output=PENDING, reason="sent at job end")
            return
        self._send_output(trial, outcome, trajectory, handle, dataset, attempt, trace_id)

    def _send_trace(
        self,
        trial: TrialRecord,
        outcome: Any,
        trajectory: dict[str, Any] | None,
        dataset: DatasetPlan | None,
        trace_id: str,
    ) -> None:
        if not self.options.traces:
            self.ledger.update_trial(trial.trial_name, trace=SKIPPED, reason="traces turned off")
            return
        if not self.client.can_send_traces:
            self.ledger.update_trial(
                trial.trial_name, trace=PENDING, trace_error="NEATLOGS_API_KEY is not set"
            )
            return
        try:
            if self.check_existing_traces and self.client.project_id:
                if self.client.trace_exists(trace_id):
                    self.ledger.update_trial(trial.trial_name, trace=SENT, trace_id=trace_id)
                    return
            instruction = (
                self._instruction(dataset, trial.task_name)
                if dataset and trial.task_name in dataset.tasks
                else None
            )
            request = build_trial_trace(
                trial,
                outcome,
                trajectory=trajectory,
                instruction=instruction,
                resource_attributes={"harbor.job.name": self.plan.job_name},
                redactor=self.redactor,
            )
            self.client.send_traces(request.SerializeToString())
        except NeatlogsApiError as exc:
            self.ledger.update_trial(trial.trial_name, trace=PENDING, trace_error=str(exc))
            return
        self.ledger.update_trial(trial.trial_name, trace=SENT, trace_id=trace_id, trace_error=None)

    def _send_output(
        self,
        trial: TrialRecord,
        outcome: Any,
        trajectory: dict[str, Any] | None,
        handle: _RunHandle,
        dataset: DatasetPlan,
        attempt: int,
        trace_id: str,
    ) -> None:
        run_id = self._ensure_run(handle)
        dataset_state = self.ledger.dataset(dataset.name)
        if run_id is None or not dataset_state.get("cases"):
            run_state = self.ledger.run(handle.uuid)
            status = SKIPPED if run_state.get("status") == SKIPPED else PENDING
            self.ledger.update_trial(
                trial.trial_name,
                output=status,
                reason=run_state.get("reason") or run_state.get("error") or "run not registered",
            )
            return
        trials = int(self.ledger.run(handle.uuid).get("trials") or MAX_RUN_TRIALS)
        if attempt > trials:
            self.ledger.update_trial(
                trial.trial_name,
                output=SKIPPED,
                reason=f"attempt {attempt} is past the run's {trials} attempts",
            )
            return
        attachments = (
            verifier_attachments(trial.trial_dir) if self.options.attachments != "none" else {}
        )
        sent_trace = self.ledger.trial(trial.trial_name).get("trace") == SENT
        body = output_body(
            trial,
            outcome,
            case_id=dataset_state["cases"][trial.task_name],
            attempt=attempt,
            trace_id=trace_id if sent_trace else None,
            trajectory=trajectory,
            attachments=attachments,
        )
        body["output"] = self.redactor.value(body["output"])
        try:
            receipt = self.client.upload_output(
                run_id, body, idempotency_key("output", handle.uuid, trial.trial_uuid, attempt)
            )
        except NeatlogsApiError as exc:
            if exc.status == 409 and exc.code in _ALREADY_FILLED:
                # The slot already holds this trial's output from an earlier send.
                self.ledger.update_trial(
                    trial.trial_name, output=SENT, output_note=exc.code, output_error=None
                )
                return
            self.ledger.update_trial(trial.trial_name, output=PENDING, output_error=str(exc))
            return
        self.ledger.update_trial(
            trial.trial_name,
            output=SENT,
            output_id=receipt.get("outputId"),
            output_error=None,
            reason=None,
        )

    # ================================================================ finish

    def attempts_for(self, trials: list[TrialRecord]) -> dict[str, int]:
        """Attempt numbers by start time, keeping slots that already hold an output."""

        groups: dict[tuple[RunKey, str], list[TrialRecord]] = defaultdict(list)
        for trial in trials:
            groups[(trial.run_key(self.plan), trial.task_name)].append(trial)
        floor = parse_time("0001-01-01T00:00:00Z")
        attempts: dict[str, int] = {}
        for members in groups.values():
            members.sort(key=lambda t: (t.started_at or floor, t.trial_name))
            taken: set[int] = set()
            for trial in members:
                state = self.ledger.trial(trial.trial_name)
                if state.get("output") == SENT and state.get("attempt"):
                    attempts[trial.trial_name] = int(state["attempt"])
                    taken.add(int(state["attempt"]))
            free = (n for n in range(1, len(members) + len(taken) + 1) if n not in taken)
            for trial in members:
                if trial.trial_name not in attempts:
                    attempts[trial.trial_name] = next(free)
        return attempts

    def send_pending(self) -> None:
        """Send every finished trial in the job folder that has not reached Neatlogs."""

        if not self.options.stream:
            for handle in self._runs.values():
                self._ensure_run(handle)
        trials = self.finished_trials_on_disk()
        attempts = self.attempts_for(trials)
        for trial in sorted(trials, key=lambda t: t.trial_name):
            state = self.ledger.trial(trial.trial_name)
            done = (SENT, SKIPPED)
            if state.get("output") in done and state.get("trace") in done:
                continue
            self.send_trial(trial, attempts[trial.trial_name])

    def finish(self) -> Summary:
        """Send what is pending, complete the runs, compare, and summarize."""

        try:
            if self._public_api_ready():
                for dataset in self.plan.datasets.values():
                    self._ensure_dataset(dataset)
            self.send_pending()
            runs = [self._finish_run(handle) for handle in self._runs.values()]
            if self.move_baseline_tag:
                for handle in self._runs.values():
                    self._tag_baseline(handle)
        except Exception as exc:  # never let a summary failure escape
            self.ledger.record_error("finish", exc)
            runs = []
        return self._summary(runs)

    def _finish_run(self, handle: _RunHandle) -> dict[str, Any]:
        state = self.ledger.run(handle.uuid)
        run_id = state.get("server_id")
        info: dict[str, Any] = {"handle": handle, "run_id": run_id}
        if not run_id:
            return info
        unsent = [
            name
            for name in self.ledger.trial_names()
            if self.ledger.trial(name).get("run") == handle.uuid
            and self.ledger.trial(name).get("output") == PENDING
        ]
        if unsent and not state.get("completed"):
            # Completing would seal the run and turn these into failures; leave it
            # open so `neatlogs harbor sync` can still fill it.
            info["open"] = len(unsent)
            return info
        if not state.get("completed"):
            try:
                status = self.client.complete_run(
                    run_id, idempotency_key("complete", handle.uuid, run_id)
                )
                self.ledger.update_run(handle.uuid, completed=True, state=status.get("state"))
            except NeatlogsApiError as exc:
                self.ledger.update_run(handle.uuid, completed=False, error=str(exc))
                info["error"] = str(exc)
                return info
        baseline = self.options.baseline or f"tag:{BASELINE_TAG}"
        try:
            info["comparison"] = self.client.comparison(run_id, baseline)
        except NeatlogsApiError as exc:
            info["comparison_error"] = exc
        return info

    def _tag_baseline(self, handle: _RunHandle) -> None:
        state = self.ledger.run(handle.uuid)
        if not state.get("server_id") or state.get("state") != "completed" or state.get("tagged"):
            return
        try:
            self.client.tag_run(
                state["server_id"],
                BASELINE_TAG,
                idempotency_key("tag", handle.uuid, state["server_id"], BASELINE_TAG),
            )
            self.ledger.update_run(handle.uuid, tagged=True)
        except NeatlogsApiError as exc:
            self.ledger.record_error("tag", exc)

    def _summary(self, runs: list[dict[str, Any]]) -> Summary:
        trials = self.finished_trials_on_disk()
        by_run: dict[RunKey, list[TrialRecord]] = defaultdict(list)
        for trial in trials:
            by_run[trial.run_key(self.plan)].append(trial)
        info_by_key = {info["handle"].key: info for info in runs}
        summaries = []
        for key, handle in self._runs.items():
            info = info_by_key.get(key, {})
            state = self.ledger.run(handle.uuid)
            dataset_state = self.ledger.dataset(key.dataset)
            summaries.append(
                RunSummary.build(
                    name=self._run_label(key),
                    dataset=key.dataset,
                    trials=by_run.get(key, []),
                    run_id=state.get("server_id"),
                    run_state=state.get("state"),
                    skipped_reason=state.get("reason"),
                    error=state.get("error"),
                    comparison=info.get("comparison"),
                    comparison_error=info.get("comparison_error"),
                    left_open=info.get("open", 0),
                    link=self._dataset_link(dataset_state),
                )
            )
        pending = self.ledger.pending_trials()
        return Summary(
            runs=summaries,
            pending_trials=len(pending),
            ledger_path=self.ledger.path,
            job_dir=self.job_dir,
            notes=self._notes(),
        )

    def _dataset_link(self, dataset_state: dict[str, Any]) -> str | None:
        if not dataset_state.get("dataset_id"):
            return None
        return (
            f"{self.options.host}/experiments/datasets/{dataset_state['dataset_id']}"
            f"?versionId={dataset_state['version_id']}"
        )

    def _notes(self) -> list[str]:
        notes = []
        if not self.client.can_send_traces and self.options.traces:
            notes.append("Traces need NEATLOGS_API_KEY (the project's ingest key).")
        if not self.client.can_use_public_api:
            notes.append("Datasets and runs need NEATLOGS_TOKEN (a service-account token).")
        return notes
