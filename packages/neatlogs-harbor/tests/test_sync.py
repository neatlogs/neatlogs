"""End to end against an in-memory Neatlogs: import, re-import, plugin, outages."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.models.job.config import JobConfig
from harbor.models.job.lock import TrialLock
from harbor.models.job.result import JobResult
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.hooks import TrialEvent, TrialHookEvent

from neatlogs_harbor import syncer as syncer_module
from neatlogs_harbor.cli import main
from neatlogs_harbor.ledger import LEDGER_FILENAME, PENDING, SENT
from neatlogs_harbor.plugin import NeatlogsPlugin

from .conftest import FAKE_GITHUB_TOKEN, FAKE_OPENAI_KEY


@pytest.fixture(autouse=True)
def fixed_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    # Task folders live in Harbor's cache on the machine that ran the job.
    monkeypatch.setattr(
        syncer_module, "read_instruction", lambda task: f"Instruction for {task.name}"
    )


def _ledger(job_dir: Path) -> dict:
    return json.loads((job_dir / LEDGER_FILENAME).read_text())


def _snapshot(fake) -> dict:
    return {
        "datasets": len(fake.datasets),
        "runs": len(fake.runs),
        "outputs": len(fake.outputs),
        "traces": len(fake.trace_ids),
        "duplicate_spans": fake.duplicate_spans,
    }


def test_import_sends_dataset_run_outputs_and_traces(env, wire, mixed_job, capsys):
    assert main(["import", str(mixed_job.parent)]) == 0
    fake = wire
    assert _snapshot(fake) == {
        "datasets": 1,
        "runs": 1,
        "outputs": 3,
        "traces": 3,
        "duplicate_spans": 0,
    }
    (run,) = fake.runs.values()
    assert run["body"]["label"] == "terminus-2 · gpt-5.6-sol"
    assert run["body"]["trials"] == 3
    assert run["body"]["deadlineSeconds"] == 3600
    assert run["state"] == "completed"

    outputs = {o["body"]["output"]["harbor"]["trialName"]: o["body"] for o in fake.all_outputs()}
    assert outputs["hello-world__Atraj01"]["trial"] == 1
    assert outputs["hello-world__Csbx003"]["trial"] == 2
    assert outputs["hello-world__Btime02"]["trial"] == 3
    assert outputs["hello-world__Csbx003"]["output"]["harbor"]["errorClass"] == "infrastructure"
    assert "reward" not in outputs["hello-world__Csbx003"]["metrics"]
    # Secrets never leave the machine.
    sent = json.dumps([o["body"] for o in fake.all_outputs()])
    assert FAKE_GITHUB_TOKEN not in sent and FAKE_OPENAI_KEY not in sent

    ledger = _ledger(mixed_job)
    assert {t["output"] for t in ledger["trials"].values()} == {SENT}
    assert {t["trace"] for t in ledger["trials"].values()} == {SENT}
    out = capsys.readouterr().out
    assert "Neatlogs summary" in out and "solved        1 / 2 trials" in out
    assert "no reward     1 trial(s)" in out


def test_importing_twice_creates_nothing_new(env, wire, mixed_job):
    assert main(["import", str(mixed_job.parent)]) == 0
    before = _snapshot(wire)
    requests = len(wire.requests)
    assert main(["import", str(mixed_job.parent)]) == 0
    assert _snapshot(wire) == before
    # The ledger short-circuits nearly everything on the second pass.
    assert len(wire.requests) - requests < 10


def test_reimport_without_the_ledger_reconciles_by_id(env, wire, mixed_job):
    assert main(["import", str(mixed_job.parent)]) == 0
    before = _snapshot(wire)
    (mixed_job / LEDGER_FILENAME).unlink()
    assert main(["import", str(mixed_job.parent)]) == 0
    assert _snapshot(wire) == before


def test_one_dataset_version_per_task_digest_set(env, wire, mixed_job, oracle_job):
    assert main(["import", str(mixed_job.parent)]) == 0
    # Same task, same digest: both jobs share one dataset version.
    assert len(wire.datasets) == 1
    assert len(wire.runs) == 2


def _job(job_dir: Path) -> SimpleNamespace:
    config = JobConfig.model_validate_json((job_dir / "config.json").read_text())
    config.jobs_dir = job_dir.parent
    hooks: dict[str, list] = {"start": [], "end": []}
    job = SimpleNamespace(
        id=json.loads((job_dir / "result.json").read_text())["id"],
        job_dir=job_dir,
        config=config,
        hooks=hooks,
    )
    job.on_trial_started = lambda cb: hooks["start"].append(cb)
    job.on_trial_ended = lambda cb: hooks["end"].append(cb)
    return job


def _event(kind: TrialEvent, trial_dir: Path) -> TrialHookEvent:
    result = TrialResult.model_validate_json((trial_dir / "result.json").read_text())
    return TrialHookEvent(
        event=kind,
        task_name=result.task_name,
        config=result.config,
        result=result,
        lock=TrialLock.model_validate_json((trial_dir / "lock.json").read_text()),
    )


def _run_plugin(job_dir: Path, max_retries: int = 0, **kwargs) -> NeatlogsPlugin:
    async def scenario() -> NeatlogsPlugin:
        job = _job(job_dir)
        job.config.retry.max_retries = max_retries
        plugin = NeatlogsPlugin(**kwargs)
        await plugin.on_job_start(job)
        trial_dirs = sorted(d for d in job_dir.iterdir() if d.is_dir())
        for trial_dir in trial_dirs:
            for callback in job.hooks["start"]:
                await callback(_event(TrialEvent.START, trial_dir))
        for trial_dir in reversed(trial_dirs):  # finish out of order
            for callback in job.hooks["end"]:
                await callback(_event(TrialEvent.END, trial_dir))
        result = JobResult.model_validate_json((job_dir / "result.json").read_text())
        await plugin.on_job_end(result)
        return plugin

    return asyncio.run(scenario())


def test_plugin_streams_and_import_afterwards_creates_no_duplicates(env, wire, mixed_job, capsys):
    _run_plugin(mixed_job)
    after_plugin = _snapshot(wire)
    assert after_plugin == {
        "datasets": 1,
        "runs": 1,
        "outputs": 3,
        "traces": 3,
        "duplicate_spans": 0,
    }
    assert "Neatlogs summary" in capsys.readouterr().out

    # Same records from disk, with and without the plugin's ledger.
    assert main(["import", str(mixed_job.parent)]) == 0
    assert _snapshot(wire) == after_plugin
    (mixed_job / LEDGER_FILENAME).unlink()
    assert main(["import", str(mixed_job.parent)]) == 0
    assert _snapshot(wire) == after_plugin


def test_plugin_numbers_attempts_like_the_importer(env, wire, mixed_job):
    _run_plugin(mixed_job)
    attempts = {
        o["body"]["output"]["harbor"]["trialName"]: o["body"]["trial"] for o in wire.all_outputs()
    }
    assert attempts == {
        "hello-world__Atraj01": 1,
        "hello-world__Csbx003": 2,
        "hello-world__Btime02": 3,
    }


def test_plugin_never_fails_the_job_and_sync_recovers(env, wire, mixed_job, capsys):
    wire.down = True
    plugin = _run_plugin(mixed_job, retries=0)
    assert plugin is not None
    ledger = _ledger(mixed_job)
    assert {t["output"] for t in ledger["trials"].values()} == {PENDING}
    assert "neatlogs harbor sync" in capsys.readouterr().out
    assert _snapshot(wire)["outputs"] == 0

    wire.down = False
    assert main(["sync", str(mixed_job)]) == 0
    assert _snapshot(wire) == {
        "datasets": 1,
        "runs": 1,
        "outputs": 3,
        "traces": 3,
        "duplicate_spans": 0,
    }
    assert {t["output"] for t in _ledger(mixed_job)["trials"].values()} == {SENT}


def test_plugin_without_credentials_buffers_then_sync_sends(
    env, wire, mixed_job, monkeypatch, capsys
):
    saved = {name: env[name] for name in ("NEATLOGS_API_KEY", "NEATLOGS_TOKEN")}
    for name in saved:
        monkeypatch.delenv(name)
    _run_plugin(mixed_job)
    assert wire.requests == []
    ledger = _ledger(mixed_job)
    assert {t["output"] for t in ledger["trials"].values()} == {PENDING}
    assert {t["trace"] for t in ledger["trials"].values()} == {PENDING}
    out = capsys.readouterr().out
    assert "NEATLOGS_API_KEY" in out and "neatlogs harbor sync" in out

    for name, value in saved.items():
        monkeypatch.setenv(name, value)
    assert main(["sync", str(mixed_job)]) == 0
    assert _snapshot(wire) == {
        "datasets": 1,
        "runs": 1,
        "outputs": 3,
        "traces": 3,
        "duplicate_spans": 0,
    }


def test_runs_wait_for_a_grading_form_but_traces_still_flow(env, wire, mixed_job, monkeypatch):
    monkeypatch.delenv("NEATLOGS_HARBOR_FORM_VERSION_ID")
    assert main(["import", str(mixed_job.parent)]) == 1
    assert len(wire.trace_ids) == 3 and len(wire.datasets) == 1
    assert wire.runs == {}
    run = next(iter(_ledger(mixed_job)["runs"].values()))
    assert "grading form" in run["reason"]


def test_sync_moves_the_baseline_and_next_run_compares(env, wire, mixed_job, oracle_job, capsys):
    assert main(["sync", str(oracle_job)]) == 0
    assert "first finished Harbor run" in capsys.readouterr().out
    assert len(wire.tags) == 1
    assert main(["sync", str(mixed_job)]) == 0
    assert "grading in progress" in capsys.readouterr().out


def test_dry_run_sends_nothing(env, wire, mixed_job, capsys):
    assert main(["import", str(mixed_job.parent), "--dry-run"]) == 0
    assert wire.requests == []
    assert "run terminus-2 · gpt-5.6-sol" in capsys.readouterr().out


def test_cli_rejects_folders_that_are_not_jobs(tmp_path, capsys):
    assert main(["sync", str(tmp_path)]) == 2
    assert "not a Harbor job folder" in capsys.readouterr().err


def test_deferred_mode_registers_and_fills_runs_at_job_end(env, wire, mixed_job):
    plugin = NeatlogsPlugin(stream="false")
    assert plugin.options.stream is False
    _run_plugin(mixed_job, stream="false")
    assert _snapshot(wire) == {
        "datasets": 1,
        "runs": 1,
        "outputs": 3,
        "traces": 3,
        "duplicate_spans": 0,
    }
    (run,) = wire.runs.values()
    assert run["state"] == "completed"


def test_trials_harbor_may_retry_are_sent_at_job_end(env, wire, mixed_job, monkeypatch):
    sent_during_trials: list[str] = []
    original = syncer_module.Syncer.send_trial

    def spy(self, trial, attempt):
        sent_during_trials.append(trial.trial_name)
        return original(self, trial, attempt)

    monkeypatch.setattr(syncer_module.Syncer, "send_trial", spy)
    _run_plugin(mixed_job, max_retries=2)
    # The sandbox failure is retryable, so the plugin waited for the final attempt.
    assert sent_during_trials.index("hello-world__Csbx003") == len(sent_during_trials) - 1
    assert _snapshot(wire)["outputs"] == 3


def test_run_stays_open_while_an_output_is_unsent(env, wire, mixed_job, capsys):
    wire.failing_trials = {"hello-world__Btime02"}
    _run_plugin(mixed_job, retries=0)
    (run,) = wire.runs.values()
    assert run["state"] == "queued"  # not sealed, so sync can still fill it
    out = capsys.readouterr().out
    assert "run left open: 1 output(s) still to send" in out

    wire.failing_trials = set()
    assert main(["sync", str(mixed_job)]) == 0
    assert run["state"] == "completed"
    assert len(wire.outputs) == 3
