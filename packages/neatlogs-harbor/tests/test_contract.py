"""Requests checked against the vendored public API contract (see contract.py)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import httpx
import pytest

from neatlogs_harbor import syncer as syncer_module
from neatlogs_harbor.cli import main
from neatlogs_harbor.client import idempotency_key
from neatlogs_harbor.ledger import LEDGER_FILENAME, SKIPPED
from neatlogs_harbor.records import (
    DatasetPlan,
    TaskEntry,
    dataset_body,
    load_trial_dir,
    outcome_of,
    output_body,
    read_trajectory,
)

from . import contract
from .conftest import FIXTURES


def _post(path: str, body: dict, key: str | None = None) -> httpx.Request:
    headers = {
        "Idempotency-Key": key or idempotency_key("test", path),
        "x-project-id": "6f1c2c1e-5d0b-4d8a-9a51-0f4c1b2a3d4e",
        "Content-Type": "application/json",
    }
    return httpx.Request("POST", f"https://neatlogs.test{path}", json=body, headers=headers)


@pytest.mark.parametrize("parts", [("run", "x"), ("output", 1, 2, 3), ("dataset", "tb@2.0", "a")])
def test_idempotency_keys_match_the_header_contract(parts):
    request = _post("/api/v1/public/runs/00000000-0000-4000-8000-000000000000/complete", {})
    request.headers["Idempotency-Key"] = idempotency_key(*parts)
    assert contract.check_request(request) == []


def test_contract_rejects_bodies_outside_the_limits():
    body = {
        "datasetVersionId": "5b1d6c4e-2f0a-4c51-9a0e-6f2d1a7c9e01",
        "label": "codex · gpt",
        "caseIds": ["5b1d6c4e-2f0a-4c51-9a0e-6f2d1a7c9e02"],
        "trials": 11,
        "grading": {
            "formVersionId": "5b1d6c4e-2f0a-4c51-9a0e-6f2d1a7c9e03",
            "evaluatorIds": ["5b1d6c4e-2f0a-4c51-9a0e-6f2d1a7c9e04"],
        },
        "deadlineSeconds": 3600,
    }
    problems = contract.check_request(_post("/api/v1/public/runs", body))
    assert any("trials" in p for p in problems)
    body["trials"] = 10
    assert contract.check_request(_post("/api/v1/public/runs", body)) == []


def test_dataset_body_for_a_benchmark_fits_the_contract():
    tasks = {
        f"terminal-bench/task-{i:03d}": TaskEntry(
            name=f"terminal-bench/task-{i:03d}",
            digest="sha256:" + f"{i:064x}",
            source="terminal-bench",
            lock={"type": "package", "version": "2.0"},
        )
        for i in range(89)
    }
    plan = DatasetPlan(name="terminal-bench@2.0", tasks=tasks)
    body = dataset_body(plan, {name: "Fix the failing build. " * 400 for name in tasks})
    request = _post("/api/v1/public/datasets", body)
    assert contract.check_request(request) == []
    assert len(request.content) <= 64 * 1024


def test_every_fixture_trial_output_fits_the_contract(mixed_job: Path):
    for trial_dir in sorted(d for d in mixed_job.iterdir() if d.is_dir()):
        trial = load_trial_dir(trial_dir)
        body = output_body(
            trial,
            outcome_of(trial.result, trial_dir),
            case_id="5b1d6c4e-2f0a-4c51-9a0e-6f2d1a7c9e05",
            attempt=1,
            trace_id="ab" * 16,
            trajectory=read_trajectory(trial_dir),
            attachments={},
        )
        run = "/api/v1/public/runs/5b1d6c4e-2f0a-4c51-9a0e-6f2d1a7c9e06/outputs"
        assert contract.check_request(_post(run, body)) == []


# ------------------------------------------------------------- benchmark-sized jobs


def make_job(root: Path, name: str, n_tasks: int, n_attempts: int) -> Path:
    """A finished job shaped like a Terminal-Bench run, built from the oracle fixture."""

    template_job = FIXTURES / "oracle-hello-world"
    template_trial = template_job / "hello-world__KMi3fnx"
    job_dir = root / "jobs" / name
    job_dir.mkdir(parents=True)
    job_id = f"7c0e9a52-3b1d-4e6f-8a2b-{n_tasks:06d}{n_attempts:06d}"
    result = json.loads((template_job / "result.json").read_text())
    result.update(id=job_id, n_total_trials=n_tasks * n_attempts)
    (job_dir / "result.json").write_text(json.dumps(result))
    config = {
        "job_name": name,
        "n_attempts": n_attempts,
        "datasets": [{"name": "terminal-bench", "version": "2.0"}],
    }
    (job_dir / "config.json").write_text(json.dumps(config))
    lock = json.loads((template_job / "lock.json").read_text())
    trial_lock_template = lock["trials"][0]
    lock["trials"] = []
    for task in range(n_tasks):
        task_name = f"terminal-bench/task-{task:03d}"
        task_lock = {
            **trial_lock_template,
            "task": {
                "name": task_name,
                "type": "package",
                "source": "terminal-bench",
                "digest": "sha256:" + f"{task:064x}",
            },
        }
        for attempt in range(n_attempts):
            lock["trials"].append(task_lock)
            trial_name = f"task-{task:03d}__a{attempt}"
            trial_dir = job_dir / trial_name
            shutil.copytree(template_trial, trial_dir)
            trial = json.loads((template_trial / "result.json").read_text())
            trial.update(
                id=f"9d1f4b2c-0a3e-4c5d-8e6f-{task:06d}{attempt:06d}",
                task_name=task_name,
                trial_name=trial_name,
                source="terminal-bench",
                started_at=f"2026-11-12T22:{10 + attempt:02d}:00.000000Z",
                finished_at=f"2026-11-12T22:{10 + attempt:02d}:30.000000Z",
            )
            trial["config"].update(trial_name=trial_name, job_id=job_id)
            trial["config"]["task"].update(name=task_name, source="terminal-bench")
            trial["verifier_result"] = {"rewards": {"reward": float((task + attempt) % 2)}}
            (trial_dir / "result.json").write_text(json.dumps(trial))
            (trial_dir / "lock.json").write_text(json.dumps(task_lock))
    (job_dir / "lock.json").write_text(json.dumps(lock))
    return job_dir


@pytest.fixture(autouse=True)
def long_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        syncer_module, "read_instruction", lambda task: f"{task.name}: fix the build. " * 300
    )


def test_terminal_bench_sized_job_imports_within_the_contract(env, wire, tmp_path):
    job_dir = make_job(tmp_path, "tb-codex", n_tasks=89, n_attempts=3)
    assert main(["import", str(job_dir)]) == 0
    (run,) = wire.runs.values()
    assert len(run["body"]["caseIds"]) == 89 and run["body"]["trials"] == 3
    assert run["body"]["label"] == "oracle"
    assert len(wire.outputs) == 89 * 3
    assert len(wire.trace_ids) == 89 * 3
    assert run["state"] == "completed"
    (dataset,) = wire.datasets.values()
    assert dataset["name"] == "terminal-bench@2.0" and len(dataset["cases"]) == 89

    # Importing again changes nothing.
    before = (len(wire.datasets), len(wire.runs), len(wire.outputs), wire.duplicate_spans)
    assert main(["import", str(job_dir)]) == 0
    assert (len(wire.datasets), len(wire.runs), len(wire.outputs), wire.duplicate_spans) == before


def test_jobs_past_the_run_limits_still_send_traces(env, wire, tmp_path):
    job_dir = make_job(tmp_path, "too-big", n_tasks=101, n_attempts=1)
    assert main(["import", str(job_dir)]) == 0
    assert wire.runs == {}
    assert len(wire.trace_ids) == 101
    ledger = json.loads((job_dir / LEDGER_FILENAME).read_text())
    (run,) = ledger["runs"].values()
    assert run["status"] == SKIPPED and "at most 100 tasks" in run["reason"]
