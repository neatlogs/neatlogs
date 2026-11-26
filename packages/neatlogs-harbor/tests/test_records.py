from __future__ import annotations

import json
from pathlib import Path

from neatlogs_harbor import ids
from neatlogs_harbor.records import (
    MAX_BODY_BYTES,
    DatasetPlan,
    TaskEntry,
    assign_attempts,
    dataset_body,
    dataset_name_for,
    load_trial_dir,
    outcome_of,
    output_body,
    plan_job,
    read_instruction,
    read_trajectory,
    trial_metrics,
)


def _plan(job_dir: Path):
    return plan_job(
        job_id=json.loads((job_dir / "result.json").read_text())["id"],
        job_config=json.loads((job_dir / "config.json").read_text()),
        job_lock=json.loads((job_dir / "lock.json").read_text()),
    )


def _trials(job_dir: Path):
    return [t for d in sorted(job_dir.iterdir()) if d.is_dir() and (t := load_trial_dir(d))]


def test_ids_are_deterministic_uuid5_values():
    first = ids.trial_uuid("job-1", "org/task", "task__abc")
    assert first == ids.trial_uuid("job-1", "org/task", "task__abc")
    assert first != ids.trial_uuid("job-2", "org/task", "task__abc")
    assert first[14] == "5"  # uuid version nibble
    assert ids.run_uuid("job-1", "codex", "openai/gpt", "tb@2.0") != ids.run_uuid(
        "job-1", "codex", None, "tb@2.0"
    )


def test_dataset_version_key_ignores_order_and_repeats_but_not_content():
    a = ids.dataset_version_key([("t1", "sha256:1"), ("t2", "sha256:2"), ("t1", "sha256:1")])
    b = ids.dataset_version_key([("t2", "sha256:2"), ("t1", "sha256:1")])
    edited = ids.dataset_version_key([("t2", "sha256:2"), ("t1", "sha256:9")])
    assert a == b
    assert a != edited


def test_trace_id_matches_what_atif2otel_derives_from_the_seed():
    from harbor_atif2otel.ids import sha256_trace_id

    trial = ids.trial_uuid("job", "task", "trial")
    assert sha256_trace_id(ids.trace_seed(trial)).hex() == ids.trace_id_hex(trial)


def test_plan_groups_runs_like_harbor(oracle_job: Path):
    plan = _plan(oracle_job)
    assert list(plan.datasets) == ["hello-world/hello-world"]
    dataset = plan.datasets["hello-world/hello-world"]
    assert list(dataset.tasks) == ["hello-world/hello-world"]
    (key,) = plan.runs
    assert (key.agent, key.model, key.dataset) == ("oracle", None, "hello-world/hello-world")
    assert key.display_name == "oracle"
    assert plan.run_attempts[key] == 2
    assert plan.harbor_version == "0.25.0"


def test_dataset_names_follow_harbor():
    config = {"datasets": [{"name": "terminal-bench", "version": "2.0"}]}
    assert dataset_name_for("terminal-bench", "t", config, []) == "terminal-bench@2.0"
    local = {"datasets": [{"path": "/data/my-tasks"}]}
    assert dataset_name_for("my-tasks", "t", local, []) == "my-tasks"
    assert dataset_name_for(None, "org/one", {}, ["org/one", "org/one"]) == "org/one"
    assert dataset_name_for(None, "org/one", {}, ["org/one", "org/two"]) == "harbor-adhoc"


def test_model_names_drop_the_provider_in_run_names(mixed_job: Path):
    (key,) = _plan(mixed_job).runs
    assert key.display_name == "terminus-2 · gpt-5.6-sol"


def test_outcomes_follow_harbors_error_policy(mixed_job: Path):
    outcomes = {t.trial_name: outcome_of(t.result, t.trial_dir) for t in _trials(mixed_job)}

    solved = outcomes["hello-world__Atraj01"]
    assert (solved.reward_valid, solved.error_class, solved.solved) == (True, None, True)

    timeout = outcomes["hello-world__Btime02"]
    assert timeout.error_class == "agent"
    assert timeout.failure_cause == "agent_exhausted_budget"
    assert timeout.reward_valid and timeout.solved is False

    sandbox = outcomes["hello-world__Csbx003"]
    assert sandbox.error_class == "infrastructure"
    assert sandbox.failure_cause == "sandbox_failed"
    assert not sandbox.reward_valid and sandbox.solved is None


def test_grader_failures_are_classified_separately():
    result = {
        "verifier_result": {"rewards": {"reward": 0.0}},
        "exception_info": {"exception_type": "VerifierTimeoutError", "exception_message": "x"},
    }
    outcome = outcome_of(result)
    assert outcome.error_class == "grader"
    assert not outcome.reward_valid


def test_attempts_are_numbered_by_start_time(mixed_job: Path):
    plan = _plan(mixed_job)
    attempts = assign_attempts(_trials(mixed_job), plan)
    # Started at :00, :05 and :02.
    assert attempts == {
        "hello-world__Atraj01": 1,
        "hello-world__Btime02": 3,
        "hello-world__Csbx003": 2,
    }


def test_metrics_leave_invalid_rewards_out(mixed_job: Path):
    trials = {t.trial_name: t for t in _trials(mixed_job)}
    solved = trials["hello-world__Atraj01"]
    metrics = trial_metrics(solved.result, outcome_of(solved.result))
    assert metrics["reward"] == 1.0 and metrics["solved"] == 1.0
    assert metrics["rewardValid"] == 1.0
    assert metrics["inputTokens"] == 1200 and metrics["outputTokens"] == 340
    assert metrics["totalTokens"] == 1540 and metrics["costUsd"] == 0.0123
    assert metrics["durationMs"] > 0 and "latencyMs" in metrics

    sandbox = trials["hello-world__Csbx003"]
    metrics = trial_metrics(sandbox.result, outcome_of(sandbox.result))
    assert metrics["rewardValid"] == 0.0
    assert "reward" not in metrics and "solved" not in metrics


def test_output_body_carries_the_trial(mixed_job: Path):
    trial = next(t for t in _trials(mixed_job) if t.trial_name == "hello-world__Atraj01")
    body = output_body(
        trial,
        outcome_of(trial.result),
        case_id="c",
        attempt=1,
        trace_id="ab" * 16,
        trajectory=read_trajectory(trial.trial_dir),
        attachments={"reward_details": {"criteria": []}},
    )
    assert set(body) == {"caseId", "trial", "output", "metrics"}
    assert body["output"]["answer"] == "Done: hello.txt contains Hello, world!"
    assert body["output"]["traceId"] == "ab" * 16
    assert body["output"]["harbor"]["steps"] == 2
    assert body["output"]["harbor"]["errorClass"] is None


def test_dataset_body_fits_the_api_limit():
    tasks = {
        f"org/task-{i:03d}": TaskEntry(
            name=f"org/task-{i:03d}",
            digest="sha256:" + f"{i:064x}",
            source="bench",
            lock={"type": "package"},
        )
        for i in range(89)
    }
    plan = DatasetPlan(name="bench@1.0", tasks=tasks)
    instructions = {name: "Do the thing. " * 300 for name in tasks}
    body = dataset_body(plan, instructions)
    size = len(json.dumps(body, ensure_ascii=False).encode())
    assert size <= MAX_BODY_BYTES
    assert [c["input"]["task"] for c in body["cases"]] == sorted(tasks)
    assert "shortened" in body["cases"][0]["input"]["instruction"]
    assert body["idempotencyKey"] == ids.dataset_publication_uuid("bench@1.0", plan.version_key)


def test_instruction_comes_from_the_task_folder(tmp_path: Path):
    task_dir = tmp_path / "my-task"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("Fix the bug.")
    entry = TaskEntry(
        name="my-task", digest="sha256:" + "0" * 64, source=None, lock={"path": str(task_dir)}
    )
    assert read_instruction(entry) == "Fix the bug."
    gone = TaskEntry(
        name="gone", digest="sha256:" + "0" * 64, source=None, lock={"path": str(tmp_path / "x")}
    )
    assert read_instruction(gone) is None
