"""The one mapper: Harbor job and trial documents → Neatlogs records.

Everything here works on the JSON form of Harbor's documents (``config.json``,
``lock.json``, ``result.json`` and the verifier files), which is exactly what a
job folder holds and what the live plugin gets by dumping Harbor's models with
``model_dump(mode="json")``. Feeding both entry points through the same
functions is what lets the plugin and the importer write identical records.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import ids

ADHOC_DATASET = "harbor-adhoc"
# Public API request bodies are capped at 64 KiB; keep headroom.
MAX_BODY_BYTES = 60_000
MAX_ATTACHMENT_CHARS = 16_000


# ---------------------------------------------------------------- plan (job start)


@dataclass(frozen=True)
class RunKey:
    """Harbor's own grouping for job stats: one run per agent, model and dataset."""

    agent: str
    model: str | None
    dataset: str

    @property
    def display_name(self) -> str:
        if not self.model:
            return self.agent
        return f"{self.agent} · {short_model_name(self.model)}"


@dataclass
class TaskEntry:
    name: str
    digest: str
    source: str | None
    lock: dict[str, Any]


@dataclass
class DatasetPlan:
    name: str
    tasks: dict[str, TaskEntry] = field(default_factory=dict)

    @property
    def version_key(self) -> str:
        return ids.dataset_version_key((t.name, t.digest) for t in self.tasks.values())


@dataclass
class JobPlan:
    job_id: str
    job_name: str
    datasets: dict[str, DatasetPlan]
    # Stable run id per (agent, model, dataset).
    runs: dict[RunKey, str]
    # Tasks each run covers, and the most attempts any of them has.
    run_tasks: dict[RunKey, list[str]] = field(default_factory=dict)
    run_attempts: dict[RunKey, int] = field(default_factory=dict)
    harbor_version: str | None = None
    _dataset_by_task: dict[tuple[str | None, str], str] = field(default_factory=dict)

    def dataset_for(self, source: str | None, task_name: str) -> str:
        return self._dataset_by_task.get((source, task_name)) or dataset_name_for(
            source, task_name, {}, []
        )


def short_model_name(model: str) -> str:
    """``openai/gpt-5.6-sol`` → ``gpt-5.6-sol``."""

    for sep in ("/", ":"):
        if sep in model:
            return model.split(sep, 1)[1]
    return model


def agent_name(agent_config: Mapping[str, Any]) -> str:
    return str(agent_config.get("name") or agent_config.get("import_path") or "agent")


def dataset_name_for(
    source: str | None,
    task_name: str,
    job_config: Mapping[str, Any],
    adhoc_task_names: Iterable[str],
) -> str:
    """Harbor's dataset name and version, such as ``terminal-bench@2.0``.

    Trials from a registry or local dataset carry the dataset name as their
    task ``source``; the version comes from the job's dataset config. Ad hoc
    tasks (``harbor run -t``) are named after the task when the job ran one, and
    share ``harbor-adhoc`` otherwise.
    """

    if source:
        for dataset in job_config.get("datasets") or []:
            names = {dataset.get("name")}
            if dataset.get("path"):
                names.add(Path(str(dataset["path"])).name)
            if source in names and dataset.get("version"):
                return f"{source}@{dataset['version']}"
        return source
    adhoc = set(adhoc_task_names)
    if len(adhoc) <= 1:
        return task_name
    return ADHOC_DATASET


def plan_job(
    *,
    job_id: str,
    job_config: Mapping[str, Any],
    job_lock: Mapping[str, Any],
) -> JobPlan:
    """Datasets (tasks keyed by digest) and runs for one Harbor job.

    ``job_lock`` is the job's ``lock.json``: one entry per planned trial with the
    resolved task digest and agent. It lists every trial, including ones that
    never ran, so the plugin and the importer see the same task set.
    """

    trial_locks = list(job_lock.get("trials") or [])
    adhoc_names = [
        str(t["task"]["name"]) for t in trial_locks if not (t.get("task") or {}).get("source")
    ]
    datasets: dict[str, DatasetPlan] = {}
    runs: dict[RunKey, str] = {}
    attempts: dict[tuple[RunKey, str], int] = defaultdict(int)
    by_task: dict[tuple[str | None, str], str] = {}
    for trial_lock in trial_locks:
        task = trial_lock.get("task") or {}
        task_name = str(task.get("name"))
        source = task.get("source")
        dataset = dataset_name_for(source, task_name, job_config, adhoc_names)
        by_task[(source, task_name)] = dataset
        plan = datasets.setdefault(dataset, DatasetPlan(name=dataset))
        plan.tasks.setdefault(
            task_name,
            TaskEntry(name=task_name, digest=str(task.get("digest")), source=source, lock=task),
        )
        agent = trial_lock.get("agent") or {}
        key = RunKey(agent=agent_name(agent), model=agent.get("model_name"), dataset=dataset)
        runs.setdefault(key, ids.run_uuid(job_id, key.agent, key.model, key.dataset))
        attempts[(key, task_name)] += 1
    run_tasks: dict[RunKey, list[str]] = defaultdict(list)
    run_attempts: dict[RunKey, int] = defaultdict(int)
    for (key, task_name), count in attempts.items():
        run_tasks[key].append(task_name)
        run_attempts[key] = max(run_attempts[key], count)
    harbor = job_lock.get("harbor") or {}
    return JobPlan(
        job_id=str(job_id),
        job_name=str(job_config.get("job_name") or ""),
        datasets=datasets,
        runs=runs,
        run_tasks={key: sorted(tasks) for key, tasks in run_tasks.items()},
        run_attempts=dict(run_attempts),
        harbor_version=harbor.get("version"),
        _dataset_by_task=by_task,
    )


def task_local_path(task_lock: Mapping[str, Any]) -> Path | None:
    """Where Harbor keeps the task folder on this machine, if it can tell."""

    try:
        from harbor.models.trial.config import TaskConfig
    except ImportError:  # pragma: no cover - harbor is a hard dependency
        return Path(task_lock["path"]) if task_lock.get("path") else None
    try:
        if task_lock.get("type") == "package":
            config = TaskConfig(name=task_lock["name"], ref=task_lock.get("digest"))
        elif task_lock.get("git_url"):
            config = TaskConfig(
                git_url=task_lock["git_url"],
                git_commit_id=task_lock.get("git_commit_id"),
                path=task_lock.get("path"),
            )
        else:
            config = TaskConfig(path=task_lock.get("path"))
        return config.get_local_path()
    except Exception:
        return None


def read_instruction(task: TaskEntry) -> str | None:
    """``instruction.md`` from the task folder; None when the folder is gone."""

    folder = task_local_path(task.lock)
    return _read_text(folder / "instruction.md") if folder is not None else None


def case_content(task: TaskEntry, instruction: str | None) -> dict[str, Any]:
    """One dataset case per task. No expected output: Harbor's verifier decides."""

    return {
        "input": {"task": task.name, "instruction": instruction},
        "metadata": {
            "source": "harbor",
            "harborTask": task.name,
            "harborTaskDigest": task.digest,
            "harborTaskType": task.lock.get("type"),
            "harborTaskVersion": task.lock.get("version"),
            "harborDataset": task.source,
            "instructionAvailable": instruction is not None,
        },
        "tags": ["harbor"],
    }


def dataset_body(
    plan: DatasetPlan, instructions: Mapping[str, str | None], max_bytes: int = MAX_BODY_BYTES
) -> dict[str, Any]:
    """``POST /datasets`` body for one dataset version, within the API's body limit.

    Cases are ordered by task name, so their positions map back to tasks. When
    the instructions do not fit, each is shortened to an equal share; the full
    text stays in the trial traces.
    """

    tasks = [plan.tasks[name] for name in sorted(plan.tasks)]
    body: dict[str, Any] = {
        "name": plan.name[:255],
        "description": f"Harbor tasks synced by neatlogs-harbor. Version {plan.version_key}",
        "idempotencyKey": ids.dataset_publication_uuid(plan.name, plan.version_key),
        "cases": [case_content(t, instructions.get(t.name)) for t in tasks],
    }
    if _size(body) <= max_bytes:
        return body
    empty = _size({**body, "cases": [case_content(t, "") for t in tasks]})
    share = max(0, (max_bytes - empty) // max(1, len(tasks)))
    while True:
        body["cases"] = [case_content(t, _shorten(instructions.get(t.name), share)) for t in tasks]
        if _size(body) <= max_bytes or share == 0:
            return body
        share //= 2


def _shorten(text: str | None, limit: int) -> str | None:
    if text is None or len(json.dumps(text)) <= limit:
        return text
    marker = "\n…[shortened; the full instruction is in each trial's trace]"
    keep = max(0, limit - len(marker) - 16)
    while keep and len(json.dumps(text[:keep] + marker)) > limit:
        keep = int(keep * 0.9)
    return text[:keep] + marker if keep else ""


def _size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


# ---------------------------------------------------------------- trials


@dataclass
class TrialRecord:
    """One finished Harbor trial (one task attempt), from disk or from a hook."""

    job_id: str
    trial_name: str
    task_name: str
    task_source: str | None
    task_digest: str | None
    agent: str
    model: str | None
    result: dict[str, Any]
    trial_dir: Path | None = None

    @property
    def trial_uuid(self) -> str:
        return ids.trial_uuid(self.job_id, self.task_name, self.trial_name)

    @property
    def started_at(self) -> datetime | None:
        return parse_time(self.result.get("started_at"))

    @property
    def finished_at(self) -> datetime | None:
        return parse_time(self.result.get("finished_at"))

    def run_key(self, plan: JobPlan) -> RunKey:
        dataset = plan.dataset_for(self.task_source, self.task_name)
        return RunKey(agent=self.agent, model=self.model, dataset=dataset)


def trial_record(
    *,
    job_id: str,
    result: Mapping[str, Any],
    lock: Mapping[str, Any] | None,
    trial_dir: Path | None,
) -> TrialRecord:
    config = result.get("config") or {}
    lock = lock or {}
    task_lock = lock.get("task") or {}
    agent_config = lock.get("agent") or config.get("agent") or {}
    task_config = config.get("task") or {}
    return TrialRecord(
        job_id=str(job_id),
        trial_name=str(result.get("trial_name") or config.get("trial_name")),
        task_name=str(task_lock.get("name") or result.get("task_name")),
        task_source=task_lock.get("source", task_config.get("source")),
        task_digest=task_lock.get("digest"),
        agent=agent_name(agent_config),
        model=agent_config.get("model_name"),
        result=dict(result),
        trial_dir=trial_dir,
    )


def load_trial_dir(trial_dir: Path, job_id: str | None = None) -> TrialRecord | None:
    """A finished trial from its folder, or None when it has no ``result.json``."""

    result = read_json(trial_dir / "result.json")
    if not isinstance(result, dict) or not result.get("trial_name"):
        return None
    lock = read_json(trial_dir / "lock.json")
    owner = job_id or (result.get("config") or {}).get("job_id")
    if not owner:
        return None
    return trial_record(
        job_id=str(owner),
        result=result,
        lock=lock if isinstance(lock, dict) else None,
        trial_dir=trial_dir,
    )


def assign_attempts(trials: Iterable[TrialRecord], plan: JobPlan) -> dict[str, int]:
    """Attempt numbers per (agent, model, dataset, task), ordered by start time.

    Harbor records no attempt number. Ordering by ``started_at`` (then trial
    name) is stable: a trial's number only depends on attempts that started
    before it, which have always started by the time it ends.
    """

    groups: dict[tuple[RunKey, str], list[TrialRecord]] = defaultdict(list)
    for trial in trials:
        groups[(trial.run_key(plan), trial.task_name)].append(trial)
    attempts: dict[str, int] = {}
    floor = datetime.min.replace(tzinfo=timezone.utc)
    for members in groups.values():
        members.sort(key=lambda t: (t.started_at or floor, t.trial_name))
        for index, trial in enumerate(members, start=1):
            attempts[trial.trial_name] = index
    return attempts


# ---------------------------------------------------------------- outcome


def _exceptions(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    found = [result["exception_info"]] if result.get("exception_info") else []
    for step in result.get("step_results") or []:
        if step.get("exception_info"):
            found.append(step["exception_info"])
    return found


def _policy(exception_type: str) -> tuple[str, bool]:
    """(failure cause, reward stays valid) from Harbor's error policy table."""

    from harbor.models.trial.error_policy import policy_for

    policy = policy_for(exception_type)
    return policy.cause.value, policy.reward_valid


def rewards_of(result: Mapping[str, Any], trial_dir: Path | None = None) -> dict[str, float]:
    """Rewards as Harbor recorded them in ``result.json``.

    Multi-step trials record rewards per step; the last step's are the trial's.
    Only when ``result.json`` has no verifier section at all (older layouts) are
    ``verifier/reward.json`` or ``verifier/reward.txt`` read instead.
    """

    verifier = result.get("verifier_result") or {}
    rewards = verifier.get("rewards") or {}
    if not rewards:
        steps = [s for s in result.get("step_results") or [] if s.get("verifier_result")]
        if steps:
            rewards = (steps[-1]["verifier_result"] or {}).get("rewards") or {}
    if not rewards and "verifier_result" not in result and trial_dir is not None:
        rewards = _reward_files(trial_dir)
    return {str(k): float(v) for k, v in rewards.items() if isinstance(v, (int, float))}


def _reward_files(trial_dir: Path) -> dict[str, Any]:
    data = read_json(trial_dir / "verifier" / "reward.json")
    if isinstance(data, dict):
        return data
    text = _read_text(trial_dir / "verifier" / "reward.txt")
    try:
        return {"reward": float(text.strip())} if text else {}
    except ValueError:
        return {}


@dataclass
class Outcome:
    rewards: dict[str, float]
    reward_valid: bool
    failure_cause: str | None
    error_class: str | None
    exception_type: str | None
    exception_message: str | None

    @property
    def reward(self) -> float | None:
        if "reward" in self.rewards:
            return self.rewards["reward"]
        if len(self.rewards) == 1:
            return next(iter(self.rewards.values()))
        return None

    @property
    def solved(self) -> bool | None:
        """Solved when the main reward is 1 on a 0/1 scale; None otherwise."""

        if not self.reward_valid or self.reward is None or self.reward not in (0.0, 1.0):
            return None
        return self.reward == 1.0


def outcome_of(result: Mapping[str, Any], trial_dir: Path | None = None) -> Outcome:
    """Rewards plus Harbor's error classification for one trial.

    * ``agent``: the agent ran out of budget or refused; graded and counted.
    * ``grader``: the verifier failed; the reward is not valid.
    * ``infrastructure``: sandbox, model provider, Harbor or cancellation; no
      valid reward, kept out of the solve rate as Harbor does.
    """

    rewards = rewards_of(result, trial_dir)
    exceptions = _exceptions(result)
    policies = [_policy(str(e.get("exception_type"))) for e in exceptions]
    reward_valid = bool(rewards) and all(valid for _, valid in policies)
    cause, error_class = None, None
    first = exceptions[0] if exceptions else None
    if policies:
        cause, valid = policies[0]
        if valid:
            error_class = "agent"
        elif cause == "verifier_failed":
            error_class = "grader"
        else:
            error_class = "infrastructure"
    return Outcome(
        rewards=rewards,
        reward_valid=reward_valid,
        failure_cause=cause,
        error_class=error_class,
        exception_type=str(first.get("exception_type")) if first else None,
        exception_message=str(first.get("exception_message")) if first else None,
    )


# ---------------------------------------------------------------- output payload


def _seconds(timing: Mapping[str, Any] | None) -> float | None:
    if not timing:
        return None
    start, end = parse_time(timing.get("started_at")), parse_time(timing.get("finished_at"))
    if start is None or end is None or end < start:
        return None
    return round((end - start).total_seconds(), 3)


def token_totals(result: Mapping[str, Any]) -> dict[str, float | None]:
    contexts = [result["agent_result"]] if result.get("agent_result") else []
    if not contexts:
        contexts = [
            s["agent_result"] for s in result.get("step_results") or [] if s.get("agent_result")
        ]
    totals: dict[str, float | None] = {
        "input_tokens": None,
        "cache_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
    }
    for ctx in contexts:
        for key, field_name in (
            ("input_tokens", "n_input_tokens"),
            ("cache_tokens", "n_cache_tokens"),
            ("output_tokens", "n_output_tokens"),
            ("cost_usd", "cost_usd"),
        ):
            value = ctx.get(field_name)
            if isinstance(value, (int, float)):
                totals[key] = (totals[key] or 0) + value
    return totals


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds * 1000.0, 1)


def trial_metrics(result: Mapping[str, Any], outcome: Outcome) -> dict[str, float]:
    """Numeric metrics for the run output, in the API's metric names.

    ``inputTokens``, ``outputTokens``, ``totalTokens``, ``costUsd``,
    ``durationMs`` and ``latencyMs`` are the names Neatlogs aggregates. Rewards
    are only reported for trials with a valid reward, so averages over them
    leave infrastructure failures out, as Harbor's own stats do.
    """

    metrics: dict[str, float] = {"rewardValid": 1.0 if outcome.reward_valid else 0.0}
    if outcome.reward_valid:
        for key, value in outcome.rewards.items():
            metrics["reward" if key == "reward" else f"reward.{key}"[:100]] = value
        if "reward" not in outcome.rewards and outcome.reward is not None:
            metrics["reward"] = outcome.reward
        if outcome.solved is not None:
            metrics["solved"] = 1.0 if outcome.solved else 0.0
    tokens = token_totals(result)
    if tokens["input_tokens"] is not None:
        metrics["inputTokens"] = float(tokens["input_tokens"])
    if tokens["output_tokens"] is not None:
        metrics["outputTokens"] = float(tokens["output_tokens"])
    if tokens["input_tokens"] is not None or tokens["output_tokens"] is not None:
        metrics["totalTokens"] = float(
            (tokens["input_tokens"] or 0) + (tokens["output_tokens"] or 0)
        )
    if tokens["cache_tokens"] is not None:
        metrics["cacheTokens"] = float(tokens["cache_tokens"])
    if tokens["cost_usd"] is not None:
        metrics["costUsd"] = float(tokens["cost_usd"])
    for key, value in (
        ("durationMs", _ms(_seconds(result))),
        ("latencyMs", _ms(_seconds(result.get("agent_execution")))),
        ("environmentSetupMs", _ms(_seconds(result.get("environment_setup")))),
        ("agentSetupMs", _ms(_seconds(result.get("agent_setup")))),
        ("verifierMs", _ms(_seconds(result.get("verifier")))),
    ):
        if value is not None:
            metrics[key] = value
    return metrics


def output_body(
    trial: TrialRecord,
    outcome: Outcome,
    *,
    case_id: str,
    attempt: int,
    trace_id: str | None,
    trajectory: Mapping[str, Any] | None,
    attachments: Mapping[str, Any],
    max_bytes: int = MAX_BODY_BYTES,
) -> dict[str, Any]:
    """``POST /runs/{id}/outputs`` body for one trial (unredacted; redact before sending).

    Every trial is sent as a value output, including ones that errored: the
    error class travels in the output, and a failure-shaped output would make
    the whole run unusable as a comparison baseline.
    """

    result = trial.result
    answer = final_agent_message(trajectory)
    harbor: dict[str, Any] = {
        "trialName": trial.trial_name,
        "trialId": result.get("id"),
        "task": trial.task_name,
        "taskDigest": trial.task_digest,
        "agent": trial.agent,
        "agentVersion": (result.get("agent_info") or {}).get("version"),
        "model": trial.model,
        "attempt": attempt,
        "reward": outcome.reward if outcome.reward_valid else None,
        "rewards": outcome.rewards,
        "rewardValid": outcome.reward_valid,
        "errorClass": outcome.error_class,
        "failureCause": outcome.failure_cause,
        "exceptionType": outcome.exception_type,
        "exceptionMessage": (outcome.exception_message or "")[:1000] or None,
        "startedAt": result.get("started_at"),
        "finishedAt": result.get("finished_at"),
        "steps": _agent_steps(trajectory),
    }
    output: dict[str, Any] = {
        "answer": answer[:4000] if answer else None,
        "traceId": trace_id,
        "harbor": harbor,
    }
    if attachments:
        output["verifier"] = dict(attachments)
    body = {
        "caseId": case_id,
        "trial": attempt,
        "output": output,
        "metrics": trial_metrics(result, outcome),
    }
    for drop in ("test_stdout_tail", "ctrf", "reward_details"):
        if _size(body) <= max_bytes:
            break
        output.get("verifier", {}).pop(drop, None)
    if _size(body) > max_bytes and output["answer"]:
        output["answer"] = output["answer"][:500]
    return body


def _agent_steps(trajectory: Mapping[str, Any] | None) -> int | None:
    if not trajectory:
        return None
    return sum(1 for step in trajectory.get("steps") or [] if step.get("source") == "agent")


def final_agent_message(trajectory: Mapping[str, Any] | None) -> str | None:
    if not trajectory:
        return None
    for step in reversed(trajectory.get("steps") or []):
        if step.get("source") == "agent" and step.get("message"):
            message = step["message"]
            if isinstance(message, list):
                return "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in message
                )
            return str(message)
    return None


def verifier_attachments(trial_dir: Path | None) -> dict[str, Any]:
    """Verifier files written during the trial: RewardKit details, ctrf and test output.

    Only files that exist when the trial ends are read, so the plugin and a later
    import send the same output. (``analysis.json`` from ``harbor analyze`` comes
    later and is left for Phase 2's review results.)
    """

    if trial_dir is None:
        return {}
    found: dict[str, Any] = {}
    for key, path in (
        ("reward_details", trial_dir / "verifier" / "reward-details.json"),
        ("ctrf", trial_dir / "verifier" / "ctrf.json"),
    ):
        data = read_json(path)
        if data is not None and len(json.dumps(data)) <= MAX_ATTACHMENT_CHARS:
            found[key] = data
    stdout = _read_text(trial_dir / "verifier" / "test-stdout.txt")
    if stdout:
        found["test_stdout_tail"] = "\n".join(stdout.splitlines()[-200:])[-MAX_ATTACHMENT_CHARS:]
    return found


def read_trajectory(trial_dir: Path | None) -> dict[str, Any] | None:
    if trial_dir is None:
        return None
    data = read_json(trial_dir / "agent" / "trajectory.json")
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------- io helpers


def parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_time(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text()
    except (OSError, UnicodeDecodeError):
        return None
