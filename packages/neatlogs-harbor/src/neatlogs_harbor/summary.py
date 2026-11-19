"""The short terminal summary printed when a job ends (or a sync finishes)."""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from typing import Any, TextIO

from .records import TrialRecord, outcome_of


def pass_at_k(attempts: dict[str, list[bool]], k: int) -> float | None:
    """Harbor's unbiased estimator, averaged over tasks: 1 - C(n-c, k) / C(n, k)."""

    scores = []
    for results in attempts.values():
        n, c = len(results), sum(results)
        if n < k:
            continue
        scores.append(1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k))
    return sum(scores) / len(scores) if scores else None


@dataclass
class RunSummary:
    name: str
    dataset: str
    n_trials: int
    n_valid: int
    n_solved: int | None
    mean_reward: float | None
    n_no_reward: int
    cost_usd: float | None
    pass_k: tuple[int, float] | None
    all_solved: float | None
    run_id: str | None
    run_state: str | None
    comparison_line: str | None
    link: str | None
    problem: str | None = None

    @classmethod
    def build(
        cls,
        *,
        name: str,
        dataset: str,
        trials: list[TrialRecord],
        run_id: str | None,
        run_state: str | None,
        skipped_reason: str | None,
        error: str | None,
        comparison: dict[str, Any] | None,
        comparison_error: Exception | None,
        link: str | None,
        left_open: int = 0,
    ) -> RunSummary:
        valid, solved, rewards, cost = 0, 0, [], None
        binary = True
        per_task: dict[str, list[bool]] = defaultdict(list)
        for trial in trials:
            outcome = outcome_of(trial.result, trial.trial_dir)
            cost_value = (trial.result.get("agent_result") or {}).get("cost_usd")
            if isinstance(cost_value, (int, float)):
                cost = (cost or 0.0) + cost_value
            if not outcome.reward_valid:
                continue
            valid += 1
            if outcome.reward is not None:
                rewards.append(outcome.reward)
            if outcome.solved is None:
                binary = False
            else:
                solved += int(outcome.solved)
                per_task[trial.task_name].append(outcome.solved)
        attempts = max((len(v) for v in per_task.values()), default=0)
        pass_k = None
        all_solved = None
        if binary and attempts > 1:
            score = pass_at_k(per_task, attempts)
            if score is not None:
                pass_k = (attempts, score)
            full = [all(v) for v in per_task.values() if len(v) == attempts]
            all_solved = sum(full) / len(full) if full else None
        return cls(
            name=name,
            dataset=dataset,
            n_trials=len(trials),
            n_valid=valid,
            n_solved=solved if binary else None,
            mean_reward=sum(rewards) / len(rewards) if rewards else None,
            n_no_reward=len(trials) - valid,
            cost_usd=cost,
            pass_k=pass_k,
            all_solved=all_solved,
            run_id=run_id,
            run_state=run_state,
            comparison_line=(
                f"run left open: {left_open} output(s) still to send"
                if left_open
                else _comparison_line(comparison, comparison_error)
            ),
            link=link,
            problem=skipped_reason or (error if not run_id else None),
        )


def _comparison_line(comparison: dict[str, Any] | None, error: Exception | None) -> str | None:
    if error is not None:
        code = getattr(error, "code", None)
        if code == "BASELINE_NOT_FOUND":
            return "first finished Harbor run on this dataset version"
        if code == "INCOMPATIBLE_RUNS":
            return "previous run used a different grading form; not compared"
        return f"comparison unavailable ({error})"
    if not comparison:
        return None
    state = comparison.get("state")
    summary = comparison.get("summary") or {}
    if state != "ready":
        return "grading in progress; the comparison fills in on the dataset page"
    changed, unchanged = summary.get("changed"), summary.get("unchanged")
    if changed is None:
        return "ready"
    return f"{changed} changed, {unchanged} unchanged across {summary.get('pairedOutputs')} pairs"


@dataclass
class Summary:
    runs: list[RunSummary]
    pending_trials: int
    ledger_path: Path
    job_dir: Path
    notes: list[str] = field(default_factory=list)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render(summary: Summary) -> str:
    lines = ["", "Neatlogs summary"]

    def row(label: str, value: str) -> None:
        lines.append(f"    {label:<14}{value}")

    for run in summary.runs:
        lines.append(f"  {run.name}  ·  {run.dataset}")
        if run.n_solved is not None and run.n_valid:
            row(
                "solved",
                f"{run.n_solved} / {run.n_valid} trials  {_pct(run.n_solved / run.n_valid)}",
            )
        elif run.mean_reward is not None:
            row("mean reward", f"{run.mean_reward:.3f} over {run.n_valid} trials")
        if run.pass_k is not None:
            k, score = run.pass_k
            all_k = (
                f"  ·  all {k} attempts solved {_pct(run.all_solved)}"
                if run.all_solved is not None
                else ""
            )
            row(f"pass@{k}", f"{_pct(score)}{all_k}")
        if run.cost_usd is not None:
            per = (
                f"  ·  ${run.cost_usd / run.n_solved:.2f} per solved trial" if run.n_solved else ""
            )
            row("cost", f"${run.cost_usd:.2f}{per}")
        if run.n_no_reward:
            row("no reward", f"{run.n_no_reward} trial(s), left out as Harbor does")
        if run.comparison_line:
            row("vs previous", run.comparison_line)
        if run.problem:
            row("not sent", run.problem)
        if run.link:
            row("open", run.link)
    for note in summary.notes:
        lines.append(f"  note: {note}")
    if summary.pending_trials:
        lines.append(
            f"  {summary.pending_trials} trial(s) not sent yet; they wait in "
            f"{summary.ledger_path}"
        )
        lines.append(f"  send them with: neatlogs harbor sync {summary.job_dir}")
    return "\n".join(lines) + "\n"


def print_summary(summary: Summary, stream: TextIO | None = None) -> None:
    (stream or sys.stdout).write(render(summary))
