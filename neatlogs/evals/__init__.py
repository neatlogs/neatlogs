"""Run a task on pinned cases and grade it with a selected evaluation form."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from contextlib import suppress
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable
from uuid import UUID, uuid4

from ..core.context import trace
from ..datasets._api import DatasetApi, encode_json
from ..replay.api import LocalRunApi
from ..replay.state import state_lock


@dataclass(frozen=True)
class Dataset:
    name: str
    version_id: str | None = None


@dataclass(frozen=True)
class EvalContext:
    expected: Any
    metadata: dict
    trial: int
    params: dict
    metrics: dict = field(default_factory=dict)

    def param(self, name, default=None):
        return self.params.get(name, default)


@dataclass(frozen=True)
class EvalResult:
    run_id: str
    comparison: dict
    gate: dict | None
    errors: int
    exit_code: int


def _with_state_lock(run):
    @wraps(run)
    def locked(*args, **kwargs):
        state_path = kwargs.get("state_path")
        if state_path is None:
            return run(*args, **kwargs)
        with state_lock(state_path):
            return run(*args, **kwargs)

    return locked


def _save(path, state):
    if path is None:
        return
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(state, handle, allow_nan=False)
    os.replace(temporary, target)


def _resolve(api, dataset, selected=None):
    try:
        dataset_id = str(UUID(dataset.name))
    except ValueError:
        matches = [
            item
            for item in api.request("GET", "/datasets")["datasets"]
            if item["name"] == dataset.name
        ]
        if len(matches) != 1:
            raise ValueError("Use a dataset UUID or one unambiguous dataset name") from None
        dataset_id = str(UUID(matches[0]["id"]))
    suffix = "?versionId=" + str(UUID(dataset.version_id)) if dataset.version_id else ""
    overview = api.request("GET", f"/datasets/{dataset_id}{suffix}")
    version_id = str(UUID(overview["version"]["id"]))
    total = overview["dataset"]["caseCount"]
    if not isinstance(total, int) or not 1 <= total <= 100000:
        raise ValueError("Dataset case count is invalid")
    if selected is None and total > 100:
        raise ValueError("Select up to 100 cases for one run")
    entries = []
    for offset in range(0, total, 100):
        page = api.request(
            "GET", f"/datasets/{dataset_id}/cases?versionId={version_id}&offset={offset}&limit=100"
        )
        if page["total"] != total or len(page["cases"]) != min(100, total - offset):
            raise ValueError("Dataset case enumeration is incomplete")
        entries.extend(page["cases"])
    if selected is not None and not set(selected).issubset({item["caseId"] for item in entries}):
        raise ValueError("Selected cases must belong to the pinned dataset")
    cases = [
        api.request(
            "GET",
            f"/datasets/{dataset_id}/cases/{str(UUID(item['caseId']))}?versionId={version_id}",
        )["case"]
        for item in entries
        if selected is None or item["caseId"] in selected
    ]
    return version_id, cases


@_with_state_lock
def run_eval(
    *,
    dataset: Dataset | str,
    task: Callable,
    label: str,
    params=None,
    trials=1,
    baseline=None,
    gates=None,
    grading=None,
    git=None,
    checks=None,
    project_id=None,
    token=None,
    base_url=None,
    wait_timeout=300,
    state_path=None,
    resume=False,
    max_error_rate=0,
    case_ids=None,
):
    """Execute each case/trial once, upload outputs, wait for existing form grading.

    ``state_path`` retains produced outputs before upload. Resume never repeats a
    task whose execution may have started without a saved result. Code checks
    belong to the dataset's eval form and run in the existing isolated runner.
    ``exit_code`` is 0 for pass, 1 for failed gate, and 4 for incomplete runs.
    """
    dataset = Dataset(dataset) if isinstance(dataset, str) else dataset
    if (
        not isinstance(dataset, Dataset)
        or not callable(task)
        or not isinstance(label, str)
        or not label.strip()
    ):
        raise ValueError("Provide a dataset, callable task and nonempty label")
    if (
        type(trials) is not int
        or not 1 <= trials <= 10
        or not 0 <= max_error_rate <= 1
        or wait_timeout <= 0
    ):
        raise ValueError("Use trials 1–10, max_error_rate 0–1 and a positive wait timeout")
    if resume and state_path is None:
        raise ValueError("Resume requires the original state_path")
    config = dict(
        project_id=project_id or os.getenv("NEATLOGS_PROJECT_ID", ""),
        token=token or os.getenv("NEATLOGS_TOKEN", ""),
        base_url=base_url or os.getenv("NEATLOGS_HOST", "https://app.neatlogs.com"),
    )
    dataset_api, run_api = DatasetApi(**config), LocalRunApi(**config)
    saved_state = json.loads(Path(state_path).read_text()) if resume else None
    if saved_state is not None:
        dataset = Dataset(dataset.name, version_id=saved_state["request"]["datasetVersionId"])
    selected = None
    if case_ids is not None:
        selected = [str(UUID(value)) for value in case_ids]
        if not selected or len(selected) > 100 or len(set(selected)) != len(selected):
            raise ValueError("Select between 1 and 100 unique cases")
    version_id, cases = _resolve(dataset_api, dataset, selected)
    request = {
        "datasetVersionId": version_id,
        "caseIds": [case["caseId"] for case in cases],
        "label": label,
        "params": params or {},
        "trials": trials,
        "deadlineSeconds": 900,
    }
    for key, value in [
        ("grading", grading),
        ("baseline", baseline),
        ("gates", gates),
        ("git", git),
        ("checks", checks),
    ]:
        if value is not None:
            request[key] = value
    encode_json(request)
    try:
        task_source = inspect.getsource(task)
    except (OSError, TypeError):
        task_source = f"{getattr(task, '__module__', '')}:{getattr(task, '__qualname__', type(task).__qualname__)}"
    task_digest = hashlib.sha256(task_source.encode()).hexdigest()
    if resume:
        state = saved_state
        if (
            state.get("request") != request
            or state.get("projectId") != config["project_id"]
            or state.get("taskDigest") != task_digest
        ):
            raise ValueError(
                "Resume must use the same project, pinned dataset and run configuration"
            )
    else:
        if state_path is not None and Path(state_path).exists():
            raise ValueError("State file exists; use resume=True or a new path")
        state = {
            "request": request,
            "projectId": config["project_id"],
            "taskDigest": task_digest,
            "registrationKey": str(uuid4()),
            "runId": None,
            "slots": {},
        }
        _save(state_path, state)
    registered_id = state["runId"]
    if registered_id is None:
        registered_id = run_api.register(request, key=state["registrationKey"])["runId"]
    run_id = str(UUID(registered_id))
    receipt = os.environ.get("NEATLOGS_EVAL_RESULT_FILE")
    if receipt:
        with suppress(OSError):
            fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                file.write(run_id)
    if state["runId"] is None:
        state["runId"] = run_id
        _save(state_path, state)
    for case in cases:
        for trial in range(1, trials + 1):
            key = f"{case['caseId']}:{trial}"
            slot = state["slots"].get(key)
            if slot and slot.get("accepted"):
                continue
            if slot and slot.get("executing"):
                raise RuntimeError(
                    "Task execution may have started without a saved output. Inspect the task before starting a new run"
                )
            if slot is None:
                slot = {"executing": True, "key": str(uuid4())}
                state["slots"][key] = slot
                _save(state_path, state)
                context = EvalContext(
                    case.get("expected"), case.get("metadata", {}), trial, params or {}
                )
                started = time.monotonic()
                body = {"caseId": case["caseId"], "trial": trial}
                try:
                    with trace(
                        label,
                        kind="WORKFLOW",
                        **{
                            "neatlogs.origin": "simulation",
                            "neatlogs.replay.run_id": run_id,
                            "neatlogs.replay.case_id": case["caseId"],
                            "neatlogs.replay.trial": trial,
                        },
                    ) as span:
                        span.set_attribute("input.value", json.dumps(case["input"]))
                        output = task(case["input"], context)
                        encode_json({"output": output})
                        span.set_attribute("output.value", json.dumps(output))
                    body.update(
                        output=output,
                        metrics={
                            **context.metrics,
                            "latencyMs": (time.monotonic() - started) * 1000,
                        },
                    )
                except Exception as error:
                    body["error"] = {
                        "code": type(error).__name__[:100],
                        "message": str(error)[:1000],
                    }
                slot.update(executing=False, body=body)
                _save(state_path, state)
            run_api.request("POST", f"/runs/{run_id}/outputs", slot["body"], key=slot["key"])
            slot["accepted"] = True
            _save(state_path, state)
    complete_key = state.setdefault("completeKey", str(uuid4()))
    _save(state_path, state)
    run_api.request("POST", f"/runs/{run_id}/complete", {}, key=complete_key)
    _save(state_path, state)
    errors = sum("error" in slot["body"] for slot in state["slots"].values())
    deadline = time.monotonic() + wait_timeout
    if grading:
        status = run_api.request("GET", f"/runs/{run_id}?includeGrading=true")
        if status.get("runId") != run_id:
            raise ValueError("Grading status belongs to a different run")
    if grading and "gradingHandoffState" not in status:
        while True:
            status = run_api.request("GET", f"/runs/{run_id}?includeGrading=true")
            if status.get("runId") != run_id:
                raise ValueError("Grading status belongs to a different run")
            batches = status.get("grading")
            if (
                batches
                or status.get("gradingHandoffState") == "failed"
                or time.monotonic() >= deadline
            ):
                break
            time.sleep(min(1, max(0, deadline - time.monotonic())))
        if not isinstance(batches, dict):
            raise RuntimeError(
                "Grading batches are not ready. Resume after checking the grading worker."
            )
        evaluation_id = str(UUID(batches["evaluationId"]))
        recorded_id = str(UUID(batches["recordedBatchId"]))
        candidate_id = str(UUID(batches["candidateBatchId"]))
        dispatch = state.setdefault(
            "gradingDispatch",
            {
                "evaluationId": evaluation_id,
                "recorded": {"batchId": recorded_id, "key": str(uuid4()), "accepted": False},
                "candidate": {"batchId": candidate_id, "key": str(uuid4()), "accepted": False},
            },
        )
        _save(state_path, state)
        if (
            dispatch["evaluationId"],
            dispatch["recorded"]["batchId"],
            dispatch["candidate"]["batchId"],
        ) != (evaluation_id, recorded_id, candidate_id):
            raise ValueError("Saved grading batches do not match this run")
        for batch in (dispatch["recorded"], dispatch["candidate"]):
            if not batch["accepted"]:
                result = run_api.request(
                    "POST",
                    f"/evaluations/{evaluation_id}/ai-dispatches",
                    {"batchId": batch["batchId"]},
                    key=batch["key"],
                    timeout=120,
                )
                if (
                    result.get("evaluationId") != evaluation_id
                    or result.get("batchId") != batch["batchId"]
                    or result.get("dispatchStatus") != "accepted"
                ):
                    raise ValueError("Grading receipt belongs to a different batch")
                batch["accepted"] = True
                _save(state_path, state)
    while True:
        comparison = run_api.request("GET", f"/runs/{run_id}/comparison", limit=17 * 1024 * 1024)
        if (
            comparison.get("state") in {"ready", "failed", "incomplete"}
            or time.monotonic() >= deadline
        ):
            break
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    if errors / (len(cases) * trials) > max_error_rate or not comparison.get("coverage", {}).get(
        "complete"
    ):
        return EvalResult(run_id, comparison, None, errors, 4)
    gate_key = state.setdefault("gateKey", str(uuid4()))
    _save(state_path, state)
    gate = run_api.request("POST", f"/runs/{run_id}/gate", {}, key=gate_key)
    code = {"pass": 0, "fail": 1, "incomplete": 4}[gate["verdict"]]
    return EvalResult(run_id, comparison, gate, errors, code)


__all__ = ["Dataset", "EvalContext", "EvalResult", "run_eval"]
