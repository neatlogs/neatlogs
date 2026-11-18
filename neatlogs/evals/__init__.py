"""Run a task on pinned cases and grade it with a selected evaluation form."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable
from uuid import UUID, uuid4

from .._task_credentials import task_credential_refresh
from .._wrap_utils import fresh_neatlogs_context, get_active_client, get_bootstrap_provider
from ..core.context import trace
from ..core.deadline import bounded_call
from ..datasets._api import DatasetApi, encode_json
from ..init import flush
from ..replay.api import LocalRunApi, LocalRunApiError
from ..replay.contracts import MAX_RECORDING_BYTES
from ..replay.environment import write_private_json
from ..replay.provider import ecmascript_json
from ..replay.state import state_lock
from ._recovery import is_uncertain, recover, terminal, uncertainty_message


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
    run_state: str | None = None


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
    write_private_json(path, state, limit=MAX_RECORDING_BYTES)


def _validate_grading(grading):
    message = "Select valid grading with a formVersionId and 1–20 evaluatorIds"
    if not isinstance(grading, dict) or set(grading) != {
        "formVersionId",
        "evaluatorIds",
    }:
        raise ValueError(message)
    evaluators = grading["evaluatorIds"]
    if not isinstance(evaluators, list) or not 1 <= len(evaluators) <= 20:
        raise ValueError(message)
    try:
        for value in [grading["formVersionId"], *evaluators]:
            if not isinstance(value, str):
                raise ValueError(message)
            parsed = UUID(value)
            if str(parsed) != value.lower() or (
                parsed.int not in {0, (1 << 128) - 1} and parsed.version not in range(1, 9)
            ):
                raise ValueError(message)
    except ValueError:
        raise ValueError(message) from None


def _resolve(api, dataset, selected=None):
    try:
        dataset_id = str(UUID(dataset.name))
    except ValueError:
        matches, seen = set(), set()
        cursor = None
        while True:
            page = api.request("GET", "/datasets" + (f"?before={cursor}" if cursor else ""))
            entries = page["datasets"]
            if not isinstance(entries, list) or len(entries) > 500:
                raise ValueError("Dataset catalog page is invalid")
            matches.update(
                str(UUID(item["id"])) for item in entries if item["name"] == dataset.name
            )
            if len(matches) > 1:
                raise ValueError("Use a dataset UUID or one unambiguous dataset name")
            cursor = page.get("nextCursor")
            if ("nextCursor" not in page and len(entries) == 500) or (cursor and cursor in seen):
                raise ValueError("Dataset catalog pagination is incomplete; use a dataset UUID")
            if cursor is None:
                break
            cursor = str(UUID(cursor))
            seen.add(cursor)
        if not matches:
            raise ValueError("Use a dataset UUID or one unambiguous dataset name")
        dataset_id = next(iter(matches))
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
            "GET",
            f"/datasets/{dataset_id}/cases?versionId={version_id}&offset={offset}&limit=100",
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


def _utc_case_time():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _flush_case_trace(timeout_millis=10_000):
    client = get_active_client()
    if client:
        return client.flush(timeout_millis=timeout_millis)
    provider = get_bootstrap_provider()
    if provider is None:
        return flush(timeout_millis=timeout_millis)
    completed, result = bounded_call(
        lambda: provider.force_flush(timeout_millis=timeout_millis),
        time.monotonic() + timeout_millis / 1000,
    )
    return bool(result) if completed else False


def _upload_case_output(run_api, run_id, slot, deadline, save_state):
    # Reserve a final value-upload/reconciliation window before the run closes.
    trace_deadline = min(deadline - 5, time.monotonic() + 60)
    if slot["body"].get("traceId"):
        remaining = min(10, trace_deadline - time.monotonic())
        if remaining > 0:
            _flush_case_trace(timeout_millis=max(1, int(remaining * 1000)))
    while True:
        if slot["body"].get("traceId") and time.monotonic() >= trace_deadline:
            slot["body"].pop("traceId")
            (slot.get("rejection") or slot)["key"] = str(uuid4())
            save_state()
        rejection = slot.get("rejection")
        body = slot["body"]
        key = slot["key"]
        if rejection:
            body = {
                name: value for name, value in body.items() if name not in ("output", "metrics")
            }
            body["error"] = rejection["error"]
            key = rejection["key"]
        try:

            def deliver(timeout):
                evidence_remaining = trace_deadline - time.monotonic()
                # Reconciliation after an ambiguous write keeps the original body/key.
                if body.get("traceId") and evidence_remaining > 0:
                    timeout = min(timeout, evidence_remaining)
                return run_api.request(
                    "POST", f"/runs/{run_id}/outputs", body, key=key, timeout=timeout
                )

            _, stopped = recover(
                deliver,
                deadline,
                run_api=run_api,
                run_id=run_id,
            )
            return stopped
        except LocalRunApiError as error:
            if (
                body.get("traceId")
                and error.code == "TRACE_NOT_READY"
                and time.monotonic() < trace_deadline
            ):
                time.sleep(min(1, max(0, trace_deadline - time.monotonic())))
                continue
            if body.get("traceId") and error.code in (
                "TRACE_NOT_READY",
                "INVALID_TRACE_IDENTITY",
            ):
                slot["body"].pop("traceId")
                (rejection or slot)["key"] = str(uuid4())
            elif (
                not rejection
                and "output" in body
                and error.status == 413
                and error.code == "GRADING_EVIDENCE_LIMIT"
            ):
                slot["rejection"] = {
                    "key": str(uuid4()),
                    "error": {
                        "code": "GRADING_EVIDENCE_LIMIT",
                        "message": "Saved input and task output exceed the grading evidence limit.",
                    },
                }
            else:
                raise
            save_state()


@_with_state_lock
def run_eval(
    *,
    dataset: Dataset | str,
    task: Callable,
    label: str,
    grading: dict,
    params=None,
    trials=1,
    baseline=None,
    gates=None,
    git=None,
    local=None,
    checks=None,
    project_id=None,
    token=None,
    base_url=None,
    wait_timeout=300,
    state_path=None,
    resume=False,
    max_error_rate=0,
    deadline_seconds=900,
    case_ids=None,
):
    """Execute each case/trial once, upload outputs, wait for existing form grading.

    ``state_path`` retains produced outputs before upload. Resume never repeats a
    task whose execution may have started without a saved result. Code checks
    belong to the dataset's eval form and run in the existing isolated runner.
    ``exit_code`` is 0 for pass, 1 for failed gate, and 4 for incomplete runs.
    ``comparison`` is one page with global coverage; use LocalRunApi.comparison
    with its nextCursor for remaining rows. Gates cover the entire run.
    """
    _validate_grading(grading)
    if max_error_rate != 0:
        raise ValueError(
            "max_error_rate must be zero: graded runs require every selected case to complete"
        )
    if type(deadline_seconds) is not int or not 60 <= deadline_seconds <= 3600:
        raise ValueError("deadline_seconds must be from 60 to 3600")
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
    config["refresh_access_token"] = task_credential_refresh(**config)
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
    if local is not None:
        if not isinstance(local, dict) or set(local) != {
            "experimentId",
            "roundId",
            "variant",
            "attempt",
        }:
            raise ValueError("Use experimentId, roundId, variant and attempt for a local round")
        for field in ("experimentId", "roundId"):
            if not isinstance(local[field], str):
                raise ValueError("Local experiment and round IDs must be UUID strings")
            UUID(local[field])
        if (
            local["variant"] not in {"A", "B"}
            or type(local["attempt"]) is not int
            or not 1 <= local["attempt"] <= 100
        ):
            raise ValueError("Use variant A or B and an attempt between 1 and 100")
    request = {
        "datasetVersionId": version_id,
        "caseIds": [case["caseId"] for case in cases],
        "label": label,
        "params": params or {},
        "trials": trials,
        "deadlineSeconds": deadline_seconds,
    }
    for key, value in [
        ("local", local),
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
    try:
        registered_id = state["runId"]
        if registered_id is None:
            registration, _ = recover(
                lambda timeout: run_api.register(
                    request, key=state["registrationKey"], timeout=timeout
                ),
                time.monotonic() + deadline_seconds,
            )
            registered_id = registration["runId"]
            state["deadlineAt"] = registration["deadlineAt"]
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
        from ..replay.cli import _remaining

        terminal_run = None
        case_output_details = False
        if not state.get("deadlineAt") or resume or local:
            current, _ = recover(
                lambda timeout: run_api.status(
                    run_id, timeout=timeout, include_case_output_protocol=bool(local)
                ),
                time.monotonic() + wait_timeout,
            )
            state["deadlineAt"] = current["deadlineAt"]
            _save(state_path, state)
            case_output_details = current.get("caseOutputProtocol") == 2
            if terminal(current):
                terminal_run = current
        server_deadline = datetime.fromisoformat(state["deadlineAt"].replace("Z", "+00:00"))
        deadline = time.monotonic() + max(
            0, (server_deadline - datetime.now(timezone.utc)).total_seconds()
        )
        for case in cases:
            if terminal_run:
                break
            for trial in range(1, trials + 1):
                key = f"{case['caseId']}:{trial}"
                slot = state["slots"].get(key)
                if slot and slot.get("accepted"):
                    continue
                if slot and slot.get("executing"):
                    raise RuntimeError(
                        "Task execution may have started without a saved output. Inspect the task before starting a new run"
                    )
                if not slot or "body" not in slot:
                    if slot is not None and not slot.get("start"):
                        raise ValueError(
                            "Saved output is missing. Inspect the private state file before resuming."
                        )
                    if _remaining(state) <= 0:
                        raise RuntimeError(
                            "Run deadline reached; no further tasks were executed. Resume only to reconcile saved outputs"
                        )
                    if (
                        state_path is not None
                        and len(json.dumps(state, allow_nan=False, separators=(",", ":")).encode())
                        + 400 * 1024
                        > MAX_RECORDING_BYTES
                    ):
                        raise RuntimeError(
                            "Run state has no room for another output; no further tasks were executed"
                        )
                    if slot is None:
                        slot = {"executing": False, "key": str(uuid4())}
                        state["slots"][key] = slot
                    if case_output_details:
                        start = slot.setdefault(
                            "start",
                            {"key": str(uuid4()), "startedAt": _utc_case_time()},
                        )
                        _save(state_path, state)
                        start_body = {
                            "caseId": case["caseId"],
                            "trial": trial,
                            "startedAt": start["startedAt"],
                        }
                        _, terminal_run = recover(
                            lambda timeout, body=start_body, key=start["key"]: (
                                run_api.request(
                                    "POST",
                                    f"/runs/{run_id}/outputs/started",
                                    body,
                                    key=key,
                                    timeout=timeout,
                                )
                            ),
                            deadline,
                            run_api=run_api,
                            run_id=run_id,
                        )
                        if terminal_run:
                            break
                    slot["executing"] = True
                    _save(state_path, state)
                    context = EvalContext(
                        copy.deepcopy(case.get("expected")),
                        copy.deepcopy(case.get("metadata", {})),
                        trial,
                        copy.deepcopy(params or {}),
                    )
                    started_at = _utc_case_time()
                    started = time.monotonic()
                    trace_id = None
                    body = {"caseId": case["caseId"], "trial": trial}
                    try:
                        task_input = copy.deepcopy(case["input"])
                        with (
                            fresh_neatlogs_context(),
                            trace(
                                label,
                                kind="WORKFLOW",
                                **{
                                    "neatlogs.origin": "simulation",
                                    "neatlogs.replay.run_id": run_id,
                                    "neatlogs.replay.case_id": case["caseId"],
                                    "neatlogs.replay.trial": trial,
                                },
                            ) as span,
                        ):
                            if case_output_details:
                                span_context = span.get_span_context()
                                if span_context.is_valid and span_context.trace_flags.sampled:
                                    trace_id = f"{span_context.trace_id:032x}"
                            span.set_attribute("input.value", json.dumps(case["input"]))
                            output = copy.deepcopy(task(task_input, context))
                            span.set_attribute("output.value", json.dumps(output))
                        body.update(
                            output=output,
                            metrics={
                                **context.metrics,
                                "latencyMs": (time.monotonic() - started) * 1000,
                            },
                        )
                        metrics = body["metrics"]
                        if any(
                            type(key) is not str or len(key.encode("utf-16-le")) > 200
                            for key in metrics
                        ):
                            raise ValueError("Task metric names must have up to 100 UTF-16 units")
                        if case_output_details:
                            body.update(
                                startedAt=started_at,
                                endedAt=max(started_at, _utc_case_time()),
                            )
                            if trace_id:
                                body["traceId"] = trace_id
                        if len(ecmascript_json(body)) > 64 * 1024:
                            raise ValueError("Complete task output request exceeds 64 KiB")
                    except Exception as error:
                        body = {
                            "caseId": case["caseId"],
                            "trial": trial,
                            "error": {
                                "code": type(error)
                                .__name__.encode("utf-16-le", errors="replace")[:200]
                                .decode("utf-16-le", errors="ignore"),
                                "message": str(error)
                                .encode("utf-16-le", errors="replace")[:2000]
                                .decode("utf-16-le", errors="ignore"),
                            },
                        }
                    if case_output_details and "error" in body:
                        body.update(
                            startedAt=started_at,
                            endedAt=max(started_at, _utc_case_time()),
                        )
                        if trace_id:
                            body["traceId"] = trace_id
                    slot.update(executing=False, body=body)
                    _save(state_path, state)
                terminal_run = _upload_case_output(
                    run_api,
                    run_id,
                    slot,
                    deadline,
                    lambda: _save(state_path, state),
                )
                if terminal_run:
                    break
                slot["accepted"] = True
                _save(state_path, state)
        status = terminal_run
        if not status:
            complete_key = state.setdefault("completeKey", str(uuid4()))
            _save(state_path, state)
            completed, recovered = recover(
                lambda timeout: run_api.request(
                    "POST",
                    f"/runs/{run_id}/complete",
                    {},
                    key=complete_key,
                    timeout=timeout,
                ),
                deadline,
                run_api=run_api,
                run_id=run_id,
            )
            status = recovered or completed
        errors = max(
            (status or {}).get("failedCount", 0),
            sum(
                "error" in slot.get("body", {}) or bool(slot.get("rejection"))
                for slot in state["slots"].values()
            ),
        )
        deadline = time.monotonic() + wait_timeout
        if grading:
            status, _ = recover(
                lambda timeout: run_api.status(run_id, include_grading=True, timeout=timeout),
                deadline,
            )
        if grading and "gradingHandoffState" not in status:
            while True:
                status, _ = recover(
                    lambda timeout: run_api.status(run_id, include_grading=True, timeout=timeout),
                    deadline,
                )
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
                    "recorded": {
                        "batchId": recorded_id,
                        "key": str(uuid4()),
                        "accepted": False,
                    },
                    "candidate": {
                        "batchId": candidate_id,
                        "key": str(uuid4()),
                        "accepted": False,
                    },
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
        # Preserve one receipt read even when handoff waiting used the poll budget.
        comparison, _ = recover(
            lambda timeout: run_api.comparison(run_id, timeout=timeout),
            max(deadline, time.monotonic() + 5),
        )
        while comparison.get("state") not in {"ready", "failed", "incomplete"}:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(1, remaining))
            if time.monotonic() >= deadline:
                break
            comparison, _ = recover(
                lambda timeout: run_api.comparison(run_id, timeout=timeout), deadline
            )
        if errors / (len(cases) * trials) > max_error_rate or not comparison.get(
            "coverage", {}
        ).get("complete"):
            return EvalResult(run_id, comparison, None, errors, 4, status.get("state"))
        if not gates:
            return EvalResult(run_id, comparison, None, errors, 0, status.get("state"))
        gate_key = state.setdefault("gateKey", str(uuid4()))
        _save(state_path, state)
        gate_deadline = max(deadline, time.monotonic() + 5)
        gate, _ = recover(
            lambda timeout: run_api.request(
                "POST",
                f"/runs/{run_id}/gate",
                {},
                key=gate_key,
                timeout=timeout,
            ),
            gate_deadline,
        )
        code = {"pass": 0, "fail": 1, "incomplete": 4}[gate["verdict"]]
        return EvalResult(run_id, comparison, gate, errors, code, status.get("state"))
    except LocalRunApiError as error:
        if is_uncertain(error):
            raise uncertainty_message(state["runId"], state_path) from error
        raise


__all__ = ["Dataset", "EvalContext", "EvalResult", "run_eval"]
