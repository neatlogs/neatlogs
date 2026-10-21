"""Enumerate a pinned dataset and resume bounded native local batches safely."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .cli import _remaining
from .contracts import MAX_RECORDING_BYTES, ReplayError, uuid
from .environment import bounded_file
from .frozen_case import replay_recording_from_case
from .local_command import parse_agent_config, run_local_command
from .session import ReplaySession
from .state import agent_digest, create_state, read_state, save_state, state_lock


def enumerate_cases(api, dataset_id, version_id):
    overview = api.request(
        "GET", f"/datasets/{dataset_id}?versionId={version_id}", limit=17 * 1024 * 1024
    )
    try:
        total = overview["dataset"]["caseCount"]
        if (
            overview["dataset"]["id"] != dataset_id
            or overview["version"]["id"] != version_id
            or type(total) is not int
            or not 1 <= total <= 2000
        ):
            raise ReplayError("The dataset API returned a different or unsupported frozen version.")
        cases, seen = [], set()
        while len(cases) < total:
            offset = len(cases)
            page = api.request(
                "GET",
                f"/datasets/{dataset_id}/cases?versionId={version_id}&offset={offset}&limit=100",
                limit=17 * 1024 * 1024,
            )
            if (
                page["total"] != total
                or type(page["cases"]) is not list
                or len(page["cases"]) != min(100, total - offset)
            ):
                raise ReplayError("The frozen dataset page is incomplete or changed.")
            for index, item in enumerate(page["cases"]):
                case_id, revision_id = uuid(item["caseId"]), uuid(item["revisionId"])
                if (
                    type(item["position"]) is not int
                    or item["position"] != offset + index
                    or case_id in seen
                ):
                    raise ReplayError("The frozen dataset has repeated or out-of-order examples.")
                seen.add(case_id)
                cases.append(
                    {"caseId": case_id, "revisionId": revision_id, "position": offset + index}
                )
        return cases
    except (KeyError, TypeError):
        raise ReplayError("The frozen dataset API response is invalid.") from None


def _upload(state, path, api, batch, slot):
    if slot["progress"]["kind"] != "upload_pending":
        return
    trace_id = slot["progress"]["traceId"]
    receipt = api.upload(
        batch["run"]["runId"],
        case_id=slot["caseId"],
        trial=slot["trial"],
        trace_id=trace_id,
        key=slot["uploadKey"],
        wait_millis=max(1, min(60_000, int(_remaining(batch["run"]) * 1000))),
    )
    slot["progress"] = {"kind": "accepted", "traceId": trace_id, "outputId": receipt["outputId"]}
    save_state(path, state)


def _slot(state, path, api, agent, batch, slot, as_json):
    kind = slot["progress"]["kind"]
    if kind == "upload_pending":
        return _upload(state, path, api, batch, slot)
    if kind != "pending":
        return
    config = state["configuration"]
    selected = next(item for item in state["cases"] if item["caseId"] == slot["caseId"])
    remaining = _remaining(batch["run"])
    if remaining < 1:
        raise ReplayError("This batch expired. Remaining examples were not executed.")
    source = api.frozen_case(config["datasetId"], config["versionId"], slot["caseId"])
    item = source.get("case")
    if (
        type(item) is not dict
        or item.get("caseId") != selected["caseId"]
        or item.get("revisionId") != selected["revisionId"]
        or type(source.get("position")) is not int
        or source["position"] != selected["position"]
    ):
        raise ReplayError("The saved example does not match the pinned selection.")
    converted = replay_recording_from_case(item)
    settings = config["replay"]
    ReplaySession(
        converted["recording"], mode=settings["mode"], until_span=settings.get("untilSpan")
    )
    if batch["run"]:
        status = api.status(batch["run"]["runId"])
        if status["state"] not in {"queued", "running"}:
            raise ReplayError("This batch no longer accepts outputs. The agent was not executed.")
    remaining = _remaining(batch["run"])
    if remaining < 1:
        raise ReplayError("This batch expired while loading evidence. The agent was not executed.")
    if len(json.dumps(state).encode()) + 16 * 1024 > MAX_RECORDING_BYTES:
        raise ReplayError(
            "Saved run state has no room for another replay report. The agent was not executed."
        )
    slot["execution"] = {
        "sourceTraceId": converted["recording"].source_trace_id,
        "runId": batch["run"]["runId"] if batch["run"] else None,
        "status": None,
        "exitCode": None,
    }
    slot["progress"] = {"kind": "executing"}
    save_state(path, state)
    replay = {**settings}
    if batch["run"]:
        replay["run"] = {
            "runId": batch["run"]["runId"],
            "caseId": slot["caseId"],
            "trial": slot["trial"],
        }
    timeout = (
        agent["timeout_seconds"]
        if not batch["run"]
        else min(agent["timeout_seconds"], max(1, int(remaining)))
    )
    try:
        result = run_local_command(
            recording=converted["recording"],
            replay=replay,
            input=item.get("input"),
            input_context={
                **item,
                "root": {"input": item.get("input"), "output": item.get("observedOutput")},
            },
            config={**agent, "timeout_seconds": timeout},
            cwd=config["cwd"],
            label=config["label"],
            params=config["params"],
        )
    except (ReplayError, OSError) as error:
        slot["progress"] = {
            "kind": "failed",
            "message": (
                str(error)[:2000]
                if isinstance(error, ReplayError)
                else "Local agent command failed."
            ),
        }
        save_state(path, state)
        return
    slot["execution"].update(status=result["status"], exitCode=result["exitCode"])
    if not as_json:
        print(result["stdout"], end="")
        print(result["stderr"], end="", file=sys.stderr)
    if result["exitCode"] != 0:
        slot["progress"] = {
            "kind": "failed",
            "message": "The agent exited unsuccessfully. It will not run again automatically.",
        }
    elif batch["run"] is None:
        slot["progress"] = {"kind": "local_complete"}
    elif len(result["status"]["traceIds"]) != 1:
        slot["progress"] = {
            "kind": "failed",
            "message": "Upload requires exactly one workflow trace. This attempt will not run again automatically.",
        }
    else:
        slot["progress"] = {"kind": "upload_pending", "traceId": result["status"]["traceIds"][0]}
    save_state(path, state)
    _upload(state, path, api, batch, slot)


def _register(state, path, api, batch):
    config = state["configuration"]
    if config["grading"] is None or batch["run"] is not None:
        return
    index = batch["offset"] // config["batchSize"] + 1
    suffix = f" [{index}/{len(state['batches'])}]"
    batch["run"] = api.register(
        {
            "datasetVersionId": config["versionId"],
            "label": config["label"][: 120 - len(suffix)] + suffix,
            "caseIds": [
                item["caseId"]
                for item in state["cases"][batch["offset"] : batch["offset"] + config["batchSize"]]
            ],
            "trials": config["trials"],
            "params": config["params"],
            "grading": config["grading"],
            "deadlineSeconds": config["deadlineSeconds"],
            "replay": config["replay"],
            "selection": {
                "kind": "all",
                "groupId": state["groupId"],
                "datasetId": config["datasetId"],
                "digest": state["selectionDigest"],
                "totalCases": len(state["cases"]),
                "batchOffset": batch["offset"],
                "batchSize": config["batchSize"],
            },
        },
        key=batch["registrationKey"],
    )
    save_state(path, state)


def _execute(state, path, api, agent, as_json):
    error = None
    try:
        if any(
            slot["progress"]["kind"] == "executing"
            for batch in state["batches"]
            for slot in batch["slots"]
        ):
            raise ReplayError(
                "A previous agent command has an unknown outcome. It will not run again automatically. Inspect that attempt before starting another run."
            )
        for batch in state["batches"]:
            _register(state, path, api, batch)
            for slot in batch["slots"]:
                _slot(state, path, api, agent, batch, slot, as_json)
    except (ReplayError, OSError, ValueError) as failure:
        error = (
            str(failure)[:2000]
            if isinstance(failure, ReplayError)
            else "Local replay could not continue."
        )
    slots = [slot for batch in state["batches"] for slot in batch["slots"]]
    completed = sum(slot["progress"]["kind"] in {"accepted", "local_complete"} for slot in slots)
    failed = sum(slot["progress"]["kind"] == "failed" for slot in slots)
    exit_code = int(error is not None or completed != len(slots))
    result = {
        "exitCode": exit_code,
        "groupId": state["groupId"],
        "stateFile": str(path),
        "selectionDigest": state["selectionDigest"],
        "totalCases": len(state["cases"]),
        "completedAttempts": completed,
        "failedAttempts": failed,
        "pendingAttempts": len(slots) - completed - failed,
        "runs": [
            {"batch": index + 1, "runId": batch["run"]["runId"]}
            for index, batch in enumerate(state["batches"])
            if batch["run"]
        ],
        "reports": [
            {
                "batch": index + 1,
                "caseId": slot["caseId"],
                "trial": slot["trial"],
                "progress": slot["progress"]["kind"],
                **slot["execution"],
            }
            for index, batch in enumerate(state["batches"])
            for slot in batch["slots"]
            if slot["execution"]
        ],
        "failures": [
            {"caseId": slot["caseId"], "trial": slot["trial"], **slot["progress"]}
            for slot in slots
            if slot["progress"]["kind"] == "failed"
        ],
        **({"error": error} if error else {}),
    }
    if as_json:
        print(json.dumps(result))
    else:
        print(
            f"{completed} of {len(slots)} attempts finished. {len(result['runs'])} saved batch runs.\nState: {path}"
        )
        for report in result["reports"]:
            status = report["status"]
            if status:
                print(
                    f"{report['caseId']}, attempt {report['trial']}: HIT {status['hits']}; MISS {int(status['firstMiss'] is not None)}; LIVE {status['liveCalls']}."
                )
        if error:
            print(error, file=sys.stderr)
    return exit_code


def start_whole_dataset(
    path,
    *,
    api,
    dataset_id,
    version_id,
    config,
    agent_file,
    replay,
    label,
    params,
    trials,
    deadline_seconds,
    grading,
    batch_size,
    as_json,
):
    with state_lock(path) as target:
        if target.exists():
            raise ReplayError("The state file already exists. Resume it or choose a new file.")
        config_values = {
            "projectId": api.project_id,
            "datasetId": dataset_id,
            "versionId": version_id,
            "baseUrl": api.base_url.removesuffix("/api/v1/public"),
            "cwd": os.getcwd(),
            "agentFile": str(Path(agent_file).absolute()),
            "agentDigest": agent_digest(config),
            "label": label,
            "trials": trials,
            "batchSize": batch_size,
            "deadlineSeconds": deadline_seconds,
            "replay": replay,
            "params": params,
            "grading": grading,
        }
        cases = enumerate_cases(api, dataset_id, version_id)
        state = create_state(config_values, cases)
        save_state(target, state, create=True)
        return _execute(state, target, api, config, as_json)


def resume_whole_dataset(path, *, api, as_json):
    with state_lock(path) as target:
        state = read_state(target)
        config = state["configuration"]
        if (
            os.getcwd() != config["cwd"]
            or api.project_id != config["projectId"]
            or api.base_url.removesuffix("/api/v1/public") != config["baseUrl"]
        ):
            raise ReplayError("Resume from the original working directory, project and API origin.")
        agent = parse_agent_config(bounded_file(config["agentFile"], 64 * 1024).decode())
        if agent_digest(agent) != config["agentDigest"]:
            raise ReplayError("Agent command settings changed. Start a new state file.")
        return _execute(state, target, api, agent, as_json)
