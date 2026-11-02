"""Private resumable run journal using the existing cross-language version-one protocol."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from .contracts import MAX_RECORDING_BYTES, ReplayError, identifier, object_fields, uuid
from .environment import bounded_file, write_private_json
from .local_command import parse_agent_config, parse_report
from .provider import ecmascript_json


def digest(value):
    return hashlib.sha256(ecmascript_json(value)).hexdigest()


def selection_digest(config, cases):
    return digest(
        [
            "neatlogs.local.all.v1",
            config["projectId"],
            config["datasetId"],
            config["versionId"],
            [[case["caseId"], case["revisionId"]] for case in cases],
        ]
    )


def agent_digest(config):
    return digest(parse_agent_config(json.dumps(config)))


def configuration(value):
    order = [
        "projectId",
        "datasetId",
        "versionId",
        "baseUrl",
        "cwd",
        "agentFile",
        "agentDigest",
        "label",
        "trials",
        "batchSize",
        "deadlineSeconds",
        "replay",
        "params",
        "grading",
    ]
    fields = object_fields(value, set(order))
    result = {key: fields[key] for key in order}
    for key in ("projectId", "datasetId", "versionId"):
        result[key] = uuid(result[key])
    for key in ("baseUrl", "cwd", "agentFile", "agentDigest", "label"):
        if type(result[key]) is not str or not result[key]:
            raise ReplayError("Saved local run configuration is invalid.")
    if len(result["label"]) > 120 or len(result["agentDigest"]) != 64:
        raise ReplayError("Saved local run configuration is invalid.")
    for key, lower, upper in (
        ("trials", 1, 10),
        ("batchSize", 1, 100),
        ("deadlineSeconds", 60, 3600),
    ):
        if type(result[key]) is not int or not lower <= result[key] <= upper:
            raise ReplayError("Saved local run limits are invalid.")
    replay = result["replay"]
    if type(replay) is not dict or replay.get("mode") not in {"tools_only", "until_span"}:
        raise ReplayError("Saved replay settings are invalid.")
    if replay["mode"] == "tools_only":
        object_fields(replay, {"mode"})
    else:
        object_fields(replay, {"mode", "untilSpan"})
        identifier(replay["untilSpan"])
    from ..ab_test.protocol import params

    result["params"] = params(result["params"])
    grading = result["grading"]
    if grading is not None:
        object_fields(grading, {"formVersionId", "evaluatorIds"})
        uuid(grading["formVersionId"])
        ids = grading["evaluatorIds"]
        if type(ids) is not list or not 1 <= len(ids) <= 20 or len(set(ids)) != len(ids):
            raise ReplayError("Saved local grading selection is invalid.")
        for evaluator in ids:
            uuid(evaluator)
    return result


def create_state(config, cases):
    config = configuration(config)
    batches = []
    for offset in range(0, len(cases), config["batchSize"]):
        batches.append(
            {
                "offset": offset,
                "registrationKey": str(uuid4()),
                "run": None,
                "slots": [
                    {
                        "caseId": case["caseId"],
                        "trial": trial,
                        "uploadKey": str(uuid4()),
                        "progress": {"kind": "pending"},
                        "execution": None,
                    }
                    for case in cases[offset : offset + config["batchSize"]]
                    for trial in range(1, config["trials"] + 1)
                ],
            }
        )
    return {
        "version": 1,
        "groupId": str(uuid4()),
        "configuration": config,
        "configurationDigest": digest(config),
        "selectionDigest": selection_digest(config, cases),
        "cases": cases,
        "batches": batches,
    }


def _validate_activity(slot, batch, config):
    object_fields(slot, {"caseId", "trial", "uploadKey", "progress", "execution"})
    uuid(slot["uploadKey"])
    progress, activity = slot["progress"], slot["execution"]
    if type(progress) is not dict:
        raise ReplayError("Saved local replay progress is invalid.")
    kind = progress.get("kind")
    fields = {
        "pending": {"kind"},
        "executing": {"kind"},
        "local_complete": {"kind"},
        "upload_pending": {"kind", "traceId"},
        "accepted": {"kind", "traceId", "outputId"},
        "failed": {"kind", "message"},
    }
    if kind not in fields:
        raise ReplayError("Saved local replay progress is invalid.")
    object_fields(progress, fields[kind])
    if kind == "failed" and (
        type(progress["message"]) is not str or len(progress["message"]) > 2000
    ):
        raise ReplayError("Saved local replay failure is invalid.")
    if kind == "pending":
        if activity is not None:
            raise ReplayError("Pending replay slots cannot contain an execution.")
        return
    object_fields(activity, {"sourceTraceId", "runId", "status", "exitCode"})
    identifier(activity["sourceTraceId"])
    if activity["runId"] != (batch["run"]["runId"] if batch["run"] else None):
        raise ReplayError("Saved replay activity belongs to another run.")
    status = activity["status"]
    if status is None:
        if activity["exitCode"] is not None or kind not in {"executing", "failed"}:
            raise ReplayError("Saved local replay execution is incomplete.")
        return
    parse_report(status)
    if (
        status["sourceTraceId"] != activity["sourceTraceId"]
        or status["mode"] != config["replay"]["mode"]
        or type(activity["exitCode"]) is not int
        or kind == "executing"
    ):
        raise ReplayError("Saved replay report does not match its execution.")
    if kind in {"accepted", "upload_pending", "local_complete"} and activity["exitCode"] != 0:
        raise ReplayError("An unsuccessful execution cannot be saved as complete.")
    if kind == "local_complete" and batch["run"] is not None:
        raise ReplayError("A registered run requires trace upload acceptance.")
    if kind in {"accepted", "upload_pending"}:
        if batch["run"] is None or status["traceIds"] != [progress["traceId"]]:
            raise ReplayError("Saved upload trace does not match its execution.")
        if kind == "accepted":
            uuid(progress["outputId"])


def read_state(path):
    try:
        state = json.loads(bounded_file(path, MAX_RECORDING_BYTES))
        object_fields(
            state,
            {
                "version",
                "groupId",
                "configuration",
                "configurationDigest",
                "selectionDigest",
                "cases",
                "batches",
            },
        )
        if type(state["version"]) is not int or state["version"] != 1:
            raise ReplayError("Unsupported saved local run version.")
        uuid(state["groupId"])
        config = configuration(state["configuration"])
        cases = state["cases"]
        if type(cases) is not list or not 1 <= len(cases) <= 2000:
            raise ReplayError("Saved local selection must contain 1 to 2000 examples.")
        for position, case in enumerate(cases):
            object_fields(case, {"caseId", "revisionId", "position"})
            uuid(case["caseId"])
            uuid(case["revisionId"])
            if type(case["position"]) is not int or case["position"] != position:
                raise ReplayError("Saved example positions are invalid.")
        if (
            len({case["caseId"] for case in cases}) != len(cases)
            or digest(config) != state["configurationDigest"]
            or selection_digest(config, cases) != state["selectionDigest"]
        ):
            raise ReplayError(
                "Saved selection or command settings changed. Start a new state file."
            )
        batches = state["batches"]
        if (
            type(batches) is not list
            or len(batches) != (len(cases) + config["batchSize"] - 1) // config["batchSize"]
        ):
            raise ReplayError("Saved local run batches are invalid.")
        from .api import LocalRunApi

        for index, batch in enumerate(batches):
            object_fields(batch, {"offset", "registrationKey", "run", "slots"})
            uuid(batch["registrationKey"])
            if batch["run"] is not None:
                LocalRunApi._receipt(batch["run"])
            expected = cases[index * config["batchSize"] : (index + 1) * config["batchSize"]]
            slots = batch["slots"]
            if (
                batch["offset"] != index * config["batchSize"]
                or type(slots) is not list
                or len(slots) != len(expected) * config["trials"]
            ):
                raise ReplayError("Saved batch selection changed.")
            for offset, slot in enumerate(slots):
                if (
                    type(slot) is not dict
                    or slot.get("caseId") != expected[offset // config["trials"]]["caseId"]
                    or type(slot.get("trial")) is not int
                    or slot["trial"] != offset % config["trials"] + 1
                ):
                    raise ReplayError("Saved attempt selection changed.")
                _validate_activity(slot, batch, config)
        state["configuration"] = config
        return state
    except (KeyError, TypeError, ValueError, RecursionError) as error:
        if isinstance(error, ReplayError):
            raise
        raise ReplayError("Saved local run state is invalid.") from None


def save_state(path, state, *, create=False):
    content = json.dumps(state, allow_nan=False, separators=(",", ":")).encode()
    if len(content) > MAX_RECORDING_BYTES:
        raise ReplayError("Saved local run state exceeds 16 MiB.")
    if create:
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            raise ReplayError(
                "The state file already exists. Resume it or choose a new file."
            ) from None
    else:
        write_private_json(path, state, limit=MAX_RECORDING_BYTES)
    if os.name != "nt":
        descriptor = os.open(Path(path).parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


@contextmanager
def state_lock(raw_path):
    path = Path(raw_path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = Path(str(path) + ".lock")
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ReplayError(
            "This saved run is locked. Inspect its process outcome before removing the lock and resuming."
        ) from None
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps({"pid": os.getpid()}))
        yield path
    finally:
        lock.unlink(missing_ok=True)
