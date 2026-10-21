"""Pinned whole-dataset batches and safe resume fixtures. No provider calls."""

import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from neatlogs.replay import ReplayError
from neatlogs.replay.all_command import enumerate_cases, resume_whole_dataset, start_whole_dataset
from neatlogs.replay.api import LocalRunApiError
from neatlogs.replay.state import create_state, read_state, save_state, state_lock

PROJECT = "00000000-0000-4000-8000-000000000001"
DATASET = "00000000-0000-4000-8000-000000000002"
VERSION = "00000000-0000-4000-8000-000000000003"
FORM = "00000000-0000-4000-8000-000000000004"
EVALUATOR = "00000000-0000-4000-8000-000000000005"


class Api:
    project_id = PROJECT
    base_url = "http://localhost:3181/api/v1/public"

    def __init__(self, count):
        self.cases = [
            {
                "caseId": str(UUID(int=index + 100)),
                "revisionId": str(UUID(int=index + 10000)),
                "position": index,
            }
            for index in range(count)
        ]
        self.registrations = []
        self.uploads = []
        self.fail_upload = False
        self.bad_revision = False

    def request(self, method, path, **kwargs):
        if "/cases?" not in path:
            return {
                "dataset": {"id": DATASET, "caseCount": len(self.cases)},
                "version": {"id": VERSION},
            }
        offset = int(path.split("offset=")[1].split("&")[0])
        return {"total": len(self.cases), "cases": self.cases[offset : offset + 100]}

    def frozen_case(self, dataset, version, case_id):
        case = next(case for case in self.cases if case["caseId"] == case_id)
        return {
            "position": case["position"],
            "case": {
                "caseId": case_id,
                "revisionId": str(uuid4()) if self.bad_revision else case["revisionId"],
                "source": {"kind": "trace", "traceId": f"source-{case_id}"},
                "input": {"value": 1},
                "observedOutput": {},
                "context": {
                    "spans": [
                        {
                            "span_id": "tool",
                            "span_type": "TOOL",
                            "tool_name": "lookup",
                            "tool_input": {"value": 1},
                            "tool_output": {"saved": True},
                        }
                    ]
                },
            },
        }

    def register(self, body, *, key):
        self.registrations.append((key, body))
        return {
            "runId": str(uuid4()),
            "state": "queued",
            "deadlineAt": "2099-01-01T00:00:00Z",
            "itemCount": len(body["caseIds"]) * body["trials"],
        }

    def status(self, run_id):
        return {"state": "running"}

    def upload(self, run_id, **kwargs):
        self.uploads.append((run_id, kwargs))
        if self.fail_upload:
            self.fail_upload = False
            raise LocalRunApiError("Engineering fixture upload interruption.")
        return {"outputId": str(uuid4())}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from neatlogs.replay import all_command

    config = {"command": ["fixture-agent"], "timeout_seconds": 10}
    agent = tmp_path / "agent.yaml"
    agent.write_text(json.dumps(config))
    calls = []

    def child(**kwargs):
        calls.append(kwargs)
        return {
            "exitCode": 0,
            "stdout": "",
            "stderr": "",
            "status": {
                "sourceTraceId": kwargs["recording"].source_trace_id,
                "mode": "tools_only",
                "hits": 1,
                "liveCalls": 0,
                "switchedToLive": False,
                "reason": None,
                "firstMiss": None,
                "traceIds": [f"{len(calls):032x}"],
            },
        }

    monkeypatch.setattr(all_command, "run_local_command", child)
    options = {
        "dataset_id": DATASET,
        "version_id": VERSION,
        "config": config,
        "agent_file": agent,
        "replay": {"mode": "tools_only"},
        "label": "Engineering fixtures",
        "params": {},
        "trials": 1,
        "deadline_seconds": 900,
        "grading": {"formVersionId": FORM, "evaluatorIds": [EVALUATOR]},
        "batch_size": 100,
        "as_json": True,
    }
    return tmp_path / "state.json", options, calls


def test_101_cases_use_100_plus_1_batches_and_completed_resume_never_reruns(setup, capsys):
    path, options, calls = setup
    api = Api(101)
    assert start_whole_dataset(path, api=api, **options) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["completedAttempts"] == 101 and len(calls) == 101
    assert [len(body["caseIds"]) for _, body in api.registrations] == [100, 1]
    assert [body["selection"]["batchOffset"] for _, body in api.registrations] == [0, 100]
    before = path.read_bytes()
    assert resume_whole_dataset(path, api=api, as_json=True) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["reports"] == first["reports"] and len(calls) == 101
    assert path.read_bytes() == before and len(api.registrations) == 2
    assert path.stat().st_mode & 0o777 == 0o600


def test_pending_upload_resumes_with_same_key_and_no_child_rerun(setup, capsys):
    path, options, calls = setup
    api = Api(1)
    api.fail_upload = True
    assert start_whole_dataset(path, api=api, **options) == 1
    capsys.readouterr()
    assert len(calls) == 1
    assert read_state(path)["batches"][0]["slots"][0]["progress"]["kind"] == "upload_pending"
    assert resume_whole_dataset(path, api=api, as_json=True) == 0
    capsys.readouterr()
    assert len(calls) == 1
    assert api.uploads[0][1]["key"] == api.uploads[1][1]["key"]
    assert len(api.registrations) == 1


def test_revision_mismatch_blocks_before_child_dispatch(setup, capsys):
    path, options, calls = setup
    api = Api(1)
    api.bad_revision = True
    assert start_whole_dataset(path, api=api, **options) == 1
    result = json.loads(capsys.readouterr().out)
    assert "pinned selection" in result["error"] and calls == []


def test_uncertain_command_cannot_run_again(setup, capsys):
    path, options, calls = setup
    api = Api(1)
    assert start_whole_dataset(path, api=api, **{**options, "grading": None}) == 0
    capsys.readouterr()
    state = read_state(path)
    slot = state["batches"][0]["slots"][0]
    slot["progress"] = {"kind": "executing"}
    slot["execution"].update(status=None, exitCode=None)
    save_state(path, state)
    assert resume_whole_dataset(path, api=api, as_json=True) == 1
    result = json.loads(capsys.readouterr().out)
    assert "unknown outcome" in result["error"] and len(calls) == 1


def test_changed_project_or_agent_settings_cannot_resume(setup, capsys):
    path, options, calls = setup
    api = Api(1)
    assert start_whole_dataset(path, api=api, **{**options, "grading": None}) == 0
    capsys.readouterr()
    api.project_id = str(uuid4())
    with pytest.raises(ReplayError, match="original working"):
        resume_whole_dataset(path, api=api, as_json=True)
    assert len(calls) == 1


def test_tampered_selection_and_lock_are_rejected(setup, capsys):
    path, options, _ = setup
    api = Api(1)
    assert start_whole_dataset(path, api=api, **{**options, "grading": None}) == 0
    capsys.readouterr()
    state = json.loads(path.read_text())
    state["cases"][0]["revisionId"] = str(uuid4())
    path.write_text(json.dumps(state))
    with pytest.raises(ReplayError, match="changed"):
        read_state(path)
    with state_lock(path):
        with pytest.raises(ReplayError, match="locked"):
            with state_lock(path):
                pass
