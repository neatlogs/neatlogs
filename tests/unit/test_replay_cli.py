"""Native Python child-process and public-API contract fixtures."""

import json
import os
import sys
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from neatlogs.__main__ import main
from neatlogs.replay import ReplayError
from neatlogs.replay.local_command import parse_agent_config, run_local_command
from neatlogs.replay.provider import read_provider_capture

PROJECT = "00000000-0000-4000-8000-000000000001"
DATASET = "00000000-0000-4000-8000-000000000002"
VERSION = "00000000-0000-4000-8000-000000000003"
CASE = "00000000-0000-4000-8000-000000000004"
REVISION = "00000000-0000-4000-8000-000000000005"
RUN = "00000000-0000-4000-8000-000000000006"
FORM = "00000000-0000-4000-8000-000000000007"
EVALUATOR = "00000000-0000-4000-8000-000000000008"
AGENT = """import json,sys,neatlogs
neatlogs.init(disable_export=True,register_shutdown_handlers=False,instrumentations=[])
import logging
logging.getLogger("neatlogs").setLevel(logging.ERROR)
@neatlogs.span(kind="TOOL",tool_name="lookup",replay=True)
def lookup(value):
    raise AssertionError("A cached tool must not execute")
@neatlogs.span(kind="WORKFLOW")
def workflow(value):
    return lookup(value)
input=json.load(sys.stdin)
print(json.dumps(workflow(input.get("value",1) if isinstance(input,dict) else 1)))
neatlogs.shutdown()
"""


def recording():
    return {
        "version": 1,
        "sourceTraceId": "source-fixture",
        "entries": [
            {
                "spanId": "tool",
                "kind": "tool",
                "name": "lookup",
                "input": {"value": 1},
                "output": {"saved": True},
            }
        ],
    }


def saved_case():
    return {
        "caseId": CASE,
        "revisionId": REVISION,
        "input": {"value": 1},
        "observedOutput": {"saved": True},
        "source": {"kind": "trace", "traceId": "source-fixture"},
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
    }


@pytest.fixture
def agent(tmp_path, monkeypatch):
    script = tmp_path / "agent.py"
    script.write_text(AGENT)
    config = tmp_path / "agent.yaml"
    config.write_text(json.dumps({"command": [sys.executable, str(script)], "timeout_seconds": 60}))
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    for name in list(os.environ):
        if name.startswith("NEATLOGS_REPLAY_"):
            monkeypatch.delenv(name)
    return config


def test_real_native_child_replays_without_credentials_and_cleans_private_files(agent):
    result = run_local_command(
        recording=recording(),
        replay={"mode": "tools_only"},
        input={"value": 1},
        config=parse_agent_config(agent.read_text()),
        cwd=os.getcwd(),
    )
    assert result["exitCode"] == 0
    assert json.loads(result["stdout"]) == {"saved": True}
    assert result["status"]["hits"] == 1
    assert result["status"]["liveCalls"] == 0
    assert len(result["status"]["traceIds"]) == 1


def test_native_cli_recording_option_uses_python_agent(agent, tmp_path, capsys):
    source = tmp_path / "case.json"
    source.write_text(json.dumps(saved_case()))
    assert (
        main(["simulate", "local", "--recording", str(source), "--agent", str(agent), "--json"])
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"]["hits"] == 1 and result["uploaded"] is False
    assert result["caseId"] is None


def test_pinned_case_upload_uses_actual_child_trace_and_stable_retry_key(
    agent, monkeypatch, capsys, respx_mock
):
    monkeypatch.setenv("NEATLOGS_PROJECT_ID", PROJECT)
    monkeypatch.setenv("NEATLOGS_TOKEN", "fixture-token")
    monkeypatch.setenv("NEATLOGS_HOST", "http://localhost:3181")
    root = "http://localhost:3181/api/v1/public"
    respx_mock.get(
        f"{root}/datasets/{DATASET}/cases/{CASE}", params={"versionId": VERSION}
    ).respond(200, json={"success": True, "data": {"case": saved_case(), "position": 0}})
    registration = respx_mock.post(f"{root}/runs").respond(
        200,
        json={
            "success": True,
            "data": {
                "runId": RUN,
                "state": "queued",
                "deadlineAt": "2099-01-01T00:00:00Z",
                "itemCount": 1,
            },
        },
    )
    uploads = []

    def uploaded(request):
        body = json.loads(request.content)
        uploads.append((request.headers["Idempotency-Key"], body))
        assert body["caseId"] == CASE and body["trial"] == 1
        if len(uploads) == 1:
            return httpx.Response(409, json={"code": "TRACE_NOT_READY"})
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "runId": RUN,
                    "itemId": str(uuid4()),
                    "outputId": str(uuid4()),
                    "traceId": body["traceId"],
                    "state": "succeeded",
                },
            },
        )

    respx_mock.post(f"{root}/runs/{RUN}/outputs").mock(side_effect=uploaded)
    assert (
        main(
            [
                "simulate",
                "local",
                "--dataset",
                DATASET,
                "--version",
                VERSION,
                "--case",
                CASE,
                "--agent",
                str(agent),
                "--upload",
                "--form",
                FORM,
                "--evaluator",
                EVALUATOR,
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["uploaded"] and result["runId"] == RUN
    assert len(uploads) == 2 and uploads[0] == uploads[1]
    assert uploads[0][1]["traceId"] == result["status"]["traceIds"][0]
    assert json.loads(registration.calls[0].request.content)["replay"] == {"mode": "tools_only"}


def test_timeout_stops_own_child_process(agent):
    config = {"command": [sys.executable, "-c", "import time;time.sleep(30)"], "timeout_seconds": 1}
    with pytest.raises(ReplayError, match="time limit"):
        run_local_command(
            recording=recording(),
            replay={"mode": "tools_only"},
            input=None,
            config=config,
            cwd=os.getcwd(),
        )


def test_agent_without_replay_report_is_not_reported_as_success(agent):
    config = {"command": [sys.executable, "-c", "print('fixture')"], "timeout_seconds": 5}
    with pytest.raises(ReplayError, match="did not report"):
        run_local_command(
            recording=recording(),
            replay={"mode": "tools_only"},
            input=None,
            config=config,
            cwd=os.getcwd(),
        )


@pytest.mark.parametrize(
    "content",
    [
        "command: [python]\ncommand: [other]",
        "command: &a [python]\ninput_from_trace: *a",
        "command: [python]\ntimeout_seconds: true",
    ],
)
def test_invalid_yaml_is_rejected_without_execution(content):
    with pytest.raises(ReplayError):
        parse_agent_config(content)


def test_ts_raw_provider_capture_digests_validate_in_python():
    golden = json.loads(
        (Path(__file__).parent / "fixtures/replay-provider-ts-golden.json").read_text()
    )
    for case in golden:
        assert read_provider_capture(case["capture"])["name"] == case["operation"]


def test_child_exception_still_has_an_honest_activity_report(agent):
    config = parse_agent_config(agent.read_text())
    result = run_local_command(
        recording=recording(),
        replay={"mode": "tools_only"},
        input={"value": 2},
        config=config,
        cwd=os.getcwd(),
    )
    assert result["exitCode"] != 0
    assert result["status"]["hits"] == 0
    assert result["status"]["liveCalls"] == 1
    assert result["status"]["firstMiss"] == {"kind": "tool", "name": "lookup"}


@pytest.mark.parametrize("defect", ["missing-result", "duplicate-id", "masked", "truncated"])
def test_incomplete_frozen_tools_are_skipped_instead_of_becoming_cache_hits(defect):
    from neatlogs.replay import replay_recording_from_case

    item = saved_case()
    tool = item["context"]["spans"][0]
    if defect == "missing-result":
        del tool["tool_output"]
    if defect == "duplicate-id":
        item["context"]["spans"].append(dict(tool))
    if defect == "masked":
        tool["attributes"] = {"neatlogs.capture_fidelity": "masked"}
    if defect == "truncated":
        tool["output_content_state"] = "truncated"
    result = replay_recording_from_case(item)
    assert result["recording"].entries == ()
    assert result["skippedTools"] >= 1
