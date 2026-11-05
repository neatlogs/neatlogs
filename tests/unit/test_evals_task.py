import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import pytest

from neatlogs.evals import Dataset, run_eval
from neatlogs.evals.cli import main as run_eval_cli
from neatlogs.replay.api import LocalRunApi, LocalRunApiError
from neatlogs.replay.contracts import ReplayError

PROJECT, DATASET, VERSION, CASE, REVISION, RUN = [str(uuid4()) for _ in range(6)]
GRADING = {"formVersionId": str(uuid4()), "evaluatorIds": [str(uuid4())]}


def test_rejects_case_outside_dataset_before_running(server):
    _, config = server
    with pytest.raises(ValueError, match="belong to the pinned dataset"):
        run_eval(
            grading=GRADING,
            dataset=DATASET,
            case_ids=[str(uuid4())],
            task=lambda *_: "answer",
            label="selection",
            **config,
        )


@pytest.fixture
def server():
    state = {
        "outputs": [],
        "dataset_pages": [],
        "fail_upload": False,
        "verdict": "pass",
        "requests": [],
        "cases": [],
        "grading": None,
        "grading_handoff_state": None,
        "dispatches": [],
        "comparison_pending": 0,
        "comparison_calls": 0,
        "dispatch_failures": 0,
        "completion_keys": [],
        "completion_failures": 0,
        "gate_keys": [],
        "gate_failures": 0,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            state["requests"].append(self.path)
            if urlsplit(self.path).path == "/api/v1/public/datasets":
                cursor = parse_qs(urlsplit(self.path).query).get("before", [None])[0]
                index = 1 if cursor else 0
                data = state["dataset_pages"][index]
            elif "/comparison" in self.path:
                state["comparison_calls"] += 1
                if state["comparison_pending"]:
                    state["comparison_pending"] -= 1
                    data = {"state": "pending", "coverage": {"complete": False}}
                else:
                    data = {"state": "ready", "coverage": {"complete": True}}
            elif "/runs/" in self.path and "includeGrading=true" in self.path:
                data = {
                    "runId": RUN,
                    "state": "completed",
                    "deadlineAt": "2030-01-01T00:00:00Z",
                    "itemCount": 1,
                    "grading": state["grading"],
                }
                if state["grading_handoff_state"] is not None:
                    data["gradingHandoffState"] = state["grading_handoff_state"]
                elif state["grading"] is None:
                    data["gradingHandoffState"] = "ready"
            elif "/cases/" in self.path:
                case_id = self.path.split("/cases/")[1].split("?")[0]
                data = {
                    "case": {
                        "caseId": case_id,
                        "revisionId": REVISION,
                        "input": {"message": "hello"},
                        "expected": {"value": "hello"},
                        "metadata": {"language": "en"},
                    }
                }
            elif "/cases?" in self.path:
                cases = state["cases"] or [CASE]
                offset = int(parse_qs(urlsplit(self.path).query)["offset"][0])
                data = {
                    "cases": [{"caseId": item} for item in cases[offset : offset + 100]],
                    "total": len(cases),
                }
            else:
                data = {
                    "dataset": {"caseCount": len(state["cases"]) or 1},
                    "version": {"id": VERSION},
                }
            self.respond(200, data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.headers["Authorization"] == "Bearer task-token"
            assert self.headers["x-project-id"] == PROJECT
            assert self.headers["Idempotency-Key"]
            if self.path.endswith("/ai-dispatches"):
                state["dispatches"].append((body["batchId"], self.headers["Idempotency-Key"]))
                if (
                    body["batchId"] == state["grading"]["candidateBatchId"]
                    and state["dispatch_failures"]
                ):
                    state["dispatch_failures"] -= 1
                    self.respond(503, {})
                    return
                data = {
                    "evaluationId": state["grading"]["evaluationId"],
                    "batchId": body["batchId"],
                    "dispatchStatus": "accepted",
                }
            elif self.path.endswith("/outputs"):
                if state["fail_upload"]:
                    self.respond(503, {})
                    return
                state["outputs"].append(body)
                data = {"state": "succeeded", "outputId": str(uuid4())}
            elif self.path.endswith("/gate"):
                state["gate_keys"].append(self.headers["Idempotency-Key"])
                if state["gate_failures"]:
                    state["gate_failures"] -= 1
                    self.respond(503, {})
                    return
                data = {"verdict": state["verdict"]}
            elif self.path.endswith("/complete"):
                state["completion_keys"].append(self.headers["Idempotency-Key"])
                if state["completion_failures"]:
                    state["completion_failures"] -= 1
                    self.respond(503, {})
                    return
                data = {}
            else:
                state["registration"] = body
                data = {
                    "runId": RUN,
                    "state": "queued",
                    "deadlineAt": "2030-01-01T00:00:00Z",
                    "itemCount": body["trials"],
                }
            self.respond(200, data)

        def respond(self, status, data):
            body = json.dumps({"success": status < 400, "data": data}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    config = {
        "project_id": PROJECT,
        "token": "task-token",
        "base_url": f"http://127.0.0.1:{http.server_port}",
    }
    yield state, config
    http.shutdown()
    worker.join()
    http.server_close()


def test_descriptive_run_has_no_quality_gate(server):
    state, config = server
    result = run_eval(dataset=DATASET, label="descriptive", task=lambda *_: "answer", **config)
    assert result.exit_code == 0
    assert result.gate is None
    assert state["gate_keys"] == []


def test_runs_real_http_task_trials_and_returns_failed_gate_exit(server):
    state, config = server
    state["verdict"] = "fail"
    seen = []

    def task(value, context):
        seen.append((value, context.trial, context.expected, context.param("model")))
        return {"answer": value["message"], "trial": context.trial}

    result = run_eval(
        grading=GRADING,
        dataset=Dataset(DATASET, version_id=VERSION),
        task=task,
        label="candidate",
        params={"model": "small"},
        trials=2,
        gates=["new_failures <= 0"],
        **config,
    )
    assert result.exit_code == 1 and result.run_id == RUN
    assert [value[1] for value in seen] == [1, 2]
    assert seen[0][2:] == ({"value": "hello"}, "small")
    assert len(state["outputs"]) == 2
    assert state["registration"]["grading"] == GRADING
    assert f"/api/v1/public/datasets/{DATASET}?versionId={VERSION}" in state["requests"]


@pytest.mark.parametrize("fail_upload, expected_exit", [(False, 1), (True, 3)])
def test_cli_reports_saved_run_id_when_eval_fails(
    server, tmp_path, monkeypatch, capsys, fail_upload, expected_exit
):
    state, config = server
    state["verdict"] = "fail"
    state["fail_upload"] = fail_upload
    task = tmp_path / "task.py"
    task.write_text("def task(value, context):\n    print('task output')\n    return value\n")
    receipt = tmp_path / "receipt"
    monkeypatch.setattr("sys.argv", ["neatlogs.evals.cli", str(task)])
    monkeypatch.setenv("NEATLOGS_EVAL_RESULT_FILE", str(receipt))
    monkeypatch.setenv(
        "NEATLOGS_EVAL_OPTIONS",
        json.dumps(
            dict(
                config,
                grading=GRADING,
                dataset={"name": DATASET, "versionId": VERSION},
                label="candidate",
                gates=["error_rate <= 0"],
            )
        ),
    )
    assert run_eval_cli() == expected_exit
    captured = capsys.readouterr()
    assert receipt.read_text() == RUN
    assert receipt.stat().st_mode & 0o077 == 0
    if fail_upload:
        assert not captured.out and "HTTP 503" in captured.err
    else:
        assert json.loads(captured.out)["runId"] == RUN
    assert "task output" in captured.err


def test_resumes_uncertain_completion_with_original_key(server, tmp_path):
    state, config = server
    state["completion_failures"] = 2

    def task(*_):
        return "answer"

    options = dict(
        grading=GRADING,
        dataset=DATASET,
        task=task,
        label="retry completion",
        state_path=tmp_path / "completion.json",
        **config,
    )
    with pytest.raises(LocalRunApiError):
        run_eval(**options)
    result = run_eval(**options, resume=True)
    assert result.exit_code == 0
    assert len(state["completion_keys"]) == 3
    assert len(set(state["completion_keys"])) == 1
    assert len(state["outputs"]) == 1


def test_resumes_uncertain_gate_with_original_key(server, tmp_path):
    state, config = server
    state["gate_failures"] = 2

    def task(*_):
        return "answer"

    options = dict(
        grading=GRADING,
        dataset=DATASET,
        task=task,
        label="retry gate",
        gates=["error_rate <= 0"],
        state_path=tmp_path / "gate.json",
        **config,
    )
    with pytest.raises(LocalRunApiError):
        run_eval(**options)
    result = run_eval(**options, resume=True)
    assert result.exit_code == 0
    assert len(state["gate_keys"]) == 3
    assert len(set(state["gate_keys"])) == 1
    assert len(state["outputs"]) == 1


def test_dispatches_both_grading_batches_with_distinct_keys(server, tmp_path):
    state, config = server
    state["grading"] = {
        "evaluationId": str(uuid4()),
        "candidateBatchId": str(uuid4()),
        "recordedBatchId": str(uuid4()),
    }
    result = run_eval(
        dataset=DATASET,
        task=lambda *_: "answer",
        label="graded",
        grading={"formVersionId": str(uuid4()), "evaluatorIds": [str(uuid4())]},
        state_path=tmp_path / "run.json",
        **config,
    )
    assert result.exit_code == 0
    assert [item[0] for item in state["dispatches"]] == [
        state["grading"]["recordedBatchId"],
        state["grading"]["candidateBatchId"],
    ]
    assert len({item[1] for item in state["dispatches"]}) == 2


@pytest.mark.parametrize("handoff,with_ids", [("pending", False), ("ready", True)])
def test_server_managed_grading_polls_comparison_without_dispatch(
    server, tmp_path, handoff, with_ids
):
    state, config = server
    state["grading_handoff_state"] = handoff
    state["grading"] = (
        {
            "evaluationId": str(uuid4()),
            "candidateBatchId": str(uuid4()),
            "recordedBatchId": str(uuid4()),
        }
        if with_ids
        else None
    )
    state["comparison_pending"] = 1
    result = run_eval(
        dataset=DATASET,
        task=lambda *_: "answer",
        label="server managed grading",
        grading={"formVersionId": str(uuid4()), "evaluatorIds": [str(uuid4())]},
        state_path=tmp_path / "run.json",
        wait_timeout=5,
        **config,
    )
    assert result.exit_code == 0
    assert state["dispatches"] == []
    assert state["comparison_calls"] == 2


def test_resumes_uncertain_grading_with_original_key(server, tmp_path):
    state, config = server
    state["grading"] = {
        "evaluationId": str(uuid4()),
        "candidateBatchId": str(uuid4()),
        "recordedBatchId": str(uuid4()),
    }
    state["dispatch_failures"] = 2

    def task(*_):
        return "answer"

    options = dict(
        dataset=DATASET,
        task=task,
        label="graded",
        grading={"formVersionId": str(uuid4()), "evaluatorIds": [str(uuid4())]},
        state_path=tmp_path / "run.json",
        **config,
    )
    with pytest.raises(LocalRunApiError):
        run_eval(**options)
    result = run_eval(**options, resume=True)
    assert result.exit_code == 0
    recorded = [
        item for item in state["dispatches"] if item[0] == state["grading"]["recordedBatchId"]
    ]
    candidate = [
        item for item in state["dispatches"] if item[0] == state["grading"]["candidateBatchId"]
    ]
    assert len(recorded) == 1
    assert len(candidate) == 3 and len({item[1] for item in candidate}) == 1


def test_selects_case_from_later_dataset_page(server):
    state, config = server
    state["cases"] = [str(uuid4()) for _ in range(101)]
    result = run_eval(
        grading=GRADING,
        dataset=DATASET,
        case_ids=[state["cases"][-1]],
        task=lambda *_: "answer",
        label="selected",
        **config,
    )
    assert result.exit_code == 0 and len(state["outputs"]) == 1
    assert any("offset=100" in path for path in state["requests"])


def test_resume_retries_saved_upload_without_reexecuting_task(server, tmp_path):
    state, config = server
    state["fail_upload"] = True
    calls = []
    path = tmp_path / "run.json"

    def task(value, context):
        calls.append(context.trial)
        return "saved answer"

    with pytest.raises(LocalRunApiError):
        run_eval(
            grading=GRADING,
            dataset=DATASET,
            task=task,
            label="retry",
            trials=2,
            state_path=path,
            **config,
        )
    assert calls == [1]
    assert path.stat().st_mode & 0o077 == 0
    state["fail_upload"] = False
    result = run_eval(
        grading=GRADING,
        dataset=DATASET,
        task=task,
        label="retry",
        trials=2,
        state_path=path,
        resume=True,
        **config,
    )
    assert result.exit_code == 0 and calls == [1, 2]
    assert len(state["outputs"]) == 2


def test_task_error_is_uploaded_and_returns_incomplete(server):
    state, config = server

    def task(value, context):
        raise ValueError("task failed")

    result = run_eval(grading=GRADING, dataset=DATASET, task=task, label="error", **config)
    assert result.exit_code == 4 and result.errors == 1
    assert state["outputs"][0]["error"] == {"code": "ValueError", "message": "task failed"}


def test_resume_rejects_changed_task_code(server, tmp_path):
    state, config = server
    state["fail_upload"] = True
    path = tmp_path / "run.json"

    def original(value, context):
        return "original"

    def changed(value, context):
        return "changed"

    with pytest.raises(LocalRunApiError):
        run_eval(
            grading=GRADING,
            dataset=DATASET,
            task=original,
            label="resume",
            state_path=path,
            **config,
        )
    state["fail_upload"] = False
    with pytest.raises(ValueError, match="same project"):
        run_eval(
            grading=GRADING,
            dataset=DATASET,
            task=changed,
            label="resume",
            state_path=path,
            resume=True,
            **config,
        )
    assert state["outputs"] == []


def test_refuses_concurrent_owner_of_eval_state(server, tmp_path):
    state, config = server
    path = tmp_path / "locked.json"
    (tmp_path / "locked.json.lock").write_text("{}")
    executions = []

    def task(*_):
        executions.append(1)
        return "answer"

    with pytest.raises(ReplayError, match="locked"):
        run_eval(
            grading=GRADING, dataset=DATASET, task=task, label="locked", state_path=path, **config
        )
    assert executions == []
    assert state["requests"] == []


def test_status_accepts_canonical_equivalent_run_id(server, monkeypatch):
    _, config = server
    api = LocalRunApi(**config)
    receipt = {
        "runId": RUN,
        "state": "completed",
        "deadlineAt": "2030-01-01T00:00:00Z",
        "itemCount": 1,
    }
    seen = []

    def request(method, path):
        seen.append((method, path))
        return receipt

    monkeypatch.setattr(api, "request", request)
    assert api.status(RUN.upper()) == receipt
    assert seen == [("GET", f"/runs/{RUN}")]


def test_dataset_and_run_api_use_same_trimmed_token(server):
    _, config = server
    config = {**config, "token": "  task-token  "}
    result = run_eval(
        grading=GRADING, dataset=DATASET, task=lambda *_: "answer", label="auth", **config
    )
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "grading",
    [
        None,
        {},
        {"formVersionId": VERSION, "evaluatorIds": []},
        *[
            {"formVersionId": value, "evaluatorIds": GRADING["evaluatorIds"]}
            for value in [
                VERSION.replace("-", ""),
                "urn:uuid:" + VERSION,
                "{" + VERSION + "}",
                "10000000-0000-0000-8000-000000000001",
                "10000000-0000-4000-0000-000000000001",
            ]
        ],
    ],
)
def test_grading_is_required_before_network_or_task(server, grading):
    state, config = server
    calls = []
    with pytest.raises(ValueError, match="grading"):
        run_eval(
            dataset=DATASET,
            task=lambda *_: calls.append(1),
            label="required",
            grading=grading,
            **config,
        )
    assert state["requests"] == []
    assert calls == []


def test_cli_rejects_missing_grading_before_importing_task(tmp_path, monkeypatch, capsys):
    task = tmp_path / "task.py"
    marker = tmp_path / "imported"
    task.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n")
    monkeypatch.setattr("sys.argv", ["neatlogs.evals.cli", str(task)])
    monkeypatch.setenv(
        "NEATLOGS_EVAL_OPTIONS", json.dumps({"dataset": {"name": DATASET}, "label": "required"})
    )
    assert run_eval_cli() == 2
    assert "grading" in capsys.readouterr().err
    assert not marker.exists()


def test_resolves_dataset_name_after_first_500(server):
    state, config = server
    cursor = str(uuid4())
    state["dataset_pages"] = [
        {
            "datasets": [{"id": str(uuid4()), "name": "other"} for _ in range(500)],
            "nextCursor": cursor,
        },
        {"datasets": [{"id": DATASET, "name": "older"}], "nextCursor": None},
    ]
    result = run_eval(
        dataset="older", task=lambda *_: "answer", label="older", grading=GRADING, **config
    )
    assert result.exit_code == 0
    assert f"/api/v1/public/datasets?before={cursor}" in state["requests"]


def test_rejects_duplicate_dataset_name_on_later_page(server):
    state, config = server
    state["dataset_pages"] = [
        {"datasets": [{"id": DATASET, "name": "same"}], "nextCursor": str(uuid4())},
        {"datasets": [{"id": str(uuid4()), "name": "same"}], "nextCursor": None},
    ]
    with pytest.raises(ValueError, match="unambiguous"):
        run_eval(
            dataset="same", task=lambda *_: "answer", label="duplicate", grading=GRADING, **config
        )
    assert len(state["requests"]) == 2
    assert state["outputs"] == []


@pytest.mark.parametrize("repeated", [False, True])
def test_rejects_incomplete_dataset_catalog(server, repeated):
    state, config = server
    cursor = str(uuid4())
    state["dataset_pages"] = [
        {
            "datasets": [{"id": DATASET, "name": "same"}] * 500,
            **({"nextCursor": cursor} if repeated else {}),
        },
        {"datasets": [], "nextCursor": cursor},
    ]
    with pytest.raises(ValueError, match="pagination is incomplete"):
        run_eval(
            dataset="same", task=lambda *_: "answer", label="catalog", grading=GRADING, **config
        )
    assert len(state["requests"]) == (2 if repeated else 1)


def test_round_identity_is_pinned_across_resume(server, tmp_path):
    state, config = server
    local = {"experimentId": str(uuid4()), "roundId": str(uuid4()), "variant": "B", "attempt": 2}
    calls = []

    def task(*_):
        calls.append(1)
        return "answer"

    options = dict(
        grading=GRADING,
        dataset=DATASET,
        task=task,
        label="round B retry",
        local=local,
        state_path=str(tmp_path / "round.json"),
        **config,
    )
    run_eval(**options)
    assert state["registration"]["local"] == local
    run_eval(**options, resume=True)
    assert len(calls) == 1
    options["local"] = {**local, "variant": "A"}
    with pytest.raises(ValueError, match="same"):
        run_eval(**options, resume=True)
