import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

import pytest

from neatlogs.evals import Dataset, run_eval
from neatlogs.replay.api import LocalRunApiError

PROJECT, DATASET, VERSION, CASE, REVISION, RUN = [str(uuid4()) for _ in range(6)]


@pytest.fixture
def server():
    state = {"outputs": [], "fail_upload": False, "verdict": "pass", "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            state["requests"].append(self.path)
            if "/comparison" in self.path:
                data = {"state": "ready", "coverage": {"complete": True}}
            elif f"/cases/{CASE}" in self.path:
                data = {
                    "case": {
                        "caseId": CASE,
                        "revisionId": REVISION,
                        "input": {"message": "hello"},
                        "expected": {"value": "hello"},
                        "metadata": {"language": "en"},
                    }
                }
            elif "/cases?" in self.path:
                data = {"cases": [{"caseId": CASE}], "total": 1}
            else:
                data = {"dataset": {"caseCount": 1}, "version": {"id": VERSION}}
            self.respond(200, data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.headers["Authorization"] == "Bearer task-token"
            assert self.headers["x-project-id"] == PROJECT
            assert self.headers["Idempotency-Key"]
            if self.path.endswith("/outputs"):
                if state["fail_upload"]:
                    self.respond(503, {})
                    return
                state["outputs"].append(body)
                data = {"state": "succeeded", "outputId": str(uuid4())}
            elif self.path.endswith("/gate"):
                data = {"verdict": state["verdict"]}
            elif self.path.endswith("/complete"):
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


def test_runs_real_http_task_trials_and_returns_failed_gate_exit(server):
    state, config = server
    state["verdict"] = "fail"
    seen = []

    def task(value, context):
        seen.append((value, context.trial, context.expected, context.param("model")))
        return {"answer": value["message"], "trial": context.trial}

    result = run_eval(
        dataset=Dataset(DATASET, tag="golden-v3"),
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
    assert "grading" not in state["registration"]
    assert any("?tag=golden-v3" in path for path in state["requests"])


def test_resume_retries_saved_upload_without_reexecuting_task(server, tmp_path):
    state, config = server
    state["fail_upload"] = True
    calls = []
    path = tmp_path / "run.json"

    def task(value, context):
        calls.append(context.trial)
        return "saved answer"

    with pytest.raises(LocalRunApiError):
        run_eval(dataset=DATASET, task=task, label="retry", trials=2, state_path=path, **config)
    assert calls == [1]
    assert path.stat().st_mode & 0o077 == 0
    state["fail_upload"] = False
    result = run_eval(
        dataset=DATASET, task=task, label="retry", trials=2, state_path=path, resume=True, **config
    )
    assert result.exit_code == 0 and calls == [1, 2]
    assert len(state["outputs"]) == 2


def test_task_error_is_uploaded_and_returns_incomplete(server):
    state, config = server

    def task(value, context):
        raise ValueError("task failed")

    result = run_eval(dataset=DATASET, task=task, label="error", **config)
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
        run_eval(dataset=DATASET, task=original, label="resume", state_path=path, **config)
    state["fail_upload"] = False
    with pytest.raises(ValueError, match="same project"):
        run_eval(
            dataset=DATASET, task=changed, label="resume", state_path=path, resume=True, **config
        )
    assert state["outputs"] == []
