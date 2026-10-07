import json

from neatlogs.__main__ import main
from neatlogs.trace_cli import check_trace, run_trace_get

GOOD = {
    "_id": "t1",
    "status": "success",
    "finalizationStatus": "finalized",
    "spanCount": 2,
    "totalTokensUsed": 5,
    "spans": [
        {"span_id": "a", "span_name": "root", "node_type": "workflow"},
        {
            "span_id": "b",
            "parent_span_id": "a",
            "span_name": "chat",
            "node_type": "llm",
        },
    ],
}


def run(response, env=None, trace_id="t1"):
    out, err, urls = [], [], []

    def fetch(url, headers, timeout):
        urls.append((url, headers))
        return (
            response
            if isinstance(response, tuple)
            else (200, json.dumps(response).encode())
        )

    code = run_trace_get(
        trace_id,
        True,
        env={"NEATLOGS_API_KEY": "secret-key"} if env is None else env,
        fetch=fetch,
        out=out.append,
        err=err.append,
    )
    return code, out, err, urls


def test_healthy_trace_passes_and_uses_existing_read_path():
    code, out, _, urls = run(GOOD)
    assert code == 0
    assert urls[0][0] == "https://ingest.neatlogs.com/api/traces/v3/t1"
    assert urls[0][1] == {"x-api-key": "secret-key"}
    assert json.loads(out[0])["result"] == "pass"


def test_missing_parent_and_unnamed_span_fail():
    bad = dict(
        GOOD, spans=[{"span_id": "a", "parent_span_id": "zzz"}, GOOD["spans"][1]]
    )
    code, out, _, _ = run(bad)
    assert code == 1
    failed = {c["name"] for c in json.loads(out[0])["checks"] if c["status"] == "fail"}
    assert {"parents_resolve", "spans_named"} <= failed


def test_exit_codes_and_key_not_printed():
    assert run((404, b""))[0] == 2
    code, _, err, _ = run((401, b""))
    assert code == 3
    assert "secret-key" not in "".join(err)
    assert run(GOOD, env={})[0] == 3
    assert run((500, b""))[0] == 5
    assert run((200, b"not json"))[0] == 5


def test_endpoint_override_and_id_encoding():
    _, _, _, urls = run(
        GOOD,
        env={
            "NEATLOGS_API_KEY": "secret-key",
            "NEATLOGS_ENDPOINT": "http://localhost:9",
        },
        trace_id="a/b",
    )
    assert urls[0][0] == "http://localhost:9/api/traces/v3/a%2Fb"


def test_check_trace_empty():
    assert check_trace({})[0]["status"] == "fail"


def test_main_dispatch_usage(capsys):
    assert main(["trace"]) == 4
