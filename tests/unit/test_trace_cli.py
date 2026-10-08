import json

from neatlogs.__main__ import main
from neatlogs.trace_cli import check_trace, run_trace_get

ENV = {
    "NEATLOGS_TOKEN": "tok-secret",
    "NEATLOGS_PROJECT_ID": "11111111-1111-4111-8111-111111111111",
}
TRACE = {
    "traceId": "t1",
    "status": "success",
    "finalizationStatus": "finalized",
    "spansCount": 2,
    "totalTokens": 5,
}
SPAN_A = {
    "spanId": "a",
    "parentSpanId": None,
    "spanName": "root",
    "spanType": "workflow",
}
SPAN_B = {"spanId": "b", "parentSpanId": "a", "spanName": "chat", "spanType": "llm"}
NO_MORE = {"hasMore": False, "limit": 50, "nextCursor": None}


def ok(data):
    return 200, json.dumps({"success": True, "data": data, "requestId": "r1"}).encode()


def healthy(url):
    if "/spans" in url:
        return ok({"spans": [SPAN_A, SPAN_B], "page": NO_MORE})
    return ok(TRACE)


def run(route=healthy, env=None, trace_id="t1"):
    out, err, calls = [], [], []

    def fetch(url, headers, timeout):
        calls.append((url, headers))
        return route(url) if callable(route) else route

    code = run_trace_get(
        trace_id,
        True,
        env=ENV if env is None else env,
        fetch=fetch,
        out=out.append,
        err=err.append,
    )
    return code, out, err, calls


def test_healthy_trace_uses_public_api_with_bearer_and_project_id():
    code, out, _, calls = run()
    assert code == 0
    assert calls[0][0] == "https://app.neatlogs.com/api/v1/public/traces/t1"
    assert (
        calls[1][0] == "https://app.neatlogs.com/api/v1/public/traces/t1/spans?limit=50"
    )
    assert calls[0][1] == {
        "Authorization": "Bearer tok-secret",
        "x-project-id": ENV["NEATLOGS_PROJECT_ID"],
    }
    assert json.loads(out[0])["result"] == "pass"


def test_follows_span_pages_until_next_cursor_is_empty():
    def route(url):
        if "/spans" not in url:
            return ok(TRACE)
        if "cursor=c2" in url:
            return ok({"spans": [SPAN_B], "page": NO_MORE})
        return ok(
            {
                "spans": [SPAN_A],
                "page": {"hasMore": True, "limit": 50, "nextCursor": "c2"},
            }
        )

    code, out, _, calls = run(route)
    assert code == 0
    assert len(calls) == 3
    assert json.loads(out[0])["span_count"] == 2


def test_missing_parent_and_unnamed_span_fail():
    def route(url):
        if "/spans" not in url:
            return ok(TRACE)
        bad = dict(SPAN_A, parentSpanId="zzz", spanName="")
        return ok({"spans": [bad, SPAN_B], "page": NO_MORE})

    code, out, _, _ = run(route)
    assert code == 1
    failed = {c["name"] for c in json.loads(out[0])["checks"] if c["status"] == "fail"}
    assert {"parents_resolve", "spans_named"} <= failed


def test_dlq_fails_and_zero_tokens_passes():
    def route(url):
        if "/spans" in url:
            return healthy(url)
        return ok(dict(TRACE, finalizationStatus="dlq", totalTokens=0))

    code, out, _, _ = run(route)
    assert code == 1
    checks = {c["name"]: c["status"] for c in json.loads(out[0])["checks"]}
    assert checks["finalized"] == "fail"
    assert checks["llm_token_usage"] == "pass"


def test_exit_codes_and_token_not_printed():
    assert run((404, b""))[0] == 2
    code, _, err, _ = run((401, b""))
    assert code == 3
    assert "tok-secret" not in "".join(err)
    assert run((403, b""))[0] == 3
    assert run((500, b""))[0] == 5
    assert run((429, b""))[0] == 5
    assert run((409, b""))[0] == 5
    assert run((200, b"not json"))[0] == 5


def test_409_on_spans_page_is_not_ready():
    code, _, _, _ = run(lambda url: (409, b"") if "/spans" in url else ok(TRACE))
    assert code == 2


def test_needs_token_and_project_id():
    assert run(env={"NEATLOGS_PROJECT_ID": "p"})[0] == 3
    assert run(env={"NEATLOGS_TOKEN": "t"})[0] == 3


def test_host_override_and_id_encoding():
    _, _, _, calls = run(
        env=dict(ENV, NEATLOGS_HOST="https://eu.app.neatlogs.com"), trace_id="a/b"
    )
    assert calls[0][0] == "https://eu.app.neatlogs.com/api/v1/public/traces/a%2Fb"


def test_check_trace_empty():
    assert check_trace({}, [])[0]["status"] == "fail"


def test_main_dispatch_usage(capsys):
    assert main(["trace"]) == 4
