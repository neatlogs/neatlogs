import json

from neatlogs.doctor import diagnose, main


def _write_jsonl(path, spans):
    path.write_text("\n".join(json.dumps(span) for span in spans), encoding="utf-8")


def _span(
    trace_id,
    span_id,
    *,
    name,
    kind,
    parent_span_id=None,
    attributes=None,
):
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "kind": kind,
        "attributes": attributes or {"neatlogs.span.kind": kind},
        "resource": {"attributes": {}},
    }


def test_diagnose_healthy_workflow_has_no_findings(tmp_path):
    path = tmp_path / "spans.log"
    _write_jsonl(
        path,
        [
            _span("trace-1", "root", name="workflow", kind="workflow"),
            _span(
                "trace-1",
                "llm",
                name="chat.completions.create",
                kind="llm",
                parent_span_id="root",
                attributes={
                    "neatlogs.span.kind": "llm",
                    "neatlogs.llm.input_messages.0.role": "user",
                    "neatlogs.llm.output_messages.0.role": "assistant",
                },
            ),
        ],
    )

    report = diagnose(path)

    assert report.spans_read == 2
    assert report.trace_count == 1
    assert report.findings == []


def test_diagnose_empty_file_returns_error(tmp_path):
    path = tmp_path / "empty.log"
    path.write_text("", encoding="utf-8")

    report = diagnose(path)

    assert report.has_errors is True
    assert [finding.code for finding in report.findings] == ["no-spans"]


def test_diagnose_rootless_http_only_trace(tmp_path):
    path = tmp_path / "http.log"
    _write_jsonl(
        path,
        [
            _span("trace-1", "http-1", name="GET", kind="http"),
            _span("trace-1", "http-2", name="POST", kind="http"),
        ],
    )

    report = diagnose(path)

    assert [finding.code for finding in report.findings] == ["rootless-http-only"]
    assert '@span(kind="WORKFLOW")' in report.findings[0].suggestion


def test_diagnose_agent_without_llm_child(tmp_path):
    path = tmp_path / "agent.log"
    _write_jsonl(
        path,
        [
            _span("trace-1", "agent", name="crewai.agent", kind="agent"),
            _span(
                "trace-1",
                "tool",
                name="lookup_user",
                kind="tool",
                parent_span_id="agent",
                attributes={
                    "neatlogs.span.kind": "tool",
                    "neatlogs.tool.parameters": '{"user_id": "u1"}',
                    "neatlogs.tool.output": '{"plan": "pro"}',
                },
            ),
        ],
    )

    report = diagnose(path)

    assert any(finding.code == "agent-without-llm" for finding in report.findings)
    assert "instrumentations" in report.findings[0].suggestion


def test_diagnose_missing_io_on_tool_span(tmp_path):
    path = tmp_path / "tool.log"
    _write_jsonl(
        path,
        [
            _span("trace-1", "root", name="workflow", kind="workflow"),
            _span(
                "trace-1",
                "tool",
                name="send_email",
                kind="tool",
                parent_span_id="root",
                attributes={
                    "neatlogs.span.kind": "tool",
                    "neatlogs.tool.parameters": '{"email": "a@example.com"}',
                },
            ),
        ],
    )

    report = diagnose(path)

    assert [finding.code for finding in report.findings] == ["tool-missing-io"]
    assert "send_email" in report.findings[0].evidence


def test_diagnose_invalid_jsonl_reports_line_number(tmp_path):
    path = tmp_path / "bad.log"
    path.write_text('{"trace_id": "trace-1"}\nnot-json\n', encoding="utf-8")

    report = diagnose(path)

    assert report.invalid_lines == [2]
    assert report.findings[0].code == "invalid-jsonl"


def test_main_prints_human_readable_report(tmp_path, capsys):
    path = tmp_path / "empty.log"
    path.write_text("", encoding="utf-8")

    exit_code = main([str(path)])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "Trace Doctor" in out
    assert "No spans found" in out
