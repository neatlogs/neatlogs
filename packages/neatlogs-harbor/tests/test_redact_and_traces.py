from __future__ import annotations

from pathlib import Path

from neatlogs_harbor import ids
from neatlogs_harbor.records import load_trial_dir, outcome_of, read_trajectory
from neatlogs_harbor.redact import REDACTED, Redactor, split_patterns
from neatlogs_harbor.traces import build_trial_trace

from .conftest import FAKE_GITHUB_TOKEN, FAKE_OPENAI_KEY


def test_token_shaped_strings_are_redacted():
    redactor = Redactor()
    aws = "AKIA" + "Q" * 16
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJlMTIz"
    text = (
        f"key {FAKE_OPENAI_KEY} gh {FAKE_GITHUB_TOKEN} aws {aws} jwt {jwt} "
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345 password=hunter2hunter2"
    )
    cleaned = redactor.text(text)
    for secret in (
        FAKE_OPENAI_KEY,
        FAKE_GITHUB_TOKEN,
        aws,
        jwt,
        "abcdefghijklmnopqrstuvwxyz012345",
    ):
        assert secret not in cleaned
    assert "password=" + REDACTED in cleaned
    assert "max_tokens: 1000" == redactor.text("max_tokens: 1000")


def test_secret_env_values_and_extra_patterns_are_redacted():
    redactor = Redactor.from_environment(
        split_patterns(r"ACME-\d{6};;internal\.example\.com"),
        environ={"MY_SERVICE_TOKEN": "s3cr3t-value-without-shape", "HOME": "/Users/someone"},
    )
    cleaned = redactor.text("t=s3cr3t-value-without-shape id ACME-123456 host internal.example.com")
    assert "s3cr3t" not in cleaned and "ACME-123456" not in cleaned
    assert "internal.example.com" not in cleaned
    assert redactor.text("/Users/someone") == "/Users/someone"


def test_redactor_walks_nested_values():
    redactor = Redactor()
    value = {"a": [f"x {FAKE_GITHUB_TOKEN}", 3], "b": {"c": FAKE_OPENAI_KEY}}
    assert FAKE_GITHUB_TOKEN not in str(redactor.value(value))
    assert redactor.value(value)["a"][1] == 3


def _spans(request):
    (resource_spans,) = request.resource_spans
    return resource_spans, list(resource_spans.scope_spans[0].spans)


def _attr(span, key):
    for kv in span.attributes:
        if kv.key == key:
            return getattr(kv.value, kv.value.WhichOneof("value"))
    return None


def test_trial_trace_nests_the_atif_trajectory_under_the_agent_phase(mixed_job: Path):
    trial = load_trial_dir(mixed_job / "hello-world__Atraj01")
    request = build_trial_trace(
        trial,
        outcome_of(trial.result),
        trajectory=read_trajectory(trial.trial_dir),
        instruction="Create hello.txt",
        resource_attributes={"harbor.job.name": "mixed"},
        redactor=Redactor(),
    )
    resource, spans = _spans(request)
    trace_ids = {span.trace_id.hex() for span in spans}
    assert trace_ids == {ids.trace_id_hex(trial.trial_uuid)}

    by_name = {span.name: span for span in spans}
    root = by_name["harbor.trial"]
    assert not root.parent_span_id
    assert [s.name for s in spans if not s.parent_span_id] == ["harbor.trial"]
    agent_phase = by_name["harbor.agent"]
    atif_root = next(s for s in spans if _attr(s, "agent.name") == "terminus-2" and s is not root)
    assert atif_root.parent_span_id == agent_phase.span_id
    assert any(_attr(s, "openinference.span.kind") == "TOOL" for s in spans)
    assert any(_attr(s, "openinference.span.kind") == "LLM" for s in spans)

    assert _attr(root, "neatlogs.origin") == "simulation"
    assert _attr(root, "harbor.reward") == 1.0
    assert {kv.key for kv in resource.resource.attributes} >= {
        "service.name",
        "neatlogs.origin",
        "deployment.environment",
    }
    # The agent exported an API key in a tool call; it never leaves the machine.
    assert FAKE_OPENAI_KEY not in request.SerializeToString().decode("latin-1")


def test_trial_without_trajectory_still_gets_a_phase_trace(oracle_job: Path):
    trial = load_trial_dir(oracle_job / "hello-world__KMi3fnx")
    request = build_trial_trace(
        trial,
        outcome_of(trial.result),
        trajectory=None,
        instruction=None,
        resource_attributes={},
        redactor=Redactor(),
    )
    _, spans = _spans(request)
    assert [s.name for s in spans] == [
        "harbor.trial",
        "harbor.environment_setup",
        "harbor.agent_setup",
        "harbor.agent",
        "harbor.verifier",
    ]


def test_failed_trials_mark_the_root_span_as_an_error(mixed_job: Path):
    trial = load_trial_dir(mixed_job / "hello-world__Csbx003")
    request = build_trial_trace(
        trial,
        outcome_of(trial.result),
        trajectory=None,
        instruction=None,
        resource_attributes={},
        redactor=Redactor(),
    )
    _, spans = _spans(request)
    root = spans[0]
    assert root.status.code == root.status.STATUS_CODE_ERROR
    assert "SandboxBuildFailedError" in root.status.message
    assert FAKE_GITHUB_TOKEN not in root.status.message
    assert _attr(root, "harbor.error_class") == "infrastructure"
