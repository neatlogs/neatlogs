from neatlogs.core.pipeline import DEFAULT_MAX_SPAN_ATTRIBUTES, default_span_limits


def test_default_limit_is_raised(monkeypatch):
    monkeypatch.delenv("OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", raising=False)
    monkeypatch.delenv("OTEL_ATTRIBUTE_COUNT_LIMIT", raising=False)
    assert default_span_limits().max_span_attributes == DEFAULT_MAX_SPAN_ATTRIBUTES


def test_env_limit_is_respected(monkeypatch):
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", "50")
    assert default_span_limits().max_span_attributes == 50


def test_client_uses_shared_limits(monkeypatch):
    monkeypatch.setenv("OTEL_ATTRIBUTE_COUNT_LIMIT", "77")
    from neatlogs.client import Client

    client = Client(api_key="nl_test", workflow_name="limits", disable_export=True)
    try:
        assert client.tracer_provider._span_limits.max_span_attributes == 77
    finally:
        client.shutdown()
