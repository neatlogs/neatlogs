import logging

import pytest

import neatlogs


@pytest.fixture(autouse=True)
def _capture_neatlogs_logger(caplog):
    # the neatlogs logger does not propagate to root, so attach the handler directly
    logger = logging.getLogger("neatlogs")
    logger.addHandler(caplog.handler)
    yield
    logger.removeHandler(caplog.handler)


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_run_with_no_key_and_no_instrumentations_warns_at_shutdown(monkeypatch, caplog):
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)
    monkeypatch.delenv("NEATLOGS_DISABLE_EXPORT", raising=False)
    with caplog.at_level(logging.WARNING, logger="neatlogs"):
        neatlogs.init(workflow_name="quiet")
        neatlogs.shutdown()

    messages = _warnings(caplog)
    assert len(messages) == 1
    assert "recorded no spans" in messages[0]
    assert "NEATLOGS_API_KEY" in messages[0]
    assert "instrumentations" in messages[0]


def test_run_that_recorded_a_span_stays_quiet(monkeypatch, caplog):
    monkeypatch.setenv("NEATLOGS_API_KEY", "test-key")
    monkeypatch.delenv("NEATLOGS_DISABLE_EXPORT", raising=False)
    with caplog.at_level(logging.WARNING, logger="neatlogs"):
        neatlogs.init(workflow_name="busy", disable_export=True)
        with neatlogs.trace("work"):
            pass
        neatlogs.shutdown()

    assert _warnings(caplog) == []
