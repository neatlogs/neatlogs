import sys
import threading
import time

import pytest

import neatlogs

init_module = sys.modules["neatlogs.init"]


class _BlockingProcessor:
    def __init__(self):
        self.release = threading.Event()

    def force_flush(self, timeout_millis=30000):
        self.release.wait(5)
        return True


def test_default_flush_returns_false_at_its_deadline(monkeypatch):
    blocker = _BlockingProcessor()
    monkeypatch.setattr(init_module, "_transport_span_processors", [blocker])
    monkeypatch.setattr(init_module, "_log_provider", None)

    started = time.monotonic()
    ok = neatlogs.flush(timeout_millis=200)
    elapsed = time.monotonic() - started
    blocker.release.set()

    assert ok is False
    assert elapsed < 2


def test_default_flush_still_returns_true_when_fast(monkeypatch):
    class Fast:
        def force_flush(self, timeout_millis=30000):
            return True

    monkeypatch.setattr(init_module, "_transport_span_processors", [Fast()])
    monkeypatch.setattr(init_module, "_log_provider", None)
    assert neatlogs.flush(timeout_millis=1000) is True


def test_timed_out_flush_never_touches_a_pipeline_created_afterwards(monkeypatch):
    release = threading.Event()
    finished = threading.Event()

    class SlowLogs:
        def force_flush(self, timeout_millis=30000):
            release.wait(5)
            finished.set()
            return True

    class NewProcessor:
        calls = 0

        def force_flush(self, timeout_millis=30000):
            NewProcessor.calls += 1
            return True

    class OldProcessor(NewProcessor):
        calls = 0

        def force_flush(self, timeout_millis=30000):
            OldProcessor.calls += 1
            return True

    monkeypatch.setattr(init_module, "_log_provider", SlowLogs())
    monkeypatch.setattr(init_module, "_transport_span_processors", [OldProcessor()])

    assert neatlogs.flush(timeout_millis=50) is False

    # shutdown() + init() would replace the default pipeline while the old flush is still stuck.
    new_processor = NewProcessor()
    monkeypatch.setattr(init_module, "_log_provider", None)
    monkeypatch.setattr(init_module, "_transport_span_processors", [new_processor])

    release.set()
    assert finished.wait(2)
    time.sleep(0.2)

    assert NewProcessor.calls == 0
    assert OldProcessor.calls == 0


@pytest.fixture
def client():
    value = neatlogs.Client(
        api_key="flush-test-key",
        workflow_name="flush-deadline",
        disable_export=True,
    )
    yield value
    value.shutdown()


@pytest.mark.parametrize("stage", ["logs", "traces"])
def test_client_flush_returns_false_at_blocking_stage_deadline(client, monkeypatch, stage):
    blocker = _BlockingProcessor()
    with monkeypatch.context() as patch:
        patch.setattr(client, "log_provider", blocker if stage == "logs" else None)
        patch.setattr(client, "_transport_processors", [blocker] if stage == "traces" else [])
        try:
            started = time.monotonic()
            assert client.flush(timeout_millis=50) is False
            assert time.monotonic() - started < 1
        finally:
            blocker.release.set()


def test_client_flush_success_preserves_stage_order(client, monkeypatch):
    calls = []

    class Fast:
        def __init__(self, name):
            self.name = name

        def force_flush(self, timeout_millis=30000):
            calls.append(self.name)
            return True

    with monkeypatch.context() as patch:
        patch.setattr(client, "log_provider", Fast("logs"))
        patch.setattr(client, "_transport_processors", [Fast("traces"), Fast("completion")])
        assert client.flush(timeout_millis=1000) is True
    assert calls == ["logs", "traces", "completion", "traces"]


@pytest.mark.parametrize("raises", [False, True])
def test_client_flush_reports_processor_failure(client, monkeypatch, raises):
    class Failed:
        def force_flush(self, timeout_millis=30000):
            if raises:
                raise RuntimeError("export failed")
            return False

    with monkeypatch.context() as patch:
        patch.setattr(client, "log_provider", None)
        patch.setattr(client, "_transport_processors", [Failed()])
        assert client.flush(timeout_millis=1000) is False


def test_timed_out_client_flush_keeps_its_pipeline_snapshot(client, monkeypatch):
    release = threading.Event()
    finished = threading.Event()
    calls = []
    original = neatlogs.Client._flush_pipeline

    class SlowLogs:
        def force_flush(self, timeout_millis=30000):
            release.wait(5)
            return True

    class Processor:
        def __init__(self, name):
            self.name = name

        def force_flush(self, timeout_millis=30000):
            calls.append(self.name)
            return True

    def tracked(*args):
        try:
            return original(*args)
        finally:
            finished.set()

    other = neatlogs.Client(
        api_key="other-flush-key", workflow_name="other-flush", disable_export=True
    )
    try:
        with monkeypatch.context() as patch:
            patch.setattr(neatlogs.Client, "_flush_pipeline", staticmethod(tracked))
            patch.setattr(client, "log_provider", SlowLogs())
            patch.setattr(client, "_transport_processors", [Processor("old-client")])
            patch.setattr(other, "_transport_processors", [Processor("other-client")])
            patch.setattr(init_module, "_transport_span_processors", [Processor("default")])
            assert client.flush(timeout_millis=50) is False
            patch.setattr(client, "log_provider", None)
            patch.setattr(client, "_transport_processors", [Processor("new-client")])
            release.set()
            assert finished.wait(2)
            assert calls == []
            assert other.flush(timeout_millis=1000) is True
            assert calls == ["other-client", "other-client"]
    finally:
        release.set()
        other.shutdown()
