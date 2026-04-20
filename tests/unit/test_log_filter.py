"""
Tests for NeatlogsLogFilter — the filtering LogRecordProcessor that drops
external-module and no-trace log records before forwarding to OTLPLogExporter.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from neatlogs.core.log_exporter import NeatlogsLogFilter, _is_external_module


# ---------------------------------------------------------------------------
# _is_external_module
# ---------------------------------------------------------------------------


def test_is_external_module_empty_string():
    assert _is_external_module("") is False


def test_is_external_module_user_logger():
    assert _is_external_module("my_app.services.auth") is False


def test_is_external_module_neatlogs_itself():
    # neatlogs is not in site-packages in the dev environment
    assert _is_external_module("neatlogs") is False


def test_is_external_module_stdlib(monkeypatch):
    # Simulate Python 3.10+ stdlib_module_names containing "asyncio"
    monkeypatch.setattr("neatlogs.core.log_exporter._STDLIB_MODULE_NAMES", frozenset({"asyncio"}))
    assert _is_external_module("asyncio") is True
    assert _is_external_module("asyncio.tasks") is True


def test_is_external_module_site_packages(monkeypatch):
    # Simulate a module whose __file__ is in site-packages
    fake_mod = MagicMock()
    fake_mod.__file__ = "/usr/lib/python3/site-packages/httpcore/_sync/http11.py"
    monkeypatch.setitem(__import__("sys").modules, "httpcore", fake_mod)
    assert _is_external_module("httpcore") is True
    assert _is_external_module("httpcore._sync.http11") is True


# ---------------------------------------------------------------------------
# Helpers to build minimal LogData-like objects
# ---------------------------------------------------------------------------


def _make_log_data(
    trace_id: int = 0xAA * (16 * 8),  # non-zero
    span_id: int = 0xBB * (8 * 8),
    scope_name: str = "my_app",
    filepath: str = "",
    body: str = "test message",
) -> MagicMock:
    log_record = MagicMock()
    log_record.trace_id = trace_id
    log_record.span_id = span_id
    log_record.attributes = {"code.filepath": filepath} if filepath else {}
    log_record.body = body

    scope = MagicMock()
    scope.name = scope_name

    log_data = MagicMock()
    log_data.log_record = log_record
    log_data.instrumentation_scope = scope
    return log_data


# ---------------------------------------------------------------------------
# NeatlogsLogFilter.emit
# ---------------------------------------------------------------------------


def test_filter_passes_normal_record():
    downstream = MagicMock()
    f = NeatlogsLogFilter(downstream)
    log_data = _make_log_data(trace_id=0xDEADBEEF, scope_name="my_app")
    f.on_emit(log_data)
    downstream.on_emit.assert_called_once_with(log_data)


def test_filter_drops_no_trace_id_zero():
    downstream = MagicMock()
    f = NeatlogsLogFilter(downstream)
    log_data = _make_log_data(trace_id=0)
    f.on_emit(log_data)
    downstream.on_emit.assert_not_called()


def test_filter_drops_no_trace_id_none():
    downstream = MagicMock()
    f = NeatlogsLogFilter(downstream)
    log_data = _make_log_data()
    log_data.log_record.trace_id = None
    f.on_emit(log_data)
    downstream.on_emit.assert_not_called()


def test_filter_drops_stdlib_scope(monkeypatch):
    monkeypatch.setattr("neatlogs.core.log_exporter._STDLIB_MODULE_NAMES", frozenset({"logging"}))
    downstream = MagicMock()
    f = NeatlogsLogFilter(downstream)
    log_data = _make_log_data(trace_id=0xDEADBEEF, scope_name="logging.handlers")
    f.on_emit(log_data)
    downstream.on_emit.assert_not_called()


def test_filter_drops_site_packages_scope(monkeypatch):
    fake_mod = MagicMock()
    fake_mod.__file__ = "/usr/lib/python3/site-packages/httpcore/__init__.py"
    monkeypatch.setitem(__import__("sys").modules, "httpcore", fake_mod)
    downstream = MagicMock()
    f = NeatlogsLogFilter(downstream)
    log_data = _make_log_data(trace_id=0xDEADBEEF, scope_name="httpcore")
    f.on_emit(log_data)
    downstream.on_emit.assert_not_called()


def test_filter_drops_site_packages_filepath():
    downstream = MagicMock()
    f = NeatlogsLogFilter(downstream)
    log_data = _make_log_data(
        trace_id=0xDEADBEEF,
        scope_name="my_app",
        filepath="/usr/lib/python3/site-packages/openai/_client.py",
    )
    f.on_emit(log_data)
    downstream.on_emit.assert_not_called()


def test_filter_drops_dist_packages_filepath():
    downstream = MagicMock()
    f = NeatlogsLogFilter(downstream)
    log_data = _make_log_data(
        trace_id=0xDEADBEEF,
        scope_name="my_app",
        filepath="/usr/lib/python3/dist-packages/urllib3/connectionpool.py",
    )
    f.on_emit(log_data)
    downstream.on_emit.assert_not_called()


# ---------------------------------------------------------------------------
# NeatlogsLogFilter delegation — shutdown / force_flush
# ---------------------------------------------------------------------------


def test_filter_shutdown_delegates():
    downstream = MagicMock()
    f = NeatlogsLogFilter(downstream)
    f.shutdown()
    downstream.shutdown.assert_called_once()


def test_filter_force_flush_delegates():
    downstream = MagicMock()
    downstream.force_flush.return_value = True
    f = NeatlogsLogFilter(downstream)
    result = f.force_flush(5000)
    downstream.force_flush.assert_called_once_with(5000)
    assert result is True
