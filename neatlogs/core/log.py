"""
Deterministic log capture — timestamped step capture inside traced code blocks.
"""

from __future__ import annotations

import io
import logging as _stdlib_logging
import sys
import time
from typing import Any

from opentelemetry._logs import LogRecord, SeverityNumber

# Loggers used internally — LoggingInstrumentor routes these to NeatlogsLogExporter
_user_logger = _stdlib_logging.getLogger("neatlogs.user")
_stdout_logger = _stdlib_logging.getLogger("neatlogs.stdout")

_SEVERITY_MAP: dict[str, SeverityNumber] = {
    "debug": SeverityNumber.DEBUG,
    "info": SeverityNumber.INFO,
    "warning": SeverityNumber.WARN,
    "warn": SeverityNumber.WARN,
    "error": SeverityNumber.ERROR,
    "critical": SeverityNumber.FATAL,
    "stdout": SeverityNumber.INFO,
}

_STDLIB_LEVEL_MAP: dict[str, int] = {
    "debug": _stdlib_logging.DEBUG,
    "info": _stdlib_logging.INFO,
    "warning": _stdlib_logging.WARNING,
    "warn": _stdlib_logging.WARNING,
    "error": _stdlib_logging.ERROR,
    "critical": _stdlib_logging.CRITICAL,
    "stdout": _stdlib_logging.INFO,
}


def _console_echo(rendered: str, data: dict[str, Any], level: str) -> None:
    """Print log entry to stderr immediately. Auto-detects TTY for color support."""
    timestamp = time.strftime("%H:%M:%S")
    use_color = sys.stderr.isatty() if hasattr(sys.stderr, "isatty") else False

    if use_color:
        RESET = "\033[0m"
        DIM = "\033[2m"
        CYAN = "\033[36m"
        YELLOW = "\033[33m"
        BOLD = "\033[1m"
        level_color = YELLOW if level in ("warning", "stdout") else CYAN
        kv_str = "  ".join(f"{BOLD}{k}{RESET}={v!r}" for k, v in data.items())
        print(
            f"{DIM}[neatlogs]{RESET} {DIM}{timestamp}{RESET}  "
            f"{level_color}LOG{RESET}  {rendered}" + (f"  {kv_str}" if kv_str else ""),
            file=sys.stderr,
        )
    else:
        kv_str = "  ".join(f"{k}={v!r}" for k, v in data.items())
        print(
            f"[neatlogs] {timestamp}  LOG  {rendered}" + (f"  {kv_str}" if kv_str else ""),
            file=sys.stderr,
        )


def log(msg_template: str, /, level: str = "info", **data: Any) -> None:
    """
    Capture a timestamped step inside a traced code block.

    Emits an OTel LogRecord via the global LoggerProvider (set up in neatlogs.init()).
    The LogRecord automatically picks up trace_id and span_id from the active span
    context, so the log entry appears as a child of the current span in the timeline.

    The message template (with {key} placeholders) is stored as span_name
    (low-cardinality, good for search/aggregation in ClickHouse). The rendered
    message is stored as input.value. Each keyword argument is stored as log.{key}.

    When neatlogs.init(debug=True), the rendered message is echoed to stderr
    immediately so developers see steps in real time without opening the dashboard.

    Args:
        msg_template: Message template with optional {key} placeholders.
                      Stored as span name. Example: "retrieved {count} docs"
        level: Log level — "info", "debug", "warning", "error", "stdout"
        **data: Structured key-value data. Keys are rendered into the template
                AND stored as individual log.{key} span attributes.
                Use _max_depth=N to limit serialization depth.

    Example:
        >>> @neatlogs.span(kind="CHAIN")
        >>> def rag_pipeline(query: str) -> str:
        ...     docs = retrieve(query)
        ...     neatlogs.log("retrieved {count} docs, top score {score:.2f}",
        ...                  count=len(docs), score=docs[0].score)
        ...     return llm.call(docs)

    Note:
        You don't need neatlogs.log() for basic capture. Any logging.info() /
        logging.warning() call inside a @span() is auto-captured by neatlogs.init().
        neatlogs.log() adds structured template support on top of that.
    """
    from ..decorators._base import _safe_json_dumps

    # Pop reserved internal kwargs
    data.pop("_max_depth", None)

    # Render the message (gracefully handle missing keys)
    try:
        rendered = msg_template.format_map(data)
    except (KeyError, ValueError):
        rendered = msg_template

    # Echo to terminal immediately when debug=True
    from ..init import is_debug_enabled

    if is_debug_enabled():
        _console_echo(rendered, data, level)

    # Build structured attributes for the LogRecord
    attributes: dict[str, Any] = {
        "log.template": msg_template,
        "log.level": level,
    }
    for key, value in data.items():
        attributes[f"log.{key}"] = _safe_json_dumps(value)

    # Emit via OTel Logs API — trace_id + span_id are auto-populated from active context
    try:
        from opentelemetry import logs as otel_logs
    except ImportError:
        from opentelemetry import _logs as otel_logs  # type: ignore[no-redef]

    otel_logger = otel_logs.get_logger(__name__)
    otel_logger.emit(
        LogRecord(
            timestamp=time.time_ns(),
            severity_number=_SEVERITY_MAP.get(level.lower(), SeverityNumber.INFO),
            severity_text=level.upper(),
            body=rendered,
            attributes=attributes,
        )
    )


class _CaptureStdoutContext:
    """
    Context manager that intercepts stdout and routes each line through
    Python's logging module (neatlogs.stdout logger), which LoggingInstrumentor
    bridges to OTel LogRecords → NeatlogsLogExporter → ClickHouse LOG spans.

    Used internally by @span(capture_stdout=True) and
    with neatlogs.trace(..., capture_stdout=True).
    """

    def __init__(self) -> None:
        self._original_stdout: Any = None
        self._writer: _LineBufferedLogWriter | None = None

    def __enter__(self) -> "_CaptureStdoutContext":
        self._original_stdout = sys.stdout
        self._writer = _LineBufferedLogWriter(self._original_stdout)
        sys.stdout = self._writer
        return self

    def __exit__(self, *_: Any) -> None:
        if self._writer is not None:
            self._writer.flush_remaining()
        sys.stdout = self._original_stdout


class _LineBufferedLogWriter(io.TextIOBase):
    """
    A stdout replacement that buffers output line-by-line and emits each line
    to _stdout_logger (neatlogs.stdout). LoggingInstrumentor captures these
    via its LoggingHandler and converts them to OTel LogRecords.

    Also mirrors output to the real stdout so developers still see it in their
    terminal.
    """

    def __init__(self, real_stdout: Any) -> None:
        self._real = real_stdout
        self._buf = ""

    def write(self, s: str) -> int:
        # Mirror to real stdout
        self._real.write(s)
        self._buf += s
        # Emit a log call for each complete line
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                _stdout_logger.info(line)
        return len(s)

    def flush(self) -> None:
        self._real.flush()

    def flush_remaining(self) -> None:
        """Emit any buffered content that didn't end with a newline."""
        if self._buf.strip():
            _stdout_logger.info(self._buf.strip())
            self._buf = ""

    @property
    def encoding(self) -> str:
        return getattr(self._real, "encoding", "utf-8")

    @property
    def errors(self) -> str:
        return getattr(self._real, "errors", "strict")

    def fileno(self) -> int:
        return self._real.fileno()

    def isatty(self) -> bool:
        return False
