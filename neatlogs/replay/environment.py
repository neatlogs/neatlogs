"""Load explicit local replay files and save a private, bounded activity report."""

from __future__ import annotations

import atexit
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

from .contracts import MAX_RECORDING_BYTES, ReplayError, parse_json_bytes
from .session import ReplaySession, set_environment_replay

_configured = None
_report = None


def bounded_file(path, limit):
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise ReplayError("Replay input must be a bounded regular file.")
            content = stream.read(limit + 1)
            if len(content) > limit:
                raise ReplayError("Replay input exceeds its size limit.")
            return content
    except OSError:
        raise ReplayError("Local replay file could not be read.") from None


def write_private_json(path, value, *, limit=16 * 1024):
    content = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(content) > limit:
        raise ReplayError("Local replay report exceeds its size limit.")
    target = Path(path).absolute()
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=".neatlogs-report-", dir=target.parent)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except OSError:
        raise ReplayError("Local replay report could not be saved.") from None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def save_environment_report():
    if _configured is not None and _report:
        write_private_json(_report, _configured.get_status())


def _exit_report():
    try:
        save_environment_report()
    except Exception:
        sys.stderr.write("Neatlogs could not save the local replay report.\n")


def configure_replay_from_environment():
    global _configured, _report
    file = os.getenv("NEATLOGS_REPLAY_FILE")
    if not file or _configured is not None:
        return
    recording = parse_json_bytes(bounded_file(file, MAX_RECORDING_BYTES))
    mode = os.getenv("NEATLOGS_REPLAY_MODE", "tools_only")
    values = [
        os.getenv(key, "")
        for key in ("NEATLOGS_REPLAY_RUN_ID", "NEATLOGS_REPLAY_CASE_ID", "NEATLOGS_REPLAY_TRIAL")
    ]
    run = None
    if any(values):
        try:
            trial = int(values[2])
        except ValueError:
            raise ReplayError("Replay run bindings require a valid trial.") from None
        run = {"runId": values[0], "caseId": values[1], "trial": trial}
    session = ReplaySession(
        recording, mode=mode, until_span=os.getenv("NEATLOGS_REPLAY_UNTIL") or None, run=run
    )
    source = os.getenv("NEATLOGS_REPLAY_TRACE_ID")
    if source and source != session.recording.source_trace_id:
        raise ReplayError("Replay source trace does not match the recording.")
    set_environment_replay(session)
    _configured = session
    _report = os.getenv("NEATLOGS_REPLAY_REPORT")
    if _report:
        atexit.register(_exit_report)
