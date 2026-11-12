"""``jobs/<job>/neatlogs-sync.json``: what reached Neatlogs and what did not.

The job folder already holds every input the integration needs, so the ledger
stores state, not payloads: server ids for the dataset and runs, and per trial
whether its output and trace were sent. ``neatlogs harbor sync <job dir>``
rebuilds anything marked pending from the files and sends it again.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LEDGER_FILENAME = "neatlogs-sync.json"
SCHEMA_VERSION = 1

SENT = "sent"
PENDING = "pending"
SKIPPED = "skipped"


class Ledger:
    def __init__(self, path: Path, job_id: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "datasets": {},
            "runs": {},
            "trials": {},
            "errors": [],
        }
        existing = _read(path)
        if existing and existing.get("job_id") == job_id:
            self._data.update(existing)

    @classmethod
    def for_job_dir(cls, job_dir: Path, job_id: str) -> Ledger:
        return cls(job_dir / LEDGER_FILENAME, job_id)

    # -- reads

    def dataset(self, name: str) -> dict[str, Any]:
        return dict(self._data["datasets"].get(name) or {})

    def run(self, run_id: str) -> dict[str, Any]:
        return dict(self._data["runs"].get(run_id) or {})

    def trial(self, trial_name: str) -> dict[str, Any]:
        return dict(self._data["trials"].get(trial_name) or {})

    def trial_names(self) -> list[str]:
        return list(self._data["trials"])

    def pending_trials(self) -> list[str]:
        return [
            name
            for name, state in self._data["trials"].items()
            if PENDING in (state.get("output"), state.get("trace"))
        ]

    @property
    def has_pending(self) -> bool:
        if self.pending_trials():
            return True
        return any(
            not run.get("server_id") or run.get("completed") is False
            for run in self._data["runs"].values()
        ) or any(not d.get("version_id") for d in self._data["datasets"].values())

    # -- writes

    def update_dataset(self, name: str, **values: Any) -> None:
        self._merge("datasets", name, values)

    def update_run(self, run_id: str, **values: Any) -> None:
        self._merge("runs", run_id, values)

    def update_trial(self, trial_name: str, **values: Any) -> None:
        self._merge("trials", trial_name, values)

    def record_error(self, where: str, error: BaseException | str) -> None:
        with self._lock:
            errors = self._data.setdefault("errors", [])
            errors.append({"at": _now(), "where": where, "error": str(error)[:500]})
            del errors[:-50]
            self._write()

    def clear_errors(self) -> None:
        with self._lock:
            self._data["errors"] = []
            self._write()

    def _merge(self, section: str, key: str, values: dict[str, Any]) -> None:
        with self._lock:
            entry = self._data[section].setdefault(key, {})
            entry.update({k: v for k, v in values.items()})
            self._write()

    def _write(self) -> None:
        self._data["updated_at"] = _now()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".neatlogs-sync.", dir=self.path.parent)
            with os.fdopen(fd, "w") as handle:
                json.dump(self._data, handle, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except OSError:
            # The ledger is best effort: losing it costs a re-send, never the run.
            pass


def _read(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
