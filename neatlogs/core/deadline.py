"""Bound potentially blocking exporter/lifecycle calls without process hangs."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from typing import Any


def bounded_call(operation: Callable[[], Any], deadline: float) -> tuple[bool, Any]:
    """Run one close operation by a monotonic deadline.

    Python cannot forcibly cancel arbitrary exporter or user callback code, so
    the worker is daemonized. A timeout lets the SDK report loss and detach the
    closed generation without keeping interpreter exit alive.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False, TimeoutError("Neatlogs shutdown deadline exceeded")
    completed: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            completed.put_nowait((True, operation()))
        except BaseException as exc:
            completed.put_nowait((False, exc))

    threading.Thread(target=run, name="neatlogs-shutdown", daemon=True).start()
    try:
        return completed.get(timeout=remaining)
    except queue.Empty:
        return False, TimeoutError("Neatlogs shutdown deadline exceeded")
