"""Bound potentially blocking exporter/lifecycle calls without process hangs."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from typing import Any


class DeadlineWorker:
    """A daemon worker started before interpreter finalization.

    Python 3.12 forbids starting a thread from an ``atexit`` callback. Pipelines
    create this worker during initialization, so exit cleanup can still abandon
    a blocking provider/exporter call at the configured deadline.
    """

    def __init__(self, name: str = "neatlogs-shutdown") -> None:
        self._jobs: queue.Queue[tuple[Callable[[], Any], queue.Queue[tuple[bool, Any]]] | None]
        self._jobs = queue.Queue()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            operation, completed = job
            try:
                result = operation()
            except BaseException as exc:
                completed.put_nowait((False, exc))
            else:
                completed.put_nowait((True, result))

    def call(self, operation: Callable[[], Any], deadline: float) -> tuple[bool, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self._thread.is_alive():
            return False, TimeoutError("Neatlogs shutdown deadline exceeded")
        completed: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)
        self._jobs.put_nowait((operation, completed))
        try:
            return completed.get(timeout=remaining)
        except queue.Empty:
            return False, TimeoutError("Neatlogs shutdown deadline exceeded")

    def is_current(self) -> bool:
        return self._thread.ident == threading.get_ident()

    def close(self) -> None:
        self._jobs.put_nowait(None)


def bounded_call(
    operation: Callable[[], Any],
    deadline: float,
    *,
    synchronous: bool = False,
    worker: DeadlineWorker | None = None,
) -> tuple[bool, Any]:
    """Run one close operation by a monotonic deadline.

    Python cannot forcibly cancel arbitrary exporter or user callback code, so
    the worker is daemonized. A timeout lets the SDK report loss and detach the
    closed generation without keeping interpreter exit alive.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False, TimeoutError("Neatlogs shutdown deadline exceeded")
    if worker is not None:
        return worker.call(operation, deadline)
    if synchronous:
        return False, RuntimeError("synchronous shutdown requires a prestarted worker")
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


class _FlushFlight:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.result: tuple[bool, Any] | None = None


class FlushGate:
    """Keep flush work single-flight for one pipeline generation.

    Python cannot cancel a blocked exporter call, so a timed-out flush leaves
    its worker running. Callers that arrive while that worker is still busy
    wait on the same outcome up to their own deadline instead of spawning
    another thread that would pile onto the same exporter. Each pipeline
    generation gets its own gate, so a stuck flush never holds the generation
    that replaces it.
    """

    def __init__(self, name: str = "neatlogs-flush") -> None:
        self._name = name
        self._lock = threading.Lock()
        self._flight: _FlushFlight | None = None

    def call(self, operation: Callable[[], Any], deadline: float) -> tuple[bool, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False, TimeoutError("Neatlogs flush deadline exceeded")
        with self._lock:
            flight = self._flight
            if flight is None:
                flight = _FlushFlight()
                self._flight = flight
                threading.Thread(
                    target=self._run,
                    args=(flight, operation),
                    name=self._name,
                    daemon=True,
                ).start()
        if not flight.done.wait(remaining):
            return False, TimeoutError("Neatlogs flush deadline exceeded")
        assert flight.result is not None
        return flight.result

    def _run(self, flight: _FlushFlight, operation: Callable[[], Any]) -> None:
        try:
            result: tuple[bool, Any] = (True, operation())
        except BaseException as exc:
            result = (False, exc)
        with self._lock:
            flight.result = result
            if self._flight is flight:
                self._flight = None
            flight.done.set()
