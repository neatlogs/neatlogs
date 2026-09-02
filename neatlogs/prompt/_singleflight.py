from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Generic, Optional, TypeVar, cast

T = TypeVar("T")


@dataclass
class _SyncFlight(Generic[T]):
    event: threading.Event = field(default_factory=threading.Event)
    error: Optional[BaseException] = None
    result: Optional[T] = None
    has_result: bool = False


class SyncSingleFlight(Generic[T]):
    """Run one synchronous operation per key and share its outcome with waiters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._flights: Dict[str, _SyncFlight[T]] = {}

    def run(self, key: str, operation: Callable[[], T]) -> T:
        with self._lock:
            flight = self._flights.get(key)
            leader = flight is None
            if flight is None:
                flight = _SyncFlight()
                self._flights[key] = flight

        if not leader:
            flight.event.wait()
            if flight.error is not None:
                raise flight.error
            if not flight.has_result:
                raise RuntimeError("Single-flight operation completed without a result")
            return cast(T, flight.result)

        try:
            flight.result = operation()
            flight.has_result = True
            return cast(T, flight.result)
        except BaseException as error:
            flight.error = error
            raise
        finally:
            flight.event.set()
            with self._lock:
                if self._flights.get(key) is flight:
                    self._flights.pop(key, None)


class AsyncSingleFlight(Generic[T]):
    """Run one event-loop task per key without transferring caller cancellation."""

    def __init__(self) -> None:
        self._tasks: Dict[str, asyncio.Task[T]] = {}

    async def run(self, key: str, operation: Callable[[], Awaitable[T]]) -> T:
        task = self._tasks.get(key)
        if task is None:
            task = asyncio.create_task(operation())
            self._tasks[key] = task

            def clear_flight(completed: asyncio.Task[T]) -> None:
                if self._tasks.get(key) is completed:
                    self._tasks.pop(key, None)
                # A shielded task can outlive every cancelled waiter. Observe its
                # terminal exception so asyncio does not report it as unhandled.
                if not completed.cancelled():
                    completed.exception()

            task.add_done_callback(clear_flight)
        return await asyncio.shield(task)
