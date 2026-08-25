"""Thread-safe weak registry of live Neatlogs client pipelines."""

import threading
import weakref
from typing import Any

_lock = threading.RLock()
_clients: weakref.WeakSet[Any] = weakref.WeakSet()


def register_client(client: Any) -> None:
    with _lock:
        _clients.add(client)


def unregister_client(client: Any) -> None:
    with _lock:
        _clients.discard(client)


def snapshot_clients() -> tuple[Any, ...]:
    with _lock:
        return tuple(_clients)
