"""Cached assignment with bounded, ordered exposure delivery."""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from dataclasses import replace
from typing import Any
from urllib.parse import quote
from uuid import UUID, uuid4

import requests
from opentelemetry import trace
from opentelemetry.instrumentation.utils import suppress_instrumentation

from .._wrap_utils import active_neatlogs_context, get_neatlogs_root_span
from ..core.logger import get_logger
from .protocol import (
    AbAssignment,
    AbDecision,
    ArmKey,
    Config,
    Params,
    assignment_bucket,
    params,
    parse_config,
    timestamp,
    valid_keys,
    validate_origin,
)


class _TransientConfigError(Exception):
    pass


class AbTestClient:
    _api_path = "/api/v1/ab-tests"

    def __init__(self, *, base_url: str, api_key: str, keys: list[str]):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("A/B api_key is required")
        self._base_url = validate_origin(base_url)
        self._api_key = api_key.strip()
        self._keys = valid_keys(keys)
        self._configs: dict[str, Config] = {}
        self._lock = threading.RLock()
        self._refresh_lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._queue: OrderedDict[str, tuple[str, dict[str, Any], float]] = OrderedDict()
        self._seen: OrderedDict[tuple, str] = OrderedDict()
        self._rejected = 0
        self._epochs: OrderedDict[tuple, int | None] = OrderedDict()

    def start(self) -> None:
        if self._stop.is_set():
            return
        self.refresh()
        with self._lock:
            if self._thread is None and not self._stop.is_set():
                self._thread = threading.Thread(
                    target=self._poll, daemon=True, name="neatlogs-ab-tests"
                )
                self._thread.start()

    def _poll(self) -> None:
        while not self._stop.wait(60):
            self.refresh()
            self.flush()

    def refresh(self) -> None:
        if self._stop.is_set() or not self._refresh_lock.acquire(blocking=False):
            return
        try:
            for key in self._keys:
                if self._stop.is_set():
                    break
                with self._lock:
                    previous = self._configs.get(key)
                try:
                    status, headers, body = self._request(
                        key, "config", headers={"If-None-Match": previous.etag} if previous else {}
                    )
                    if status == 304 and previous:
                        config = replace(
                            previous, expires_at=timestamp(headers.get("X-AB-Config-Expires-At"))
                        )
                    else:
                        if status in {408, 429} or status >= 500:
                            raise _TransientConfigError("A/B settings temporarily unavailable")
                        if status != 200:
                            raise ValueError("A/B settings unavailable")
                        config = parse_config(body)
                    now = time.time()
                    if (
                        config.key != key
                        or not now < config.expires_at <= now + 65
                        or previous
                        and (
                            config.id != previous.id
                            or config.salt != previous.salt
                            or config.revision < previous.revision
                            or config.allocation_version < previous.allocation_version
                        )
                    ):
                        raise ValueError("Invalid A/B settings revision or expiry")
                    with self._lock:
                        if not self._stop.is_set():
                            self._configs[key] = replace(
                                config, expires_at=min(config.expires_at, now + 60)
                            )
                except (
                    _TransientConfigError,
                    requests.ConnectionError,
                    requests.Timeout,
                    requests.exceptions.ChunkedEncodingError,
                    TimeoutError,
                ):
                    with self._lock:
                        if previous is None or previous.expires_at <= time.time():
                            self._configs.pop(key, None)
                except Exception:
                    with self._lock:
                        self._configs.pop(key, None)
        finally:
            self._refresh_lock.release()

    def assign(self, key: str, *, unit: str, fallback: Params | None = None) -> AbAssignment:
        decision = self.choose(key, unit=unit, fallback=fallback)
        if decision.reason == "assigned":
            self.record_run(decision)
        return AbAssignment(
            decision.key, decision.arm, decision.params, decision.reason,
            decision.revision, decision.allocation_version,
        )

    def choose(self, key: str, *, unit: str, fallback: Params | None = None) -> AbDecision:
        """Select from the local cache without recording unserved traffic."""
        with self._lock:
            config = None if self._stop.is_set() else self._configs.get(key)
        if (
            not config
            or config.expires_at <= time.time()
            or not isinstance(unit, str)
            or not 1 <= len(unit) <= 1024
        ):
            return AbDecision(key, unit, "a", params(fallback or {}), "unavailable", None)
        enabled = config.enabled and config.ends_at > time.time()
        index = (
            (0 if assignment_bucket(config.salt, unit) < config.arms[0].weight else 1)
            if enabled
            else (0 if config.default_arm == "a" else 1)
        )
        arm = config.arms[index]
        return AbDecision(
            key,
            unit,
            arm.key,
            dict(arm.params),
            "assigned" if enabled else "default",
            config.revision,
            config.allocation_version,
        )

    def record_run(self, decision: AbDecision) -> bool:
        """Record inside the trace, immediately before invoking the selected agent."""
        if decision.reason != "assigned" or decision.allocation_version is None:
            return False
        return self._queue_exposure(
            decision.key, unit_id=decision.unit_id, arm=decision.arm,
            source="sdk", allocation_version=decision.allocation_version,
        )

    def complete_decision(self, decision: AbDecision, trace_id: str) -> bool:
        """Close the unit after its final trace has been exported."""
        if decision.reason != "assigned" or decision.allocation_version is None:
            return False
        return self.complete_unit(
            decision.key, unit_id=decision.unit_id, arm=decision.arm,
            trace_id=trace_id, allocation_version=decision.allocation_version,
        )

    def log_exposure(
        self,
        key: str,
        *,
        unit_id: str,
        arm: ArmKey,
        trace_id: str | None = None,
        unit_complete: bool = False,
        event_id: str | None = None,
        allocation_version: int | None = None,
    ) -> bool:
        return self._queue_exposure(
            key,
            unit_id=unit_id,
            arm=arm,
            trace_id=trace_id,
            unit_complete=unit_complete,
            event_id=event_id,
            source="external",
            allocation_version=allocation_version,
        )

    def complete_unit(
        self,
        key: str,
        *,
        unit_id: str,
        arm: ArmKey,
        trace_id: str,
        source: str = "sdk",
        allocation_version: int | None = None,
    ) -> bool:
        """Call only after the session/user's final trace has been exported."""
        return self._queue_exposure(
            key,
            unit_id=unit_id,
            arm=arm,
            trace_id=trace_id,
            unit_complete=True,
            source=source,
            allocation_version=allocation_version,
        )

    def _queue_exposure(
        self,
        key: str,
        *,
        unit_id: str,
        arm: ArmKey,
        source: str,
        trace_id: str | None = None,
        unit_complete: bool = False,
        event_id: str | None = None,
        allocation_version: int | None = None,
    ) -> bool:
        from ..replay.session import current_replay

        if current_replay() is not None:
            return False
        active = active_neatlogs_context()
        span = get_neatlogs_root_span() or (
            trace.get_current_span(active) if active is not None else None
        )
        active_id = format(span.get_span_context().trace_id, "032x") if span else None
        trace_id = trace_id or active_id
        event_id = event_id or str(uuid4())
        try:
            UUID(event_id)
        except (ValueError, TypeError, AttributeError):
            return False
        if (
            not isinstance(unit_id, str)
            or not 1 <= len(unit_id) <= 1024
            or not isinstance(trace_id, str)
            or not 1 <= len(trace_id) <= 255
            or arm not in {"a", "b"}
            or source not in {"sdk", "external"}
            or type(unit_complete) is not bool
        ):
            return False
        remembered_identity = (key, unit_id, trace_id, arm, source)
        with self._lock:
            if self._stop.is_set() or key not in self._keys:
                return False
            if allocation_version is None:
                if remembered_identity in self._epochs:
                    allocation_version = self._epochs[remembered_identity]
                elif not unit_complete:
                    config = self._configs.get(key)
                    if (
                        config
                        and config.enabled
                        and min(config.expires_at, config.ends_at) > time.time()
                    ):
                        allocation_version = config.allocation_version
            if type(allocation_version) is not int or not 1 <= allocation_version <= 2147483647:
                return False
            identity = (*remembered_identity, unit_complete, allocation_version)
            if identity in self._seen:
                return True
            if len(self._queue) >= 1000 or event_id in self._queue:
                self._rejected += 1
                get_logger().warning("A/B exposure queue rejected an event")
                return False
            if span and active_id == trace_id:
                span.set_attribute(f"neatlogs.ab.{key}", arm)
            if (
                remembered_identity in self._epochs
                and self._epochs[remembered_identity] != allocation_version
            ):
                self._epochs[remembered_identity] = None
            else:
                self._epochs[remembered_identity] = allocation_version
            if len(self._epochs) > 10000:
                self._epochs.popitem(last=False)
            self._seen[identity] = event_id
            if len(self._seen) > 10000:
                self._seen.popitem(last=False)
            exposure = {
                "eventId": event_id,
                "unitId": unit_id,
                "traceId": trace_id,
                "arm": arm,
                "source": source,
                "unitComplete": unit_complete,
                "allocationVersion": allocation_version,
            }
            self._queue[event_id] = (key, exposure, time.time())
        return True

    def flush(self, timeout_seconds: float = 5) -> bool:
        if self._stop.is_set() or not self._flush_lock.acquire(blocking=False):
            return self.get_diagnostics() == {"pending": 0, "rejected": 0}
        deadline = time.monotonic() + max(0, min(timeout_seconds, 5))
        try:
            while not self._stop.is_set() and time.monotonic() < deadline:
                with self._lock:
                    item = next(iter(self._queue.items()), None)
                if item is None:
                    break
                event_id, (key, exposure, queued_at) = item
                rejected = time.time() - queued_at > 86400
                if not rejected:
                    try:
                        status, _, body = self._request(
                            key,
                            "exposures",
                            payload=exposure,
                            timeout=min(3, deadline - time.monotonic()),
                        )
                        if (
                            status == 200
                            and isinstance(body, dict)
                            and body.get("accepted") is True
                        ):
                            pass
                        elif (
                            status in {400, 401, 403, 404, 422}
                            or status == 409
                            and isinstance(body, dict)
                            and body.get("code") not in {None, "AB_TRACE_NOT_READY"}
                        ):
                            rejected = True
                        else:
                            break
                    except Exception:
                        break
                with self._lock:
                    self._queue.pop(event_id, None)
                    self._rejected += int(rejected)
                if rejected:
                    get_logger().warning("A/B exposure rejected; inspect exposure diagnostics")
            return self.get_diagnostics() == {"pending": 0, "rejected": 0}
        finally:
            self._flush_lock.release()

    def get_diagnostics(self) -> dict[str, int]:
        with self._lock:
            return {"pending": len(self._queue), "rejected": self._rejected}

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            self._configs.clear()

    def _request(
        self,
        key: str,
        operation: str,
        *,
        headers: dict[str, str] | None = None,
        payload: dict | None = None,
        timeout: float = 3,
    ):
        deadline = time.monotonic() + timeout
        with (
            suppress_instrumentation(),
            requests.request(
                "POST" if payload is not None else "GET",
                f"{self._base_url}{self._api_path}/{quote(key, safe='')}/{operation}",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    **(headers or {}),
                },
                json=payload,
                timeout=max(0.001, timeout),
                allow_redirects=False,
                stream=True,
            ) as response,
        ):
            if response.status_code == 304 or (
                operation == "config" and response.status_code != 200
            ):
                return response.status_code, response.headers, None
            content = bytearray()
            for chunk in response.iter_content(8192):
                if time.monotonic() >= deadline:
                    raise TimeoutError("A/B response deadline exceeded")
                content.extend(chunk)
                if len(content) > 65536:
                    raise ValueError("A/B response exceeds 64 KiB")
            try:
                body = json.loads(content)
            except (ValueError, UnicodeDecodeError):
                body = None
            return response.status_code, response.headers, body
