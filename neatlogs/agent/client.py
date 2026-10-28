"""Versioned agent adapter using the existing bounded transport and ordered outbox."""
import asyncio
from collections import OrderedDict
from dataclasses import replace
import threading
import time
from uuid import uuid4

import requests

from ..ab_test.client import AbTestClient, _TransientConfigError
from ..ab_test.protocol import params, timestamp
from .._wrap_utils import get_neatlogs_root_span
from ..replay.session import current_replay
from .protocol import AgentDecision, attributes, unit, parse_assignment, parse_config, decision, _params

_TRANSIENT = (_TransientConfigError, requests.ConnectionError, requests.Timeout,
              requests.exceptions.ChunkedEncodingError, TimeoutError)


class AgentClient(AbTestClient):
    """Session assignment may perform HTTP; call choose_agent before opening the root trace."""
    _api_path = "/api/v2/experiment-agents"

    def __init__(self, *, base_url: str, api_key: str, keys: list[str]):
        super().__init__(base_url=base_url, api_key=api_key, keys=keys)
        self._publications = {}
        self._leases = OrderedDict()
        self._recorded = OrderedDict()
        self._choose_lock = threading.Lock()

    def _request(self, key, operation, **kwargs):
        response = super()._request(key, operation, **kwargs)
        if response[0] in {401, 403}:
            with self._lock:
                self._revoke(key)
        return response

    def _revoke(self, key):
        self._publications.pop(key, None)
        for identity in list(self._leases):
            if identity[0] == key:
                self._leases.pop(identity)

    def refresh(self):
        with self._refresh_lock:
            for key in self._keys:
                self._refresh_agent(key)

    async def refresh_async(self):
        await asyncio.to_thread(self.refresh)

    def _refresh_agent(self, key):
        with self._lock:
            if self._stop.is_set():
                return
            previous = self._publications.get(key)
        try:
            status, headers, body = self._request(key, "config", headers={"If-None-Match": previous[1]} if previous and previous[1] else {})
            if status in {408, 429} or status >= 500:
                raise _TransientConfigError()
            if status == 304 and previous:
                body = {**previous[0], "expiresAt": headers.get("X-AB-Config-Expires-At")}
            elif status != 200:
                raise ValueError("Agent config unavailable")
            config = parse_config(body)
            now = time.time()
            if config["agentKey"] != key or not now < timestamp(config["expiresAt"]) <= now + 65 or previous and config["revision"] < previous[0]["revision"]:
                raise ValueError("Invalid config identity or expiry")
            expiry = min(timestamp(config["expiresAt"]), now + 60)
            with self._lock:
                if not self._stop.is_set():
                    if not previous or {**previous[0], "expiresAt": ""} != {**config, "expiresAt": ""} or not config["live"] or config["live"]["status"] != "running":
                        self._revoke(key)
                    self._publications[key] = (config, headers.get("ETag"), expiry)
        except _TRANSIENT:
            with self._lock:
                if not previous or previous[2] <= time.time():
                    self._revoke(key)
        except Exception:
            with self._lock:
                self._revoke(key)

    async def choose_agent(self, key, *, unit: str, attributes: dict | None = None, fallback: dict):
        return await asyncio.to_thread(self._choose_agent, key, unit, attributes or {}, fallback)

    def _choose_agent(self, key, unit_id, context, fallback):
        unavailable = AgentDecision(key, unit_id, "a", _params(fallback), "unavailable")
        try:
            unit(unit_id)
            attributes(context)
        except (TypeError, ValueError, UnicodeError):
            return unavailable
        if self._stop.is_set() or key not in self._keys or current_replay() is not None:
            return unavailable
        identity = (key, unit_id)
        with self._choose_lock:
            for attempt in range(2):
                with self._lock:
                    publication = self._publications.get(key)
                if not publication or publication[2] <= time.time():
                    with self._refresh_lock:
                        self._refresh_agent(key)
                    with self._lock:
                        publication = self._publications.get(key)
                if not publication or publication[2] <= time.time() or self._stop.is_set():
                    return unavailable
                config, _, config_expiry = publication
                live = config["live"]
                if not live or live["status"] != "running":
                    return AgentDecision(key, unit_id, "a", dict(config["current"]["params"]), "paused" if live else "current", config["revision"], config_expiry, config["current"]["fingerprint"])
                with self._lock:
                    lease = self._leases.get(identity)
                if lease and lease.revision == config["revision"] and min(lease.expires_at, config_expiry) > time.time():
                    return replace(lease, params=dict(lease.params), expires_at=min(lease.expires_at, config_expiry))
                try:
                    status, _, body = self._request(key, "assignments", payload={"protocolVersion": 2, "revision": config["revision"], "unitId": unit_id, "attributes": context})
                    if status == 409 and isinstance(body, dict) and body.get("code") == "CONFIG_CHANGED" and attempt == 0:
                        with self._lock:
                            self._revoke(key)
                        continue
                    if status in {408, 429} or status >= 500:
                        return unavailable
                    if status != 200:
                        raise ValueError("Agent assignment unavailable")
                    result = parse_assignment(body)
                    with self._lock:
                        if self._stop.is_set() or self._publications.get(key) is not publication:
                            return unavailable
                        now = time.time()
                        if result["agentKey"] != key or result["revision"] != config["revision"] or not now < timestamp(result["expiresAt"]) <= now + 65:
                            raise ValueError("Invalid assignment identity or expiry")
                        if result["reason"] == "assigned":
                            if any(result[field] != live[field] for field in ("testId", "experimentId", "candidateRevision", "allocationVersion")) or result["fingerprint"] != (config["current"]["fingerprint"] if result["arm"] == "a" else live["candidateFingerprint"]):
                                raise ValueError("Invalid assignment version")
                        elif result["fingerprint"] != config["current"]["fingerprint"]:
                            raise ValueError("Invalid current fingerprint")
                        expires = min(config_expiry, timestamp(result["expiresAt"]), now + 60)
                        if expires <= now:
                            return unavailable
                        selected = decision(result, unit_id, expires)
                        self._leases[identity] = selected
                        if len(self._leases) > 1000:
                            self._leases.popitem(last=False)
                        return replace(selected, params=dict(selected.params))
                except _TRANSIENT:
                    return unavailable
                except Exception:
                    with self._lock:
                        self._revoke(key)
                    return unavailable
        return unavailable

    def record_agent_run(self, selected: AgentDecision):
        if selected.reason != "assigned" or current_replay() is not None:
            return False
        root = get_neatlogs_root_span()
        if root is None or not root.is_recording():
            return False
        trace_id = format(root.get_span_context().trace_id, "032x")
        identity = (selected.agent_key, trace_id)
        with self._lock:
            publication = self._publications.get(selected.agent_key)
            if not publication or not publication[0]["live"] or publication[0]["live"]["status"] != "running" or publication[0]["revision"] != selected.revision or min(publication[2], selected.expires_at or 0) <= time.time():
                return False
            if identity in self._recorded and self._recorded[identity] != selected.assignment_id:
                return False
            if not self._queue_agent(selected, trace_id, False):
                return False
            root.set_attribute(f"neatlogs.agent.{selected.agent_key}.assignment_id", selected.assignment_id)
            root.set_attribute(f"neatlogs.agent.{selected.agent_key}.fingerprint", selected.fingerprint)
            self._recorded[identity] = selected.assignment_id
            if len(self._recorded) > 10000:
                self._recorded.popitem(last=False)
        return True

    def complete_agent_decision(self, selected: AgentDecision, trace_id: str):
        if selected.reason != "assigned" or current_replay() is not None:
            return False
        with self._lock:
            if self._recorded.get((selected.agent_key, trace_id)) != selected.assignment_id:
                return False
            return self._queue_agent(selected, trace_id, True)

    def _queue_agent(self, selected, trace_id, complete):
        if self._stop.is_set() or not isinstance(trace_id, str) or not 1 <= len(trace_id) <= 128:
            return False
        identity = ("agent", selected.agent_key, selected.assignment_id, trace_id, complete)
        if identity in self._seen:
            return True
        if len(self._queue) >= 1000:
            self._rejected += 1
            return False
        event_id = str(uuid4())
        payload = {"protocolVersion": 2, "eventId": event_id, "assignmentId": selected.assignment_id, "unitId": selected.unit_id, "traceId": trace_id, "unitComplete": complete}
        self._seen[identity] = event_id
        if len(self._seen) > 10000:
            self._seen.popitem(last=False)
        self._queue[event_id] = (selected.agent_key, payload, time.time())
        return True

    async def flush_async(self, timeout_seconds=5):
        return await asyncio.to_thread(self.flush, timeout_seconds)

    def close(self):
        super().close()
        with self._lock:
            self._publications.clear()
            self._leases.clear()
            self._recorded.clear()
