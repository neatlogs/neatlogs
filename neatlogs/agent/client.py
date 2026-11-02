"""Versioned agent adapter using the existing bounded transport and ordered outbox."""

import asyncio
import threading
import time
from collections import OrderedDict
from dataclasses import replace
from uuid import uuid4

import requests

from .._wrap_utils import get_neatlogs_root_span
from ..ab_test.client import AbTestClient, _TransientConfigError
from ..ab_test.protocol import params, timestamp
from ..replay.session import current_replay
from .fingerprint import AgentCanonicalError, attributes_digest, version_fingerprint
from .protocol import (
    CAPABILITY_HEADER,
    PROFILE,
    AgentDecision,
    _params,
    attributes,
    decision,
    parse_assignment,
    parse_config,
    unit,
)

_TRANSIENT = (
    _TransientConfigError,
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
    TimeoutError,
)


class AgentClient(AbTestClient):
    """Customer or session assignment may perform HTTP; call choose_agent before opening the root trace."""

    _api_path = "/api/v2/experiment-agents"

    def __init__(self, *, base_url: str, api_key: str, keys: list[str]):
        super().__init__(base_url=base_url, api_key=api_key, keys=keys)
        self._publications = {}
        self._leases = OrderedDict()
        self._serving_leases = OrderedDict()
        self._observed = OrderedDict()
        self._refresh_causes = {}
        self._recorded = OrderedDict()
        self._choose_lock = threading.Lock()

    def _request(self, key, operation, **kwargs):
        kwargs["headers"] = {**kwargs.get("headers", {}), CAPABILITY_HEADER: PROFILE}
        response = super()._request(key, operation, **kwargs)
        if response[0] in {401, 403}:
            with self._lock:
                self._revoke(key)
        return response

    def _revoke(self, key):
        self._publications.pop(key, None)
        for leases in (self._leases, self._serving_leases):
            for identity in list(leases):
                if identity[0] == key:
                    leases.pop(identity)

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
            status, headers, body = self._request(
                key,
                "config",
                headers={"If-None-Match": previous[1]} if previous and previous[1] else {},
            )
            if status in {408, 429} or status >= 500:
                raise _TransientConfigError()
            acknowledged = headers.get(CAPABILITY_HEADER) == PROFILE
            if (
                status == 304
                and previous
                and previous[0].get("profile") == PROFILE
                and not acknowledged
            ):
                raise ValueError("Agent profile not acknowledged")
            if status == 304 and previous:
                body = {**previous[0], "expiresAt": headers.get("X-AB-Config-Expires-At")}
            elif status != 200:
                raise ValueError("Agent config unavailable")
            config = parse_config(body)
            if (config.get("profile") == PROFILE) != acknowledged:
                raise ValueError("Agent profile mismatch")
            self._validate_config_fingerprints(config)
            now = time.time()
            if (
                config["agentKey"] != key
                or not now < timestamp(config["expiresAt"]) <= now + 65
                or previous
                and config["revision"] < previous[0]["revision"]
            ):
                raise ValueError("Invalid config identity or expiry")
            expiry = min(timestamp(config["expiresAt"]), now + 60)
            with self._lock:
                if not self._stop.is_set():
                    if (
                        not previous
                        or {**previous[0], "expiresAt": ""} != {**config, "expiresAt": ""}
                        or not config["live"]
                        or config["live"]["status"] != "running"
                    ):
                        self._revoke(key)
                    self._refresh_causes.pop(key, None)
                    self._publications[key] = (config, headers.get("ETag"), expiry)
        except _TRANSIENT:
            with self._lock:
                self._refresh_causes[key] = "transport_unavailable"
                if not previous or previous[2] <= time.time():
                    self._revoke(key)
        except Exception as error:
            with self._lock:
                self._refresh_causes[key] = (
                    "unsupported_canonical_params"
                    if isinstance(error, AgentCanonicalError)
                    else "invalid_identity"
                )
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

        def failure(cause=None):
            return replace(unavailable, cause=cause or self._refresh_causes.get(key))

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
                    return failure()
                config, _, config_expiry = publication
                live = config["live"]
                serving = config.get("serving")
                unavailable = replace(
                    unavailable,
                    known_revision=config["revision"],
                    known_unit=serving["unit"] if serving else live["unit"] if live else None,
                )
                if (not live or live["status"] != "running") and not serving:
                    return AgentDecision(
                        key,
                        unit_id,
                        "a",
                        dict(config["current"]["params"]),
                        "paused" if live else "current",
                        config["revision"],
                        config_expiry,
                        config["current"]["fingerprint"],
                        profile=config.get("profile"),
                        known_unit=live["unit"] if live else None,
                    )
                try:
                    serving_identity = (
                        (
                            key,
                            config["revision"],
                            serving["revisionId"],
                            serving["digest"],
                            unit_id,
                            attributes_digest(context, serving["requiredAttributeKeys"]),
                        )
                        if serving
                        else None
                    )
                except (ValueError, UnicodeError):
                    return failure()
                with self._lock:
                    lease = self._leases.get(identity) or (
                        self._serving_leases.get(serving_identity) if serving_identity else None
                    )
                if (
                    lease
                    and lease.revision == config["revision"]
                    and min(lease.expires_at, config_expiry) > time.time()
                ):
                    return replace(
                        lease,
                        params=dict(lease.params),
                        expires_at=min(lease.expires_at, config_expiry),
                        serving=dict(lease.serving) if lease.serving else None,
                    )
                try:
                    status, response_headers, body = self._request(
                        key,
                        "assignments",
                        payload={
                            "protocolVersion": 2,
                            "revision": config["revision"],
                            "unitId": unit_id,
                            "attributes": context,
                        },
                    )
                    if (
                        status == 409
                        and isinstance(body, dict)
                        and body.get("code") == "CONFIG_CHANGED"
                        and attempt == 0
                    ):
                        with self._lock:
                            self._revoke(key)
                        continue
                    if status in {408, 429} or status >= 500:
                        return failure("transport_unavailable")
                    if status != 200:
                        raise ValueError("Agent assignment unavailable")
                    result = parse_assignment(body)
                    if (
                        (result.get("profile") == PROFILE) != (config.get("profile") == PROFILE)
                        or config.get("profile") == PROFILE
                        and response_headers.get(CAPABILITY_HEADER) != PROFILE
                    ):
                        raise ValueError("Agent decision profile mismatch")
                    with self._lock:
                        if self._stop.is_set() or self._publications.get(key) is not publication:
                            return failure()
                        now = time.time()
                        if (
                            result["agentKey"] != key
                            or result["revision"] != config["revision"]
                            or not now < timestamp(result["expiresAt"]) <= now + 65
                        ):
                            raise ValueError("Invalid assignment identity or expiry")
                        if result["reason"] == "assigned":
                            if not live or live["status"] != "running":
                                raise ValueError("Inactive live assignment")
                            if any(
                                result[field] != live[field]
                                for field in (
                                    "testId",
                                    "experimentId",
                                    "candidateRevision",
                                    "allocationVersion",
                                    "unit",
                                    "completionPolicy",
                                )
                            ) or result["fingerprint"] != (
                                (
                                    live["current"]["fingerprint"]
                                    if "current" in live
                                    else live["currentFingerprint"]
                                )
                                if result["arm"] == "a"
                                else live["candidateFingerprint"]
                            ):
                                raise ValueError("Invalid assignment version")
                            if (
                                live.get("agentProfile") == PROFILE
                                and version_fingerprint(key, result["params"])
                                != result["fingerprint"]
                            ):
                                raise ValueError("Invalid assignment params")
                        elif result["reason"] == "published":
                            self._validate_published(config, result, unit_id, context)
                        elif (
                            result["fingerprint"] != config["current"]["fingerprint"]
                            or config.get("profile") == PROFILE
                            and (serving or not live or live.get("agentProfile") != "legacy-v2")
                            and version_fingerprint(key, result["params"]) != result["fingerprint"]
                        ):
                            raise ValueError("Invalid current fingerprint")
                        expires = min(config_expiry, timestamp(result["expiresAt"]), now + 60)
                        if expires <= now:
                            return failure()
                        selected = decision(result, unit_id, expires)
                        cache = (
                            self._serving_leases
                            if serving_identity and selected.reason != "assigned"
                            else self._leases
                        )
                        cache[serving_identity if cache is self._serving_leases else identity] = (
                            selected
                        )
                        if len(cache) > 1000:
                            cache.popitem(last=False)
                        return replace(
                            selected,
                            params=dict(selected.params),
                            serving=dict(selected.serving) if selected.serving else None,
                        )
                except _TRANSIENT:
                    return failure("transport_unavailable")
                except Exception as error:
                    with self._lock:
                        self._revoke(key)
                    return failure(
                        "unsupported_canonical_params"
                        if isinstance(error, AgentCanonicalError)
                        else "invalid_identity"
                    )
        return failure()

    @staticmethod
    def _validate_config_fingerprints(config):
        if config.get("profile") != PROFILE:
            return
        live, serving = config["live"], config["serving"]
        if live and live["agentProfile"] == "legacy-v2" and not serving:
            return
        if (
            version_fingerprint(config["agentKey"], config["current"]["params"])
            != config["current"]["fingerprint"]
        ):
            raise ValueError("Invalid Current params")
        if live:
            if (
                version_fingerprint(config["agentKey"], live["current"]["params"])
                != live["current"]["fingerprint"]
            ):
                raise ValueError("Invalid live Current params")
            baseline = live["baselineServing"]
            if live["agentProfile"] == PROFILE and (
                bool(baseline) != bool(serving)
                or baseline
                and any(
                    baseline[key] != serving[key] for key in ("revisionId", "revision", "digest")
                )
            ):
                raise ValueError("Invalid baseline serving identity")

    @staticmethod
    def _validate_published(config, result, unit_id, context):
        serving = config.get("serving")
        if (
            not serving
            or result["unit"] != serving["unit"]
            or result["unitId"] != unit_id
            or any(
                result["serving"][key] != serving[key]
                for key in ("revisionId", "revision", "digest")
            )
            or result["attributesDigest"]
            != attributes_digest(context, serving["requiredAttributeKeys"])
            or not any(
                item["publicationId"] == result["serving"]["publicationId"]
                and item["fingerprint"] == result["fingerprint"]
                for item in serving["allowedIdentities"]
            )
            or version_fingerprint(config["agentKey"], result["params"]) != result["fingerprint"]
        ):
            raise ValueError("Invalid published identity")

    def record_agent_run(self, selected: AgentDecision):
        if current_replay() is not None:
            return False
        root = get_neatlogs_root_span()
        if root is None or not root.is_recording():
            return False
        trace_id = format(root.get_span_context().trace_id, "032x")
        identity = (selected.agent_key, trace_id)
        actual = None
        try:
            actual = version_fingerprint(selected.agent_key, dict(selected.params))
        except (ValueError, UnicodeError, OverflowError):
            pass
        prefix = f"neatlogs.agent.{selected.agent_key}."
        revision, kind = (
            selected.revision or selected.known_revision,
            selected.unit or selected.known_unit,
        )
        observed = (selected.reason, actual, selected.unit_id, revision, kind)
        with self._lock:
            observed_publication = self._publications.get(selected.agent_key)
            signed_prior = self._recorded.get(identity)
            observation_prior = self._observed.get(identity)
        observed_live = observed_publication[0]["live"] if observed_publication else None
        strict = (
            selected.reason == "published"
            or observed_live
            and observed_live.get("agentProfile") == PROFILE
        )
        observed_config = observed_publication[0] if observed_publication else None
        if (
            selected.reason == "assigned"
            and observed_config
            and observed_live
            and observed_live["status"] == "running"
            and observed_config["revision"] == selected.revision
            and min(observed_publication[2], selected.expires_at or 0) > time.time()
            and not (strict and actual != selected.fingerprint)
            and signed_prior in (None, selected.assignment_id)
            and observation_prior in (None, observed)
        ):
            # Signed identity keeps its legacy priority over best-effort ordinary tags.
            root.set_attribute(prefix + "assignment_id", selected.assignment_id)
            root.set_attribute(prefix + "fingerprint", selected.fingerprint)
        root.set_attribute(prefix + "serving_trace_version", 1)
        root.set_attribute(prefix + "unit_id", selected.unit_id)
        root.set_attribute(prefix + "decision_reason", selected.reason)
        cause = selected.cause or (
            "unsupported_canonical_params"
            if actual is None
            else "invalid_identity" if strict and actual != selected.fingerprint else None
        )
        if cause:
            root.set_attribute(prefix + "decision_cause", cause)
        if actual:
            root.set_attribute(prefix + "served_fingerprint", actual)
        if revision:
            root.set_attribute(prefix + "config_revision", revision)
        if kind:
            root.set_attribute(prefix + "unit_kind", kind)
        with self._lock:
            observed = (selected.reason, actual, selected.unit_id, revision, kind)
            if identity in self._observed and self._observed[identity] != observed:
                root.set_attribute(prefix + "decision_reason", "unavailable")
                root.set_attribute(prefix + "decision_cause", "invalid_identity")
                for name in (
                    "assignment_id",
                    "fingerprint",
                    "serving_policy_id",
                    "serving_policy_digest",
                ):
                    root.set_attribute(prefix + name, "")
                return False
            self._observed[identity] = observed
            if len(self._observed) > 10000:
                self._observed.popitem(last=False)
            publication = self._publications.get(selected.agent_key)
            config = publication[0] if publication else None
            if selected.reason == "published":
                serving = config.get("serving") if config else None
                if (
                    actual == selected.fingerprint
                    and serving
                    and selected.serving
                    and config["revision"] == selected.revision
                    and min(publication[2], selected.expires_at or 0) > time.time()
                    and all(
                        selected.serving[key] == serving[key]
                        for key in ("revisionId", "revision", "digest")
                    )
                    and any(
                        item["publicationId"] == selected.serving["publicationId"]
                        and item["fingerprint"] == actual
                        for item in serving["allowedIdentities"]
                    )
                ):
                    root.set_attribute(
                        prefix + "serving_policy_id", selected.serving["publicationId"]
                    )
                    root.set_attribute(
                        prefix + "serving_policy_revision", selected.serving["revision"]
                    )
                    root.set_attribute(prefix + "serving_policy_digest", selected.serving["digest"])
                return False
            if (
                selected.reason != "assigned"
                or not config
                or not config["live"]
                or config["live"]["status"] != "running"
                or config["revision"] != selected.revision
                or min(publication[2], selected.expires_at or 0) <= time.time()
            ):
                return False
            if config["live"].get("agentProfile") == PROFILE and actual != selected.fingerprint:
                return False
            if identity in self._recorded and self._recorded[identity] != selected.assignment_id:
                return False
            recorded_attributes = getattr(root, "attributes", None)
            if recorded_attributes is not None and (
                recorded_attributes.get(prefix + "assignment_id") != selected.assignment_id
                or recorded_attributes.get(prefix + "fingerprint") != selected.fingerprint
            ):
                return False
            if not self._queue_agent(selected, trace_id, False):
                return False
            self._recorded[identity] = selected.assignment_id
            if len(self._recorded) > 10000:
                self._recorded.popitem(last=False)
        return True

    def complete_agent_decision(self, selected: AgentDecision, trace_id: str):
        """Complete an exported conversation; fixed-horizon customer units stay open."""
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
        payload = {
            "protocolVersion": 2,
            "eventId": event_id,
            "assignmentId": selected.assignment_id,
            "unitId": selected.unit_id,
            "traceId": trace_id,
            "conversationComplete": complete,
            "unitComplete": complete
            and selected.unit == "session"
            and selected.completion_policy == "explicit",
        }
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
            self._serving_leases.clear()
            self._observed.clear()
            self._refresh_causes.clear()
            self._recorded.clear()
