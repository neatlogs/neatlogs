"""OTLP HTTP session policy that complements (without duplicating) OTel retry."""

from __future__ import annotations

import math
import os
import random
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter

_MAX_RATE_LIMIT_ATTEMPTS = 3
_MAX_RETRY_AFTER_SECONDS = 5.0
_DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 5
_DEFAULT_CIRCUIT_BREAKER_COOLDOWN_S = 30.0


class CircuitOpenError(requests.exceptions.RequestException):
    """Raised immediately while the fail-open OTLP circuit breaker is open.

    Deliberately does NOT subclass ``ConnectionError`` so OTel's OTLP HTTP
    exporters treat it as non-retryable and drop the batch in <0.1 ms without
    sleeping through 6 exponential-backoff retries.
    """


class _CircuitBreaker:
    """Thread-safe CLOSED -> OPEN -> HALF_OPEN fail-open breaker for OTLP export."""

    def __init__(
        self,
        *,
        threshold: int = _DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
        cooldown_s: float = _DEFAULT_CIRCUIT_BREAKER_COOLDOWN_S,
        clock=time.monotonic,
    ) -> None:
        self._threshold = max(0, int(threshold))
        self._cooldown_s = max(0.0, float(cooldown_s))
        self._clock = clock
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._open_until = 0.0
        self._half_open_probe = False

    @property
    def enabled(self) -> bool:
        return self._threshold > 0

    def state(self) -> str:
        if not self.enabled:
            return "closed"
        with self._lock:
            if self._open_until <= 0.0:
                return "closed"
            return "open" if self._clock() < self._open_until else "half_open"

    def _acquire(self) -> tuple[bool, bool]:
        if not self.enabled:
            return True, False
        with self._lock:
            if self._open_until <= 0.0:
                return True, False
            now = self._clock()
            if now < self._open_until:
                return False, False
            if self._half_open_probe:
                return False, False
            self._half_open_probe = True
            return True, True

    def allow_request(self) -> bool:
        allowed, _ = self._acquire()
        return allowed

    def record_success(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._consecutive_failures = 0
            self._open_until = 0.0
            self._half_open_probe = False

    def record_failure(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._consecutive_failures += 1
            if self._half_open_probe or self._consecutive_failures >= self._threshold:
                self._half_open_probe = False
                self._open_until = self._clock() + self._cooldown_s


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def _parse_retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        parsed = float(value)
        if not math.isfinite(parsed):
            return None
        return max(0.0, parsed)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _retry_after_seconds(value: str | None, fallback: float) -> float:
    parsed = _parse_retry_after_seconds(value)
    return fallback if parsed is None else parsed


def _trace_receipt(response, url):
    if (
        not 200 <= response.status_code < 300
        or not urlsplit(str(url)).path.endswith("/v1/traces")
        or not response.content
    ):
        return response
    from google.protobuf.message import DecodeError
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceResponse

    try:
        complete = (
            ExportTraceServiceResponse.FromString(response.content).partial_success.rejected_spans
            == 0
        )
    except DecodeError:
        complete = False
    if not complete:
        # Upstream OTel treats every HTTP 200 as success, including partial rejection.
        response.status_code = 422
        response.reason = "OTLP trace export was not fully accepted"
        response._content = b""
    return response


class _OtlpSession(requests.Session):
    """Retry only definitely-rejected 429 requests inside the caller's deadline.

    The OpenTelemetry exporter owns connection/408/5xx retries. In particular,
    read timeouts are not retried here because the receiver may already have
    accepted the POST. Keeping urllib3 retries disabled also supports every
    urllib3 version allowed by ``requests>=2.31`` without relying on newer
    ``Retry`` constructor arguments.
    """

    def __init__(
        self,
        *,
        circuit_breaker_threshold: int | None = None,
        circuit_breaker_cooldown_s: float | None = None,
        clock=time.monotonic,
    ) -> None:
        super().__init__()
        threshold = (
            _env_int("NEATLOGS_CIRCUIT_BREAKER_THRESHOLD", _DEFAULT_CIRCUIT_BREAKER_THRESHOLD)
            if circuit_breaker_threshold is None
            else circuit_breaker_threshold
        )
        cooldown_s = (
            _env_float("NEATLOGS_CIRCUIT_BREAKER_COOLDOWN_S", _DEFAULT_CIRCUIT_BREAKER_COOLDOWN_S)
            if circuit_breaker_cooldown_s is None
            else circuit_breaker_cooldown_s
        )
        self._circuit_breaker = _CircuitBreaker(
            threshold=threshold,
            cooldown_s=cooldown_s,
            clock=clock,
        )

    @property
    def circuit_breaker_state(self) -> str:
        return self._circuit_breaker.state()

    def request(self, method, url, **kwargs):
        timeout = kwargs.get("timeout")
        if method.upper() != "POST" or not isinstance(timeout, (int, float)):
            return super().request(method, url, **kwargs)

        allowed, is_probe = self._circuit_breaker._acquire()
        if not allowed:
            raise CircuitOpenError("OTLP export circuit breaker is open")

        settled = False
        try:
            deadline = time.monotonic() + max(0.0, float(timeout))
            for attempt in range(_MAX_RATE_LIMIT_ATTEMPTS):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._circuit_breaker.record_failure()
                    settled = True
                    raise requests.exceptions.Timeout("OTLP export deadline exceeded")
                kwargs["timeout"] = remaining
                try:
                    response = super().request(method, url, **kwargs)
                except requests.exceptions.RequestException:
                    self._circuit_breaker.record_failure()
                    settled = True
                    raise
                if response.status_code != 429 or attempt + 1 == _MAX_RATE_LIMIT_ATTEMPTS:
                    checked = _trace_receipt(response, url)
                    if 200 <= checked.status_code < 300:
                        self._circuit_breaker.record_success()
                        settled = True
                    elif checked.status_code >= 500 or checked.status_code in (408, 429):
                        self._circuit_breaker.record_failure()
                        settled = True
                    return checked

                server_delay = _parse_retry_after_seconds(response.headers.get("Retry-After"))
                remaining = deadline - time.monotonic()
                if server_delay is not None:
                    # Never shorten a server-provided Retry-After minimum via cap or negative
                    # jitter. If the required wait exceeds our delay cap or remaining deadline,
                    # skip the internal retry immediately rather than retrying early.
                    if server_delay > _MAX_RETRY_AFTER_SECONDS or server_delay >= remaining:
                        self._circuit_breaker.record_failure()
                        settled = True
                        return response
                    delay = server_delay
                else:
                    base_delay = min(_MAX_RETRY_AFTER_SECONDS, 0.1 * (2**attempt))
                    delay = min(_MAX_RETRY_AFTER_SECONDS, base_delay * random.uniform(0.8, 1.2))
                    if delay >= remaining:
                        self._circuit_breaker.record_failure()
                        settled = True
                        return response
                response.close()
                if delay:
                    time.sleep(delay)
            return response
        finally:
            if is_probe and not settled:
                self._circuit_breaker.record_failure()


def build_otlp_session(
    *,
    circuit_breaker_threshold: int | None = None,
    circuit_breaker_cooldown_s: float | None = None,
) -> requests.Session:
    """Build a dedicated OTLP session with no urllib3-level POST retries."""
    session = _OtlpSession(
        circuit_breaker_threshold=circuit_breaker_threshold,
        circuit_breaker_cooldown_s=circuit_breaker_cooldown_s,
    )
    adapter = HTTPAdapter(max_retries=0)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
