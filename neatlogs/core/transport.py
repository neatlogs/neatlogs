"""OTLP HTTP session policy that complements (without duplicating) OTel retry."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter

_MAX_RATE_LIMIT_ATTEMPTS = 3
_MAX_RETRY_AFTER_SECONDS = 5.0


def _retry_after_seconds(value: str | None, fallback: float) -> float:
    if not value:
        return fallback
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return fallback


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

    def request(self, method, url, **kwargs):
        timeout = kwargs.get("timeout")
        if method.upper() != "POST" or not isinstance(timeout, (int, float)):
            return super().request(method, url, **kwargs)

        deadline = time.monotonic() + max(0.0, float(timeout))
        for attempt in range(_MAX_RATE_LIMIT_ATTEMPTS):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise requests.exceptions.Timeout("OTLP export deadline exceeded")
            kwargs["timeout"] = remaining
            response = super().request(method, url, **kwargs)
            if response.status_code != 429 or attempt + 1 == _MAX_RATE_LIMIT_ATTEMPTS:
                return _trace_receipt(response, url)

            delay = min(
                _MAX_RETRY_AFTER_SECONDS,
                _retry_after_seconds(response.headers.get("Retry-After"), 0.1 * (2**attempt)),
            )
            if delay >= deadline - time.monotonic():
                return response
            response.close()
            if delay:
                time.sleep(delay)
        return response


def build_otlp_session() -> requests.Session:
    """Build a dedicated OTLP session with no urllib3-level POST retries."""
    session = _OtlpSession()
    adapter = HTTPAdapter(max_retries=0)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
