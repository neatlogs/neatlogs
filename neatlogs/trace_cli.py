"""``neatlogs trace get``: read a trace back and check it. Standard library only."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any
from urllib.parse import quote, urlparse

DEFAULT_ENDPOINT = "https://ingest.neatlogs.com"
EXIT_OK, EXIT_CHECKS_FAILED, EXIT_NOT_READY, EXIT_AUTH, EXIT_USAGE, EXIT_ERROR = (
    0,
    1,
    2,
    3,
    4,
    5,
)

Fetcher = Callable[[str, dict, float], "tuple[int, Any]"]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def _default_fetch(url: str, headers: dict, timeout: float) -> tuple[int, Any]:
    request = urllib.request.Request(url, headers=headers, method="GET")
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read(1 << 20)
    except urllib.error.HTTPError as exc:
        return exc.code, b""


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def check_trace(trace: dict) -> list[dict]:
    """Pure checks over a persisted trace."""

    raw = trace.get("spans")
    spans = [s for s in raw if isinstance(s, dict)] if isinstance(raw, list) else []
    checks: list[dict] = []

    def add(name: str, ok: bool, message: str) -> None:
        checks.append(
            {"name": name, "status": "pass" if ok else "fail", "message": message}
        )

    add("has_spans", len(spans) > 0, f"{len(spans)} span(s) returned")
    count = trace.get("spanCount")
    if isinstance(count, int) and not isinstance(count, bool):
        add(
            "span_count_matches",
            count == len(spans),
            f"spanCount={count}, returned={len(spans)}",
        )
    ids = {_text(s.get("span_id")) for s in spans} - {None}
    orphans = [s for s in spans if _text(s.get("parent_span_id")) not in (None, *ids)]
    add(
        "parents_resolve",
        not orphans,
        "every parent span is present"
        if not orphans
        else f"{len(orphans)} span(s) point at a missing parent",
    )
    unnamed = [
        s
        for s in spans
        if not _text(s.get("span_name")) and not _text(s.get("node_name"))
    ]
    add(
        "spans_named",
        not unnamed,
        "every span has a name"
        if not unnamed
        else f"{len(unnamed)} span(s) have no name",
    )
    llm = [
        s
        for s in spans
        if "llm" in str(s.get("node_type") or s.get("span_type") or "").lower()
    ]
    total = trace.get("totalTokensUsed")
    if llm and isinstance(total, (int, float)) and not isinstance(total, bool):
        add(
            "llm_token_usage",
            total > 0,
            f"LLM span(s): {len(llm)}, totalTokensUsed={total}",
        )
    if "finalizationStatus" in trace:
        status = trace["finalizationStatus"]
        add("finalized", status == "finalized", f"finalizationStatus={status}")
    return checks


def usage() -> str:
    return (
        "Usage: neatlogs trace get <trace_id> [--json]\n"
        "Reads a trace back with NEATLOGS_API_KEY (and optional NEATLOGS_ENDPOINT) and checks it."
    )


def run_trace_get(
    trace_id: str,
    as_json: bool,
    *,
    env: dict | None = None,
    fetch: Fetcher | None = None,
    timeout: float = 5.0,
    out: Callable[[str], None] = print,
    err: Callable[[str], None] | None = None,
) -> int:
    """Exit codes: 0 pass, 1 check failed, 2 not ready/not found, 3 key, 4 usage, 5 error."""

    import sys

    env = os.environ if env is None else env
    fetch = fetch or _default_fetch
    err = err or (lambda line: print(line, file=sys.stderr))
    key = (env.get("NEATLOGS_API_KEY") or "").strip()
    if not key:
        err("NEATLOGS_API_KEY is not set")
        return EXIT_AUTH
    endpoint = (env.get("NEATLOGS_ENDPOINT") or DEFAULT_ENDPOINT).strip()
    parsed = urlparse(endpoint)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        err("NEATLOGS_ENDPOINT must be an absolute http(s) URL")
        return EXIT_USAGE
    url = f"{parsed.scheme}://{parsed.netloc}/api/traces/v3/{quote(trace_id, safe='')}"
    try:
        status, body = fetch(url, {"x-api-key": key}, timeout)
    except (TimeoutError, urllib.error.URLError, OSError):
        err("Could not reach the trace read API")
        return EXIT_ERROR
    if status in (401, 403):
        err("Trace read rejected the API key")
        return EXIT_AUTH
    if status in (202, 404, 409):
        err(
            f"Trace not ready or not found (HTTP {status}); retry after the app flushes"
        )
        return EXIT_NOT_READY
    if not 200 <= status < 300:
        err(f"Trace read failed (HTTP {status})")
        return EXIT_ERROR
    try:
        trace = json.loads(body)
    except (TypeError, ValueError):
        err("Trace read returned invalid JSON")
        return EXIT_ERROR
    if not isinstance(trace, dict):
        err("Trace read returned an unexpected response")
        return EXIT_ERROR
    checks = check_trace(trace)
    failed = [c for c in checks if c["status"] == "fail"]
    spans = [s for s in trace.get("spans") or [] if isinstance(s, dict)]
    summary = {
        "trace_id": _text(trace.get("_id")) or trace_id,
        "status": trace.get("status"),
        "span_count": len(spans),
        "total_tokens": trace.get("totalTokensUsed"),
        "spans": [
            {
                "span_id": s.get("span_id"),
                "parent_span_id": s.get("parent_span_id"),
                "name": s.get("span_name") or s.get("node_name"),
                "type": s.get("node_type") or s.get("span_type"),
            }
            for s in spans
        ],
        "checks": checks,
        "result": "pass" if not failed else "fail",
    }
    if as_json:
        out(json.dumps(summary, indent=2))
    else:
        out(
            f"trace {summary['trace_id']}: {summary['result']} ({summary['span_count']} spans)"
        )
        for c in checks:
            out(
                f"  {'ok  ' if c['status'] == 'pass' else 'FAIL'} {c['name']}: {c['message']}"
            )
    return EXIT_OK if not failed else EXIT_CHECKS_FAILED
