"""``neatlogs trace get``: read a trace back and check it. Standard library only."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any
from urllib.parse import quote, urlparse

DEFAULT_HOST = "https://app.neatlogs.com"
PAGE_LIMIT = 50
MAX_PAGES = 200
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


def check_trace(trace: dict, spans: list[dict]) -> list[dict]:
    checks: list[dict] = []

    def add(name: str, ok: bool, message: str) -> None:
        checks.append({"name": name, "status": "pass" if ok else "fail", "message": message})

    add("has_spans", len(spans) > 0, f"{len(spans)} span(s) returned")
    count = trace.get("spansCount")
    if isinstance(count, int) and not isinstance(count, bool):
        add(
            "span_count_matches",
            count == len(spans),
            f"spansCount={count}, returned={len(spans)}",
        )
    ids = {_text(s.get("spanId")) for s in spans} - {None}
    orphans = [s for s in spans if _text(s.get("parentSpanId")) not in (None, *ids)]
    add(
        "parents_resolve",
        not orphans,
        (
            "every parent span is present"
            if not orphans
            else f"{len(orphans)} span(s) point at a missing parent"
        ),
    )
    unnamed = [s for s in spans if not _text(s.get("spanName"))]
    add(
        "spans_named",
        not unnamed,
        "every span has a name" if not unnamed else f"{len(unnamed)} span(s) have no name",
    )
    llm = [s for s in spans if "llm" in str(s.get("spanType") or "").lower()]
    total = trace.get("totalTokens")
    if llm and isinstance(total, (int, float)) and not isinstance(total, bool):
        add(
            "llm_token_usage",
            total >= 0,
            f"LLM span(s): {len(llm)}, totalTokens={total} "
            "(0 can mean the provider sent no usage)",
        )
    if "finalizationStatus" in trace:
        status = trace["finalizationStatus"]
        add("finalized", status == "finalized", f"finalizationStatus={status}")
    return checks


def usage() -> str:
    return (
        "Usage: neatlogs trace get <trace_id> [--json]\n"
        "Reads a trace from the public API and checks it.\n"
        "Needs NEATLOGS_TOKEN (service-account token with observability:read) "
        "and NEATLOGS_PROJECT_ID.\n"
        "NEATLOGS_HOST sets the app origin (default https://app.neatlogs.com, "
        "EU: https://eu.app.neatlogs.com)."
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
    # exit: 0 pass, 1 check failed, 2 not ready or not found, 3 credentials, 4 usage, 5 error
    env = os.environ if env is None else env
    fetch = fetch or _default_fetch
    err = err or (lambda line: print(line, file=sys.stderr))
    token = (env.get("NEATLOGS_TOKEN") or "").strip()
    project_id = (env.get("NEATLOGS_PROJECT_ID") or "").strip()
    if not token:
        err("NEATLOGS_TOKEN is not set")
        return EXIT_AUTH
    if not project_id:
        err("NEATLOGS_PROJECT_ID is not set")
        return EXIT_AUTH
    host = (env.get("NEATLOGS_HOST") or DEFAULT_HOST).strip()
    parsed = urlparse(host)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        err("NEATLOGS_HOST must be an absolute http(s) URL")
        return EXIT_USAGE
    base = f"{parsed.scheme}://{parsed.netloc}/api/v1/public/traces/{quote(trace_id, safe='')}"
    headers = {"Authorization": f"Bearer {token}", "x-project-id": project_id}

    def get(url: str, spans_call: bool) -> dict | int:
        try:
            status, body = fetch(url, headers, timeout)
        except (TimeoutError, urllib.error.URLError, OSError):
            err("Could not reach the trace read API")
            return EXIT_ERROR
        if status in (401, 403):
            err(
                f"Trace read rejected the credentials (HTTP {status}); "
                "check the token scope, project id and host"
            )
            return EXIT_AUTH
        if status == 404 or (spans_call and status == 409):
            err(f"Trace not ready or not found (HTTP {status}); retry after the app flushes")
            return EXIT_NOT_READY
        if status in (429, 503):
            err(f"Trace read is rate limited or unavailable (HTTP {status}); retry later")
            return EXIT_ERROR
        if not 200 <= status < 300:
            err(f"Trace read failed (HTTP {status})")
            return EXIT_ERROR
        try:
            data = json.loads(body).get("data")
        except (TypeError, ValueError, AttributeError):
            err("Trace read returned invalid JSON")
            return EXIT_ERROR
        if not isinstance(data, dict):
            err("Trace read returned an unexpected response")
            return EXIT_ERROR
        return data

    trace = get(base, False)
    if isinstance(trace, int):
        return trace
    if trace.get("finalizationStatus") == "dlq":
        err("Trace ingestion failed for good (finalizationStatus=dlq); " "retrying will not help")
        return EXIT_ERROR
    spans: list[dict] = []
    cursor = None
    for page_number in range(MAX_PAGES):
        query = f"?limit={PAGE_LIMIT}" + (f"&cursor={quote(cursor, safe='')}" if cursor else "")
        data = get(f"{base}/spans{query}", True)
        if isinstance(data, int):
            return data
        spans.extend(s for s in data.get("spans") or [] if isinstance(s, dict))
        page = data.get("page")
        cursor = _text(page.get("nextCursor")) if isinstance(page, dict) else None
        if not cursor:
            break
        if page_number == MAX_PAGES - 1:
            err(
                f"Span pagination incomplete: still more spans after {MAX_PAGES} pages, "
                "so the trace was not checked"
            )
            return EXIT_ERROR
    checks = check_trace(trace, spans)
    failed = [c for c in checks if c["status"] == "fail"]
    summary = {
        "trace_id": _text(trace.get("traceId")) or trace_id,
        "status": trace.get("status"),
        "span_count": len(spans),
        "total_tokens": trace.get("totalTokens"),
        "spans": [
            {
                "span_id": s.get("spanId"),
                "parent_span_id": s.get("parentSpanId"),
                "name": s.get("spanName"),
                "type": s.get("spanType"),
            }
            for s in spans
        ],
        "checks": checks,
        "result": "pass" if not failed else "fail",
    }
    if as_json:
        out(json.dumps(summary, indent=2))
    else:
        out(f"trace {summary['trace_id']}: {summary['result']} ({summary['span_count']} spans)")
        for c in checks:
            out(f"  {'ok  ' if c['status'] == 'pass' else 'FAIL'} {c['name']}: {c['message']}")
    return EXIT_OK if not failed else EXIT_CHECKS_FAILED
