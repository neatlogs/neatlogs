#!/usr/bin/env python3
"""Poll and validate one exact packaged-release trace through the product API."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _normalized_span(span: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(span.get("node_name") or span.get("span_name") or ""),
        "kind": str(span.get("node_type") or span.get("span_type") or "").lower(),
        "span_id": span.get("span_id"),
        "parent": span.get("parent_span_id") or None,
        "data": span.get("data") if isinstance(span.get("data"), dict) else {},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    api_key = os.environ.get("NEATLOGS_API_KEY")
    endpoint = os.environ.get("NEATLOGS_ENDPOINT")
    if not api_key or not endpoint:
        raise SystemExit("NEATLOGS_API_KEY and NEATLOGS_ENDPOINT are required")

    expected = json.loads(args.expected.read_text())
    trace_id = expected["trace_id"]
    url = f"{endpoint.rstrip('/')}/api/traces/v3/{quote(trace_id, safe='')}"
    deadline = time.monotonic() + args.timeout
    payload: dict[str, Any] | None = None

    while time.monotonic() < deadline:
        response = requests.get(url, headers={"x-api-key": api_key}, timeout=5)
        if response.status_code in {401, 403}:
            response.raise_for_status()
        if response.ok and response.status_code != 202:
            value = response.json()
            _require(isinstance(value, dict), "trace readback must be an object")
            payload = value
            break
        if response.status_code not in {202, 404}:
            response.raise_for_status()
        time.sleep(1)

    _require(payload is not None, "timed out waiting for exact trace")
    args.output.with_name("readback-raw.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    _require(payload.get("_id") == trace_id, "readback trace ID mismatch")
    _require(str(payload.get("status") or "").lower() == "success", "trace did not finalize")

    raw_spans = [item for item in payload.get("spans", []) if isinstance(item, dict)]
    actual = [_normalized_span(item) for item in raw_spans]
    by_id = {item["span_id"]: item for item in actual}
    expected_by_id = {item["span_id"]: item for item in expected["spans"]}

    _require(len(by_id) == len(actual), "duplicate span IDs")
    _require(set(by_id) == set(expected_by_id), "missing or unexpected span IDs")
    _require(payload.get("spanCount") == expected["expected_span_count"], "span count mismatch")

    for span_id, wanted in expected_by_id.items():
        got = by_id[span_id]
        expected_display_name = wanted.get("display_name", wanted["name"])
        _require(
            got["name"] == expected_display_name,
            f"name mismatch for {span_id}: expected {expected_display_name!r}, got {got['name']!r}",
        )
        _require(got["kind"] == wanted["kind"], f"kind mismatch for {wanted['name']}")
        _require(got["parent"] == wanted["parent"], f"parent mismatch for {wanted['name']}")
        _require("input_value" in got["data"], f"missing input for {wanted['name']}")
        _require("output_value" in got["data"], f"missing output for {wanted['name']}")

    roots = [item for item in actual if item["parent"] is None]
    _require(len(roots) == 1, "expected one meaningful root")
    _require(payload.get("promptTokens") == expected["expected_prompt_tokens"], "prompt tokens")
    _require(
        payload.get("completionTokens") == expected["expected_completion_tokens"],
        "completion tokens",
    )
    _require(payload.get("totalTokensUsed") == expected["expected_total_tokens"], "total tokens")

    sanitized = {
        "format_version": expected["format_version"],
        "status": "pass",
        "trace_id": trace_id,
        "readback_status": payload.get("status"),
        "span_count": len(actual),
        "root_count": len(roots),
        "prompt_tokens": payload.get("promptTokens"),
        "completion_tokens": payload.get("completionTokens"),
        "total_tokens": payload.get("totalTokensUsed"),
        "names": sorted(item["name"] for item in actual),
    }
    args.output.write_text(json.dumps(sanitized, indent=2, sort_keys=True) + "\n")
    print(json.dumps(sanitized, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
