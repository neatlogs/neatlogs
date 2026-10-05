"""Fail a scheduled compatibility run only for regressions or broken automation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def read_status(path: str) -> dict[str, Any] | None:
    try:
        value = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def failure_reason(
    summary: dict[str, Any] | None,
    proposal: dict[str, Any] | None,
    validation: dict[str, Any] | None,
    published: dict[str, Any] | None,
    jobs: dict[str, str],
) -> str | None:
    if not isinstance(summary, dict) or not isinstance(summary.get("smokeRegressions"), int):
        return "Compatibility comparison summary is missing or malformed"
    if summary["smokeRegressions"]:
        return f"{summary['smokeRegressions']} baseline-passing activation smoke regression(s)"
    for name in ("discover", "analyze", "smoke", "report", "propose"):
        if jobs.get(name) != "success":
            return f"Compatibility {name} job did not complete"
    for name in ("validate", "publish"):
        if jobs.get(name) == "failure":
            return f"Compatibility {name} job failed"
    if proposal is None:
        return "Fix proposal status is missing or malformed"
    state = proposal.get("status")
    if state == "rejected" and proposal.get("kind") == "unsafe-proposal":
        return None if validation is None and published is None and jobs.get("validate") == "skipped" and jobs.get("publish") == "skipped" else "Unexpected fix processing after safe proposal rejection"
    if state == "no-safe-fix":
        return None if validation is None and published is None and jobs.get("validate") == "skipped" and jobs.get("publish") == "skipped" else "Unexpected fix processing without a proposal"
    if state != "proposed":
        return "Gemini fix proposal request did not complete"
    if jobs.get("validate") != "success":
        return "Generated fix validation job did not complete"
    if validation is None:
        return "Generated fix validation status is missing"
    if validation.get("status") == "rejected" and validation.get("kind") == "candidate-rejected":
        return None if published is None else "Rejected generated fix was unexpectedly published"
    if validation.get("status") != "validated":
        return "Generated fix validation did not complete"
    if jobs.get("publish") != "success":
        return "Validated fix PR publication job did not complete"
    if published is None or published.get("status") not in {"created", "ready", "already-covered"}:
        return "Validated fix PR publication did not complete"
    return None


def main() -> int:
    reason = failure_reason(
        read_status("compatibility-smoke-summary.json"),
        read_status("compatibility-fix-status.json"),
        read_status("compatibility-validation-status.json"),
        read_status("compatibility-publish-status.json"),
        {name: os.environ.get(f"COMPAT_{name.upper()}_RESULT", "") for name in (
            "discover", "analyze", "smoke", "report", "propose", "validate", "publish"
        )},
    )
    if reason:
        print(reason)
        return 1
    print("Compatibility run complete: no activation regression or automation failure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
