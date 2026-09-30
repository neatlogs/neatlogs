from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


def optional_json(path: str) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def workflow_url() -> str | None:
    values = (
        os.environ.get("GITHUB_SERVER_URL"),
        os.environ.get("GITHUB_REPOSITORY"),
        os.environ.get("GITHUB_RUN_ID"),
    )
    if not all(values):
        return None
    return f"{values[0].rstrip('/')}/{values[1]}/actions/runs/{values[2]}"


def slack_message(
    status: str,
    report: dict[str, Any] | None,
    analysis: dict[str, Any] | None,
    run_url: str | None,
    upstream_issue: dict[str, Any] | None = None,
    review_issue_url: str | None = None,
    gemini_status: str | None = None,
    smoke_summary: dict[str, Any] | None = None,
) -> str:
    links = []
    if review_issue_url:
        links.append(f"<{review_issue_url}|Review issue>")
    if run_url:
        links.append(f"<{run_url}|Workflow run and evidence>")
    link = f" {' · '.join(links)}." if links else ""
    issue = (
        f" Referenced upstream issue: <{upstream_issue['url']}|"
        f"{upstream_issue.get('title', upstream_issue['url'])}>."
        if upstream_issue and upstream_issue.get("url")
        else ""
    )
    changes = (report or {}).get("changes", [])
    packages = ", ".join(
        f"{item['package']} {item.get('previouslyAnalyzed') or 'untracked'} → {item['latest']}"
        for item in changes[:4]
    )
    remaining = f", +{len(changes) - 4} more" if len(changes) > 4 else ""
    if gemini_status == "failure":
        reason = str((analysis or {}).get("error", "Gemini request failed"))[:180]
        advisory = f"Gemini advisory failed: {reason}."
    elif analysis and analysis.get("riskLevel"):
        advisory = f"Gemini advisory: {analysis['riskLevel']} potential risk (unverified)."
    else:
        advisory = "Gemini advisory unavailable."
    counts = (smoke_summary or {}).get("counts", {})
    regressions = (smoke_summary or {}).get("smokeRegressions", 0)
    if smoke_summary:
        verification = (
            f"Latest-version package/integration checks: {counts.get('pass', 0)} pass, "
            f"{counts.get('fail', 0)} fail, {counts.get('blocked', 0)} blocked installs, "
            f"{counts.get('not-tested', 0)} not tested. "
        )
        if regressions:
            verification += f"{regressions} baseline-passing activation smoke regression(s). "
        elif counts.get("fail", 0):
            verification += "Latest failures need triage; no baseline-passing regression established. "
    else:
        verification = "Latest-version smoke results unavailable. "
    if regressions:
        headline = ":red_circle: *Python SDK activation smoke regression detected.*"
    elif status != "success":
        headline = ":red_circle: *Python SDK compatibility workflow failed.*"
    else:
        headline = ":warning: *Python SDK: upstream releases newer than the monitored baseline.*"
    release_count = f"{len(changes)} upstream release{'s' if len(changes) != 1 else ''}"
    release_detail = f" {release_count}: {packages}{remaining}." if changes else ""
    return (
        f"{headline}{release_detail} {verification}"
        f"Checks cover install, dependencies, and instrumentation activation only. "
        f"{advisory} Alerts repeat until the baseline is updated.{issue}{link}"
    )


def main() -> int:
    webhook = os.environ.get("COMPAT_SLACK_WEBHOOK_URL")
    if not webhook:
        print("Slack notification skipped: COMPAT_SLACK_WEBHOOK_URL is not configured")
        return 0
    status = os.environ.get("COMPAT_JOB_STATUS", "unknown")
    if status == "success" and os.environ.get("COMPAT_CHANGES_FOUND") != "true":
        return 0
    body = json.dumps(
        {
            "text": slack_message(
                status,
                optional_json("compatibility-release-report.json"),
                optional_json("compatibility-llm-analysis.json"),
                workflow_url(),
                optional_json(
                    os.environ.get(
                        "COMPAT_UPSTREAM_ISSUE_FILE",
                        "compatibility-upstream-issue.json",
                    )
                ),
                os.environ.get("COMPAT_REVIEW_ISSUE_URL"),
                os.environ.get("COMPAT_GEMINI_STEP_STATUS"),
                optional_json("compatibility-smoke-summary.json"),
            )
        }
    ).encode()
    request = Request(
        webhook,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(f"Slack webhook returned {response.status}")
    print("Slack compatibility alert sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
