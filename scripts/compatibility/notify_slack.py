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
    return "/".join(values) if all(values) else None


def slack_message(
    status: str,
    report: dict[str, Any] | None,
    analysis: dict[str, Any] | None,
    run_url: str | None,
    upstream_issue: dict[str, Any] | None = None,
) -> str:
    link = f" <{run_url}|Open workflow run>." if run_url else ""
    issue = (
        f" Reproduced upstream issue: <{upstream_issue['url']}|"
        f"{upstream_issue.get('title', upstream_issue['url'])}>."
        if upstream_issue and upstream_issue.get("url")
        else ""
    )
    if status != "success":
        return f":red_circle: *Python SDK compatibility workflow failed.*{issue}{link}"
    changes = (report or {}).get("changes", [])
    packages = ", ".join(
        f"{item['package']} {item.get('previouslyAnalyzed') or 'untracked'} → {item['latest']}"
        for item in changes[:8]
    )
    remaining = f", +{len(changes) - 8} more" if len(changes) > 8 else ""
    risk = (
        f" Advisory risk: *{analysis['riskLevel']}*."
        if analysis and analysis.get("riskLevel")
        else ""
    )
    return (
        f":warning: *Python SDK compatibility review required:* "
        f"{len(changes)} upstream release(s). {packages}{remaining}.{risk}{issue}{link}"
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
