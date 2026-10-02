from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


def optional_json(path: str) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def short_reason(value: Any, width: int) -> str:
    result = textwrap.shorten(str(value), width=width, placeholder="…")
    return result if result.endswith("…") else result.rstrip(".") + "."


def workflow_url() -> str | None:
    values = (
        os.environ.get("GITHUB_SERVER_URL"),
        os.environ.get("GITHUB_REPOSITORY"),
        os.environ.get("GITHUB_RUN_ID"),
    )
    if not all(values):
        return None
    return f"{values[0].rstrip('/')}/{values[1]}/actions/runs/{values[2]}"


def should_notify(
    status: str,
    report: dict[str, Any] | None,
    analysis: dict[str, Any] | None,
    gemini_status: str | None,
    smoke_summary: dict[str, Any] | None,
    proposal_status: dict[str, Any] | None,
    validation_status: dict[str, Any] | None,
    publish_status: dict[str, Any] | None,
    upstream_issue: dict[str, Any] | None = None,
) -> bool:
    """Silence only a complete, passing release review with no proposed fix."""
    if (
        status != "success"
        or gemini_status != "success"
        or upstream_issue
        or not report
        or not analysis
        or not smoke_summary
        or not proposal_status
        or proposal_status.get("status") != "no-safe-fix"
        or validation_status is not None
        or publish_status is not None
    ):
        return True
    if any(
        item.get("state") == "OPEN" and item.get("isDraft")
        for item in proposal_status.get("alreadyCoveredPullRequests", [])
    ):
        return True
    try:
        expected = {
            (change["package"], integration):
                (change.get("previouslyAnalyzed"), change["latest"])
            for change in report["changes"]
            for integration in change["integrations"]
        }
        results = smoke_summary["results"]
        counts = smoke_summary["counts"]
        if (
            not expected
            or len(expected) != len(results)
            or smoke_summary["pairCount"] != len(expected)
            or smoke_summary["smokeRegressions"] != 0
            or counts["pass"] != len(expected)
            or any(counts[key] for key in ("fail", "blocked", "not-tested"))
        ):
            return True
        for result in results:
            pair = (result["package"], result["integration"])
            if (
                pair not in expected
                or (result["baselineVersion"], result["latestVersion"]) != expected[pair]
                or result["baseline"]["status"] != "pass"
                or result["latest"]["status"] != "pass"
                or result["comparison"] != "latest-smoke-passed"
            ):
                return True
    except (KeyError, TypeError, ValueError):
        return True
    return False


def slack_message(
    status: str,
    report: dict[str, Any] | None,
    analysis: dict[str, Any] | None,
    run_url: str | None,
    upstream_issue: dict[str, Any] | None = None,
    review_issue_url: str | None = None,
    gemini_status: str | None = None,
    smoke_summary: dict[str, Any] | None = None,
    proposal_status: dict[str, Any] | None = None,
    validation_status: dict[str, Any] | None = None,
    publish_status: dict[str, Any] | None = None,
) -> str:
    covered_drafts = [
        item for item in (proposal_status or {}).get("alreadyCoveredPullRequests", [])
        if item.get("state") == "OPEN" and item.get("isDraft")
        and not (
            publish_status and publish_status.get("status") == "ready"
            and item.get("url") == publish_status.get("prUrl")
        )
    ]
    links = []
    if review_issue_url:
        links.append(f"<{review_issue_url}|Review issue>")
    if run_url:
        links.append(f"<{run_url}|Workflow run and evidence>")
    if publish_status and publish_status.get("prUrl"):
        if publish_status.get("isDraft"):
            label = "Existing draft fix PR"
        elif publish_status.get("state") in {"CLOSED", "MERGED"}:
            label = "Previous fix PR"
        else:
            label = "Fix PR for review"
        links.append(f"<{publish_status['prUrl']}|{label}>")
    elif covered_drafts and covered_drafts[0].get("url"):
        links.append(f"<{covered_drafts[0]['url']}|Existing draft fix PR>")
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
        reason = short_reason((analysis or {}).get("error", "Gemini request failed"), 180)
        advisory = f"Gemini advisory failed: {reason}"
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
            affected = [
                f"{item['package']}/{item['integration']}"
                for item in smoke_summary.get("results", [])
                if item.get("comparison") == "smoke-regression"
            ]
            names = (
                f" ({', '.join(affected[:3])}{', +more' if len(affected) > 3 else ''})"
                if affected else ""
            )
            verification += f"{regressions} baseline-passing activation smoke regression(s){names}. "
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
    proposal = ""
    if publish_status and publish_status.get("status") == "failed":
        if publish_status.get("isDraft") and publish_status.get("prUrl"):
            proposal = f" Existing bot draft PR could not be marked ready: {short_reason(publish_status.get('reason', 'unknown'), 150)}"
        else:
            proposal = f" Fix PR could not be opened: {short_reason(publish_status.get('reason', 'unknown'), 150)}"
    elif validation_status and validation_status.get("status") == "failed":
        proposal = f" Gemini proposed a fix, but validation failed: {short_reason(validation_status.get('reason', 'unknown'), 150)} No PR opened."
    elif publish_status and publish_status.get("status") == "already-covered":
        if publish_status.get("state") in {"CLOSED", "MERGED"}:
            proposal = f" Previous fix PR is {publish_status['state'].lower()}; no new PR opened."
        elif publish_status.get("isDraft"):
            proposal = " Existing fix PR remains draft; a human must mark it ready for review."
        else:
            proposal = " Existing fix PR is open for human code review."
    elif publish_status and publish_status.get("status") == "created":
        if validation_status and validation_status.get("redGreen") == "red-before-green-after":
            proposal = " Fix PR opened for review; focused red/green test passed. Human code review required."
        else:
            proposal = " Gemini-proposed fix PR opened for review; behavior fix unverified. Human code review required."
    elif publish_status and publish_status.get("status") == "ready":
        proposal = " Existing validated bot fix PR marked ready for human review."
    elif proposal_status and proposal_status.get("status") == "proposed":
        if validation_status and validation_status.get("status") == "validated":
            proposal = " Gemini proposed a fix and validation passed, but PR publication is unconfirmed."
        else:
            proposal = " Gemini proposed a fix; validation result unavailable. No PR confirmed."
    elif proposal_status:
        proposal_labels = {
            "no-safe-fix": "Gemini did not produce a concrete safe fix",
            "rejected": "Gemini fix proposal failed or was rejected",
            "unavailable": "Gemini fix proposal is unavailable",
        }
        label = proposal_labels.get(
            proposal_status.get("status"), "Gemini fix proposal status unknown"
        )
        proposal = f" {label}: {short_reason(proposal_status.get('reason', 'unknown'), 120)} No PR opened."
    if covered_drafts and not (
        publish_status and publish_status.get("status") == "already-covered"
        and publish_status.get("isDraft")
    ):
        proposal += f" {len(covered_drafts)} matching fix PR(s) remain draft; mark ready manually."
    deferred = len((proposal_status or {}).get("deferredCandidates", []))
    if deferred:
        proposal += f" {deferred} candidate(s) deferred to later runs."
    no_surface = (proposal_status or {}).get("noSdkPatchSurface", [])
    if no_surface:
        examples = ", ".join(
            f"{item['package']}/{item['integration']}" for item in no_surface[:2]
        )
        remainder = f", +{len(no_surface) - 2} more" if len(no_surface) > 2 else ""
        proposal += (
            f" No integration-specific SDK patch source for {len(no_surface)} pair(s)"
            f" ({examples}{remainder}); automatic fix generation skipped those pairs."
        )
    return (
        f"{headline}{release_detail} {verification}"
        f"Checks cover install, dependencies, and instrumentation activation only. "
        f"{advisory}{proposal}{issue}{link}"
    )


def main() -> int:
    webhook = os.environ.get("COMPAT_SLACK_WEBHOOK_URL")
    if not webhook:
        print("Slack notification skipped: COMPAT_SLACK_WEBHOOK_URL is not configured")
        return 0
    status = os.environ.get("COMPAT_JOB_STATUS", "unknown")
    if status == "success" and os.environ.get("COMPAT_CHANGES_FOUND") != "true":
        return 0
    report = optional_json("compatibility-release-report.json")
    analysis = optional_json("compatibility-llm-analysis.json")
    upstream_issue = optional_json(
        os.environ.get("COMPAT_UPSTREAM_ISSUE_FILE", "compatibility-upstream-issue.json")
    )
    gemini_status = os.environ.get("COMPAT_GEMINI_STEP_STATUS")
    smoke_summary = optional_json("compatibility-smoke-summary.json")
    proposal_status = optional_json("compatibility-fix-status.json")
    validation_status = optional_json("compatibility-validation-status.json")
    publish_status = optional_json("compatibility-publish-status.json")
    if not should_notify(
        status, report, analysis, gemini_status, smoke_summary, proposal_status,
        validation_status, publish_status, upstream_issue,
    ):
        print("Slack notification suppressed: both version checks passed and no fix was proposed")
        return 0
    body = json.dumps(
        {
            "text": slack_message(
                status,
                report,
                analysis,
                workflow_url(),
                upstream_issue,
                os.environ.get("COMPAT_REVIEW_ISSUE_URL"),
                gemini_status,
                smoke_summary,
                proposal_status,
                validation_status,
                publish_status,
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
