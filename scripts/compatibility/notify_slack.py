from __future__ import annotations

import hashlib
import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


ALERT_MARKER = re.compile(r"<!-- compatibility-slack-fingerprint:v1:([0-9a-f]{64}) -->")


def blocker_classification(outcome: dict[str, Any]) -> str | None:
    if outcome.get("status") != "blocked":
        return None
    reason = str(outcome.get("reason", "")).lower()
    if "timed out" in reason:
        return "timeout"
    if "resolutionimpossible" in reason or "conflicting dependencies" in reason:
        return "dependency-conflict"
    if "no matching distribution" in reason or "could not find a version" in reason:
        return "package-unavailable"
    if "requires-python" in reason or "requires python" in reason:
        return "python-version"
    if "connection" in reason or "network" in reason:
        return "network"
    return "other-install-block"


def repeatable_review_fingerprint(
    status: str,
    report: dict[str, Any] | None,
    analysis: dict[str, Any] | None,
    gemini_status: str | None,
    smoke_summary: dict[str, Any] | None,
    proposal_status: dict[str, Any] | None,
    validation_status: dict[str, Any] | None,
    publish_status: dict[str, Any] | None,
    upstream_issue: dict[str, Any] | None,
) -> str | None:
    """Identify complete, unchanged inconclusive reviews; never silence actionable events."""
    if (
        status != "success" or gemini_status != "success" or upstream_issue
        or not report or not analysis or not smoke_summary
        or not proposal_status or proposal_status.get("status") != "no-safe-fix"
        or proposal_status.get("alreadyCoveredPullRequests")
        or validation_status is not None or publish_status is not None
    ):
        return None
    try:
        changes = report["changes"]
        results = smoke_summary["results"]
        counts = smoke_summary["counts"]
        expected = {
            (change["package"], integration):
                (change.get("previouslyAnalyzed"), change["latest"])
            for change in changes for integration in change["integrations"]
        }
        if (
            not expected or len(expected) != len(results)
            or smoke_summary["pairCount"] != len(expected)
            or smoke_summary["smokeRegressions"] != 0
            or counts["not-tested"] != 0
            or counts["fail"] + counts["blocked"] == 0
            or sum(counts[key] for key in ("pass", "fail", "blocked", "not-tested")) != len(expected)
        ):
            return None
        observed = {}
        for result in results:
            pair = result["package"], result["integration"]
            baseline = result["baseline"]
            latest = result["latest"]
            latest_status = latest["status"]
            if (
                pair not in expected or pair in observed
                or (result["baselineVersion"], result["latestVersion"]) != expected[pair]
                or latest_status not in ("pass", "fail", "blocked")
                or (latest_status == "fail" and baseline["status"] != "fail")
                or result["comparison"] == "smoke-regression"
            ):
                return None
            observed[pair] = (
                *expected[pair], baseline["status"], baseline.get("stage"),
                latest_status, latest.get("stage"), result["comparison"],
                blocker_classification(latest),
            )
        if {key: sum(item["latest"]["status"] == key for item in results)
            for key in ("pass", "fail", "blocked")} != {
                key: counts[key] for key in ("pass", "fail", "blocked")
            }:
            return None
        payload = [[*pair, *observed[pair]] for pair in sorted(observed)]
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
    except (KeyError, TypeError, ValueError):
        return None


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
    counts = (smoke_summary or {}).get("counts", {})
    regressions = (smoke_summary or {}).get("smokeRegressions", 0)
    results = (smoke_summary or {}).get("results", [])
    preexisting = sum(
        item.get("latest", {}).get("status") == "fail"
        and item.get("baseline", {}).get("status") == "fail"
        for item in results
    )
    published = (publish_status or {}).get("status")
    pr_url = (publish_status or {}).get("prUrl")
    covered_drafts = [
        item for item in (proposal_status or {}).get("alreadyCoveredPullRequests", [])
        if item.get("state") == "OPEN" and item.get("isDraft") and item.get("url")
        and not (published == "ready" and item["url"] == pr_url)
    ]
    if published in {"created", "ready"}:
        headline = ":large_green_circle: *Python SDK: validated fix PR ready — review code and tests.*"
    elif regressions:
        headline = f":red_circle: *Python SDK: {regressions} activation regression candidate(s) — review the evidence.*"
    elif status != "success" or gemini_status == "failure":
        headline = ":red_circle: *Python SDK: compatibility automation failed — inspect the run.*"
    elif any(counts.get(key, 0) for key in ("fail", "blocked", "not-tested")):
        headline = ":warning: *Python SDK: no confirmed regression — triage failed or blocked checks.*"
    elif smoke_summary is None:
        headline = ":warning: *Python SDK: check evidence incomplete — inspect the run.*"
    elif (validation_status or {}).get("status") == "failed" or (proposal_status or {}).get("status") == "rejected":
        headline = ":warning: *Python SDK: fix attempt rejected — review validation evidence.*"
    else:
        headline = ":warning: *Python SDK: new upstream releases — review the recorded checks.*"

    if smoke_summary:
        checked = (
            f"*Checked:* {counts.get('pass', 0)} pass · {counts.get('fail', 0)} fail"
            f" ({preexisting} also failed at baseline) · {counts.get('blocked', 0)} blocked"
            f" · {counts.get('not-tested', 0)} not tested. "
        )
        if regressions:
            checked += f"*Regression:* {regressions} baseline-pass/latest-fail activation candidate(s)."
        elif counts.get("fail", 0) > preexisting:
            checked += "*Regression:* other latest failures need comparison; none confirmed."
        else:
            checked += "*Regression:* none found in the tested activation scope."
    else:
        checked = "*Checked:* results unavailable; regression status unknown."
    changes = (report or {}).get("changes", [])
    if changes:
        examples = ", ".join(
            f"{item['package']} {item.get('previouslyAnalyzed') or 'untracked'} → {item['latest']}"
            for item in changes[:2]
        )
        release_word = "release" if len(changes) == 1 else "releases"
        checked += f" {len(changes)} newer upstream {release_word} (e.g. {examples})."

    if published in {"created", "ready"} and pr_url:
        pr = f"*Fix PR:* <{pr_url}|open for human review>."
        if validation_status and validation_status.get("redGreen") == "red-before-green-after":
            pr += " Focused red/green test passed."
        elif validation_status and validation_status.get("reproduction") == "activation-smoke-baseline-pass-latest-fail-patched-pass":
            pr += " Activation check passed at baseline and after patch, failed on latest before patch."
    elif published == "already-covered":
        state = (publish_status or {}).get("state", "OPEN").lower()
        if pr_url:
            pr = f"*Fix PR:* <{pr_url}|existing PR> ({state}{', draft' if publish_status.get('isDraft') else ''}); no new PR opened."
        else:
            pr = f"*Fix PR:* existing PR ({state}); no new PR opened."
    elif published == "failed":
        reason = short_reason(publish_status.get("reason", "unknown"), 110)
        if publish_status.get("isDraft") and pr_url:
            pr = f"*Fix PR:* <{pr_url}|existing draft> could not be marked ready: {reason}"
        else:
            pr = f"*Fix PR:* not opened — publication failed: {reason}"
    elif validation_status and validation_status.get("status") == "failed":
        pr = f"*Fix PR:* not opened — proposed fix failed validation: {short_reason(validation_status.get('reason', 'unknown'), 110)}"
    elif proposal_status and proposal_status.get("status") == "rejected":
        pr = f"*Fix PR:* not opened — proposal rejected: {short_reason(proposal_status.get('reason', 'unknown'), 110)}"
    elif proposal_status and proposal_status.get("status") == "no-safe-fix":
        pr = "*Fix PR:* not opened — no safe code-specific fix was generated."
    elif proposal_status and proposal_status.get("status") == "proposed":
        pr = "*Fix PR:* unconfirmed — proposal exists, but publication was not confirmed."
    else:
        pr = "*Fix PR:* not opened — no validated fix available."
    no_surface = (proposal_status or {}).get("noSdkPatchSurface", [])
    if no_surface and published not in {"created", "ready"}:
        item = no_surface[0]
        pr += (
            f" No integration-specific SDK patch source for {len(no_surface)} pair(s)"
            f" (e.g. {item['package']}/{item['integration']})."
        )
    if covered_drafts and published not in {"created", "ready", "already-covered"}:
        pr += f" <{covered_drafts[0]['url']}|Existing draft fix PR> remains draft."

    if published in {"created", "ready"}:
        action = "Review the PR and its validation evidence."
    elif regressions:
        action = "Investigate the candidate regression and fix attempt."
    elif status != "success" or gemini_status == "failure":
        action = "Inspect the failed workflow step and retry after correction."
    elif counts.get("fail", 0) or counts.get("blocked", 0) or counts.get("not-tested", 0):
        work = []
        if counts.get("fail", 0):
            work.append("preexisting failures" if counts["fail"] == preexisting else "failed checks")
        if counts.get("blocked", 0):
            work.append("blocked installs")
        if counts.get("not-tested", 0):
            work.append("untested checks")
        items = f"{', '.join(work[:-1])} and {work[-1]}" if len(work) > 1 else work[0]
        action = f"Triage {items} in the issue."
    elif smoke_summary is None or (validation_status or {}).get("status") == "failed" or (proposal_status or {}).get("status") == "rejected":
        action = "Inspect the issue and failed fix attempt."
    else:
        action = "Review the release findings in the issue."
    if gemini_status == "failure":
        action += f" Gemini analysis failed: {short_reason((analysis or {}).get('error', 'request failed'), 90)}"
    links = []
    if review_issue_url:
        links.append(f"<{review_issue_url}|Review issue>")
    if run_url:
        links.append(f"<{run_url}|Run and evidence>")
    if upstream_issue and upstream_issue.get("url"):
        links.append(f"<{upstream_issue['url']}|Upstream issue>")
    if pr_url and published not in {"created", "ready", "already-covered", "failed"}:
        links.append(f"<{pr_url}|Fix PR>")
    link_text = f" {' · '.join(links)}" if links else ""
    return f"{headline}\n{checked}\n{pr}\n*Action:* {action}{link_text}"


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
    review_issue_url = os.environ.get("COMPAT_REVIEW_ISSUE_URL")
    if not should_notify(
        status, report, analysis, gemini_status, smoke_summary, proposal_status,
        validation_status, publish_status, upstream_issue,
    ):
        if review_issue_url:
            # A clean review ends the previous inconclusive finding's lifetime.
            Path("compatibility-sent-slack-marker.txt").write_text("")
        print("Slack notification suppressed: both version checks passed and no fix was proposed")
        return 0
    fingerprint = repeatable_review_fingerprint(
        status, report, analysis, gemini_status, smoke_summary, proposal_status,
        validation_status, publish_status, upstream_issue,
    )
    prior_marker = Path("compatibility-prior-slack-marker.txt")
    if fingerprint and review_issue_url and prior_marker.exists():
        match = ALERT_MARKER.fullmatch(prior_marker.read_text().strip())
        if match and match.group(1) == fingerprint:
            print("Slack notification suppressed: unchanged inconclusive review; issue and artifacts updated")
            return 0
    body = json.dumps(
        {
            "text": slack_message(
                status,
                report,
                analysis,
                workflow_url(),
                upstream_issue,
                review_issue_url,
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
    if review_issue_url:
        # Empty content clears an older fingerprint after an actionable alert.
        Path("compatibility-sent-slack-marker.txt").write_text(
            f"<!-- compatibility-slack-fingerprint:v1:{fingerprint} -->\n" if fingerprint else ""
        )
    print("Slack compatibility alert sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
