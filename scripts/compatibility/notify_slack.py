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
    """Notify only about regression candidates, fix PRs, or incomplete automation."""
    if (
        status != "success"
        or gemini_status != "success"
        or not isinstance(report, dict)
        or not isinstance(analysis, dict)
        or analysis.get("failed")
        or analysis.get("skipped")
        or not isinstance(smoke_summary, dict)
        or not isinstance(proposal_status, dict)
        or (validation_status is not None and not isinstance(validation_status, dict))
        or (publish_status is not None and not isinstance(publish_status, dict))
    ):
        return True
    if proposal_status.get("status") not in {"no-safe-fix", "proposed"}:
        return True
    if validation_status and validation_status.get("status") != "validated":
        return True
    published = (publish_status or {}).get("status")
    if published in {"created", "ready", "failed"}:
        return True
    if published not in {None, "already-covered"}:
        return True
    if proposal_status.get("status") == "proposed" and published != "already-covered":
        return True
    if validation_status and published != "already-covered":
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
            or counts["not-tested"] != 0
            or sum(counts[key] for key in ("pass", "fail", "blocked", "not-tested")) != len(expected)
        ):
            return True
        observed = set()
        for result in results:
            pair = (result["package"], result["integration"])
            baseline = result["baseline"]["status"]
            latest = result["latest"]["status"]
            if (
                pair not in expected
                or pair in observed
                or (result["baselineVersion"], result["latestVersion"]) != expected[pair]
                or latest not in {"pass", "fail", "blocked"}
                or (latest == "fail" and baseline != "fail")
                or result["comparison"] != {
                    "pass": "latest-smoke-passed",
                    "fail": "latest-failure-needs-triage",
                    "blocked": "latest-install-blocked",
                }[latest]
            ):
                return True
            observed.add(pair)
        if any(
            counts[key] != sum(result["latest"]["status"] == key for result in results)
            for key in ("pass", "fail", "blocked")
        ):
            return True
    except (AttributeError, KeyError, TypeError, ValueError):
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
    automation_incomplete = (
        report is None or analysis is None or proposal_status is None
        or bool((analysis or {}).get("failed") or (analysis or {}).get("skipped"))
    )
    if published in {"created", "ready"}:
        headline = ":large_green_circle: *Python SDK: validated fix PR ready — review code and tests.*"
    elif regressions:
        headline = f":red_circle: *Python SDK: {regressions} activation regression candidate(s) — review the evidence.*"
    elif published == "failed":
        headline = ":red_circle: *Python SDK: fix PR publication failed — inspect the run.*"
    elif (validation_status or {}).get("status") == "failed":
        headline = ":warning: *Python SDK: proposed fix failed validation — review evidence.*"
    elif (proposal_status or {}).get("status") == "rejected":
        headline = ":warning: *Python SDK: Gemini fix proposal rejected — inspect AI review.*"
    elif status != "success" or gemini_status not in {None, "success"} or automation_incomplete:
        headline = ":red_circle: *Python SDK: compatibility automation failed — inspect the run.*"
    elif smoke_summary is None:
        headline = ":warning: *Python SDK: check evidence incomplete — inspect the run.*"
    elif any(counts.get(key, 0) for key in ("fail", "blocked", "not-tested")):
        headline = ":warning: *Python SDK: check evidence incomplete — inspect the run.*"
    else:
        headline = ":warning: *Python SDK: compatibility review requires attention — inspect the run.*"

    if smoke_summary:
        checked = (
            f"*Checks:* {counts.get('pass', 0)} pass · {counts.get('fail', 0)} fail"
            f" ({preexisting} also failed at baseline) · {counts.get('blocked', 0)} blocked"
            f" · {counts.get('not-tested', 0)} not tested."
        )
        if regressions:
            regression = f"*Regression:* {regressions} baseline-pass/latest-fail activation candidate(s)."
        elif counts.get("fail", 0) > preexisting:
            regression = "*Regression:* other latest failures need comparison; none confirmed."
        else:
            regression = "*Regression:* none found in the tested activation scope."
    else:
        checked = "*Checks:* results unavailable."
        regression = "*Regression:* status unknown."
    why_alerted = None
    if not regressions:
        if published == "failed":
            why_alerted = "*Why alerted:* a validated fix could not be published. This is not a confirmed activation regression."
        elif (validation_status or {}).get("status") == "failed":
            why_alerted = "*Why alerted:* the proposed fix failed validation. This is not a confirmed activation regression."
        elif (proposal_status or {}).get("status") in {"rejected", "unavailable"}:
            why_alerted = "*Why alerted:* Gemini's fix proposal could not be used. The bounded checks did not establish an SDK regression."
        elif gemini_status == "failure" or (analysis or {}).get("failed"):
            why_alerted = "*Why alerted:* Gemini review did not complete. Passing bounded checks cannot establish full SDK compatibility."
        elif status != "success" or automation_incomplete:
            why_alerted = "*Why alerted:* compatibility automation did not complete; this is not a confirmed SDK regression."
    changes = (report or {}).get("changes", [])
    if changes:
        examples = ", ".join(
            f"{item['package']} {item.get('previouslyAnalyzed') or 'untracked'} → {item['latest']}"
            for item in changes[:2]
        )
        release_word = "release" if len(changes) == 1 else "releases"
        upstream = f"*Upstream:* {len(changes)} newer {release_word} (e.g. {examples})."
    else:
        upstream = None

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
    elif published == "failed" or (validation_status or {}).get("status") == "failed" or (proposal_status or {}).get("status") == "rejected":
        action = "Inspect the failed fix attempt in the run and issue."
    elif status != "success" or gemini_status not in {None, "success"} or automation_incomplete:
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
    evidence = "\n".join(item for item in (checked, regression, why_alerted, upstream) if item)
    next_step = f"*Next step:* {action}"
    link_text = f"\n{' · '.join(links)}" if links else ""
    return f"{headline}\n\n{evidence}\n\n{pr}\n{next_step}{link_text}"


def slack_payload(message: str) -> dict[str, Any]:
    """Keep a readable fallback and render each alert decision as a separate Slack block."""
    headline, evidence, outcome = message.split("\n\n", 2)
    pr, action = outcome.split("\n", 1)
    return {
        "text": message,
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": headline}},
            {"type": "section", "text": {"type": "mrkdwn", "text": evidence}},
            {"type": "divider"},
            {"type": "section", "text": {"type": "mrkdwn", "text": pr}},
            {"type": "section", "text": {"type": "mrkdwn", "text": action}},
        ],
    }


def main() -> int:
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
        print("Slack notification suppressed: complete review found no regression or automation failure; issue and artifacts updated")
        return 0
    webhook = os.environ.get("COMPAT_SLACK_WEBHOOK_URL")
    if not webhook:
        raise RuntimeError("Actionable compatibility alert could not be delivered: COMPAT_SLACK_WEBHOOK_URL is not configured")
    # A malformed artifact is itself alert-worthy; render the remaining evidence safely.
    report = report if isinstance(report, dict) else None
    analysis = analysis if isinstance(analysis, dict) else None
    smoke_summary = smoke_summary if isinstance(smoke_summary, dict) else None
    proposal_status = proposal_status if isinstance(proposal_status, dict) else None
    validation_status = validation_status if isinstance(validation_status, dict) else None
    publish_status = publish_status if isinstance(publish_status, dict) else None
    upstream_issue = upstream_issue if isinstance(upstream_issue, dict) else None
    message = slack_message(
        status, report, analysis, workflow_url(), upstream_issue, review_issue_url,
        gemini_status, smoke_summary, proposal_status, validation_status, publish_status,
    )
    body = json.dumps(slack_payload(message)).encode()
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
