"""Slack deduplication of repeated, inconclusive compatibility reviews."""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.compatibility.notify_slack import (
    ALERT_MARKER, main, repeatable_review_fingerprint,
)


def october_review():
    """Representative October 3-4 finding set: 50 pass, 6 preexisting fail, 5 blocked."""
    changes = []
    results = []
    for index in range(61):
        package = f"package-{index}"
        integration = f"integration-{index}"
        changes.append({
            "package": package, "previouslyAnalyzed": "1", "latest": "2",
            "integrations": [integration],
        })
        latest_status = "pass" if index < 50 else "fail" if index < 56 else "blocked"
        baseline_status = "pass" if index < 50 else "fail" if index < 56 else "blocked"
        results.append({
            "package": package, "integration": integration,
            "baselineVersion": "1", "latestVersion": "2",
            "baseline": {"status": baseline_status, "stage": "instrumentation-activation"},
            "latest": {
                "status": latest_status,
                "stage": "install" if latest_status == "blocked" else "instrumentation-activation",
                "reason": "Timed out after 240s" if index == 56 else
                          "ERROR: ResolutionImpossible: conflicting dependencies" if index >= 57 else "",
            },
            "comparison": "latest-smoke-passed" if latest_status == "pass" else
                          "latest-failure-needs-triage" if latest_status == "fail" else
                          "latest-install-blocked",
        })
    return {
        "compatibility-release-report.json": {"changes": changes},
        "compatibility-llm-analysis.json": {"riskLevel": "high"},
        "compatibility-smoke-summary.json": {
            "pairCount": 61, "counts": {"pass": 50, "fail": 6, "blocked": 5, "not-tested": 0},
            "smokeRegressions": 0, "results": results,
        },
        "compatibility-fix-status.json": {
            "status": "no-safe-fix", "selectedCandidate": {"package": "package-1"},
        },
    }


def fingerprint(files, *, status="success", gemini_status="success"):
    return repeatable_review_fingerprint(
        status, files.get("compatibility-release-report.json"),
        files.get("compatibility-llm-analysis.json"), gemini_status,
        files.get("compatibility-smoke-summary.json"),
        files.get("compatibility-fix-status.json"),
        files.get("compatibility-validation-status.json"),
        files.get("compatibility-publish-status.json"), None,
    )


class NotifySlackDedupeTests(unittest.TestCase):
    def test_repeated_findings_ignore_rotating_gemini_and_pip_log_noise(self):
        first = october_review()
        second = copy.deepcopy(first)
        second["compatibility-llm-analysis.json"] = {"riskLevel": "low", "summary": "new advisory"}
        second["compatibility-fix-status.json"]["selectedCandidate"] = {"package": "package-50"}
        second["compatibility-smoke-summary.json"]["results"][57]["latest"]["reason"] = (
            "ERROR: ResolutionImpossible: dependency conflict; different pip trace"
        )
        self.assertRegex(fingerprint(first), r"^[0-9a-f]{64}$")
        self.assertEqual(fingerprint(first), fingerprint(second))

    def test_new_version_status_or_blocker_class_changes_fingerprint(self):
        first = october_review()
        prior = fingerprint(first)
        version = copy.deepcopy(first)
        version["compatibility-release-report.json"]["changes"][0]["latest"] = "3"
        version["compatibility-smoke-summary.json"]["results"][0]["latestVersion"] = "3"
        status = copy.deepcopy(first)
        result = status["compatibility-smoke-summary.json"]["results"][56]
        result["baseline"]["status"] = "fail"
        result["latest"] = {"status": "fail", "stage": "instrumentation-activation"}
        result["comparison"] = "latest-failure-needs-triage"
        status["compatibility-smoke-summary.json"]["counts"].update(fail=7, blocked=4)
        blocker = copy.deepcopy(first)
        blocker["compatibility-smoke-summary.json"]["results"][56]["latest"]["reason"] = (
            "ERROR: ResolutionImpossible: conflicting dependencies"
        )
        for changed in (version, status, blocker):
            self.assertNotEqual(prior, fingerprint(changed))

    def test_regression_automation_failure_and_pr_are_never_suppressed(self):
        files = october_review()
        self.assertIsNone(fingerprint(files, status="failure"))
        self.assertIsNone(fingerprint(files, gemini_status="failure"))
        for filename, value in (
            ("compatibility-validation-status.json", {"status": "failed"}),
            ("compatibility-publish-status.json", {"status": "created", "prUrl": "https://example.test/pr"}),
        ):
            changed = copy.deepcopy(files)
            changed[filename] = value
            self.assertIsNone(fingerprint(changed))
        regression = copy.deepcopy(files)
        regression["compatibility-smoke-summary.json"]["smokeRegressions"] = 1
        self.assertIsNone(fingerprint(regression))
        incomplete = copy.deepcopy(files)
        incomplete["compatibility-smoke-summary.json"]["counts"]["not-tested"] = 1
        self.assertIsNone(fingerprint(incomplete))

    def test_posts_once_for_same_finding_then_posts_new_version(self):
        files = october_review()
        env = {
            "COMPAT_SLACK_WEBHOOK_URL": "https://slack.example.invalid/webhook",
            "COMPAT_JOB_STATUS": "success", "COMPAT_CHANGES_FOUND": "true",
            "COMPAT_GEMINI_STEP_STATUS": "success",
            "COMPAT_REVIEW_ISSUE_URL": "https://github.com/neatlogs/neatlogs/issues/125",
        }
        with tempfile.TemporaryDirectory() as directory:
            old_cwd = os.getcwd()
            try:
                os.chdir(directory)
                with patch.dict("os.environ", env, clear=True), \
                        patch("scripts.compatibility.notify_slack.optional_json", side_effect=files.get), \
                        patch("scripts.compatibility.notify_slack.urlopen") as post:
                    post.return_value.__enter__.return_value.status = 200
                    self.assertEqual(main(), 0)
                    self.assertEqual(post.call_count, 1)
                    sent = Path("compatibility-sent-slack-marker.txt").read_text().strip()
                    self.assertTrue(ALERT_MARKER.fullmatch(sent))
                    Path("compatibility-prior-slack-marker.txt").write_text(sent)
                    files["compatibility-llm-analysis.json"]["riskLevel"] = "low"
                    files["compatibility-fix-status.json"]["selectedCandidate"] = {"package": "package-51"}
                    self.assertEqual(main(), 0)
                    self.assertEqual(post.call_count, 1)
                    files["compatibility-release-report.json"]["changes"][0]["latest"] = "3"
                    files["compatibility-smoke-summary.json"]["results"][0]["latestVersion"] = "3"
                    self.assertEqual(main(), 0)
                    self.assertEqual(post.call_count, 2)
                    self.assertNotEqual(sent, Path("compatibility-sent-slack-marker.txt").read_text().strip())
            finally:
                os.chdir(old_cwd)

    def test_webhook_failure_never_records_finding(self):
        files = october_review()
        env = {
            "COMPAT_SLACK_WEBHOOK_URL": "https://slack.example.invalid/webhook",
            "COMPAT_JOB_STATUS": "success", "COMPAT_CHANGES_FOUND": "true",
            "COMPAT_GEMINI_STEP_STATUS": "success",
            "COMPAT_REVIEW_ISSUE_URL": "https://github.com/neatlogs/neatlogs/issues/125",
        }
        with tempfile.TemporaryDirectory() as directory:
            old_cwd = os.getcwd()
            try:
                os.chdir(directory)
                with patch.dict("os.environ", env, clear=True), \
                        patch("scripts.compatibility.notify_slack.optional_json", side_effect=files.get), \
                        patch("scripts.compatibility.notify_slack.urlopen") as post:
                    post.return_value.__enter__.return_value.status = 500
                    with self.assertRaisesRegex(RuntimeError, "Slack webhook returned 500"):
                        main()
                    self.assertFalse(Path("compatibility-sent-slack-marker.txt").exists())
            finally:
                os.chdir(old_cwd)

    def test_intervening_failure_clears_prior_fingerprint_after_post(self):
        files = october_review()
        prior = f"<!-- compatibility-slack-fingerprint:v1:{fingerprint(files)} -->"
        env = {
            "COMPAT_SLACK_WEBHOOK_URL": "https://slack.example.invalid/webhook",
            "COMPAT_JOB_STATUS": "failure", "COMPAT_CHANGES_FOUND": "true",
            "COMPAT_GEMINI_STEP_STATUS": "success",
            "COMPAT_REVIEW_ISSUE_URL": "https://github.com/neatlogs/neatlogs/issues/125",
        }
        with tempfile.TemporaryDirectory() as directory:
            old_cwd = os.getcwd()
            try:
                os.chdir(directory)
                Path("compatibility-prior-slack-marker.txt").write_text(prior)
                with patch.dict("os.environ", env, clear=True), \
                        patch("scripts.compatibility.notify_slack.optional_json", side_effect=files.get), \
                        patch("scripts.compatibility.notify_slack.urlopen") as post:
                    post.return_value.__enter__.return_value.status = 200
                    self.assertEqual(main(), 0)
                    self.assertEqual(post.call_count, 1)
                    self.assertEqual(Path("compatibility-sent-slack-marker.txt").read_text(), "")
                    # The workflow removes the old issue marker after this successful post.
                    Path("compatibility-prior-slack-marker.txt").unlink()
                    os.environ["COMPAT_JOB_STATUS"] = "success"
                    self.assertEqual(main(), 0)
                    self.assertEqual(post.call_count, 2)
            finally:
                os.chdir(old_cwd)

    def test_clean_review_clears_prior_fingerprint_without_slack_post(self):
        files = october_review()
        for result in files["compatibility-smoke-summary.json"]["results"]:
            result["baseline"]["status"] = "pass"
            result["latest"]["status"] = "pass"
            result["comparison"] = "latest-smoke-passed"
        files["compatibility-smoke-summary.json"]["counts"].update({"pass": 61, "fail": 0, "blocked": 0})
        env = {
            "COMPAT_SLACK_WEBHOOK_URL": "https://slack.example.invalid/webhook",
            "COMPAT_JOB_STATUS": "success", "COMPAT_CHANGES_FOUND": "true",
            "COMPAT_GEMINI_STEP_STATUS": "success",
            "COMPAT_REVIEW_ISSUE_URL": "https://github.com/neatlogs/neatlogs/issues/125",
        }
        with tempfile.TemporaryDirectory() as directory:
            old_cwd = os.getcwd()
            try:
                os.chdir(directory)
                with patch.dict("os.environ", env, clear=True), \
                        patch("scripts.compatibility.notify_slack.optional_json", side_effect=files.get), \
                        patch("scripts.compatibility.notify_slack.urlopen") as post:
                    self.assertEqual(main(), 0)
                    post.assert_not_called()
                    self.assertEqual(Path("compatibility-sent-slack-marker.txt").read_text(), "")
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
