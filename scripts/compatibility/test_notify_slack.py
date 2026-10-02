import copy
import json
import unittest
from unittest.mock import patch

from scripts.compatibility.notify_slack import main, short_reason, slack_message, workflow_url


class NotifySlackTests(unittest.TestCase):
    @staticmethod
    def live_review_files():
        report = {"changes": [{
            "package": "google-genai", "previouslyAnalyzed": "2.23.0",
            "latest": "2.27.0", "integrations": ["google-genai", "vertex-google-genai"],
        }]}
        results = [{
            "package": "google-genai", "integration": integration,
            "baselineVersion": "2.23.0", "latestVersion": "2.27.0",
            "baseline": {"status": "pass"}, "latest": {"status": "pass"},
            "comparison": "latest-smoke-passed",
        } for integration in report["changes"][0]["integrations"]]
        return {
            "compatibility-release-report.json": report,
            "compatibility-llm-analysis.json": {"riskLevel": "medium"},
            "compatibility-smoke-summary.json": {
                "pairCount": 2, "counts": {"pass": 2, "fail": 0, "blocked": 0, "not-tested": 0},
                "smokeRegressions": 0, "results": results,
            },
            "compatibility-fix-status.json": {"status": "no-safe-fix", "reason": "No concrete safe fix"},
        }

    def webhook_posts(self, files, *, status="success", gemini_status="success"):
        env = {
            "COMPAT_SLACK_WEBHOOK_URL": "https://slack.example.invalid/webhook",
            "COMPAT_JOB_STATUS": status,
            "COMPAT_CHANGES_FOUND": "true",
            "COMPAT_GEMINI_STEP_STATUS": gemini_status,
        }
        with patch.dict("os.environ", env, clear=True), \
                patch("scripts.compatibility.notify_slack.optional_json", side_effect=files.get), \
                patch("scripts.compatibility.notify_slack.urlopen") as post:
            post.return_value.__enter__.return_value.status = 200
            self.assertEqual(main(), 0)
            if post.called:
                return [json.loads(call.args[0].data)["text"] for call in post.call_args_list]
            return []

    def test_live_passing_review_without_safe_fix_sends_no_slack_post(self):
        self.assertEqual(self.webhook_posts(self.live_review_files()), [])

    def test_actionable_or_incomplete_reviews_still_send_slack(self):
        cases = {}
        cases["workflow failure"] = (self.live_review_files(), "failure", "success")
        regression = self.live_review_files()
        regression["compatibility-smoke-summary.json"]["results"][0]["latest"]["status"] = "fail"
        regression["compatibility-smoke-summary.json"]["results"][0]["comparison"] = "smoke-regression"
        regression["compatibility-smoke-summary.json"]["counts"] = {
            "pass": 1, "fail": 1, "blocked": 0, "not-tested": 0,
        }
        regression["compatibility-smoke-summary.json"]["smokeRegressions"] = 1
        cases["activation regression"] = (regression, "success", "success")
        blocked = self.live_review_files()
        blocked["compatibility-smoke-summary.json"]["results"][0]["latest"]["status"] = "blocked"
        blocked["compatibility-smoke-summary.json"]["counts"] = {
            "pass": 1, "fail": 0, "blocked": 1, "not-tested": 0,
        }
        cases["blocked install"] = (blocked, "success", "success")
        missing = self.live_review_files()
        missing.pop("compatibility-smoke-summary.json")
        cases["missing smoke evidence"] = (missing, "success", "success")
        cases["Gemini failure"] = (self.live_review_files(), "success", "failure")
        missing_advisory = self.live_review_files()
        missing_advisory.pop("compatibility-llm-analysis.json")
        cases["missing Gemini evidence"] = (missing_advisory, "success", "success")
        rejected = self.live_review_files()
        rejected["compatibility-fix-status.json"]["status"] = "rejected"
        cases["proposal failure"] = (rejected, "success", "success")
        validation = self.live_review_files()
        validation["compatibility-fix-status.json"]["status"] = "proposed"
        validation["compatibility-validation-status.json"] = {"status": "failed", "reason": "test failed"}
        cases["validation failure"] = (validation, "success", "success")
        publisher = copy.deepcopy(validation)
        publisher["compatibility-validation-status.json"] = {"status": "validated"}
        publisher["compatibility-publish-status.json"] = {"status": "failed", "reason": "GitHub denied PR"}
        cases["publisher failure"] = (publisher, "success", "success")
        opened = copy.deepcopy(publisher)
        opened["compatibility-publish-status.json"] = {
            "status": "created", "prUrl": "https://github.com/neatlogs/neatlogs/pull/200",
        }
        cases["fix PR opened"] = (opened, "success", "success")
        for name, (files, status, gemini_status) in cases.items():
            with self.subTest(name=name):
                self.assertEqual(len(self.webhook_posts(files, status=status, gemini_status=gemini_status)), 1)

    def test_long_reason_is_shortened_at_a_word_boundary(self):
        self.assertEqual(short_reason("alpha beta gamma", 10), "alpha…")
        self.assertEqual(short_reason("clear reason.", 20), "clear reason.")

    def test_release_message_contains_evidence_summary(self):
        message = slack_message(
            "success",
            {
                "changes": [
                    {
                        "package": "openai",
                        "previouslyAnalyzed": "1",
                        "latest": "2",
                    }
                ]
            },
            {"riskLevel": "high"},
            "https://example.test/run",
            {
                "title": "multi-hop usage undercounts tokens",
                "url": "https://github.com/example/sdk/issues/2",
            },
        )
        self.assertIn("1 upstream release", message)
        self.assertIn("openai 1 → 2", message)
        self.assertIn("high", message)
        self.assertIn("multi-hop usage undercounts tokens", message)
        self.assertIn("github.com/example/sdk/issues/2", message)
        self.assertIn("https://example.test/run", message)
        self.assertIn("unverified", message)
        self.assertIn("Latest-version smoke results unavailable", message)

    def test_failure_message_does_not_require_report(self):
        self.assertIn("workflow failed", slack_message("failure", None, None, None))

    def test_failure_message_keeps_upstream_issue_context(self):
        message = slack_message(
            "failure",
            None,
            None,
            "https://example.test/run",
            {
                "title": "multi-hop usage undercounts tokens",
                "url": "https://github.com/example/sdk/issues/2",
            },
        )
        self.assertIn("multi-hop usage undercounts tokens", message)
        self.assertIn("github.com/example/sdk/issues/2", message)

    def test_gemini_failure_has_reason_and_review_issue(self):
        message = slack_message(
            "failure", None, {"failed": True, "error": "Gemini HTTP 400: input too long"},
            "https://github.com/neatlogs/neatlogs/actions/runs/123", None,
            "https://github.com/neatlogs/neatlogs/issues/42", "failure",
        )
        self.assertIn("Gemini advisory failed", message)
        self.assertIn("HTTP 400", message)
        self.assertIn("issues/42", message)
        self.assertIn("actions/runs/123", message)

    def test_verified_smoke_regression_is_separate_from_gemini(self):
        message = slack_message(
            "success", {"changes": [{"package": "openai", "previouslyAnalyzed": "1", "latest": "2"}]},
            {"riskLevel": "low"}, "https://example.test/run", smoke_summary={
                "counts": {"pass": 0, "fail": 1, "blocked": 0, "not-tested": 0},
                "smokeRegressions": 1,
            },
        )
        self.assertIn("activation smoke regression detected", message)
        self.assertIn("1 baseline-passing", message)
        self.assertIn("Gemini advisory: low potential risk (unverified)", message)
        self.assertIn("Checks cover install, dependencies, and instrumentation activation only", message)

    def test_workflow_url_uses_actions_runs_path(self):
        with patch.dict("os.environ", {
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_REPOSITORY": "neatlogs/neatlogs",
            "GITHUB_RUN_ID": "36571240360",
        }):
            self.assertEqual(
                workflow_url(),
                "https://github.com/neatlogs/neatlogs/actions/runs/36571240360",
            )

    def test_review_pr_is_linked_with_unverified_scope(self):
        message = slack_message(
            "success", {"changes": [{"package": "openai", "previouslyAnalyzed": "1", "latest": "2"}]},
            {"riskLevel": "high"}, "https://example.test/run",
            smoke_summary={"counts": {"pass": 1, "fail": 0, "blocked": 0, "not-tested": 0}, "smokeRegressions": 0},
            proposal_status={"status": "proposed", "deferredCandidates": [{"package": "other"}]},
            validation_status={"status": "validated"},
            publish_status={"status": "created", "prUrl": "https://github.com/neatlogs/neatlogs/pull/200"},
        )
        self.assertIn("Fix PR for review", message)
        self.assertIn("/pull/200", message)
        self.assertIn("Gemini advisory: high potential risk (unverified)", message)
        self.assertIn("behavior fix unverified", message)
        self.assertIn("1 candidate(s) deferred", message)

    def test_validation_failure_explains_no_pr_and_keeps_links(self):
        message = slack_message(
            "failure", None, {"riskLevel": "high"},
            "https://github.com/neatlogs/neatlogs/actions/runs/123",
            review_issue_url="https://github.com/neatlogs/neatlogs/issues/42",
            proposal_status={"status": "proposed"},
            validation_status={"status": "failed", "reason": "focused test failed"},
        )
        self.assertIn("Gemini proposed a fix, but validation failed", message)
        self.assertIn("focused test failed", message)
        self.assertIn("No PR opened", message)
        self.assertIn("issues/42", message)
        self.assertIn("actions/runs/123", message)

    def test_pr_creation_failure_is_distinct_from_a_test_regression(self):
        message = slack_message(
            "failure", None, {"riskLevel": "low"}, "https://example.test/run",
            smoke_summary={
                "counts": {"pass": 0, "fail": 1, "blocked": 0, "not-tested": 0},
                "smokeRegressions": 1,
                "results": [{"package": "openai", "integration": "openai", "comparison": "smoke-regression"}],
            },
            proposal_status={"status": "proposed"},
            validation_status={"status": "validated", "redGreen": "red-before-green-after"},
            publish_status={"status": "failed", "reason": "GitHub denied PR creation"},
        )
        self.assertIn("activation smoke regression detected", message)
        self.assertIn("openai/openai", message)
        self.assertIn("Gemini advisory: low potential risk (unverified)", message)
        self.assertIn("Fix PR could not be opened", message)
        self.assertIn("GitHub denied PR creation", message)

    def test_gemini_proposal_without_validation_does_not_claim_a_pr(self):
        message = slack_message(
            "success", None, {"riskLevel": "high"}, "https://example.test/run",
            proposal_status={"status": "proposed"},
        )
        self.assertIn("Gemini proposed a fix; validation result unavailable", message)
        self.assertIn("No PR confirmed", message)
        self.assertIn("Gemini advisory: high potential risk (unverified)", message)

    def test_reused_draft_or_closed_pr_is_described_accurately(self):
        draft_url = "https://github.com/neatlogs/neatlogs/pull/201"
        draft = slack_message(
            "success", None, None, "https://example.test/run",
            publish_status={"status": "already-covered", "prUrl": draft_url,
                            "state": "OPEN", "isDraft": True},
        )
        self.assertIn("remains draft", draft)
        self.assertIn("Existing draft fix PR", draft)
        self.assertIn(draft_url, draft)
        closed = slack_message(
            "success", None, None, "https://example.test/run",
            publish_status={"status": "already-covered", "prUrl": draft_url,
                            "state": "CLOSED", "isDraft": False},
        )
        self.assertIn("Previous fix PR is closed; no new PR opened", closed)
        self.assertNotIn("human code review required", closed)

    def test_covered_draft_is_visible_when_candidate_is_skipped(self):
        message = slack_message(
            "success", None, None, "https://example.test/run",
            proposal_status={"status": "no-safe-fix", "reason": "Already covered",
                             "alreadyCoveredPullRequests": [{
                                 "url": "https://github.com/neatlogs/neatlogs/pull/202",
                                 "state": "OPEN", "isDraft": True,
                             }]},
        )
        self.assertIn("matching fix PR(s) remain draft", message)
        self.assertIn("/pull/202", message)

    def test_matching_bot_draft_marked_ready_is_not_reported_as_still_draft(self):
        url = "https://github.com/neatlogs/neatlogs/pull/203"
        message = slack_message(
            "success", None, None, "https://example.test/run",
            proposal_status={"status": "proposed", "alreadyCoveredPullRequests": [{
                "url": url, "state": "OPEN", "isDraft": True,
            }]},
            validation_status={"status": "validated"},
            publish_status={"status": "ready", "prUrl": url, "state": "OPEN", "isDraft": False},
        )
        self.assertIn("marked ready for human review", message)
        self.assertIn("Fix PR for review", message)
        self.assertNotIn("remain draft", message)

    def test_bot_draft_ready_failure_retains_link_and_reason(self):
        message = slack_message(
            "failure", None, None, "https://example.test/run",
            publish_status={"status": "failed", "prUrl": "https://github.com/neatlogs/neatlogs/pull/204",
                            "state": "OPEN", "isDraft": True, "reason": "GitHub denied readiness"},
        )
        self.assertIn("could not be marked ready", message)
        self.assertIn("GitHub denied readiness", message)
        self.assertIn("Existing draft fix PR", message)
        self.assertIn("/pull/204", message)


if __name__ == "__main__":
    unittest.main()
