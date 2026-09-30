import unittest
from unittest.mock import patch

from scripts.compatibility.notify_slack import slack_message, workflow_url


class NotifySlackTests(unittest.TestCase):
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

    def test_draft_pr_is_linked_with_unverified_scope(self):
        message = slack_message(
            "success", {"changes": [{"package": "openai", "previouslyAnalyzed": "1", "latest": "2"}]},
            {"riskLevel": "high"}, "https://example.test/run",
            smoke_summary={"counts": {"pass": 1, "fail": 0, "blocked": 0, "not-tested": 0}, "smokeRegressions": 0},
            proposal_status={"status": "proposed", "deferredCandidates": [{"package": "other"}]},
            validation_status={"status": "validated"},
            publish_status={"status": "created", "prUrl": "https://github.com/neatlogs/neatlogs/pull/200"},
        )
        self.assertIn("Draft fix PR", message)
        self.assertIn("/pull/200", message)
        self.assertIn("Gemini advisory: high potential risk (unverified)", message)
        self.assertIn("behavior fix unverified", message)
        self.assertIn("1 candidate(s) deferred", message)


if __name__ == "__main__":
    unittest.main()
