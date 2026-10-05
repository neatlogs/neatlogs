import copy
import json
import unittest
from unittest.mock import patch

from scripts.compatibility.notify_slack import main, short_reason, slack_message, slack_payload, workflow_url


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
        files = self.live_review_files()
        files["compatibility-fix-status.json"]["noSdkPatchSurface"] = [
            {"package": "google-genai", "integration": "unmapped", "latestVersion": "2.27.0"},
        ]
        self.assertEqual(self.webhook_posts(files), [])

    def test_manual_upstream_issue_is_evidence_not_an_alert_by_itself(self):
        files = self.live_review_files()
        files["compatibility-upstream-issue.json"] = {
            "title": "Potential upstream behavior change",
            "url": "https://github.com/example/upstream/issues/123",
        }
        self.assertEqual(self.webhook_posts(files), [])
        files["compatibility-smoke-summary.json"]["results"][0]["latest"]["status"] = "fail"
        files["compatibility-smoke-summary.json"]["results"][0]["comparison"] = "smoke-regression"
        files["compatibility-smoke-summary.json"]["smokeRegressions"] = 1
        files["compatibility-smoke-summary.json"]["counts"] = {
            "pass": 1, "fail": 1, "blocked": 0, "not-tested": 0,
        }
        messages = self.webhook_posts(files)
        self.assertEqual(len(messages), 1)
        self.assertIn("Upstream issue", messages[0])

    def test_actionable_alert_without_slack_webhook_fails_the_job(self):
        files = self.live_review_files()
        with patch.dict("os.environ", {
            "COMPAT_JOB_STATUS": "failure",
            "COMPAT_CHANGES_FOUND": "true",
            "COMPAT_GEMINI_STEP_STATUS": "success",
        }, clear=True), patch(
            "scripts.compatibility.notify_slack.optional_json", side_effect=files.get,
        ), self.assertRaisesRegex(RuntimeError, "could not be delivered"):
            main()

    def test_actionable_alert_explains_unmapped_patch_surface(self):
        files = self.live_review_files()
        files["compatibility-fix-status.json"]["noSdkPatchSurface"] = [
            {"package": "groq", "integration": "groq", "latestVersion": "2"},
        ]
        messages = self.webhook_posts(files, status="failure")
        self.assertEqual(len(messages), 1)
        self.assertIn("No integration-specific SDK patch source for 1 pair(s) (e.g. groq/groq)", messages[0])

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
        missing = self.live_review_files()
        missing.pop("compatibility-smoke-summary.json")
        cases["missing smoke evidence"] = (missing, "success", "success")
        cases["Gemini failure"] = (self.live_review_files(), "success", "failure")
        missing_advisory = self.live_review_files()
        missing_advisory.pop("compatibility-llm-analysis.json")
        cases["missing Gemini evidence"] = (missing_advisory, "success", "success")
        malformed = self.live_review_files()
        malformed["compatibility-llm-analysis.json"] = []
        cases["malformed Gemini evidence"] = (malformed, "success", "success")
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

    def test_baseline_failures_and_blocked_installs_never_alert_without_regression(self):
        files = self.live_review_files()
        smoke = files["compatibility-smoke-summary.json"]
        smoke["results"][0]["baseline"]["status"] = "fail"
        smoke["results"][0]["latest"]["status"] = "fail"
        smoke["results"][0]["comparison"] = "latest-failure-needs-triage"
        smoke["results"][1]["latest"]["status"] = "blocked"
        smoke["results"][1]["comparison"] = "latest-install-blocked"
        smoke["counts"] = {"pass": 0, "fail": 1, "blocked": 1, "not-tested": 0}
        self.assertEqual(self.webhook_posts(files), [])
        files["compatibility-release-report.json"]["changes"][0]["latest"] = "2.28.0"
        for result in smoke["results"]:
            result["latestVersion"] = "2.28.0"
        self.assertEqual(self.webhook_posts(files), [], "New release alone is not an alert")
        smoke["results"][0]["baseline"]["status"] = "pass"
        smoke["results"][0]["comparison"] = "smoke-regression"
        smoke["smokeRegressions"] = 1
        self.assertEqual(len(self.webhook_posts(files)), 1)

    def test_inconsistent_or_missing_comparison_evidence_alerts(self):
        files = self.live_review_files()
        smoke = files["compatibility-smoke-summary.json"]
        smoke["results"][0]["latest"]["status"] = "fail"
        smoke["counts"] = {"pass": 1, "fail": 1, "blocked": 0, "not-tested": 0}
        self.assertEqual(len(self.webhook_posts(files)), 1)
        smoke["results"][0]["comparison"] = "smoke-regression"
        smoke["smokeRegressions"] = 1
        self.assertEqual(len(self.webhook_posts(files)), 1)

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
        self.assertIn("1 newer release", message)
        self.assertIn("openai 1 → 2", message)
        self.assertNotIn("high potential risk", message)
        self.assertIn("Upstream issue", message)
        self.assertIn("github.com/example/sdk/issues/2", message)
        self.assertIn("https://example.test/run", message)
        self.assertIn("*Regression:* status unknown", message)

    def test_failure_message_does_not_require_report(self):
        self.assertIn("compatibility automation failed", slack_message("failure", None, None, None))

    def test_missing_analysis_does_not_look_like_a_release_review(self):
        files = self.live_review_files()
        message = slack_message(
            "success", files["compatibility-release-report.json"], None,
            "https://example.test/run", gemini_status="success",
            smoke_summary=files["compatibility-smoke-summary.json"],
            proposal_status=files["compatibility-fix-status.json"],
        )
        self.assertIn("compatibility automation failed", message)
        self.assertIn("Inspect the failed workflow step", message)

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
        self.assertIn("Upstream issue", message)
        self.assertIn("github.com/example/sdk/issues/2", message)

    def test_gemini_failure_has_reason_and_review_issue(self):
        message = slack_message(
            "failure", None, {"failed": True, "error": "Gemini HTTP 400: input too long"},
            "https://github.com/neatlogs/neatlogs/actions/runs/123", None,
            "https://github.com/neatlogs/neatlogs/issues/42", "failure",
        )
        self.assertIn("Gemini analysis failed", message)
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
        self.assertIn("activation regression candidate", message)
        self.assertIn("1 baseline-pass/latest-fail", message)
        self.assertNotIn("Gemini advisory", message)
        self.assertIn("tested activation scope", slack_message(
            "success", None, None, None,
            smoke_summary={"counts": {"pass": 1, "fail": 0, "blocked": 0, "not-tested": 0}, "smokeRegressions": 0},
        ))

    def test_inconclusive_failures_and_blocked_installs_name_the_assessed_candidate(self):
        message = slack_message(
            "success", {"changes": [{"package": "google-genai", "previouslyAnalyzed": "2.23.0",
                                      "latest": "2.27.0"}]},
            {"riskLevel": "high"}, "https://example.test/run",
            smoke_summary={
                "counts": {"pass": 0, "fail": 1, "blocked": 1, "not-tested": 0},
                "smokeRegressions": 0,
                "results": [
                    {"latest": {"status": "fail"}, "baseline": {"status": "fail"}},
                    {"latest": {"status": "blocked"}, "baseline": {"status": "pass"}},
                ],
            },
            proposal_status={"status": "rejected", "reason": "Regression test has no test function",
                             "selectedCandidate": {"package": "google-genai", "integration": "google-genai",
                                                   "latestVersion": "2.27.0",
                                                   "basis": "upstream-and-adapter-evidence-review"}},
        )
        self.assertIn("fix attempt rejected", message)
        self.assertIn("1 fail (1 also failed at baseline) · 1 blocked", message)
        self.assertIn("none found in the tested activation scope", message)
        self.assertIn("proposal rejected: Regression test has no test function", message)
        self.assertNotIn("high potential risk", message)

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

    def test_review_pr_is_linked_with_reproduction_scope(self):
        message = slack_message(
            "success", {"changes": [{"package": "openai", "previouslyAnalyzed": "1", "latest": "2"}]},
            {"riskLevel": "high"}, "https://example.test/run",
            smoke_summary={"counts": {"pass": 1, "fail": 0, "blocked": 0, "not-tested": 0}, "smokeRegressions": 0},
            proposal_status={"status": "proposed", "deferredCandidates": [{"package": "other"}]},
            validation_status={"status": "validated", "redGreen": "red-before-green-after",
                               "reproduction": "focused-red-green"},
            publish_status={"status": "created", "prUrl": "https://github.com/neatlogs/neatlogs/pull/200"},
        )
        self.assertIn("validated fix PR ready", message)
        self.assertIn("/pull/200", message)
        self.assertNotIn("Gemini advisory", message)
        self.assertIn("Focused red/green test passed", message)
        self.assertNotIn("deferred", message)

    def test_validation_failure_explains_no_pr_and_keeps_links(self):
        message = slack_message(
            "failure", None, {"riskLevel": "high"},
            "https://github.com/neatlogs/neatlogs/actions/runs/123",
            review_issue_url="https://github.com/neatlogs/neatlogs/issues/42",
            proposal_status={"status": "proposed"},
            validation_status={"status": "failed", "reason": "focused test failed"},
        )
        self.assertIn("proposed fix failed validation", message)
        self.assertIn("focused test failed", message)
        self.assertIn("not opened", message)
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
        self.assertIn("activation regression candidate", message)
        self.assertIn("baseline-pass/latest-fail", message)
        self.assertNotIn("Gemini advisory", message)
        self.assertIn("publication failed", message)
        self.assertIn("GitHub denied PR creation", message)

    def test_gemini_proposal_without_validation_does_not_claim_a_pr(self):
        message = slack_message(
            "success", None, {"riskLevel": "high"}, "https://example.test/run",
            proposal_status={"status": "proposed"},
        )
        self.assertIn("proposal exists, but publication was not confirmed", message)
        self.assertNotIn("Gemini advisory", message)

    def test_reused_draft_or_closed_pr_is_described_accurately(self):
        draft_url = "https://github.com/neatlogs/neatlogs/pull/201"
        draft = slack_message(
            "success", None, None, "https://example.test/run",
            publish_status={"status": "already-covered", "prUrl": draft_url,
                            "state": "OPEN", "isDraft": True},
        )
        self.assertIn("open, draft", draft)
        self.assertIn("existing PR", draft)
        self.assertIn(draft_url, draft)
        closed = slack_message(
            "success", None, None, "https://example.test/run",
            publish_status={"status": "already-covered", "prUrl": draft_url,
                            "state": "CLOSED", "isDraft": False},
        )
        self.assertIn("existing PR> (closed); no new PR opened", closed)
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
        self.assertIn("Existing draft fix PR> remains draft", message)
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
        self.assertIn("validated fix PR ready", message)
        self.assertIn("open for human review", message)
        self.assertNotIn("remain draft", message)

    def test_bot_draft_ready_failure_retains_link_and_reason(self):
        message = slack_message(
            "failure", None, None, "https://example.test/run",
            publish_status={"status": "failed", "prUrl": "https://github.com/neatlogs/neatlogs/pull/204",
                            "state": "OPEN", "isDraft": True, "reason": "GitHub denied readiness"},
        )
        self.assertIn("could not be marked ready", message)
        self.assertIn("GitHub denied readiness", message)
        self.assertIn("existing draft", message)
        self.assertIn("/pull/204", message)

    def test_october_3_alert_puts_outcome_and_no_pr_reason_first(self):
        changes = [
            {"package": "agno", "previouslyAnalyzed": "3.0.9", "latest": "3.1.1"},
            {"package": "anthropic", "previouslyAnalyzed": "1.6.0", "latest": "1.11.0"},
        ] + [
            {"package": f"dependency-{index}", "previouslyAnalyzed": "1", "latest": "2"}
            for index in range(49)
        ]
        results = [
            {"baseline": {"status": "fail"}, "latest": {"status": "fail"}}
            for _ in range(6)
        ] + [
            {"baseline": {"status": "blocked"}, "latest": {"status": "blocked"}}
            for _ in range(5)
        ]
        message = slack_message(
            "success", {"changes": changes}, {"riskLevel": "high"},
            "https://github.com/neatlogs/neatlogs/actions/runs/37085105195",
            review_issue_url="https://github.com/neatlogs/neatlogs/issues/125",
            smoke_summary={
                "counts": {"pass": 50, "fail": 6, "blocked": 5, "not-tested": 0},
                "smokeRegressions": 0, "results": results,
            },
            proposal_status={"status": "no-safe-fix", "reason": "No actionable SDK patch",
                             "deferredCandidates": [{"package": "other"}] * 37},
        )
        lines = message.splitlines()
        self.assertTrue(any("*Checks:* 50 pass · 6 fail (6 also failed at baseline) · 5 blocked" in line for line in lines))
        self.assertIn("*Regression:* none found in the tested activation scope.", lines)
        self.assertIn("*Upstream:* 51 newer releases (e.g. agno 3.0.9 → 3.1.1, anthropic 1.6.0 → 1.11.0).", lines)
        self.assertTrue(any("not opened — no safe code-specific fix" in line for line in lines))
        self.assertTrue(any("Triage preexisting failures" in line for line in lines))
        self.assertTrue(any("issues/125" in line for line in lines))
        self.assertTrue(any("actions/runs/37085105195" in line for line in lines))
        self.assertNotIn("high potential risk", message)
        self.assertNotIn("37 candidate", message)

    def test_actionable_alert_posts_block_kit_with_text_fallback(self):
        files = self.live_review_files()
        smoke = files["compatibility-smoke-summary.json"]
        smoke["results"][0]["latest"]["status"] = "fail"
        smoke["results"][0]["comparison"] = "smoke-regression"
        smoke["counts"] = {"pass": 1, "fail": 1, "blocked": 0, "not-tested": 0}
        smoke["smokeRegressions"] = 1
        env = {
            "COMPAT_SLACK_WEBHOOK_URL": "https://slack.example.invalid/webhook",
            "COMPAT_JOB_STATUS": "success", "COMPAT_CHANGES_FOUND": "true",
            "COMPAT_GEMINI_STEP_STATUS": "success",
            "COMPAT_REVIEW_ISSUE_URL": "https://github.com/neatlogs/neatlogs/issues/125",
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_REPOSITORY": "neatlogs/neatlogs", "GITHUB_RUN_ID": "123",
        }
        with patch.dict("os.environ", env, clear=True), \
                patch("scripts.compatibility.notify_slack.optional_json", side_effect=files.get), \
                patch("scripts.compatibility.notify_slack.urlopen") as post:
            post.return_value.__enter__.return_value.status = 200
            self.assertEqual(main(), 0)
            payload = json.loads(post.call_args.args[0].data)
        self.assertEqual(payload["blocks"][2]["type"], "divider")
        sections = [block["text"]["text"] for block in payload["blocks"] if block["type"] == "section"]
        self.assertEqual(len(sections), 4)
        self.assertIn("regression candidate", sections[0])
        self.assertIn("*Checks:*", sections[1])
        self.assertIn("*Regression:*", sections[1])
        self.assertIn("*Fix PR:*", sections[2])
        self.assertIn("*Next step:*", sections[3])
        self.assertIn("actions/runs/123", sections[3])
        self.assertIn("\n\n", payload["text"])
        self.assertEqual(slack_payload(payload["text"]), payload)


if __name__ == "__main__":
    unittest.main()
