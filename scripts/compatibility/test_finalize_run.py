import unittest

from scripts.compatibility.finalize_run import failure_reason


class FinalizeRunTests(unittest.TestCase):
    def setUp(self):
        self.summary = {"smokeRegressions": 0}
        self.jobs = {
            "discover": "success", "analyze": "success", "smoke": "success",
            "report": "success", "propose": "success", "validate": "skipped",
            "publish": "skipped", "issue_update": "success",
        }

    def test_safe_rejections_complete_without_a_failed_run(self):
        self.assertIsNone(failure_reason(
            self.summary, {"status": "rejected", "kind": "unsafe-proposal"},
            None, None, self.jobs,
        ))
        self.assertIsNone(failure_reason(
            self.summary, {"status": "proposed"},
            {"status": "rejected", "kind": "candidate-rejected"}, None,
            {**self.jobs, "validate": "success"},
        ))
        self.assertIn("Unexpected publication", failure_reason(
            self.summary, {"status": "proposed"},
            {"status": "rejected", "kind": "candidate-rejected"}, None,
            {**self.jobs, "validate": "success", "publish": "success"},
        ))

    def test_core_request_failure_or_missing_status_fails_the_run(self):
        for proposal in (
            {"status": "unavailable"},
            {"status": "rejected", "kind": "request-failure"},
            None,
        ):
            with self.subTest(proposal=proposal):
                self.assertIsNotNone(failure_reason(
                    self.summary, proposal, None, None, self.jobs,
                ))

    def test_regression_tool_failure_or_unpublished_validated_fix_fails(self):
        self.assertIn("regression", failure_reason(
            {"smokeRegressions": 1}, {"status": "no-safe-fix"}, None, None,
            self.jobs,
        ))
        self.assertIn("smoke job", failure_reason(
            self.summary, {"status": "no-safe-fix"}, None, None,
            {**self.jobs, "smoke": "failure"},
        ))
        self.assertIn("issue update", failure_reason(
            self.summary, {"status": "no-safe-fix"}, None, None,
            {**self.jobs, "issue_update": "failure"},
        ))
        self.assertIn("publication", failure_reason(
            self.summary, {"status": "proposed"}, {"status": "validated"},
            None, {**self.jobs, "validate": "success"},
        ))
        self.assertIsNone(failure_reason(
            self.summary, {"status": "proposed"}, {"status": "validated"},
            {"status": "created"},
            {**self.jobs, "validate": "success", "publish": "success"},
        ))


if __name__ == "__main__":
    unittest.main()
