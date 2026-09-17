import unittest

from scripts.compatibility.notify_slack import slack_message


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

    def test_failure_message_does_not_require_report(self):
        self.assertIn("workflow failed", slack_message("failure", None, None, None))


if __name__ == "__main__":
    unittest.main()
