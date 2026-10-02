import unittest

from scripts.compatibility.propose_fix import is_bot_draft_pr
from scripts.compatibility.publish_fix_pr import BOT_AUTHOR_EMAIL, ready_proof


class PublishFixPrTests(unittest.TestCase):
    def test_only_open_bot_drafts_can_be_reconsidered(self):
        pull = {
            "state": "OPEN", "isDraft": True,
            "author": {"login": "github-actions[bot]", "is_bot": True},
        }
        self.assertTrue(is_bot_draft_pr(pull))
        self.assertTrue(is_bot_draft_pr({**pull, "author": {"login": "app/neatlogs", "is_bot": True}}))
        self.assertFalse(is_bot_draft_pr({**pull, "state": "CLOSED"}))
        self.assertFalse(is_bot_draft_pr({**pull, "isDraft": False}))
        self.assertFalse(is_bot_draft_pr({**pull, "author": {"login": "reviewer", "is_bot": False}}))

    def test_ready_requires_exact_bot_branch_and_validated_tree(self):
        pull = {
            "state": "OPEN", "isDraft": True, "headRefOid": "commit-a",
            "author": {"login": "github-actions[bot]", "is_bot": True},
        }
        facts = {
            "fetched_sha": "commit-a", "fetched_tree": "validated-tree",
            "fetched_parent": "base-a", "fetched_author_email": BOT_AUTHOR_EMAIL,
            "fetched_subject": "fix: Python openai 3.22.1 compatibility",
            "base_sha": "base-a", "local_tree": "validated-tree",
            "expected_subject": "fix: Python openai 3.22.1 compatibility",
        }
        self.assertTrue(ready_proof(pull, **facts))
        for changed in (
            {"fetched_tree": "human-edited-tree"},
            {"fetched_parent": "different-base"},
            {"fetched_sha": "new-commit"},
            {"fetched_author_email": "human@example.com"},
            {"fetched_subject": "human edit"},
        ):
            with self.subTest(changed=changed):
                self.assertFalse(ready_proof(pull, **{**facts, **changed}))
        self.assertFalse(ready_proof({**pull, "author": {"login": "human", "is_bot": False}}, **facts))


if __name__ == "__main__":
    unittest.main()
