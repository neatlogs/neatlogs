import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.compatibility.propose_fix import is_bot_draft_pr
from scripts.compatibility.publish_fix_pr import BOT_AUTHOR_EMAIL, publish, ready_proof


class PublishFixPrTests(unittest.TestCase):
    def test_validated_proposal_reaches_regular_pr_create_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "neatlogs" / "openai.py"
            target.parent.mkdir()
            target.write_text("original\n")
            proposal = {
                "baseSha": "base-sha",
                "candidate": {"package": "openai", "latestVersion": "3.0.0",
                              "integration": "openai", "basis": "upstream-and-adapter-evidence-review"},
                "edits": [{"path": "neatlogs/openai.py", "oldText": "original", "newText": "patched"}],
                "evidence": "Upstream API signature changed",
                "rationale": "Update adapter call",
            }
            proposal_bytes = json.dumps(proposal).encode()
            (root / "compatibility-fix-proposal.json").write_bytes(proposal_bytes)
            (root / "compatibility-evidence.json").write_text(json.dumps({"packages": []}))
            (root / "compatibility-validation-status.json").write_text(json.dumps({
                "status": "validated",
                "proposalSha256": hashlib.sha256(proposal_bytes).hexdigest(),
                "changedFiles": ["neatlogs/openai.py"],
                "changedFileSha256": {"neatlogs/openai.py": hashlib.sha256(b"patched\n").hexdigest()},
                "validationLimit": "Focused test passed; behavior still needs human review",
                "postPatchSmoke": "pass",
            }))
            commands = []

            def mocked_command(*args, check=True):
                commands.append(args)
                if args[:3] == ("git", "rev-parse", "HEAD"):
                    output = "base-sha\n"
                elif args[:3] == ("git", "rev-parse", "HEAD^{tree}"):
                    output = "patched-tree\n"
                elif args[:3] == ("git", "diff", "--cached") and "--name-only" in args:
                    output = "neatlogs/openai.py\n"
                elif args[:3] == ("git", "ls-remote", "--heads"):
                    output = ""
                elif args[:3] == ("gh", "pr", "list"):
                    output = "[]"
                elif args[:3] == ("gh", "repo", "view"):
                    output = "main\n"
                elif args[:3] == ("gh", "pr", "create"):
                    output = "https://github.com/neatlogs/neatlogs/pull/999\n"
                else:
                    output = ""
                return subprocess.CompletedProcess(args, 0, output, "")

            def mocked_apply(*args):
                target.write_text("patched\n")
                return ["neatlogs/openai.py"]

            previous_directory = Path.cwd()
            try:
                os.chdir(root)
                with patch("scripts.compatibility.publish_fix_pr.ROOT", root), \
                     patch("scripts.compatibility.publish_fix_pr.command", side_effect=mocked_command), \
                     patch("scripts.compatibility.publish_fix_pr.apply_proposal", side_effect=mocked_apply), \
                     patch.dict(os.environ, {"GITHUB_REPOSITORY": "neatlogs/neatlogs"}):
                    result = publish()
            finally:
                os.chdir(previous_directory)
            self.assertEqual(result["status"], "created")
            create = next(args for args in commands if args[:3] == ("gh", "pr", "create"))
            self.assertNotIn("--draft", create)
            self.assertIn(("git", "push", "origin", "HEAD:refs/heads/compat/python/openai-3-0-0-openai"), commands)

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
