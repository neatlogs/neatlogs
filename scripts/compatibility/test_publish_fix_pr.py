import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.compatibility.propose_fix import is_bot_draft_pr
from scripts.compatibility.publish_fix_pr import (
    BOT_AUTHOR_EMAIL, publish, ready_proof, require_complete_integration_checks,
    require_reproduced_fix,
)


class PublishFixPrTests(unittest.TestCase):
    def test_validated_proposal_reaches_regular_pr_create_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "neatlogs" / "openai.py"
            target.parent.mkdir()
            target.write_text("original\n")
            lock_path = root / ".compatibility" / "versions.lock.json"
            lock_path.parent.mkdir()
            lock_path.write_text(json.dumps({"schemaVersion": 1, "packages": {"openai": "2.0.0"}}) + "\n")
            updated_lock = json.dumps({"schemaVersion": 1, "packages": {"openai": "3.0.0"}}, indent=2).encode() + b"\n"
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
            (root / "compatibility-evidence.json").write_text(json.dumps({"packages": [{
                "package": "openai", "previousVersion": "2.0.0", "latestVersion": "3.0.0",
                "integrations": [{"id": "openai"}],
            }]}))
            (root / "compatibility-validation-status.json").write_text(json.dumps({
                "status": "validated",
                "proposalSha256": hashlib.sha256(proposal_bytes).hexdigest(),
                "changedFiles": ["neatlogs/openai.py", ".compatibility/versions.lock.json"],
                "changedFileSha256": {
                    "neatlogs/openai.py": hashlib.sha256(b"patched\n").hexdigest(),
                    ".compatibility/versions.lock.json": hashlib.sha256(updated_lock).hexdigest(),
                },
                "validationLimit": "Focused test passed; behavior still needs human review",
                "postPatchSmoke": "pass",
                "postPatchIntegrationResults": [{"integration": "openai", "status": "pass"}],
                "basis": "upstream-and-adapter-evidence-review",
                "redGreen": "red-before-green-after",
                "reproduction": "focused-red-green",
            }))
            commands = []

            def mocked_command(*args, check=True):
                commands.append(args)
                if args[:3] == ("git", "rev-parse", "HEAD"):
                    output = "base-sha\n"
                elif args[:3] == ("git", "rev-parse", "HEAD^{tree}"):
                    output = "patched-tree\n"
                elif args[:3] == ("git", "diff", "--cached") and "--name-only" in args:
                    output = "neatlogs/openai.py\n.compatibility/versions.lock.json\n"
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
            self.assertEqual(json.loads(lock_path.read_text())["packages"], {"openai": "3.0.0"})
            create = next(args for args in commands if args[:3] == ("gh", "pr", "create"))
            self.assertNotIn("--draft", create)
            self.assertIn(("git", "push", "origin", "HEAD:refs/heads/compat/python/openai-3-0-0-openai"), commands)

    def test_publisher_requires_every_affected_integration_to_pass(self):
        candidate = {"package": "openai"}
        evidence = {"packages": [{"package": "openai", "integrations": [
            {"id": "openai"}, {"id": "azure-openai"},
        ]}]}
        validation = {"postPatchSmoke": "pass", "postPatchIntegrationResults": [
            {"integration": "openai", "status": "pass"},
            {"integration": "azure-openai", "status": "pass"},
        ]}
        require_complete_integration_checks(candidate, evidence, validation)
        for bad in (
            {**validation, "postPatchIntegrationResults": validation["postPatchIntegrationResults"][:1]},
            {**validation, "postPatchIntegrationResults": [validation["postPatchIntegrationResults"][0],
                                                        {"integration": "azure-openai", "status": "blocked"}]},
        ):
            with self.assertRaisesRegex(ValueError, "Not every affected integration"):
                require_complete_integration_checks(candidate, evidence, bad)

    def test_publisher_rejects_unreproduced_advisory_fix(self):
        advisory = {"basis": "upstream-and-adapter-evidence-review"}
        with self.assertRaisesRegex(ValueError, "no reproducible"):
            require_reproduced_fix(advisory, {"basis": advisory["basis"],
                                             "redGreen": "not-proven", "reproduction": "not-proven"})
        require_reproduced_fix(advisory, {"basis": advisory["basis"],
                                          "redGreen": "red-before-green-after",
                                          "reproduction": "focused-red-green"})
        smoke = {"basis": "activation-smoke-regression", "package": "openai",
                 "integration": "openai", "latestVersion": "3", "smoke": {
                     "package": "openai", "integration": "openai", "latestVersion": "3",
                     "comparison": "smoke-regression", "baseline": {"status": "pass"},
                     "latest": {"status": "fail"},
                 }}
        require_reproduced_fix(smoke, {"basis": smoke["basis"], "redGreen": "not-proven",
                                        "reproduction": "activation-smoke-baseline-pass-latest-fail-patched-pass"})

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
