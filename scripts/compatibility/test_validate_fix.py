import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.compatibility.publish_fix_pr import require_validated_content
from scripts.compatibility.validate_fix import (
    changed_paths,
    check_affected_integrations,
    require_post_patch_smoke,
    require_reproduction,
    workspace_snapshot,
)


class ValidateFixTests(unittest.TestCase):
    def test_every_affected_integration_must_pass_before_lock_advances(self):
        candidate = {"package": "openai", "latestVersion": "3", "integration": "openai"}
        evidence = {"packages": [{"package": "openai", "integrations": [
            {"id": "openai"}, {"id": "azure-openai"},
        ]}]}
        config = {"integrations": [
            {"id": "openai", "packages": ["openai"], "extra": "openai"},
            {"id": "azure-openai", "packages": ["openai"], "extra": "azure-openai"},
        ]}
        with patch("scripts.compatibility.validate_fix.check_version",
                   side_effect=[{"status": "pass"}, {"status": "pass"}]) as check:
            results = check_affected_integrations(candidate, evidence, config, Path("wheel.whl"))
        self.assertEqual([item["integration"] for item in results], ["openai", "azure-openai"])
        self.assertEqual(check.call_count, 2)
        with patch("scripts.compatibility.validate_fix.check_version",
                   side_effect=[{"status": "pass"}, {"status": "blocked", "reason": "constraint"}]):
            with self.assertRaisesRegex(RuntimeError, "azure-openai.*blocked"):
                check_affected_integrations(candidate, evidence, config, Path("wheel.whl"))

    def test_only_passing_post_patch_smoke_can_validate(self):
        require_post_patch_smoke({"status": "pass"})
        for status in ("blocked", "fail", "not-tested"):
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, status):
                require_post_patch_smoke({"status": status, "reason": "upstream install failed"})

    def test_reproduction_requires_red_green_or_recorded_smoke_regression(self):
        advisory = {"package": "openai", "integration": "openai", "latestVersion": "3",
                    "basis": "upstream-and-adapter-evidence-review"}
        self.assertEqual(require_reproduction(advisory, "red-before-green-after"), "focused-red-green")
        with self.assertRaisesRegex(RuntimeError, "No reproducible regression"):
            require_reproduction(advisory, "not-proven")
        smoke = {**advisory, "basis": "activation-smoke-regression", "smoke": {
            "package": "openai", "integration": "openai", "latestVersion": "3",
            "comparison": "smoke-regression", "baseline": {"status": "pass"},
            "latest": {"status": "fail"},
        }}
        self.assertEqual(
            require_reproduction(smoke, "not-proven"),
            "activation-smoke-baseline-pass-latest-fail-patched-pass",
        )
        smoke["smoke"]["baseline"]["status"] = "fail"
        with self.assertRaisesRegex(RuntimeError, "No reproducible regression"):
            require_reproduction(smoke, "not-proven")

    def test_workspace_snapshot_detects_generated_test_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / "adapter.py").write_text("before\n")
            subprocess.run(["git", "add", "adapter.py"], cwd=root, check=True)
            (root / "compatibility-evidence.json").write_text("{}\n")
            original = workspace_snapshot(root)
            (root / "adapter.py").write_text("after\n")
            (root / "unexpected.py").write_text("pass\n")
            self.assertEqual(
                changed_paths(original, workspace_snapshot(root)),
                {"adapter.py", "unexpected.py"},
            )
            (root / "compatibility-evidence.json").write_text('{"tampered": true}\n')
            self.assertIn(
                "compatibility-evidence.json",
                changed_paths(original, workspace_snapshot(root)),
            )

    def test_workspace_snapshot_records_symlink_without_reading_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / "checkout"
            checkout.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
            outside = root / "outside-secret.txt"
            outside.write_text("first\n")
            (checkout / "outside-link").symlink_to(outside)
            first = workspace_snapshot(checkout)
            outside.write_text("second\n")
            self.assertEqual(first, workspace_snapshot(checkout))
            (checkout / "outside-link").unlink()
            (checkout / "outside-link").symlink_to(root / "different-target")
            self.assertNotEqual(first, workspace_snapshot(checkout))

    def test_publisher_rejects_content_different_from_validated_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "adapter.py"
            target.write_text("validated patch\n")
            validation = {
                "changedFiles": ["adapter.py"],
                "changedFileSha256": {
                    "adapter.py": hashlib.sha256(target.read_bytes()).hexdigest(),
                },
            }
            require_validated_content(["adapter.py"], validation, root)
            target.write_text("different patch\n")
            with self.assertRaisesRegex(ValueError, "content differs"):
                require_validated_content(["adapter.py"], validation, root)
            with self.assertRaisesRegex(ValueError, "paths differ"):
                require_validated_content(["other.py"], validation, root)
            target.unlink()
            target.symlink_to(root / "elsewhere.py")
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                require_validated_content(["adapter.py"], validation, root)


if __name__ == "__main__":
    unittest.main()
