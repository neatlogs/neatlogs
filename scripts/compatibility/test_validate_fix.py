import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.compatibility.publish_fix_pr import require_validated_content
from scripts.compatibility.validate_fix import (
    changed_paths,
    require_post_patch_smoke,
    workspace_snapshot,
)


class ValidateFixTests(unittest.TestCase):
    def test_only_passing_post_patch_smoke_can_validate(self):
        require_post_patch_smoke({"status": "pass"})
        for status in ("blocked", "fail", "not-tested"):
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, status):
                require_post_patch_smoke({"status": status, "reason": "upstream install failed"})

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


if __name__ == "__main__":
    unittest.main()
