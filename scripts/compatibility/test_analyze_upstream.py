import io
import unittest
import zipfile

from scripts.compatibility.analyze_upstream import (
    archive_manifest,
    diff_files,
    diff_objects,
)


class AnalyzeUpstreamTests(unittest.TestCase):
    def test_diff_objects(self):
        self.assertEqual(
            diff_objects({"requires_python": ">=3.9"}, {"requires_python": ">=3.10"}),
            [
                {
                    "key": "requires_python",
                    "before": ">=3.9",
                    "after": ">=3.10",
                }
            ],
        )

    def test_diff_files(self):
        self.assertEqual(
            diff_files(
                [{"path": "old.py", "size": 1}, {"path": "changed.py", "size": 2}],
                [{"path": "new.py", "size": 1}, {"path": "changed.py", "size": 3}],
            ),
            {
                "added": ["new.py"],
                "removed": ["old.py"],
                "sizeChanged": [{"path": "changed.py", "beforeBytes": 2, "afterBytes": 3}],
            },
        )

    def test_archive_manifest_reads_wheel_without_extracting(self):
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w") as archive:
            archive.writestr("package/__init__.py", "value = 1\n")
        self.assertEqual(
            archive_manifest(content.getvalue(), "package.whl"),
            [{"path": "package/__init__.py", "size": 10}],
        )


if __name__ == "__main__":
    unittest.main()
