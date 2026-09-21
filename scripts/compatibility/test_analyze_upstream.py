import io
import unittest
import zipfile

from scripts.compatibility.analyze_upstream import (
    archive_manifest,
    archive_text_files,
    diff_api,
    diff_files,
    diff_objects,
    diff_text_files,
    documentation_text,
    extract_python_api,
    official_documentation_urls,
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
                "sizeChanged": [
                    {"path": "changed.py", "beforeBytes": 2, "afterBytes": 3}
                ],
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

    def test_archive_text_files_reads_real_source_from_wheel(self):
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w") as archive:
            archive.writestr(
                "package/__init__.py",
                "def public(value: str) -> str:\n    return value\n",
            )
            archive.writestr("package/data.bin", b"ignored")
        self.assertEqual(
            archive_text_files(content.getvalue(), "package.whl"),
            {
                "package/__init__.py": "def public(value: str) -> str:\n    return value\n"
            },
        )

    def test_extract_python_api_and_diff_are_substantive(self):
        before = {
            "package/__init__.py": "def instrument(context: str) -> None:\n    pass\n"
        }
        after = {
            "package/__init__.py": "def instrument(runtime_context: dict) -> None:\n    pass\n"
        }
        previous_api = extract_python_api(before)
        latest_api = extract_python_api(after)
        self.assertEqual(
            diff_api(previous_api, latest_api),
            {
                "added": [
                    "package/__init__.py: def instrument(runtime_context: dict) -> None"
                ],
                "removed": [
                    "package/__init__.py: def instrument(context: str) -> None"
                ],
            },
        )
        self.assertEqual(
            diff_text_files(before, after),
            [
                {
                    "path": "package/__init__.py",
                    "addedLines": ["def instrument(runtime_context: dict) -> None:"],
                    "removedLines": ["def instrument(context: str) -> None:"],
                }
            ],
        )

    def test_official_documentation_comes_from_pypi_project_metadata(self):
        self.assertEqual(
            official_documentation_urls(
                {
                    "info": {
                        "project_urls": {
                            "Documentation": "https://sdk.example.dev/docs",
                            "Source": "https://github.com/example/sdk",
                        }
                    }
                }
            ),
            [
                {"kind": "documentation", "url": "https://sdk.example.dev/docs"},
                {"kind": "source", "url": "https://github.com/example/sdk"},
                {
                    "kind": "release-notes",
                    "url": "https://github.com/example/sdk/releases",
                },
            ],
        )

    def test_documentation_text_extracts_visible_html(self):
        self.assertEqual(
            documentation_text(
                "<html><script>ignore()</script><body><h1>Migration</h1><p>Use runtime_context.</p></body></html>",
                "text/html",
            ),
            "Migration Use runtime_context.",
        )


if __name__ == "__main__":
    unittest.main()
