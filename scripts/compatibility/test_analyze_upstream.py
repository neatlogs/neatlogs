import io
import json
import tempfile
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from scripts.compatibility.analyze_upstream import (
    archive_manifest,
    archive_text_files,
    analyze_with_gemini,
    compact_package,
    diff_api,
    diff_files,
    diff_objects,
    diff_text_files,
    documentation_text,
    extract_python_api,
    evidence_batches,
    official_documentation_urls,
    _analyze_gemini_batch,
    MAX_GEMINI_BATCH_BYTES,
    MAX_GEMINI_PACKAGE_BYTES,
    main,
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

    def test_large_evidence_is_bounded_and_every_package_is_reviewed(self):
        packages = []
        for index in range(7):
            packages.append({
                "package": f"package-{index}",
                "previousVersion": "1",
                "latestVersion": "2",
                "integrations": [{"id": "adapter", "contracts": ["streaming"], "adapterSource": [{"path": "adapter.py", "content": "a" * 50000}]}],
                "packageSurfaceChanges": [{"key": "requires_dist", "before": ["a" * 50000], "after": ["b" * 50000]}],
                "publicApiChanges": {"added": ["def new(): ..."] * 200, "removed": []},
                "sourceContentChanges": [{"path": "source.py", "addedLines": ["x" * 5000] * 50, "removedLines": []}] * 40,
                "officialDocumentation": [{"url": "https://example.test", "content": "d" * 50000}],
                "artifactFileChanges": {"added": ["file.py"] * 200, "removed": [], "sizeChanged": []},
            })
        evidence = {"ecosystem": "pypi", "packages": packages}
        self.assertLessEqual(len(json.dumps(compact_package(packages[0])).encode()), MAX_GEMINI_PACKAGE_BYTES)
        batches = evidence_batches(evidence)
        self.assertEqual(
            [p["package"] for batch in batches for p in batch["packages"]],
            [p["package"] for p in packages],
        )
        self.assertTrue(all(len(batch["packages"]) <= 3 for batch in batches))
        self.assertTrue(all(len(json.dumps(batch).encode()) <= MAX_GEMINI_BATCH_BYTES for batch in batches))

    def test_gemini_http_error_includes_api_reason_without_key(self):
        error = HTTPError("https://example.test", 400, "Bad Request", {}, io.BytesIO(b'{"error":{"message":"Input is too long"}}'))
        with patch("scripts.compatibility.analyze_upstream.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "HTTP 400.*Input is too long") as caught:
                _analyze_gemini_batch({"packages": [{"package": "openai"}]}, "secret-key", "gemini-2.5-flash")
        self.assertNotIn("secret-key", str(caught.exception))

    def test_batch_risks_are_aggregated(self):
        evidence = {"ecosystem": "pypi", "packages": [{"package": f"package-{index}"} for index in range(4)]}
        responses = [
            {"summary": "Low risk evidence", "riskLevel": "low", "findings": ["package-0 detail"], "recommendedTests": []},
            {"summary": "Possible break", "riskLevel": "high", "findings": ["package-3 detail"], "recommendedTests": ["package-3 smoke"]},
        ]
        with patch("scripts.compatibility.analyze_upstream._analyze_gemini_batch", side_effect=responses) as send:
            result = analyze_with_gemini(evidence, "key")
        self.assertEqual(send.call_count, 2)
        self.assertEqual(result["riskLevel"], "high")
        self.assertEqual(result["findings"], ["package-0 detail", "package-3 detail"])
        self.assertEqual(len(result["packagesReviewed"]), 4)

    def test_failed_analysis_is_written_for_artifact_and_slack(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "compatibility-evidence.json").write_text('{"packages": []}')
            args = Namespace(evidence="compatibility-evidence.json", llm_output="analysis.json", llm_only=True)
            with patch("scripts.compatibility.analyze_upstream.REPOSITORY_ROOT", root), \
                 patch("scripts.compatibility.analyze_upstream.parse_args", return_value=args), \
                 patch.dict("os.environ", {"COMPAT_GEMINI_API_KEY": "key"}), \
                 patch("scripts.compatibility.analyze_upstream.analyze_with_gemini", side_effect=RuntimeError("Gemini HTTP 400: input too long")):
                with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
                    main()
            self.assertEqual(
                json.loads((root / "analysis.json").read_text()),
                {"failed": True, "error": "Gemini HTTP 400: input too long"},
            )


if __name__ == "__main__":
    unittest.main()
