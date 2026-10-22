import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.compatibility.run_latest_version_checks import (
    check_version,
    compare_results,
    instrumented_library,
    not_tested_pair,
)
from scripts.compatibility.summarize_latest_version_checks import summarize


class LatestVersionChecksTests(unittest.TestCase):
    def test_accurate_instrumentation_aliases(self):
        self.assertEqual(instrumented_library("qdrant-langchain"), "qdrant")
        self.assertEqual(instrumented_library("google-genai"), "google_genai")

    def test_only_baseline_pass_and_latest_fail_is_smoke_regression(self):
        passed = {"status": "pass"}
        failed = {"status": "fail"}
        blocked = {"status": "blocked"}
        self.assertEqual(compare_results(passed, failed), "smoke-regression")
        self.assertEqual(compare_results(failed, failed), "latest-failure-needs-triage")
        self.assertEqual(compare_results(passed, blocked), "latest-install-blocked")
        self.assertEqual(compare_results(failed, passed), "latest-smoke-passed")

    def test_exact_version_and_sdk_extra_are_installed_before_probe(self):
        commands = []

        def fake_run(command, **kwargs):
            commands.append((command, kwargs))
            return {"ok": True, "exitCode": 0, "output": "ok"}

        with patch.dict("os.environ", {"PYTHONPATH": "/checkout"}), patch(
            "scripts.compatibility.run_latest_version_checks.run_command", side_effect=fake_run,
        ):
            result = check_version(
                package="openai", version="2.0.0", integration="openai",
                extra="openai", wheel=Path("dist/neatlogs-test.whl"),
            )
        self.assertEqual(result["status"], "pass")
        self.assertIn("openai==2.0.0", commands[1][0])
        self.assertTrue(any(arg.endswith("neatlogs-test.whl[openai]") for arg in commands[1][0]))
        self.assertEqual(commands[3][1]["env"]["COMPAT_LIBRARY"], "openai")
        self.assertEqual(commands[3][1]["cwd"].name, ".venv")
        self.assertTrue(all("PYTHONPATH" not in kwargs["env"] for _, kwargs in commands))
        self.assertIn("Neatlogs was imported from the checkout", commands[3][0][2])

    def test_missing_extra_is_explicitly_not_tested(self):
        result = not_tested_pair("package", "1", "2", "adapter", "No extra")
        self.assertEqual(result["latest"]["status"], "not-tested")
        self.assertEqual(result["comparison"], "not-tested")

    def test_install_failure_is_blocked_with_reason(self):
        outcomes = [
            {"ok": True, "exitCode": 0, "output": "venv created"},
            {"ok": False, "exitCode": 1, "output": "ResolutionImpossible: SDK constraint"},
        ]
        with patch("scripts.compatibility.run_latest_version_checks.run_command", side_effect=outcomes):
            result = check_version(
                package="openai", version="2.0.0", integration="openai",
                extra="openai", wheel=Path("dist/neatlogs-test.whl"),
            )
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["stage"], "install")
        self.assertIn("SDK constraint", result["reason"])

    def test_summary_counts_missing_and_blocked_checks(self):
        report = {"changes": [
            {"package": "alpha", "previouslyAnalyzed": "1", "latest": "2", "integrations": ["openai"]},
            {"package": "beta", "previouslyAnalyzed": "1", "latest": "2", "integrations": ["mcp"]},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alpha.json"
            path.write_text(json.dumps({
                "package": "alpha", "integration": "openai", "baselineVersion": "1",
                "latestVersion": "2", "baseline": {"status": "pass"},
                "latest": {"status": "blocked", "stage": "install"},
                "comparison": "latest-install-blocked",
            }))
            summary = summarize(report, Path(directory))
        self.assertEqual(summary["counts"], {"pass": 0, "fail": 0, "blocked": 1, "not-tested": 1})
        self.assertEqual(summary["smokeRegressions"], 0)
        self.assertEqual(summary["results"][1]["comparison"], "not-tested")


if __name__ == "__main__":
    unittest.main()
