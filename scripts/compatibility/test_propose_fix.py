import json
import tempfile
import unittest
from pathlib import Path

from scripts.compatibility.propose_fix import (
    apply_proposal,
    advance_version_lock,
    allowed_adapter_paths,
    candidate_options,
    choose_candidate,
    deferred_candidates,
    generated_test_path,
    no_sdk_patch_surface,
    proposal_branch,
    validate_proposal,
)


def evidence():
    return {"packages": [
        {"package": package, "latestVersion": "2", "integrations": [
            {"id": "openai", "adapterSource": [{"path": f"neatlogs/{package}.py", "content": "def run():\n    return 1\n"}]}
        ]}
        for package in ("alpha", "beta")
    ]}


class ProposeFixTests(unittest.TestCase):
    def test_version_lock_advances_only_evidence_bound_fixed_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / ".compatibility/versions.lock.json"
            target.parent.mkdir()
            target.write_text(json.dumps({
                "schemaVersion": 1, "packages": {"alpha": "1", "beta": "9"},
            }) + "\n")
            release = {"packages": [{"package": "alpha", "previousVersion": "1",
                                     "latestVersion": "2", "integrations": []}]}
            self.assertEqual(
                advance_version_lock({"package": "alpha", "latestVersion": "2"}, release, root),
                ".compatibility/versions.lock.json",
            )
            self.assertEqual(json.loads(target.read_text())["packages"], {"alpha": "2", "beta": "9"})
            with self.assertRaisesRegex(ValueError, "baseline does not match"):
                advance_version_lock({"package": "alpha", "latestVersion": "2"}, release, root)

    def test_unmapped_integration_is_reported_without_allowing_shared_registry_edits(self):
        package_evidence = evidence()
        package_evidence["packages"][0]["integrations"] = [
            {"id": "groq", "adapterSource": []},
        ]
        self.assertEqual(no_sdk_patch_surface(package_evidence), [
            {"package": "alpha", "integration": "groq", "latestVersion": "2"},
        ])
        self.assertEqual(
            [(item["package"], item["integration"]) for item in candidate_options({}, {}, package_evidence)],
            [("beta", "openai")],
        )

    def test_candidates_do_not_depend_on_high_advisory_score_and_rotate(self):
        items = candidate_options({"results": []}, {"riskLevel": "low", "findings": []}, evidence())
        self.assertEqual([item["package"] for item in items], ["alpha", "beta"])
        self.assertEqual(choose_candidate({}, {}, evidence(), rotation=0)["package"], "alpha")
        self.assertEqual(choose_candidate({}, {}, evidence(), rotation=1)["package"], "beta")
        branch = proposal_branch(items[0])
        self.assertEqual(choose_candidate({}, {}, evidence(), {branch}, rotation=0)["package"], "beta")

    def test_activation_regression_takes_priority_until_covered(self):
        summary = {"results": [{
            "package": "beta", "integration": "openai", "comparison": "smoke-regression",
        }]}
        selected = choose_candidate(summary, {}, evidence(), rotation=0)
        self.assertEqual(selected["package"], "beta")
        self.assertEqual(selected["basis"], "activation-smoke-regression")
        self.assertEqual(
            choose_candidate(summary, {}, evidence(), {proposal_branch(selected)}, rotation=0)["package"],
            "alpha",
        )

    def test_deferred_candidates_exclude_selected_candidate_by_branch(self):
        selected = choose_candidate({}, {}, evidence(), rotation=0)
        # Candidate selection builds a separate list from the reporting path.
        options = candidate_options({}, {}, evidence())
        self.assertIsNot(selected, options[0])
        self.assertEqual(
            deferred_candidates(options, selected, set()),
            [{"package": "beta", "integration": "openai", "latestVersion": "2",
              "basis": "upstream-and-adapter-evidence-review"}],
        )

    def test_only_exact_adapter_source_replacement_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "neatlogs").mkdir()
            target = root / "neatlogs/alpha.py"
            target.write_text("def run():\n    return 1\n")
            candidate = choose_candidate({}, {}, evidence(), rotation=0)
            proposal = {
                "decision": "propose_fix", "package": "alpha", "integration": "openai",
                "rationale": "The upstream call changed its return value.",
                "evidence": "The upstream API diff removed the old signature.",
                "edits": [{"path": "neatlogs/alpha.py", "oldText": "return 1", "newText": "return 2"}],
            }
            validate_proposal(proposal, candidate, evidence(), root)
            self.assertEqual(apply_proposal(proposal, candidate, evidence(), root), ["neatlogs/alpha.py"])
            self.assertIn("return 2", target.read_text())

    def test_workflow_or_unmatched_edits_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "neatlogs").mkdir()
            (root / "neatlogs/alpha.py").write_text("def run():\n    return 1\n")
            candidate = choose_candidate({}, {}, evidence(), rotation=0)
            proposal = {
                "decision": "propose_fix", "package": "alpha", "integration": "openai",
                "rationale": "The upstream call changed its return value.",
                "evidence": "The upstream API diff removed the old signature.",
                "edits": [{"path": ".github/workflows/ci.yml", "oldText": "x", "newText": "y"}],
            }
            with self.assertRaisesRegex(ValueError, "outside the candidate adapter"):
                validate_proposal(proposal, candidate, evidence(), root)
            proposal["edits"][0] = {"path": "neatlogs/alpha.py", "oldText": "missing", "newText": "value"}
            with self.assertRaisesRegex(ValueError, "unique exact replacement"):
                validate_proposal(proposal, candidate, evidence(), root)

    def test_generated_test_path_is_per_release_and_adapter_paths_cannot_escape(self):
        candidate = choose_candidate({}, {}, evidence(), rotation=0)
        self.assertEqual(
            generated_test_path(candidate),
            "tests/unit/test_compat_generated_alpha_2_openai.py",
        )
        bad_evidence = evidence()
        bad_evidence["packages"][0]["integrations"][0]["adapterSource"].append(
            {"path": "neatlogs/../.github/workflows/unsafe.py", "content": "pass"}
        )
        self.assertEqual(allowed_adapter_paths(candidate, bad_evidence), {"neatlogs/alpha.py"})


if __name__ == "__main__":
    unittest.main()
