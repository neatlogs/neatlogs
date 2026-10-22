import unittest

from scripts.compatibility.discover_releases import (
    compare_versions,
    validate_configuration,
    watched_packages,
)

CONFIG = {
    "schemaVersion": 1,
    "integrations": [
        {
            "id": "one",
            "displayName": "One",
            "packages": ["one", "shared"],
            "documentationUrls": ["https://one.example/docs"],
        },
        {
            "id": "two",
            "displayName": "Two",
            "packages": ["shared"],
            "documentationUrls": ["https://two.example/docs"],
        },
    ],
}
LOCK = {"schemaVersion": 1, "packages": {"one": "1.0.0", "shared": "2.0.0"}}


class DiscoverReleasesTest(unittest.TestCase):
    def test_validation_and_package_deduplication(self) -> None:
        validate_configuration(CONFIG, LOCK)
        self.assertEqual(watched_packages(CONFIG), ["one", "shared"])

    def test_only_new_versions_are_reported(self) -> None:
        self.assertEqual(
            compare_versions(CONFIG, LOCK, {"one": "1.0.0", "shared": "3.0.0"}),
            [
                {
                    "package": "shared",
                    "previouslyAnalyzed": "2.0.0",
                    "latest": "3.0.0",
                    "integrations": ["one", "two"],
                }
            ],
        )

    def test_missing_package_baseline_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing baselines for: shared"):
            validate_configuration(CONFIG, {"schemaVersion": 1, "packages": {"one": "1.0.0"}})

    def test_duplicate_ids_are_rejected(self) -> None:
        invalid = {
            **CONFIG,
            "integrations": [*CONFIG["integrations"], CONFIG["integrations"][0]],
        }
        with self.assertRaisesRegex(ValueError, "duplicate integration id"):
            validate_configuration(invalid, LOCK)


if __name__ == "__main__":
    unittest.main()
