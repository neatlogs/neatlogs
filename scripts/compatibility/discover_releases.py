from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def validate_configuration(config: dict[str, Any], lock: dict[str, Any]) -> None:
    if config.get("schemaVersion") != 1 or not isinstance(config.get("integrations"), list):
        raise ValueError("integrations.json must use schemaVersion 1 and contain integrations[]")
    if lock.get("schemaVersion") != 1 or not isinstance(lock.get("packages"), dict):
        raise ValueError("versions.lock.json must use schemaVersion 1 and contain packages{}")

    ids: set[str] = set()
    for integration in config["integrations"]:
        if not integration.get("id") or not integration.get("displayName"):
            raise ValueError("every integration requires id and displayName")
        if not isinstance(integration.get("packages"), list):
            raise ValueError(f"{integration['id']} requires packages[]")
        if integration["id"] in ids:
            raise ValueError(f"duplicate integration id: {integration['id']}")
        ids.add(integration["id"])
        if any(not isinstance(item, str) or not item for item in integration["packages"]):
            raise ValueError(f"invalid package name for {integration['id']}")


def watched_packages(config: dict[str, Any]) -> list[str]:
    return sorted(
        {
            package
            for integration in config["integrations"]
            if integration.get("releaseMonitoring", True)
            for package in integration["packages"]
        }
    )


def compare_versions(
    config: dict[str, Any],
    lock: dict[str, Any],
    registry_versions: dict[str, str],
) -> list[dict[str, Any]]:
    integrations_by_package: dict[str, list[str]] = {}
    for integration in config["integrations"]:
        if not integration.get("releaseMonitoring", True):
            continue
        for package in integration["packages"]:
            integrations_by_package.setdefault(package, []).append(integration["id"])

    changes: list[dict[str, Any]] = []
    for package in watched_packages(config):
        latest = registry_versions.get(package)
        analyzed = lock["packages"].get(package)
        if not latest or latest == analyzed:
            continue
        changes.append(
            {
                "package": package,
                "previouslyAnalyzed": analyzed,
                "latest": latest,
                "integrations": integrations_by_package.get(package, []),
            }
        )
    return changes


def fetch_latest_version(package: str) -> str:
    request = Request(
        f"https://pypi.org/pypi/{quote(package, safe='')}/json",
        headers={"User-Agent": "neatlogs-compatibility-monitor/1"},
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed trusted host
        metadata = json.load(response)
    latest = metadata.get("info", {}).get("version")
    if not isinstance(latest, str) or not latest:
        raise ValueError(f"PyPI returned no latest version for {package}")
    return latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--report", default="compatibility-release-report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads((REPOSITORY_ROOT / ".compatibility/integrations.json").read_text())
    lock = json.loads((REPOSITORY_ROOT / ".compatibility/versions.lock.json").read_text())
    validate_configuration(config, lock)

    if args.validate_only:
        print(f"Validated {len(config['integrations'])} Python integrations")
        return 0

    packages = watched_packages(config)
    registry_versions = {package: fetch_latest_version(package) for package in packages}
    changes = compare_versions(config, lock, registry_versions)
    report = {
        "schemaVersion": 1,
        "ecosystem": "pypi",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "checkedPackages": len(packages),
        "changes": changes,
    }
    report_path = (REPOSITORY_ROOT / args.report).resolve()
    report_path.write_text(f"{json.dumps(report, indent=2)}\n")
    print(json.dumps(report, indent=2))

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a") as output:
            output.write(f"changes_found={'true' if changes else 'false'}\n")
            output.write(f"report_path={report_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
