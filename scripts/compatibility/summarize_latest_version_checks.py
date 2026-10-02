"""Collect matrix smoke results without hiding missing or failed jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def missing_result(change: dict[str, Any], integration: str) -> dict[str, Any]:
    return {
        "package": change["package"],
        "integration": integration,
        "baselineVersion": change.get("previouslyAnalyzed"),
        "latestVersion": change["latest"],
        "baseline": {"status": "not-tested", "reason": "Matrix result missing"},
        "latest": {"status": "not-tested", "reason": "Matrix result missing"},
        "comparison": "not-tested",
        "scope": "Exact-version install, dependency check, and instrumentation activation",
    }


def summarize(report: dict[str, Any], results_directory: Path) -> dict[str, Any]:
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for path in results_directory.glob("*.json"):
        try:
            item = json.loads(path.read_text())
            key = (item["package"], item["integration"])
            found[key] = item
        except (OSError, ValueError, KeyError, TypeError):
            continue
    results = []
    for change in report.get("changes", []):
        for integration in change.get("integrations", []):
            key = (change["package"], integration)
            item = found.get(key)
            if (
                item is None
                or item.get("latestVersion") != change["latest"]
                or item.get("baselineVersion") != change.get("previouslyAnalyzed")
            ):
                item = missing_result(change, integration)
            results.append(item)
    counts = {status: 0 for status in ("pass", "fail", "blocked", "not-tested")}
    for item in results:
        status = item["latest"].get("status", "not-tested")
        counts[status if status in counts else "not-tested"] += 1
    return {
        "schemaVersion": 1,
        "scope": "Exact-version install, dependency check, and Neatlogs instrumentation activation; no provider call or full regression suite",
        "pairCount": len(results),
        "counts": counts,
        "smokeRegressions": sum(item["comparison"] == "smoke-regression" for item in results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="compatibility-release-report.json")
    parser.add_argument("--results-dir", default="smoke-results")
    parser.add_argument("--output", default="compatibility-smoke-summary.json")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    summary = summarize(report, Path(args.results_dir))
    Path(args.output).write_text(f"{json.dumps(summary, indent=2)}\n")
    print(f"Latest-version smoke checks: {summary['counts']}; smoke regressions: {summary['smokeRegressions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
