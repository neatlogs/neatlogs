"""Validate generated code in a read-only job without model or write credentials."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.compatibility.propose_fix import apply_proposal, validate_proposal
from scripts.compatibility.run_latest_version_checks import check_version


def run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                          check=False, timeout=timeout)


def validate() -> dict[str, object]:
    proposal_bytes = Path("compatibility-fix-proposal.json").read_bytes()
    proposal = json.loads(proposal_bytes)
    evidence = json.loads(Path("compatibility-evidence.json").read_text())
    candidate = proposal["candidate"]
    base = run(["git", "rev-parse", "HEAD"], 10).stdout.strip()
    if proposal["baseSha"] != base:
        raise ValueError("Proposal base SHA does not match validation checkout")
    validate_proposal(proposal, candidate, evidence)
    red_green = "not-proven"
    test = proposal.get("regressionTest")
    if test:
        test_path = ROOT / test["path"]
        test_path.write_text(test["content"])
        try:
            red = run([sys.executable, "-m", "pytest", "-q", test["path"]], 120)
            if red.returncode == 1:
                red_green = "red-before-patch"
            else:
                raise RuntimeError(
                    "Supplied regression test did not fail as a test on the original SDK"
                )
        finally:
            test_path.unlink(missing_ok=True)
    changed = apply_proposal(proposal, candidate, evidence)
    Path("compatibility-validated-proposal.json").write_text(json.dumps(proposal, indent=2) + "\n")
    diff = run(["git", "diff", "--check"], 20)
    if diff.returncode != 0:
        raise RuntimeError(f"Patch whitespace check failed: {diff.stderr[-1000:]}")
    automation = run([sys.executable, "-m", "unittest", "discover", "-s", "scripts/compatibility", "-p", "test_*.py"], 120)
    if automation.returncode != 0:
        raise RuntimeError(f"Compatibility automation tests failed: {(automation.stdout + automation.stderr)[-2000:]}")
    stems = {Path(edit["path"]).stem for edit in proposal["edits"]}
    focused = sorted({
        path for stem in stems
        for path in (ROOT / "tests/unit").glob(f"test_*{stem}*.py")
    })[:5]
    if proposal.get("regressionTest"):
        focused.append(ROOT / proposal["regressionTest"]["path"])
    if focused:
        tests = run([sys.executable, "-m", "pytest", "-p", "pytest_asyncio.plugin", "-q",
                     *[str(path.relative_to(ROOT)) for path in focused]], 240)
        if tests.returncode != 0:
            raise RuntimeError(f"Focused tests failed: {(tests.stdout + tests.stderr)[-2000:]}")
    if proposal.get("regressionTest"):
        red_green = "red-before-green-after"
    config = json.loads((ROOT / ".compatibility/integrations.json").read_text())
    integration = next(item for item in config["integrations"] if item["id"] == candidate["integration"])
    with tempfile.TemporaryDirectory(prefix="neatlogs-compat-fixed-wheel-") as directory:
        built = run([sys.executable, "-m", "build", "--wheel", "--outdir", directory], 180)
        if built.returncode != 0:
            raise RuntimeError(f"Patched SDK wheel build failed: {(built.stdout + built.stderr)[-2000:]}")
        wheel = next(Path(directory).glob("neatlogs-*.whl"))
        smoke = check_version(
            package=candidate["package"], version=candidate["latestVersion"],
            integration=candidate["integration"], extra=integration["extra"], wheel=wheel,
        )
    if smoke["status"] in {"fail", "not-tested"} or (
        candidate["basis"] == "activation-smoke-regression" and smoke["status"] != "pass"
    ):
        raise RuntimeError(f"Patched latest-version activation check did not pass: {smoke['status']}: {smoke.get('reason', '')[-1000:]}")
    scope = ["Compatibility automation tests passed"]
    if red_green == "red-before-green-after":
        scope.append("focused generated test failed before and passed after the patch")
    else:
        scope.append("no focused red/green reproduction")
    existing_count = len(focused) - int(bool(test))
    if existing_count:
        scope.append(f"{existing_count} existing adapter test file(s) passed")
    else:
        scope.append("no existing adapter test file matched")
    scope.append(f"patched latest-version activation: {smoke['status']}")
    if red_green != "red-before-green-after" or smoke["status"] != "pass":
        scope.append("behavior change remains unverified")
    return {
        "status": "validated",
        "proposalSha256": hashlib.sha256(proposal_bytes).hexdigest(),
        "changedFiles": changed,
        "redGreen": red_green,
        "postPatchSmoke": smoke["status"],
        "focusedTests": [str(path.relative_to(ROOT)) for path in focused],
        "compatibilityAutomationTests": "passed",
        "basis": candidate["basis"],
        "validationLimit": "; ".join(scope),
    }


def main() -> int:
    try:
        status = validate()
        code = 0
    except Exception as error:
        status = {"status": "failed", "reason": str(error)[:2000]}
        code = 1
    Path("compatibility-validation-status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(f"Fix validation: {status['status']}: {status.get('reason', '')}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
