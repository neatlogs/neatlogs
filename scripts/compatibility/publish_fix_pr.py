"""Publish a validated text patch as one idempotent draft review PR."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.compatibility.propose_fix import apply_proposal, proposal_branch


def command(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False, timeout=60)
    if check and result.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:3])} failed: {result.stderr[-1000:]}")
    return result


def publish() -> dict[str, str]:
    proposal_bytes = Path("compatibility-fix-proposal.json").read_bytes()
    proposal = json.loads(proposal_bytes)
    evidence = json.loads(Path("compatibility-evidence.json").read_text())
    validation = json.loads(Path("compatibility-validation-status.json").read_text())
    if validation.get("status") != "validated":
        raise ValueError("No successful isolated validation result")
    if validation.get("proposalSha256") != hashlib.sha256(proposal_bytes).hexdigest():
        raise ValueError("Validation result does not match the original proposal")
    candidate = proposal["candidate"]
    base_sha = command("git", "rev-parse", "HEAD").stdout.strip()
    if base_sha != proposal.get("baseSha"):
        raise ValueError("Proposal base SHA differs from publisher checkout")
    branch = proposal_branch(candidate)
    repository = os.environ["GITHUB_REPOSITORY"]
    existing = json.loads(command(
        "gh", "pr", "list", "--repo", repository, "--head", branch,
        "--state", "all", "--limit", "20", "--json", "url,state",
    ).stdout)
    if existing:
        return {"status": "already-covered", "prUrl": existing[0]["url"],
                "reason": f"Existing PR is {existing[0]['state']}"}
    changed = apply_proposal(proposal, candidate, evidence)
    if set(changed) != set(validation.get("changedFiles", [])):
        raise ValueError("Publisher paths differ from isolated validation")
    command("git", "switch", "-c", branch)
    command("git", "add", "--", *changed)
    staged = set(command("git", "diff", "--cached", "--name-only").stdout.splitlines())
    if staged != set(changed):
        raise ValueError("Staged paths differ from validated proposal")
    command("git", "diff", "--cached", "--check")
    command("git", "config", "user.name", "neatlogs-compatibility-bot")
    command("git", "config", "user.email", "compatibility-bot@users.noreply.github.com")
    command("git", "-c", "core.hooksPath=/dev/null", "commit", "-m",
            f"fix: Python {candidate['package']} {candidate['latestVersion']} compatibility")
    local_tree = command("git", "rev-parse", "HEAD^{tree}").stdout.strip()
    command("gh", "auth", "setup-git")
    remote = command("git", "ls-remote", "--heads", "origin", branch).stdout.strip()
    if remote:
        command("git", "fetch", "origin", branch)
        remote_tree = command("git", "rev-parse", "FETCH_HEAD^{tree}").stdout.strip()
        if remote_tree != local_tree:
            raise RuntimeError("Existing remote branch has different content; refusing to overwrite it")
    else:
        command("git", "push", "origin", f"HEAD:refs/heads/{branch}")
    issue = os.environ.get("COMPAT_REVIEW_ISSUE_URL", "")
    run_url = os.environ.get("COMPAT_WORKFLOW_RUN_URL", "")
    default_branch = command(
        "gh", "repo", "view", repository, "--json", "defaultBranchRef",
        "--jq", ".defaultBranchRef.name",
    ).stdout.strip()
    if not default_branch:
        raise RuntimeError("Could not resolve the repository default branch")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as body:
        body.write(
            "## Gemini-proposed compatibility fix for human review\n\n"
            f"Package: `{candidate['package']}` {candidate['latestVersion']} / integration `{candidate['integration']}`.\n\n"
            f"Basis: {candidate['basis']}.\n\n"
            f"Evidence: {str(proposal['evidence'])[:1000]}\n\n"
            f"Rationale: {str(proposal['rationale'])[:1000]}\n\n"
            f"Validation: {validation['validationLimit']}.\n\n"
            f"Latest-version activation after patch: {validation.get('postPatchSmoke', 'unknown')}.\n\n"
            f"Related review issue: {issue}\n\nWorkflow run: {run_url}\n\n"
            "This is a draft for human code review. It must not be merged without reviewing the upstream evidence and the generated diff.\n"
        )
        body_path = body.name
    try:
        created = command(
            "gh", "pr", "create", "--repo", repository, "--base", default_branch, "--head", branch,
            "--draft", "--title",
            f"fix: review Python {candidate['package']} {candidate['latestVersion']} compatibility",
            "--body-file", body_path,
        )
    finally:
        Path(body_path).unlink(missing_ok=True)
    return {"status": "created", "prUrl": created.stdout.strip(), "branch": branch}


def main() -> int:
    try:
        status = publish()
        code = 0
    except Exception as error:
        status = {"status": "failed", "reason": str(error)[:1000]}
        code = 1
    Path("compatibility-publish-status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(f"Draft PR: {status['status']}: {status.get('prUrl', status.get('reason', ''))}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
