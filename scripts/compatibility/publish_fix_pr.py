"""Publish a validated text patch as one idempotent review PR."""

from __future__ import annotations

import json
import hashlib
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.compatibility.propose_fix import (
    apply_proposal, is_bot_draft_pr, proposal_branch,
)

BOT_AUTHOR_EMAIL = "compatibility-bot@users.noreply.github.com"


def command(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False, timeout=60)
    if check and result.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:3])} failed: {result.stderr[-1000:]}")
    return result


def ready_proof(
    pull: dict[str, Any], *, fetched_sha: str, fetched_tree: str,
    fetched_parent: str, fetched_author_email: str, fetched_subject: str,
    base_sha: str, local_tree: str, expected_subject: str,
) -> bool:
    return (
        is_bot_draft_pr(pull)
        and pull.get("headRefOid") == fetched_sha
        and fetched_tree == local_tree
        and fetched_parent == base_sha
        and fetched_author_email == BOT_AUTHOR_EMAIL
        and fetched_subject == expected_subject
    )


def require_validated_content(
    changed: list[str], validation: dict[str, Any], root: Path = ROOT,
) -> None:
    if set(changed) != set(validation.get("changedFiles", [])):
        raise ValueError("Publisher paths differ from isolated validation")
    validated_hashes = validation.get("changedFileSha256")
    if not isinstance(validated_hashes, dict) or set(validated_hashes) != set(changed):
        raise ValueError("Isolated validation did not record every changed file hash")
    actual_hashes = {}
    for path in changed:
        target = root / path
        if not target.resolve().is_relative_to(root.resolve()) or not stat.S_ISREG(target.lstat().st_mode):
            raise ValueError(f"Publisher patch path is not a regular file inside checkout: {path}")
        actual_hashes[path] = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual_hashes != validated_hashes:
        raise ValueError("Publisher patch content differs from isolated validation")


def publish() -> dict[str, Any]:
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
        "--state", "all", "--limit", "20", "--json", "url,state,isDraft,author,headRefOid",
    ).stdout)
    existing.sort(key=lambda item: item["state"] != "OPEN")
    changed = apply_proposal(proposal, candidate, evidence)
    require_validated_content(changed, validation, ROOT)
    command("git", "switch", "-c", branch)
    command("git", "add", "--", *changed)
    staged = set(command("git", "diff", "--cached", "--name-only").stdout.splitlines())
    if staged != set(changed):
        raise ValueError("Staged paths differ from validated proposal")
    command("git", "diff", "--cached", "--check")
    command("git", "config", "user.name", "neatlogs-compatibility-bot")
    command("git", "config", "user.email", BOT_AUTHOR_EMAIL)
    subject = f"fix: Python {candidate['package']} {candidate['latestVersion']} compatibility"
    command("git", "-c", "core.hooksPath=/dev/null", "commit", "-m", subject)
    local_tree = command("git", "rev-parse", "HEAD^{tree}").stdout.strip()
    command("gh", "auth", "setup-git")
    remote = command("git", "ls-remote", "--heads", "origin", branch).stdout.strip()
    if existing:
        pull = existing[0]
        status: dict[str, Any] = {
            "status": "already-covered", "prUrl": pull["url"],
            "state": pull["state"], "isDraft": pull["isDraft"],
            "reason": f"Existing PR is {pull['state']}",
        }
        if not is_bot_draft_pr(pull):
            return status
        if not remote:
            status["reason"] = "Existing bot draft has no remote branch; human review required"
            return status
        command("git", "fetch", "origin", branch)
        fresh = json.loads(command(
            "gh", "pr", "view", pull["url"], "--repo", repository,
            "--json", "url,state,isDraft,author,headRefOid",
        ).stdout)
        status["state"] = fresh["state"]
        status["isDraft"] = fresh["isDraft"]
        proof = ready_proof(
            fresh,
            fetched_sha=command("git", "rev-parse", "FETCH_HEAD").stdout.strip(),
            fetched_tree=command("git", "rev-parse", "FETCH_HEAD^{tree}").stdout.strip(),
            fetched_parent=command("git", "show", "-s", "--format=%P", "FETCH_HEAD").stdout.strip(),
            fetched_author_email=command("git", "show", "-s", "--format=%ae", "FETCH_HEAD").stdout.strip(),
            fetched_subject=command("git", "show", "-s", "--format=%s", "FETCH_HEAD").stdout.strip(),
            base_sha=base_sha, local_tree=local_tree, expected_subject=subject,
        )
        if not proof:
            status["reason"] = "Existing draft is not proven bot-owned and unchanged; human must mark ready"
            return status
        try:
            command("gh", "pr", "ready", pull["url"], "--repo", repository)
        except RuntimeError as error:
            return {"status": "failed", "prUrl": pull["url"], "state": "OPEN",
                    "isDraft": True, "reason": f"Could not mark existing bot PR ready: {error}"}
        return {"status": "ready", "prUrl": pull["url"], "state": "OPEN", "isDraft": False,
                "reason": "Validated, unchanged bot PR marked ready for review"}
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
            "This Gemini-proposed patch needs human code review. Automation does not approve or merge it.\n"
        )
        body_path = body.name
    try:
        created = command(
            "gh", "pr", "create", "--repo", repository, "--base", default_branch, "--head", branch,
            "--title",
            f"fix: review Python {candidate['package']} {candidate['latestVersion']} compatibility",
            "--body-file", body_path,
        )
    finally:
        Path(body_path).unlink(missing_ok=True)
    return {"status": "created", "prUrl": created.stdout.strip(), "branch": branch,
            "state": "OPEN", "isDraft": False}


def main() -> int:
    try:
        status = publish()
        code = 1 if status.get("status") == "failed" else 0
    except Exception as error:
        status = {"status": "failed", "reason": str(error)[:1000]}
        code = 1
    Path("compatibility-publish-status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(f"Fix PR: {status['status']}: {status.get('prUrl', status.get('reason', ''))}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
