"""Plan and optionally apply a direct weekly PyPI release."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "pyproject.toml"
LOCK = ROOT / "uv.lock"
VERSION = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
RELEASE_PATHS = ("neatlogs", "pyproject.toml", "MANIFEST.in", "README.MD")


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=check)


def version() -> str:
    project = PROJECT.read_text().split("[project]", 1)[1].split("\n[", 1)[0]
    match = re.search(r'^version = "([^"]+)"$', project, re.MULTILINE)
    if not match or not VERSION.fullmatch(match.group(1)):
        raise ValueError("[project].version must be major.minor.patch")
    return match.group(1)


def registry() -> tuple[str, str, dict]:
    request = urllib.request.Request("https://pypi.org/pypi/neatlogs/json", headers={"User-Agent": "neatlogs-weekly-release/1"})
    with urllib.request.urlopen(request, timeout=15) as response:
        data = json.load(response)
    latest = data["info"]["version"]
    if not VERSION.fullmatch(latest) or not data["releases"].get(latest):
        raise ValueError("PyPI latest release is missing or not a stable numeric version")
    uploaded = max(item["upload_time_iso_8601"] for item in data["releases"][latest])
    return latest, uploaded, data["releases"]


def compare(left: str, right: str) -> int:
    a, b = tuple(map(int, left.split("."))), tuple(map(int, right.split(".")))
    return (a > b) - (a < b)


def baseline(published: str, uploaded: str) -> tuple[str, str]:
    tag = f"v{published}"
    found = git("rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}", check=False)
    if found.returncode == 0:
        sha = found.stdout.strip()
        git("merge-base", "--is-ancestor", sha, "HEAD")
        return sha, tag
    # Older releases lack tags. This estimate is only used once, until this
    # workflow creates its first exact source tag.
    found = git("rev-list", "--first-parent", "-1", f"--before={uploaded}", "HEAD")
    sha = found.stdout.strip()
    if not sha:
        raise ValueError("No mainline commit predates the PyPI upload; create a verified release tag")
    return sha, f"mainline before {uploaded} (unverified)"


def apply_version(old: str, new: str) -> None:
    text = PROJECT.read_text()
    updated, count = re.subn(rf'(?m)^version = "{re.escape(old)}"$', f'version = "{new}"', text, count=1)
    if count != 1:
        raise ValueError("Could not update [project].version")
    lock = LOCK.read_text()
    updated_lock, count = re.subn(
        rf'(?m)^(\[\[package\]\]\nname = "neatlogs"\nversion = "){re.escape(old)}("$)',
        lambda match: f"{match.group(1)}{new}{match.group(2)}", lock, count=1,
    )
    if count != 1:
        raise ValueError("Could not update neatlogs version in uv.lock")
    PROJECT.write_text(updated)
    LOCK.write_text(updated_lock)


def plan(apply: bool) -> dict:
    current = version()
    published, uploaded, releases = registry()
    relation = compare(current, published)
    if relation < 0:
        raise ValueError(f"source {current} is behind PyPI {published}")
    if relation > 0:
        if current in releases:
            raise ValueError(f"{current} already exists on PyPI")
        target, bump, description = current, False, "version already patched in source"
        tagged = git("rev-parse", "--verify", f"refs/tags/v{current}^{{commit}}", check=False)
        retry_tag = f"v{current}" if tagged.returncode == 0 else ""
        if retry_tag:
            git("merge-base", "--is-ancestor", tagged.stdout.strip(), "HEAD")
    else:
        source, description = baseline(published, uploaded)
        diff = git("diff", "--quiet", source, "HEAD", "--", *RELEASE_PATHS, check=False)
        if diff.returncode not in (0, 1):
            raise RuntimeError(diff.stderr)
        if diff.returncode == 0:
            return {"action": "skip", "version": current, "published": published, "baseline": description}
        major, minor, patch = map(int, current.split("."))
        target, bump = f"{major}.{minor}.{patch + 1}", True
        retry_tag = ""
        if target in releases:
            raise ValueError(f"{target} already exists on PyPI")
    lock_text = LOCK.read_text()
    if f'[[package]]\nname = "neatlogs"\nversion = "{current}"' not in lock_text:
        raise ValueError("uv.lock does not match pyproject.toml")
    if apply and bump:
        apply_version(current, target)
    return {"action": "publish", "version": target, "published": published, "bump": bump,
            "commit": bump, "baseline": description, "retry_tag": retry_tag}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(plan(args.apply), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, subprocess.CalledProcessError, OSError) as error:
        print(f"weekly release planning failed: {error}", file=sys.stderr)
        sys.exit(1)
