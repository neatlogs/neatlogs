"""Refresh the LiteLLM pricing snapshot used by the parity test.

Fetches the LiteLLM community catalog at the SHA pinned in
`tests/data/pricing_model_map.json` and writes the result to
`tests/fixtures/litellm_pricing_snapshot.json`. The parity test loads the
fixture from disk; the live URL is not hit during pytest.

Usage:
    python scripts/refresh_parity_fixture.py

When to run:
    When you change `litellm_pinned_sha` in
    `tests/data/pricing_model_map.json` and want to advance the snapshot to a
    new LiteLLM commit. Or on a schedule (e.g. weekly) to keep the fixture
    current with the upstream catalog.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
MAPPING_PATH = REPO_ROOT / "tests" / "data" / "pricing_model_map.json"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "litellm_pricing_snapshot.json"
LITELLM_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/BerriAI/litellm/{sha}/"
    "model_prices_and_context_window.json"
)


def _load_pinned_sha() -> str:
    mapping = json.loads(MAPPING_PATH.read_text())
    sha = mapping.get("_meta", {}).get("litellm_pinned_sha", "")
    if not sha:
        sys.exit(
            "tests/data/pricing_model_map.json has no litellm_pinned_sha; "
            "set one in _meta before refreshing."
        )
    return sha


def _fetch(sha: str) -> bytes:
    url = LITELLM_URL_TEMPLATE.format(sha=sha)
    req = Request(url, headers={"User-Agent": "neatlogs-parity-refresh/1.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            return resp.read()
    except (HTTPError, URLError) as exc:
        sys.exit(f"Failed to fetch {url}: {exc}")


def main() -> int:
    sha = _load_pinned_sha()
    print(f"Fetching LiteLLM pricing snapshot at sha {sha[:12]}...")
    data = _fetch(sha)
    parsed = json.loads(data)
    FIXTURE_PATH.write_text(json.dumps(parsed, indent=2, sort_keys=True))
    print(
        f"Wrote {FIXTURE_PATH.relative_to(REPO_ROOT)} "
        f"({len(data):,} bytes, {len(parsed)} model entries)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
