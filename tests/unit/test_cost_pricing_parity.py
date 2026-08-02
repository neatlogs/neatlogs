"""Parity test: bundled pricing catalog vs. LiteLLM community catalog.

Catches drift in `neatlogs/config/pricing.json` against a pinned snapshot of
LiteLLM's `model_prices_and_context_window.json`. Default test runs offline
against the local fixture; `NEATLOGS_RUN_PARITY=1` would re-fetch the pinned
URL, but the live mode is not exercised in this test file (the fixture is the
authoritative source here; live refresh is a separate workflow).

Fixture:  tests/fixtures/litellm_pricing_snapshot.json
Mapping:  tests/data/pricing_model_map.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neatlogs.cost import BuiltinProvider

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "litellm_pricing_snapshot.json"
MAPPING_PATH = REPO_ROOT / "tests" / "data" / "pricing_model_map.json"

# Per-token drift tolerance. 1% covers minor rounding in LiteLLM and per-million
# conversions on our side without masking real pricing errors.
TOLERANCE = 0.01

# Fields compared between our catalog and LiteLLM, with their canonical
# attribute names in each schema. Our `usage_types` is USD per 1M tokens.
# LiteLLM's `*_cost_per_token` is USD per token, so we multiply by 1_000_000.
FIELDS = (
    ("input", "input_cost_per_token"),
    ("output", "output_cost_per_token"),
    ("cache_read", "cache_read_input_token_cost"),
    ("cache_write", "cache_creation_input_token_cost"),
    ("reasoning", "output_cost_per_reasoning_token"),
)


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _load_mapping() -> dict:
    return json.loads(MAPPING_PATH.read_text())


def _our_rates_per_million(bundled) -> dict[str, float | None]:
    """Our schema: usage_types maps field name to USD per 1M tokens."""
    return {name: bundled.usage_types.get(name) for name, _ in FIELDS}


def _litellm_rates_per_million(entry: dict) -> dict[str, float | None]:
    """LiteLLM schema: *_cost_per_token in USD per token, multiply by 1M."""
    out = {}
    for our_name, litellm_name in FIELDS:
        val = entry.get(litellm_name)
        out[our_name] = (val * 1_000_000) if val is not None else None
    return out


def _check_pair(our_key: str, bundled_rates: dict, litellm_rates: dict) -> list[str]:
    """Compare one model's rates; return list of drift lines."""
    drift_lines: list[str] = []
    for field_name, _ in FIELDS:
        ours = bundled_rates[field_name]
        theirs = litellm_rates[field_name]
        if ours is None and theirs is None:
            continue
        if ours is None and theirs is not None and theirs != 0:
            # We don't ship the field but LiteLLM does. Document but don't fail;
            # a future v2 can add the field to our catalog.
            drift_lines.append(
                f"  {our_key} {field_name}: missing in bundled " f"(litellm={theirs:.4f})"
            )
            continue
        if theirs is None and ours is not None and ours != 0:
            # We ship a field but LiteLLM doesn't. Usually fine; surface it.
            drift_lines.append(
                f"  {our_key} {field_name}: missing in litellm " f"(bundled={ours:.4f})"
            )
            continue
        if ours is None or theirs is None:
            continue
        if ours == 0 and theirs == 0:
            continue
        # Compare. Use absolute drift relative to the larger of the two values
        # so that a 0.005 vs 0.000 drift is not masked by the larger base.
        denom = max(abs(ours), abs(theirs), 1e-9)
        drift = abs(ours - theirs) / denom
        if drift > TOLERANCE:
            drift_lines.append(
                f"  {our_key} {field_name}: bundled={ours:.4f} "
                f"litellm={theirs:.4f} drift={drift * 100:.2f}%"
            )
    return drift_lines


def test_pricing_parity():
    """Each bundled model matches the LiteLLM pinned snapshot within 1% drift."""
    fixture = _load_fixture()
    mapping = _load_mapping()
    builtin = BuiltinProvider()

    skipped: list[str] = []
    drift_lines: list[str] = []
    compared = 0
    missing_in_bundled: list[str] = []
    missing_in_litellm: list[str] = []

    for our_key, litellm_key in mapping.items():
        if our_key.startswith("_"):
            continue
        if litellm_key is None:
            skip_reason = mapping.get("_skip_reasons", {}).get(our_key, "no reason given")
            skipped.append(f"  {our_key}: {skip_reason}")
            continue
        bundled = builtin.lookup(our_key)
        if bundled is None:
            missing_in_bundled.append(our_key)
            continue
        litellm_entry = fixture.get(litellm_key)
        if litellm_entry is None:
            missing_in_litellm.append(f"{our_key} -> {litellm_key}")
            continue
        compared += 1
        bundled_rates = _our_rates_per_million(bundled)
        litellm_rates = _litellm_rates_per_million(litellm_entry)
        drift_lines.extend(_check_pair(our_key, bundled_rates, litellm_rates))

    # The test is allowed to skip entries (with a warning, not a failure) but
    # we surface the skipped list so the maintainer sees which models are not
    # covered by the parity check.
    if skipped:
        print("\nParity check skipped for these models:")
        print("\n".join(skipped))
    if missing_in_bundled:
        print(
            "\nParity check warning: mapping lists these models but they are "
            "missing from the bundled catalog:"
        )
        print("\n".join(f"  {k}" for k in missing_in_bundled))
    if missing_in_litellm:
        print(
            "\nParity check warning: mapping points to LiteLLM keys that do not "
            "exist in the pinned fixture:"
        )
        print("\n".join(f"  {k}" for k in missing_in_litellm))

    if drift_lines:
        pytest.fail(
            "\nPricing drift between bundled catalog and LiteLLM snapshot "
            f"(tolerance {TOLERANCE * 100:.0f}%):\n"
            + "\n".join(drift_lines)
            + f"\n\n{compared} models compared, {len(skipped)} skipped. "
            "Update neatlogs/config/pricing.json to match the LiteLLM source "
            "or update tests/data/pricing_model_map.json to use a different "
            "LiteLLM key."
        )

    # The test passing is silent; the human-readable breakdown lives in the
    # pytest -v output via the prints above.
    assert compared > 0, "Parity test compared 0 models; mapping is empty or wrong."
