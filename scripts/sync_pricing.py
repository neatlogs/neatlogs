#!/usr/bin/env python3
"""
Sync LLM pricing data from litellm's community-maintained model database.

Usage:
    python scripts/sync_pricing.py                    # writes neatlogs/config/pricing.json
    python scripts/sync_pricing.py --dry-run          # print stats, don't write
    python scripts/sync_pricing.py --output /tmp/p.json

Source: https://github.com/BerriAI/litellm  (model_prices_and_context_window.json)

The SDK loads pricing.json at startup as a static file — this script is a
developer/CI tool, never called at runtime.
"""

import argparse
import json
import math
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from datetime import date

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "neatlogs" / "config" / "pricing.json"

# ---------------------------------------------------------------------------
# Provider / platform classification
# ---------------------------------------------------------------------------

# litellm_provider → neatlogs pricing section
PROVIDER_SECTION_MAP = {
    # Direct API providers → "chat" (flat pricing)
    "openai": "chat",
    "anthropic": "chat",
    "deepseek": "chat",
    "groq": "chat",
    "mistral": "chat",
    "cohere": "chat",
    "cohere_chat": "chat",
    "ai21": "chat",
    "fireworks_ai": "chat",
    "together_ai": "chat",
    "perplexity": "chat",
    "xai": "chat",
    "sambanova": "chat",
    "cerebras": "chat",
    "deepinfra": "chat",
    "openrouter": "chat",
    "anyscale": "chat",
    "replicate": "chat",
    "nlp_cloud": "chat",
    "aleph_alpha": "chat",
    "text-completion-openai": "chat",
    "text-completion-codestral": "chat",
    "codestral": "chat",
    "gemini": "chat",
    "voyage": "embeddings",

    # Platform providers → dedicated sections
    "azure": "azure_openai",
    "azure_ai": "azure_openai",
    "bedrock": "bedrock",
    "bedrock_converse": "bedrock",
    "sagemaker": "bedrock",
    "vertex_ai": "vertex_ai",
    "vertex_ai-chat-models": "vertex_ai",
    "vertex_ai-text-models": "vertex_ai",
    "vertex_ai-language-models": "vertex_ai",
    "vertex_ai-vision-models": "vertex_ai",
    "vertex_ai-anthropic_models": "vertex_ai",
    "vertex_ai-llama-models": "vertex_ai",
    "vertex_ai-mistral_models": "vertex_ai",
    "vertex_ai-ai21_models": "vertex_ai",
    "vertex_ai-embedding-models": "vertex_ai",
    "vertex_ai-image-models": "vertex_ai",
}

# litellm key prefixes that should be stripped to get the clean model name
PROVIDER_PREFIXES = [
    "azure/", "azure_ai/",
    "bedrock/", "bedrock_converse/",
    "vertex_ai/", "vertex_ai-chat-models/",
    "vertex_ai-text-models/", "vertex_ai-language-models/",
    "vertex_ai-vision-models/", "vertex_ai-anthropic_models/",
    "vertex_ai-llama-models/", "vertex_ai-mistral_models/",
    "vertex_ai-ai21_models/", "vertex_ai-embedding-models/",
    "vertex_ai-image-models/",
    "openai/", "anthropic/", "groq/", "mistral/", "cohere/", "cohere_chat/",
    "deepseek/", "fireworks_ai/", "together_ai/", "perplexity/", "xai/",
    "anyscale/", "replicate/", "nlp_cloud/", "aleph_alpha/",
    "text-completion-openai/", "text-completion-codestral/", "codestral/",
    "sambanova/", "cerebras/", "deepinfra/", "openrouter/", "ai21/",
    "voyage/", "gemini/", "sagemaker/",
]

# Azure regional suffixes → neatlogs tier names
AZURE_REGION_MAP = {
    "/eu/": "eu_standard",
    "/us/": "us_standard",
    "/au/": "au_standard",
    "/global/": "global_standard",
    # Default (no region) → "global_standard"
}


def _per_token_to_per_1k(cost_per_token):
    """Convert litellm's per-token price to our per-1K-token price."""
    if cost_per_token is None or cost_per_token == 0:
        return None
    return round(cost_per_token * 1000, 10)


def _strip_provider_prefix(key):
    """Remove provider prefix from litellm key to get clean model name."""
    for prefix in sorted(PROVIDER_PREFIXES, key=len, reverse=True):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _detect_azure_tier(litellm_key):
    """Detect Azure region/tier from the litellm key pattern like azure/eu/gpt-4o."""
    for pattern, tier in AZURE_REGION_MAP.items():
        if pattern in litellm_key:
            return tier
    return "global_standard"


def _build_price_entry(entry):
    """Build a neatlogs price dict from a litellm model entry."""
    result = {}

    # Core pricing
    prompt = _per_token_to_per_1k(entry.get("input_cost_per_token"))
    completion = _per_token_to_per_1k(entry.get("output_cost_per_token"))

    if prompt is not None:
        result["promptPrice"] = prompt
    if completion is not None:
        result["completionPrice"] = completion

    # Cache pricing
    cache_read = _per_token_to_per_1k(entry.get("cache_read_input_token_cost"))
    cache_write = _per_token_to_per_1k(entry.get("cache_creation_input_token_cost"))
    if cache_read is not None:
        result["cacheReadPrice"] = cache_read
    if cache_write is not None:
        result["cacheWritePrice"] = cache_write

    # Tiered pricing — detect threshold dynamically from ANY litellm field name
    # containing "_above_Nk_tokens". Handles 200k, 272k, or any future threshold.
    import re as _re
    tier_threshold = None
    tier_suffix = None  # e.g., "_above_200k_tokens"
    for key in entry:
        m = _re.search(r"_above_(\d+k?)_tokens$", key)
        if m and "1hr" not in key:
            raw = m.group(1)
            expanded = raw.replace("k", "000").replace("m", "000000")
            try:
                tier_threshold = int(expanded)
                tier_suffix = f"_above_{raw}_tokens"
            except ValueError:
                pass
            break

    if tier_suffix:
        prompt_above_tier = _per_token_to_per_1k(
            entry.get(f"input_cost_per_token{tier_suffix}"))
        completion_above_tier = _per_token_to_per_1k(
            entry.get(f"output_cost_per_token{tier_suffix}"))
        cache_read_above_tier = _per_token_to_per_1k(
            entry.get(f"cache_read_input_token_cost{tier_suffix}"))
        cache_write_above_tier = _per_token_to_per_1k(
            entry.get(f"cache_creation_input_token_cost{tier_suffix}"))

        if tier_threshold and (prompt_above_tier is not None or completion_above_tier is not None
                               or cache_read_above_tier is not None):
            result["tierTokenThreshold"] = tier_threshold
        if prompt_above_tier is not None:
            result["promptPriceAboveTier"] = prompt_above_tier
        if completion_above_tier is not None:
            result["completionPriceAboveTier"] = completion_above_tier
        if cache_read_above_tier is not None:
            result["cacheReadPriceAboveTier"] = cache_read_above_tier
        if cache_write_above_tier is not None:
            result["cacheWritePriceAboveTier"] = cache_write_above_tier

    # Cache write pricing above 1hr (Anthropic-specific, independent of token tier)
    cache_write_above_1hr = _per_token_to_per_1k(entry.get("cache_creation_input_token_cost_above_1hr"))
    if cache_write_above_1hr is not None:
        result["cacheWritePriceAbove1hr"] = cache_write_above_1hr

    # Audio tokens
    prompt_audio = _per_token_to_per_1k(entry.get("input_cost_per_audio_token"))
    completion_audio = _per_token_to_per_1k(entry.get("output_cost_per_audio_token"))
    if prompt_audio is not None:
        result["promptAudioPrice"] = prompt_audio
    if completion_audio is not None:
        result["completionAudioPrice"] = completion_audio

    # Batch pricing
    prompt_batch = _per_token_to_per_1k(entry.get("input_cost_per_token_batches"))
    completion_batch = _per_token_to_per_1k(entry.get("output_cost_per_token_batches"))
    if prompt_batch is not None:
        result["promptBatchPrice"] = prompt_batch
    if completion_batch is not None:
        result["completionBatchPrice"] = completion_batch

    # Context metadata
    if entry.get("max_input_tokens"):
        result["maxInputTokens"] = entry["max_input_tokens"]
    if entry.get("max_output_tokens"):
        result["maxOutputTokens"] = entry["max_output_tokens"]

    return result


def _build_embedding_entry(entry):
    """Build embedding price from litellm entry."""
    cost = _per_token_to_per_1k(entry.get("input_cost_per_token"))
    if cost is not None:
        return cost
    return None


def fetch_litellm_data():
    """Fetch and parse the litellm pricing JSON."""
    print(f"Fetching {LITELLM_URL} ...")
    req = urllib.request.Request(LITELLM_URL, headers={"User-Agent": "neatlogs-sync/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    print(f"  Fetched {len(data)} entries")
    return data


def _build_rerank_entry(entry):
    """Build rerank price from litellm entry. Rerankers are priced per search query."""
    cost = entry.get("input_cost_per_query")
    if cost is not None:
        return {"costPerQuery": cost}
    # Some rerankers use per-token pricing
    input_cost = _per_token_to_per_1k(entry.get("input_cost_per_token"))
    if input_cost is not None:
        return {"promptPrice": input_cost}
    return None


def build_pricing(raw_data):
    """Transform litellm data into neatlogs pricing.json format."""
    chat = {}
    embeddings = {}
    rerank = {}
    azure_chat = {}
    azure_embeddings = {}
    bedrock_chat = {}
    bedrock_embeddings = {}
    vertex_chat = {}
    vertex_embeddings = {}

    # Detailed skip tracking
    from collections import Counter
    skipped_by_mode = Counter()
    skipped_no_provider = 0
    skipped_no_price = 0
    duplicates_merged = 0

    # Modes we handle (per-token or per-query pricing)
    handled_modes = {"chat", "completion", "embedding", "rerank"}
    # Modes we intentionally skip (different pricing units)
    known_skip_modes = {
        "image_generation", "image_edit", "audio_transcription", "audio_speech",
        "video_generation", "search", "ocr", "moderations", "moderation",
        "realtime", "vector_store", "responses",
    }

    for litellm_key, entry in raw_data.items():
        if litellm_key.startswith("sample_spec") or litellm_key.startswith("_"):
            continue
        if not isinstance(entry, dict):
            continue

        mode = entry.get("mode", "")
        provider = entry.get("litellm_provider", "")

        # Skip non-token-priced modes
        if mode and mode not in handled_modes:
            skipped_by_mode[mode] += 1
            continue

        section = PROVIDER_SECTION_MAP.get(provider)
        if not section:
            skipped_no_provider += 1
            continue

        model_name = _strip_provider_prefix(litellm_key)
        # Remove duplicate region segments for azure (e.g., "eu/gpt-4o" → "gpt-4o")
        for region_slug in ("/eu/", "/us/", "/au/", "/global/"):
            if region_slug[1:] in model_name:
                model_name = model_name.replace(region_slug[1:], "")

        # --- Rerank models ---
        if mode == "rerank":
            rerank_entry = _build_rerank_entry(entry)
            if rerank_entry:
                rerank[model_name] = rerank_entry
            else:
                skipped_no_price += 1
            continue

        # --- Embedding models ---
        if mode == "embedding":
            emb_price = _build_embedding_entry(entry)
            if emb_price is None:
                skipped_no_price += 1
                continue

            if section == "chat":
                embeddings[model_name] = emb_price
            elif section == "azure_openai":
                azure_embeddings[model_name] = emb_price
            elif section == "bedrock":
                bedrock_embeddings[model_name] = emb_price
            elif section == "vertex_ai":
                vertex_embeddings[model_name] = emb_price
            continue

        # --- Chat / completion models ---
        price_entry = _build_price_entry(entry)
        if not price_entry.get("promptPrice") and not price_entry.get("completionPrice"):
            skipped_no_price += 1
            continue

        if section == "chat":
            if model_name in chat:
                existing = chat[model_name]
                if len(price_entry) > len(existing):
                    chat[model_name] = price_entry
                duplicates_merged += 1
            else:
                chat[model_name] = price_entry

        elif section == "azure_openai":
            tier = _detect_azure_tier(litellm_key)
            if model_name not in azure_chat:
                azure_chat[model_name] = {}
            azure_chat[model_name][tier] = price_entry

        elif section == "bedrock":
            if model_name in bedrock_chat:
                existing = bedrock_chat[model_name]
                if len(price_entry) > len(existing):
                    bedrock_chat[model_name] = price_entry
                duplicates_merged += 1
            else:
                bedrock_chat[model_name] = price_entry

        elif section == "vertex_ai":
            if model_name in vertex_chat:
                existing = vertex_chat[model_name]
                if len(price_entry) > len(existing):
                    vertex_chat[model_name] = price_entry
                duplicates_merged += 1
            else:
                vertex_chat[model_name] = price_entry

    total_output = (len(chat) + len(embeddings) + len(rerank)
                    + len(azure_chat) + len(azure_embeddings)
                    + len(bedrock_chat) + len(bedrock_embeddings)
                    + len(vertex_chat) + len(vertex_embeddings))
    total_skipped = sum(skipped_by_mode.values()) + skipped_no_provider + skipped_no_price

    return {
        "_metadata": {
            "source": "litellm/model_prices_and_context_window.json (community-maintained)",
            "source_url": LITELLM_URL,
            "last_synced": date.today().isoformat(),
            "note": "Prices are per 1,000 tokens. Auto-generated by scripts/sync_pricing.py — do not edit manually.",
            "stats": {
                "total_models_output": total_output,
                "duplicates_merged": duplicates_merged,
                "skipped_total": total_skipped,
                "skipped_by_mode": dict(skipped_by_mode.most_common()),
                "skipped_no_provider": skipped_no_provider,
                "skipped_no_price": skipped_no_price,
                "breakdown": {
                    "chat": len(chat),
                    "embeddings": len(embeddings),
                    "rerank": len(rerank),
                    "azure_chat": len(azure_chat),
                    "azure_embeddings": len(azure_embeddings),
                    "bedrock_chat": len(bedrock_chat),
                    "bedrock_embeddings": len(bedrock_embeddings),
                    "vertex_chat": len(vertex_chat),
                    "vertex_embeddings": len(vertex_embeddings),
                },
            },
        },
        "chat": dict(sorted(chat.items())),
        "embeddings": dict(sorted(embeddings.items())),
        "rerank": dict(sorted(rerank.items())),
        "azure_openai": {
            "_metadata": {
                "note": "Azure pricing varies by region/tier. Keys are model names, values contain tier-specific pricing.",
                "tiers": ["global_standard", "eu_standard", "us_standard", "au_standard"],
            },
            "chat": dict(sorted(azure_chat.items())),
            "embeddings": dict(sorted(azure_embeddings.items())),
        },
        "bedrock": {
            "_metadata": {
                "note": "AWS Bedrock pricing. Model keys use Bedrock model IDs (e.g., anthropic.claude-3-5-sonnet-20241022-v2:0).",
            },
            "chat": dict(sorted(bedrock_chat.items())),
            "embeddings": dict(sorted(bedrock_embeddings.items())),
        },
        "vertex_ai": {
            "_metadata": {
                "note": "Google Vertex AI pricing.",
            },
            "chat": dict(sorted(vertex_chat.items())),
            "embeddings": dict(sorted(vertex_embeddings.items())),
        },
    }


def print_stats(pricing):
    meta = pricing["_metadata"]["stats"]
    bd = meta["breakdown"]
    print(f"\n  Stats:")
    print(f"    Total models output: {meta['total_models_output']}")
    print(f"    Duplicates merged:   {meta['duplicates_merged']}")
    print(f"    Skipped total:       {meta['skipped_total']}")
    print(f"      - no provider:     {meta['skipped_no_provider']}")
    print(f"      - no price:        {meta['skipped_no_price']}")
    if meta.get("skipped_by_mode"):
        for mode, count in meta["skipped_by_mode"].items():
            print(f"      - mode={mode}: {count}")
    print(f"\n  Breakdown:")
    print(f"    chat:               {bd['chat']}")
    print(f"    embeddings:         {bd['embeddings']}")
    print(f"    rerank:             {bd['rerank']}")
    print(f"    azure_chat:         {bd['azure_chat']}")
    print(f"    azure_embeddings:   {bd['azure_embeddings']}")
    print(f"    bedrock_chat:       {bd['bedrock_chat']}")
    print(f"    bedrock_embeddings: {bd['bedrock_embeddings']}")
    print(f"    vertex_chat:        {bd['vertex_chat']}")
    print(f"    vertex_embeddings:  {bd['vertex_embeddings']}")
    total_check = sum(bd.values())
    print(f"    ─────────────────────────────")
    print(f"    sum of breakdown:   {total_check}  {'✓' if total_check == meta['total_models_output'] else '✗ MISMATCH'}")

    # Show sample entries
    print(f"\n  Sample chat entries:")
    for name in list(pricing["chat"].keys())[:5]:
        entry = pricing["chat"][name]
        p = entry.get("promptPrice", "?")
        c = entry.get("completionPrice", "?")
        cr = entry.get("cacheReadPrice")
        tier_thresh = entry.get("tierTokenThreshold")
        extras = []
        if cr:
            extras.append(f"cache_read={cr}")
        if tier_thresh:
            extras.append(f"tiered_{tier_thresh//1000}k={entry.get('promptPriceAboveTier')}")
        extra_str = f" [{', '.join(extras)}]" if extras else ""
        print(f"    {name}: prompt={p}, completion={c}{extra_str}")

    print(f"\n  Sample rerank entries:")
    for name in list(pricing["rerank"].keys())[:5]:
        entry = pricing["rerank"][name]
        print(f"    {name}: {entry}")

    print(f"\n  Sample Azure entries:")
    for name in list(pricing["azure_openai"]["chat"].keys())[:3]:
        tiers = pricing["azure_openai"]["chat"][name]
        tier_names = list(tiers.keys())
        print(f"    {name}: tiers={tier_names}")

    print(f"\n  Sample Bedrock entries:")
    for name in list(pricing["bedrock"]["chat"].keys())[:3]:
        entry = pricing["bedrock"]["chat"][name]
        p = entry.get("promptPrice", "?")
        c = entry.get("completionPrice", "?")
        cr = entry.get("cacheReadPrice")
        extra = f" [cache_read={cr}]" if cr else ""
        print(f"    {name}: prompt={p}, completion={c}{extra}")


def main():
    parser = argparse.ArgumentParser(description="Sync LLM pricing from litellm")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing")
    args = parser.parse_args()

    raw_data = fetch_litellm_data()
    pricing = build_pricing(raw_data)
    print_stats(pricing)

    if args.dry_run:
        print("\n  Dry run — not writing file.")
        return

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pricing, f, indent=2)
        f.write("\n")

    size_kb = output_path.stat().st_size / 1024
    print(f"\n  Wrote {output_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
