"""
End-to-end pricing test with Azure OpenAI (gpt-5-nano).

Verifies that:
1. Azure platform is detected
2. Azure-specific pricing is used (not direct API)
3. Cost fields are computed correctly
4. Cache-read tokens (if present) get the discounted rate

Usage:
    cd neatlogs && uv run --env-file .env python -m examples.test_pricing_azure_e2e
"""

import os
import sys
import json

# Disable export — we only want to inspect the processed spans locally
os.environ["NEATLOGS_DISABLE_EXPORT"] = "true"
os.environ["NEATLOGS_LOG_SPANS"] = "true"
os.environ["NEATLOGS_LOG_SPANS_FILE"] = "spans_pricing_azure_e2e.log"
os.environ["NEATLOGS_LOG_RAW_SPANS"] = "true"
os.environ["NEATLOGS_LOG_RAW_SPANS_FILE"] = "spans_raw_pricing_azure_e2e.log"

import neatlogs

neatlogs.init(
    api_key="test-key",
    endpoint="http://localhost:4100/api/data/v4/batch",
    workflow_name="pricing-azure-e2e-test",
    tags=["pricing-test"],
    instrumentations=["openai"],
    debug=True,
)

from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5-nano")
print(f"\n{'='*60}")
print(f"Testing Azure OpenAI pricing with deployment: {deployment}")
print(f"{'='*60}")

# ── Test 1: Simple call ──
print("\n--- Test 1: Simple Azure call ---")
resp = client.chat.completions.create(
    model=deployment,
    messages=[{"role": "user", "content": "What is 2+2? Answer in one word."}],
    max_completion_tokens=10,
)
print(f"Response: {resp.choices[0].message.content}")
print(f"Model:    {resp.model}")
print(f"Tokens:   prompt={resp.usage.prompt_tokens}, completion={resp.usage.completion_tokens}, total={resp.usage.total_tokens}")
if hasattr(resp.usage, 'prompt_tokens_details') and resp.usage.prompt_tokens_details:
    print(f"Cache:    cached_tokens={getattr(resp.usage.prompt_tokens_details, 'cached_tokens', 0)}")

# ── Test 2: Same prompt again (may trigger cache) ──
print("\n--- Test 2: Repeated prompt (potential cache hit) ---")
resp2 = client.chat.completions.create(
    model=deployment,
    messages=[{"role": "user", "content": "What is 2+2? Answer in one word."}],
    max_completion_tokens=10,
)
print(f"Response: {resp2.choices[0].message.content}")
print(f"Tokens:   prompt={resp2.usage.prompt_tokens}, completion={resp2.usage.completion_tokens}")
if hasattr(resp2.usage, 'prompt_tokens_details') and resp2.usage.prompt_tokens_details:
    print(f"Cache:    cached_tokens={getattr(resp2.usage.prompt_tokens_details, 'cached_tokens', 0)}")

# ── Flush and check ──
neatlogs.flush()
neatlogs.shutdown()

print(f"\n{'='*60}")
print("Checking processed spans for cost fields...")
print(f"{'='*60}")

try:
    with open("spans_pricing_azure_e2e.log") as f:
        lines = f.readlines()

    llm_spans = []
    for line in lines:
        try:
            span = json.loads(line.strip())
            attrs = span.get("attributes", {})
            kind = attrs.get("neatlogs.span.kind", "").lower()
            if kind == "llm":
                llm_spans.append(span)
        except json.JSONDecodeError:
            continue

    if not llm_spans:
        print("WARNING: No LLM spans found in processed log!")
    else:
        for i, span in enumerate(llm_spans):
            attrs = span.get("attributes", {})
            print(f"\n  LLM Span #{i+1}: {span.get('name', '?')}")
            print(f"    model:           {attrs.get('neatlogs.llm.model_name', '?')}")
            print(f"    platform:        {attrs.get('neatlogs.platform', '(not set)')}")
            print(f"    provider:        {attrs.get('neatlogs.provider', '(not set)')}")
            print(f"    prompt_tokens:   {attrs.get('neatlogs.llm.token_count.prompt', '?')}")
            print(f"    compl_tokens:    {attrs.get('neatlogs.llm.token_count.completion', '?')}")
            print(f"    cache_read:      {attrs.get('neatlogs.llm.token_count.cache_read', '(none)')}")
            print(f"    cost.prompt:     {attrs.get('neatlogs.llm.cost.prompt', '(none)')}")
            print(f"    cost.completion: {attrs.get('neatlogs.llm.cost.completion', '(none)')}")
            print(f"    cost.total:      {attrs.get('neatlogs.llm.cost.total', '(none)')}")
            print(f"    cost.cache_read: {attrs.get('neatlogs.llm.cost.cache_read', '(none)')}")

            # Manual verification
            prompt_t = attrs.get('neatlogs.llm.token_count.prompt')
            compl_t = attrs.get('neatlogs.llm.token_count.completion')
            total_cost = attrs.get('neatlogs.llm.cost.total')
            if prompt_t and compl_t and total_cost:
                # Azure gpt-5-nano global_standard: prompt=0.00005, completion=0.0004
                expected_prompt_cost = (float(prompt_t) / 1000) * 0.00005
                expected_compl_cost = (float(compl_t) / 1000) * 0.0004
                expected_total = round(expected_prompt_cost + expected_compl_cost, 8)
                cache_read = float(attrs.get('neatlogs.llm.token_count.cache_read', 0) or 0)
                if cache_read > 0:
                    # Adjust: uncached at full rate, cached at discount
                    expected_prompt_cost = ((float(prompt_t) - cache_read) / 1000) * 0.00005 + (cache_read / 1000) * 0.000005
                    expected_total = round(expected_prompt_cost + expected_compl_cost, 8)
                print(f"    --- MANUAL CHECK ---")
                print(f"    expected_total:  {expected_total}")
                match = abs(float(total_cost) - expected_total) < 1e-8
                print(f"    MATCH: {'YES' if match else 'NO <<<< MISMATCH!'}")

    print(f"\n  Total LLM spans: {len(llm_spans)}")

except FileNotFoundError:
    print("Log file not found — spans may not have been flushed.")

print(f"\n  Raw log: spans_raw_pricing_azure_e2e.log")
print(f"  Processed log: spans_pricing_azure_e2e.log")
print(f"{'='*60}")
