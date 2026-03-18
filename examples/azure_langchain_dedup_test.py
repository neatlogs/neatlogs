"""
Azure + LangChain Dedup Test
=============================
Minimal reproduction for the 4x duplicate LLM span issue when using
LangChain's AzureChatOpenAI with both "langchain" and "openai" instrumentations.

Run:
    cd neatlogs-sdk && uv run --env-file ../.env python -m examples.azure_langchain_dedup_test
"""

import os
import sys

# The neatlogs-sdk directory IS the package, but named "neatlogs-sdk" not "neatlogs".
# Create a symlink at /tmp/neatlogs -> neatlogs-sdk/ so Python resolves the import.
_sdk_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_symlink = os.path.join("/tmp", "neatlogs")
if not os.path.exists(_symlink) or os.readlink(_symlink) != _sdk_dir:
    if os.path.islink(_symlink):
        os.unlink(_symlink)
    os.symlink(_sdk_dir, _symlink)
sys.path.insert(0, "/tmp")
# Remove stale neatlogs paths (e.g., from editable installs) so /tmp/neatlogs wins
sys.path = ["/tmp"] + [p for p in sys.path[1:] if "neatlogs" not in p or "site-packages" in p]

from dotenv import load_dotenv

load_dotenv(os.path.join(_sdk_dir, "..", ".env"))

os.environ["NEATLOGS_LOG_RAW_SPANS"] = "true"
os.environ["NEATLOGS_LOG_RAW_SPANS_FILE"] = "spans_raw_azure_dedup_test.log"
os.environ["NEATLOGS_LOG_SPANS"] = "true"
os.environ["NEATLOGS_LOG_SPANS_FILE"] = "spans_azure_dedup_test.log"
os.environ["NEATLOGS_DISABLE_EXPORT"] = "true"

import neatlogs

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY", "3Rlq5Ltv7d59It_zccsGxScE3liQ0QBG"),
    endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://52.53.40.222:4100/api/data/v4/batch"),
    workflow_name="azure-langchain-dedup-test",
    tags=["dedup-test"],
    instrumentations=["langchain", "openai"],  # Both layers -> creates duplicates
    debug=True,
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
)

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant."), ("user", "{query}")]
)
chain = prompt | model | StrOutputParser()

# Test 1: Chain call with user's neatlogs.trace() wrapper (replicates real usage)
# This creates the outer "langchain_get_ai_response" LLM span seen in production
print("=== Test 1: Chain call with neatlogs.trace() wrapper ===")
system_tpl = neatlogs.PromptTemplate("You are a helpful assistant.")

with neatlogs.trace(
    name="langchainAzureChatOpenAI", kind="LLM", prompt_template=system_tpl
):
    result = chain.invoke({"query": "What is 2+2? Answer in one word."})
print(f"Result: {result}")

# Test 2: Direct model.invoke with trace wrapper
print("\n=== Test 2: Direct model.invoke with neatlogs.trace() wrapper ===")
with neatlogs.trace(name="directAzureInvoke", kind="LLM"):
    result2 = model.invoke("Say hello in one word.")
print(f"Result2: {result2.content}")

# Test 3: Chain call WITHOUT trace wrapper (tests auto-instrumentation only)
print("\n=== Test 3: Chain call without trace wrapper ===")
result3 = chain.invoke({"query": "What is the capital of France? One word."})
print(f"Result3: {result3}")

neatlogs.flush()
neatlogs.shutdown()

print("\n" + "=" * 60)
print("Done. Analyze these files for duplicate LLM spans:")
print("  Raw spans:       spans_raw_azure_dedup_test.log")
print("  Processed spans: spans_azure_dedup_test.log")
print("=" * 60)
