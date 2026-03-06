"""
GobbleCube Agent - Shared Configuration
========================================
Initialises the Azure OpenAI LLM client and Neatlogs SDK.
All agents import `llm` from here so credentials are loaded once.
"""

import os
import sys

# Allow running directly from the gobblecube/ directory OR from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dotenv import load_dotenv

# Load .env from this directory first, then fall back to process env.
# `override=True` ensures this example's local env is deterministic and
# not accidentally shadowed by a parent shell export.
_env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(_env_path if os.path.exists(_env_path) else None, override=True)

# ---------------------------------------------------------------------------
# Neatlogs – initialise ONCE before any LLM import/call
# ---------------------------------------------------------------------------
import neatlogs

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("NEATLOGS_LOG_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_METRICS", "true")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_SPANS_FILE", "spans_gobblecube.log")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS_FILE", "spans_raw_gobblecube.log")
os.environ.setdefault("NEATLOGS_LOG_METRICS_FILE", "metrics_gobblecube.log")

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY"),
    endpoint="http://localhost:4100/api/data/v4/batch",
    tags=["gobblecube", "langgraph", "demo"],
    instrumentations=["langchain", "azure_ai_inference"],   # auto-instruments LangChain + LangGraph
    workflow_name="gobblecube_v7",
    debug=True,
)

# ---------------------------------------------------------------------------
# Azure OpenAI LLM (used by all agents)
# ---------------------------------------------------------------------------
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0,
)

print(
    "[gobblecube] Azure config:"
    f" endpoint={os.getenv('AZURE_OPENAI_ENDPOINT')}"
    f" deployment={os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}"
    f" api_version={os.getenv('AZURE_OPENAI_API_VERSION', '2024-08-01-preview')}"
)

BRAND = "Demo Brand"
CATEGORY = "health_snacks"
