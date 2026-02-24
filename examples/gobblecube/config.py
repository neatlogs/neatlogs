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

# Load .env from this directory first, then fall back to repo root
_env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(_env_path if os.path.exists(_env_path) else None)

# ---------------------------------------------------------------------------
# Neatlogs – initialise ONCE before any LLM import/call
# ---------------------------------------------------------------------------
import neatlogs

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY", "test-key"),
    endpoint=os.getenv("NEATLOGS_ENDPOINT", "https://api.neatlogs.com/v4/batch"),
    tags=["gobblecube", "langgraph", "demo"],
    instrumentations=["langchain"],   # auto-instruments LangChain + LangGraph
    debug=False,
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

BRAND = "Demo Brand"
CATEGORY = "health_snacks"
