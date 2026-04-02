"""
Configuration and neatlogs initialization for the support bot.

Must be imported first — neatlogs.init() must run before any LLM/CrewAI imports
so that OpenInference instrumentation is registered before the first LLM call.
"""

import logging
import os
import sys

# ---------------------------------------------------------------------------
# Add the local neatlogs SDK (python_sdk_new/neatlogs) to sys.path so that
# `import neatlogs` always resolves to the dev version, not a pip install.
# This file lives at: python_sdk_new/neatlogs/neatlogs/examples/neatlogs_support_bot/config.py
# Three levels up is: python_sdk_new/neatlogs/  (the repo root that contains the neatlogs/ package)
# ---------------------------------------------------------------------------
_sdk_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _sdk_root not in sys.path:
    sys.path.insert(0, _sdk_root)

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
# Suppress noisy library loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("crewai").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

logger = logging.getLogger("support_bot")

# ---------------------------------------------------------------------------
# neatlogs — must be called before any LLM/CrewAI imports
#
# The new SDK uses OTLPSpanExporter (OTLP HTTP protobuf) instead of the old
# custom JSON exporter.  Backend needs an OTLP receiver at /v1/traces before
# disable_export can be set to False.
#
# For now we log spans to files so you can inspect what gets captured.
# ---------------------------------------------------------------------------

import neatlogs

# ---------------------------------------------------------------------------
# Span file logging — set defaults before init() reads them.
# Override any of these in your .env file.
#
#   NEATLOGS_LOG_SPANS=true            Write normalized spans to SPANS_FILE
#   NEATLOGS_LOG_SPANS_FILE            Path for normalized span JSON lines
#   NEATLOGS_LOG_RAW_SPANS=true        Write raw OTel spans to RAW_SPANS_FILE
#   NEATLOGS_LOG_RAW_SPANS_FILE        Path for raw span JSON lines
#   NEATLOGS_DISABLE_EXPORT=true       Disable HTTP export (local dev / no backend)
#   NEATLOGS_API_KEY                   Your neatlogs API key
#   NEATLOGS_ENDPOINT                  Backend base URL (e.g. http://localhost:4100)
#   DEBUG=true                         Verbose SDK + CrewAI debug output
# ---------------------------------------------------------------------------
os.environ.setdefault("NEATLOGS_LOG_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_SPANS_FILE", "support_bot_spans.log")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS_FILE", "support_bot_raw_spans.log")

neatlogs.init(
    api_key="",
    # The SDK normalises this to {base_url}/v1/traces automatically.
    # The backend OTLP receiver is at POST /v1/traces (protobuf).
    endpoint=os.getenv(
        "NEATLOGS_ENDPOINT",
        "http://localhost:4100",
    ),
    workflow_name="neatlogs_support_bot",
    tags=["support-bot", "crewai", "openai"],
    # litellm: CrewAI routes all LLM calls through LiteLLM internally
    # openai:  direct OpenAI calls in KB embedding
    instrumentations=["openai", "crewai", "azure_ai_inference", "langchain"],
    debug=True,
)

# ---------------------------------------------------------------------------
# Azure OpenAI config
# ---------------------------------------------------------------------------

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_LLM_DEPLOYMENT = os.getenv("AZURE_LLM_DEPLOYMENT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION", "2023-05-15")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-08-01-preview")

if not AZURE_OPENAI_API_KEY:
    raise EnvironmentError(
        "AZURE_OPENAI_API_KEY is not set. Add it to your .env file or environment."
    )
if not AZURE_OPENAI_ENDPOINT:
    raise EnvironmentError(
        "AZURE_OPENAI_ENDPOINT is not set. Add it to your .env file or environment."
    )

# Model used by CrewAI agents (via LiteLLM)
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4.1")

# Model for KB embeddings (direct Azure OpenAI call)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
