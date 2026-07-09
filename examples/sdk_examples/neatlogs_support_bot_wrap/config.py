"""Configuration for the support bot."""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
for noisy in (
    "httpx",
    "openai",
    "litellm",
    "crewai",
    "azure.core",
    "azure.core.pipeline.policies.http_logging_policy",
):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("support_bot")

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
    raise EnvironmentError("AZURE_OPENAI_API_KEY is not set.")
if not AZURE_OPENAI_ENDPOINT:
    raise EnvironmentError("AZURE_OPENAI_ENDPOINT is not set.")

AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4.1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
