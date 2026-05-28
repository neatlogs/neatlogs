"""
Shared settings and NeatLogs initialization for the GobbleCube demo.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

_env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(_env_path if os.path.exists(_env_path) else None, override=False)

import neatlogs  # noqa: E402

llm = None

BRAND = "Demo Brand"
CATEGORY = "health_snacks"


@dataclass(frozen=True)
class Settings:
    neatlogs_api_key: str
    neatlogs_endpoint: str | None
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_deployment: str
    azure_openai_api_version: str


def load_settings() -> Settings:
    api_key = os.getenv("NEATLOGS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing NEATLOGS_API_KEY. Copy .env.example to .env and fill in your keys."
        )

    missing = [
        name
        for name, val in (
            ("AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY")),
            ("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT")),
            ("AZURE_OPENAI_DEPLOYMENT_NAME", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")),
        )
        if not val
    ]
    if missing:
        raise RuntimeError(f"Missing Azure OpenAI env vars: {', '.join(missing)}")

    return Settings(
        neatlogs_api_key=api_key,
        neatlogs_endpoint=os.getenv("NEATLOGS_ENDPOINT"),
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
    )


def init_neatlogs(settings: Settings) -> None:
    """Initialize NeatLogs before importing modules that pull in LangChain."""
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

    neatlogs.init(
        api_key=settings.neatlogs_api_key,
        endpoint=settings.neatlogs_endpoint,
        workflow_name="gobblecube",
        tags=["sdk-examples", "gobblecube", "langgraph", "multi-agent"],
        instrumentations=["langchain", "openai", "azure_ai_inference"],
    )


def setup_llm(settings: Settings):
    """Create the shared Azure OpenAI client after neatlogs.init()."""
    global llm
    from langchain_openai import AzureChatOpenAI

    llm = AzureChatOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        azure_deployment=settings.azure_openai_deployment,
        api_version=settings.azure_openai_api_version,
    )
    return llm
