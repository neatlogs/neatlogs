"""
Shared settings and NeatLogs initialization for detection demo workflows.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

import neatlogs  # noqa: E402


@dataclass(frozen=True)
class Settings:
    neatlogs_api_key: str
    neatlogs_endpoint: str | None
    openai_api_key: str | None
    openai_model: str
    azure_openai_api_key: str | None
    azure_openai_endpoint: str | None
    azure_openai_deployment: str | None
    azure_openai_api_version: str
    use_azure: bool


def load_settings() -> Settings:
    api_key = os.getenv("NEATLOGS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing NEATLOGS_API_KEY. Copy .env.example to .env and fill in your keys."
        )

    use_azure = os.getenv("USE_AZURE", "true").lower() in ("1", "true", "yes")

    if use_azure:
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
            raise RuntimeError(
                f"Missing Azure OpenAI env vars: {', '.join(missing)}"
            )

    return Settings(
        neatlogs_api_key=api_key,
        neatlogs_endpoint=os.getenv("NEATLOGS_ENDPOINT"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        use_azure=use_azure,
    )


def init_neatlogs(settings: Settings) -> None:
    """Initialize NeatLogs before importing workflow modules that pull in LLM libraries."""
    neatlogs.init(
        api_key=settings.neatlogs_api_key,
        endpoint=settings.neatlogs_endpoint,
        workflow_name="detection-demo",
        tags=["sdk-examples", "detection-demo", "multi-framework"],
        instrumentations=[
            "langchain",
            "openai",
            "crewai",
            "azure_ai_inference",
            "google_genai",
        ],
    )
