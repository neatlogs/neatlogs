"""
Configuration for Detection Demo Workflows
===========================================
Shared settings across all 3 workflows (LangGraph, CrewAI, LangChain).
No external dependencies (Qdrant, Cohere) - uses simulated retrieval.
"""

import os
import sys
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env file from current directory
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Add parent directory to path for neatlogs import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import neatlogs

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("NEATLOGS_LOG_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_METRICS", "true")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_SPANS_FILE", "spans_detection_demo_new.log")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS_FILE", "spans_raw_detection_demo_new.log")
os.environ.setdefault("NEATLOGS_LOG_METRICS_FILE", "metrics_detection_demo.log")


@dataclass(frozen=True)
class Settings:
    """Shared settings for all workflows."""
    
    # Neatlogs Configuration
    neatlogs_api_key: str
    neatlogs_endpoint: str
    
    # LLM Provider
    openai_api_key: str
    openai_model: str
    
    # Azure OpenAI (optional)
    azure_openai_api_key: str = None
    azure_openai_endpoint: str = None
    azure_openai_deployment: str = None
    azure_openai_api_version: str = "2025-01-01-preview"
    use_azure: bool = False
    
    # Workflow Settings
    debug: bool = True


def load_settings() -> Settings:
    """Load settings from environment variables."""
    
    # Required env vars check
    # required_vars = [
    #     "NEATLOGS_API_KEY",
    #     "OPENAI_API_KEY",
    # ]
    
    # missing = [var for var in required_vars if not os.getenv(var)]
    # if missing:
    #     raise RuntimeError(
    #         f"Missing required environment variables: {', '.join(missing)}\n"
    #         f"Please copy .env.example to .env and fill in your keys."
    #     )
    
    # Check if Azure OpenAI should be used
    use_azure = True
    
    return Settings(
        # Neatlogs
        neatlogs_endpoint=os.getenv(
            "NEATLOGS_ENDPOINT",
            "https://staging-api.neatlogs.com/api/data/v4/batch"
        ),
        
        # LLM
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        
        
        # Azure OpenAI
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        use_azure=use_azure,
        neatlogs_api_key="",
    )


def init_neatlogs(settings: Settings):
    """Initialize Neatlogs SDK with instrumentation."""

    
    # neatlogs.init(
    #     api_key="",
    #     endpoint="http://localhost:4100/api/data/v4/batch",
    #     workflow_name="sales-qualified-1",
    #     tags=["detection-demo", "multi-framework"],
    #     instrumentations=["langchain", "openai", "crewai", "azure_ai_inference"],
    #     debug=True,
    # )
    neatlogs.init(
        api_key="",
        endpoint="http://52.53.40.222:4100/api/data/v4/batch",
        workflow_name="sales-nsfw-1",
        tags=["detection-demo", "multi-framework"],
        instrumentations=["langchain", "openai", "crewai", "azure_ai_inference"],
        debug=True,
    )
    
    print(f"✓ Neatlogs initialized")
    print(f"  - Endpoint: {settings.neatlogs_endpoint}")
    print(f"  - Tags: detection-demo, multi-framework")
    if settings.use_azure:
        print(f"  - LLM: Azure OpenAI ({settings.azure_openai_deployment})")
    else:
        print(f"  - LLM: OpenAI ({settings.openai_model})")
