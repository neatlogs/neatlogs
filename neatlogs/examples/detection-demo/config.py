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


@dataclass(frozen=True)
class Settings:
    """Shared settings for all workflows."""
    
    # Neatlogs Configuration
    neatlogs_api_key: str
    neatlogs_endpoint: str
    project_id: str
    
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
    required_vars = [
        "NEATLOGS_API_KEY",
        "PROJECT_ID",
        "OPENAI_API_KEY",
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Please copy .env.example to .env and fill in your keys."
        )
    
    # Check if Azure OpenAI should be used
    use_azure = os.getenv("USE_AZURE", "false").lower() == "true"
    
    return Settings(
        # Neatlogs
        neatlogs_api_key=os.getenv("NEATLOGS_API_KEY"),
        neatlogs_endpoint=os.getenv(
            "NEATLOGS_ENDPOINT",
            "https://staging-api.neatlogs.com/api/data/v4/batch"
        ),
        project_id=os.getenv("PROJECT_ID"),
        
        # LLM
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        
        # Azure OpenAI
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        use_azure=use_azure,
        
        # Debug
        debug=os.getenv("DEBUG", "true").lower() == "true",
    )


def init_neatlogs(settings: Settings):
    """Initialize Neatlogs SDK with instrumentation."""
    
    # Set debug logging if enabled
    if settings.debug:
        os.environ.setdefault("NEATLOGS_LOG_SPANS", "true")
        os.environ.setdefault("NEATLOGS_LOG_METRICS", "true")
    
    neatlogs.init(
        api_key=settings.neatlogs_api_key,
        endpoint=settings.neatlogs_endpoint,
        workflow_name="detection-demo",
        tags=["sales-demo", "langgraph", "investor-demo"],
        instrumentations=["langchain", "openai", "crewai"],
        debug=settings.debug,
    )
    
    print(f"✓ Neatlogs initialized")
    print(f"  - Endpoint: {settings.neatlogs_endpoint}")
    print(f"  - Project ID: {settings.project_id}")
    print(f"  - Tags: sales-demo, langgraph, investor-demo")
    if settings.use_azure:
        print(f"  - LLM: Azure OpenAI ({settings.azure_openai_deployment})")
    else:
        print(f"  - LLM: OpenAI ({settings.openai_model})")
