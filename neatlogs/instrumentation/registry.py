"""
Registry of available instrumentations.
"""

INSTRUMENTATION_REGISTRY = {
    "tags": {
        "llm": [
            "azure_ai_inference",
            "openai",
            "anthropic",
            "cohere",
            "bedrock",
            "groq",
            "together",
            "vertexai",
            "google_generativeai",
            "mistralai",
            "ollama",
            "watsonx",
            "alephalpha",
            "replicate",
            "sagemaker",
            "huggingface_hub",
            "litellm",
            "google_genai",
            "portkey",
        ],
        "embedding": ["openai", "cohere", "huggingface", "vertexai", "mistralai", "ollama"],
        "retrieval": [
            "chromadb",
            "pinecone",
            "weaviate",
            "qdrant",
            "milvus",
            "opensearch",
            "elasticsearch",
            "redis",
            "marqo",
        ],
        "agent": [
            "langchain",
            "langgraph",
            "llamaindex",
            "crewai",
            "autogen",
            "haystack",
            "dspy",
            "agno",
            "beeai",
            "openai_agents",
            "pydantic_ai",
            "smolagents",
            "strands",
            "pipecat",
        ],
        "tool": ["langchain", "llamaindex", "haystack", "mcp"],
        "http": ["requests", "httpx", "urllib3", "aiohttp"],
        "framework": ["instructor", "guardrails", "promptflow", "google_adk"],
    },
    "libraries": {
        "azure_ai_inference": {
            "openllmetry": None,
            "openinference": None,
            "neatlogs": "neatlogs_instrumentation_azure_ai_inference",
            "default_span_kind": "LLM",
        },
        "openai": {
            "neatlogs": "neatlogs.openai",
            "openinference": "openinference.instrumentation.openai",
            "openllmetry": "opentelemetry.instrumentation.openai",
            "default_span_kind": "LLM",
        },
        "anthropic": {
            "neatlogs": "neatlogs.anthropic",
            "openllmetry": "opentelemetry.instrumentation.anthropic",
            "openinference": "openinference.instrumentation.anthropic",
            "default_span_kind": "LLM",
        },
        "cohere": {
            "openllmetry": "opentelemetry.instrumentation.cohere",
            "openinference": None,
            "default_span_kind": "LLM",
        },
        "bedrock": {
            "openllmetry": "opentelemetry.instrumentation.bedrock",
            "openinference": "openinference.instrumentation.bedrock",
            "default_span_kind": "LLM",
        },
        "groq": {
            "openllmetry": "opentelemetry.instrumentation.groq",
            "openinference": "openinference.instrumentation.groq",
            "default_span_kind": "LLM",
        },
        "together": {
            "openllmetry": "opentelemetry.instrumentation.together",
            "openinference": None,
            "default_span_kind": "LLM",
        },
        "vertexai": {
            "openllmetry": "opentelemetry.instrumentation.vertexai",
            "openinference": "openinference.instrumentation.vertexai",
            "default_span_kind": "LLM",
        },
        "google_generativeai": {
            "openllmetry": "opentelemetry.instrumentation.google_generativeai",
            "openinference": None,
            "default_span_kind": "LLM",
        },
        "mistralai": {
            "openllmetry": "opentelemetry.instrumentation.mistralai",
            "openinference": "openinference.instrumentation.mistralai",
            "default_span_kind": "LLM",
        },
        "ollama": {
            "openllmetry": "opentelemetry.instrumentation.ollama",
            "openinference": None,
            "default_span_kind": "LLM",
        },
        "watsonx": {
            "openllmetry": "opentelemetry.instrumentation.watsonx",
            "openinference": None,
            "default_span_kind": "LLM",
        },
        "alephalpha": {
            "openllmetry": "opentelemetry.instrumentation.alephalpha",
            "openinference": None,
            "default_span_kind": "LLM",
        },
        "replicate": {
            "openllmetry": "opentelemetry.instrumentation.replicate",
            "openinference": None,
            "default_span_kind": "LLM",
        },
        "sagemaker": {
            "openllmetry": "opentelemetry.instrumentation.sagemaker",
            "openinference": None,
            "default_span_kind": "LLM",
        },
        "huggingface_hub": {
            "openllmetry": None,
            "openinference": None,
            "default_span_kind": "LLM",
        },
        "litellm": {
            "openllmetry": None,
            "openinference": "openinference.instrumentation.litellm",
            "default_span_kind": "LLM",
        },
        "langchain": {
            "openllmetry": "opentelemetry.instrumentation.langchain",
            "openinference": "openinference.instrumentation.langchain",
            "default_span_kind": "CHAIN",
        },
        "langgraph": {
            "openllmetry": None,
            "openinference": None,
            "default_span_kind": "WORKFLOW",
        },
        "llamaindex": {
            "openllmetry": "opentelemetry.instrumentation.llamaindex",
            "openinference": "openinference.instrumentation.llama_index",
            "default_span_kind": "CHAIN",
        },
        "crewai": {
            "openllmetry": "opentelemetry.instrumentation.crewai",
            "openinference": "openinference.instrumentation.crewai",
            "default_span_kind": "AGENT",
            "auto_load": ["litellm"],
        },
        "autogen": {
            "openllmetry": None,
            "openinference": "openinference.instrumentation.autogen",
            "default_span_kind": "AGENT",
        },
        "haystack": {
            "openllmetry": "opentelemetry.instrumentation.haystack",
            "openinference": "openinference.instrumentation.haystack",
            "default_span_kind": "CHAIN",
        },
        "dspy": {
            "openllmetry": None,
            "openinference": "openinference.instrumentation.dspy",
            "default_span_kind": "CHAIN",
        },
        "requests": {
            "openllmetry": "opentelemetry.instrumentation.requests",
            "openinference": None,
            "default_span_kind": "TOOL",
        },
        "httpx": {
            "openllmetry": "opentelemetry.instrumentation.httpx",
            "openinference": None,
            "default_span_kind": "TOOL",
        },
        "urllib3": {
            "openllmetry": "opentelemetry.instrumentation.urllib3",
            "openinference": None,
            "default_span_kind": "TOOL",
        },
        "aiohttp": {
            "openllmetry": "opentelemetry.instrumentation.aiohttp_client",
            "openinference": None,
            "default_span_kind": "TOOL",
        },
        "chromadb": {
            "openllmetry": "opentelemetry.instrumentation.chromadb",
            "openinference": None,
            "default_span_kind": "RETRIEVER",
        },
        "pinecone": {
            "openllmetry": "opentelemetry.instrumentation.pinecone",
            "openinference": None,
            "default_span_kind": "RETRIEVER",
        },
        "weaviate": {
            "openllmetry": "opentelemetry.instrumentation.weaviate",
            "openinference": "openinference.instrumentation.weaviate",
            "default_span_kind": "RETRIEVER",
        },
        "qdrant": {
            "openllmetry": "opentelemetry.instrumentation.qdrant",
            "openinference": None,
            "default_span_kind": "RETRIEVER",
        },
        "milvus": {
            "openllmetry": "opentelemetry.instrumentation.milvus",
            "openinference": None,
            "default_span_kind": "RETRIEVER",
        },
        "opensearch": {
            "openllmetry": None,
            "openinference": None,
            "default_span_kind": "RETRIEVER",
        },
        "elasticsearch": {
            "openllmetry": "opentelemetry.instrumentation.elasticsearch",
            "openinference": None,
            "default_span_kind": "RETRIEVER",
        },
        "redis": {
            "openllmetry": "opentelemetry.instrumentation.redis",
            "openinference": None,
            "default_span_kind": "RETRIEVER",
        },
        "marqo": {
            "openllmetry": "opentelemetry.instrumentation.marqo",
            "openinference": None,
            "default_span_kind": "RETRIEVER",
        },
        "instructor": {
            "openllmetry": None,
            "openinference": "openinference.instrumentation.instructor",
            "default_span_kind": "CHAIN",
        },
        "guardrails": {
            "openllmetry": None,
            "openinference": "openinference.instrumentation.guardrails",
            "default_span_kind": "GUARDRAIL",
        },
        "google_genai": {
            "neatlogs": "neatlogs.google_genai",
            "openllmetry": None,
            "openinference": "openinference.instrumentation.google_genai",
            "default_span_kind": "LLM",
        },
        "google_adk": {
            "openllmetry": None,
            "openinference": "openinference.instrumentation.google_adk",
            "default_span_kind": "CHAIN",
        },
        "agno": {
            "openllmetry": None,
            "openinference": "openinference.instrumentation.agno",
            "default_span_kind": "AGENT",
        },
        "beeai": {
            "openllmetry": None,
            "openinference": "openinference.instrumentation.beeai",
            "default_span_kind": "AGENT",
        },
        "openai_agents": {
            "openllmetry": None,
            "openinference": "openinference.instrumentation.openai_agents",
            "default_span_kind": "AGENT",
        },
        "pydantic_ai": {
            "openllmetry": None,
            "openinference": "openinference.instrumentation.pydantic_ai",
            "default_span_kind": "AGENT",
        },
        "smolagents": {
            "openllmetry": None,
            "openinference": "openinference.instrumentation.smolagents",
            "default_span_kind": "AGENT",
        },
        "strands": {
            "openllmetry": None,
            "openinference": "openinference.instrumentation.strands",
            "default_span_kind": "AGENT",
        },
        "pipecat": {
            "openllmetry": None,
            "openinference": "openinference.instrumentation.pipecat",
            "default_span_kind": "AGENT",
        },
        "portkey": {
            "openllmetry": None,
            "openinference": "openinference.instrumentation.portkey",
            "default_span_kind": "LLM",
        },
        "promptflow": {
            "openllmetry": None,
            "openinference": "openinference.instrumentation.promptflow",
            "default_span_kind": "CHAIN",
        },
        "mcp": {
            "openllmetry": "opentelemetry.instrumentation.mcp",
            "openinference": "openinference.instrumentation.mcp",
            "default_span_kind": "TOOL",
        },
    },
}


def get_libraries_by_tag(tag: str) -> list:
    """
    Get list of library names for a given semantic tag.

    Args:
        tag: Semantic tag (e.g., "llm", "agent", "http")

    Returns:
        List of library names matching the tag
    """
    return INSTRUMENTATION_REGISTRY["tags"].get(tag, [])


def get_library_info(library: str) -> dict:
    """
    Get instrumentation info for a specific library.

    Args:
        library: Library name (e.g., "openai", "langchain")

    Returns:
        Dictionary with instrumentation package names and metadata
    """
    return INSTRUMENTATION_REGISTRY["libraries"].get(library, {})
