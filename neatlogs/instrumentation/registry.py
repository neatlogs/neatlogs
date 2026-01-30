"""
Registry of available instrumentations.

Maps semantic tags and library names to instrumentation packages from both
OpenInference and OpenLLMetry ecosystems.
"""

# Registry of all available instrumentations from:
# - OpenLLMetry: https://github.com/traceloop/openllmetry/tree/main/packages
# - OpenInference: https://github.com/Arize-ai/openinference/tree/main/python/instrumentation
INSTRUMENTATION_REGISTRY = {
    # Tag-based grouping for convenient selection
    "tags": {
        "llm": [
            "openai", "anthropic", "cohere", "bedrock", "groq", "together", "vertexai",
            "google_generativeai", "mistralai", "ollama", "watsonx", "alephalpha",
            "replicate", "sagemaker", "huggingface_hub", "litellm", "google_genai", "portkey"
        ],
        "embedding": ["openai", "cohere", "huggingface", "vertexai", "mistralai", "ollama"],
        "retrieval": [
            "chromadb", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
            "elasticsearch", "redis", "marqo"
        ],
        "agent": [
            "langchain", "llamaindex", "crewai", "autogen", "haystack", "dspy",
            "agno", "beeai", "openai_agents", "pydantic_ai", "smolagents", "strands",
            "pipecat"
        ],
        "tool": ["langchain", "llamaindex", "haystack", "mcp"],
        "http": ["requests", "httpx", "urllib3", "aiohttp"],
        "framework": [
            "instructor", "guardrails", "letta", "promptflow", "google_adk"
        ],
    },
    
    # Library-specific instrumentations
    # For each library, we specify:
    # - openllmetry: Package name for OpenLLMetry instrumentation
    # - openinference: Package name for OpenInference instrumentation
    # - default_span_kind: Hint for span kind inference
    "libraries": {
        # ===== LLM Providers (OpenAI, Anthropic, etc.) =====
        "openai": {
            "openinference": "openinference.instrumentation.openai",
            "openllmetry": "opentelemetry.instrumentation.openai",
            # "neatlogs": "neatlogs.neatlogs_instrumentation_openai",
            "default_span_kind": "LLM",
        },
        "anthropic": {
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
            "openllmetry": "opentelemetry.instrumentation.huggingface_hub",
            "openinference": None,
            "default_span_kind": "LLM",
        },
        "litellm": {
            "openllmetry": "opentelemetry.instrumentation.litellm",
            "openinference": "openinference.instrumentation.litellm",
            "default_span_kind": "LLM",
        },
        
        # ===== AI Frameworks =====
        "langchain": {
            "openllmetry": "opentelemetry.instrumentation.langchain",
            "openinference": "openinference.instrumentation.langchain",
            "default_span_kind": "CHAIN",
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
        },
        "autogen": {
            "openllmetry": "opentelemetry.instrumentation.autogen",
            "openinference": "openinference.instrumentation.autogen",
            "default_span_kind": "AGENT",
        },
        "haystack": {
            "openllmetry": "opentelemetry.instrumentation.haystack",
            "openinference": "openinference.instrumentation.haystack",
            "default_span_kind": "CHAIN",
        },
        "dspy": {
            "openllmetry": "opentelemetry.instrumentation.dspy",
            "openinference": "openinference.instrumentation.dspy",
            "default_span_kind": "CHAIN",
        },
        
        # ===== HTTP Libraries (CRITICAL for context propagation) =====
        "requests": {
            "openllmetry": "opentelemetry.instrumentation.requests",
            "openinference": None,  # No OpenInference HTTP instrumentation
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
        
        # ===== Vector Databases & Search =====
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
            "openinference": "openinference.instrumentation.qdrant",
            "default_span_kind": "RETRIEVER",
        },
        "milvus": {
            "openllmetry": "opentelemetry.instrumentation.milvus",
            "openinference": None,
            "default_span_kind": "RETRIEVER",
        },
        "opensearch": {
            "openllmetry": "opentelemetry.instrumentation.opensearch",
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
        
        # ===== Other Frameworks & Tools =====
        "instructor": {
            "openllmetry": "opentelemetry.instrumentation.instructor",
            "openinference": "openinference.instrumentation.instructor",
            "default_span_kind": "CHAIN",
        },
        "guardrails": {
            "openllmetry": "opentelemetry.instrumentation.guardrails",
            "openinference": "openinference.instrumentation.guardrails",
            "default_span_kind": "GUARDRAIL",
        },
        "letta": {
            "openllmetry": "opentelemetry.instrumentation.letta",
            "openinference": None,
            "default_span_kind": "AGENT",
        },
        
        # ===== New OpenInference Packages =====
        "google_genai": {
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
            "openllmetry": "opentelemetry.instrumentation.mcp",  # Span creation
            "openinference": "openinference.instrumentation.mcp",  # Context propagation
            "default_span_kind": "TOOL",
        },
    }
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
