"""
Mapping between OpenInference and OpenLLMetry (Traceloop) span kinds.

OpenInference uses 9 granular AI-specific span kinds:
- LLM, EMBEDDING, RETRIEVER, RERANKER, CHAIN, AGENT, TOOL, GUARDRAIL, EVALUATOR

OpenLLMetry (Traceloop) uses 5 generic span kinds:
- workflow, task, agent, tool, unknown
"""

# Map OpenInference → OpenLLMetry (Traceloop)
# Used for backward compatibility when exporting to systems expecting Traceloop format
OPENINFERENCE_TO_TRACELOOP = {
    "LLM": "task",
    "EMBEDDING": "task",
    "RETRIEVER": "task",
    "RERANKER": "task",
    "CHAIN": "workflow",
    "AGENT": "agent",
    "TOOL": "tool",
    "GUARDRAIL": "task",
    "EVALUATOR": "task",
    "UNKNOWN": "unknown",
}

# Map OpenLLMetry (Traceloop) → OpenInference
# Used when reading Traceloop spans and converting to OpenInference format
TRACELOOP_TO_OPENINFERENCE = {
    "workflow": "CHAIN",
    "task": "CHAIN",
    "agent": "AGENT",
    "tool": "TOOL",
    "unknown": "UNKNOWN",
}


def infer_span_kind_from_name(span_name: str) -> str:
    """
    Infer OpenInference span kind from span name.
    
    This is a FALLBACK for AI/LLM spans that don't have span kind set by instrumentation.
    Should NOT be called for HTTP/infrastructure spans.
    
    Typically used for:
    - Custom user spans
    - Legacy instrumentation
    - Manually created spans without semantic attributes
    
    Args:
        span_name: The name of the span
        
    Returns:
        OpenInference span kind string (e.g., "LLM", "RETRIEVER", "CHAIN")
    """
    name_lower = span_name.lower()
    
    # LLM patterns
    if any(
        keyword in name_lower
        for keyword in ["openai", "anthropic", "cohere", "bedrock", "chat", "completion", "llm"]
    ):
        return "LLM"
    
    # Embedding patterns
    elif "embed" in name_lower:
        return "EMBEDDING"
    
    # Retrieval patterns
    elif any(
        keyword in name_lower
        for keyword in [
            "retriev",
            "search",
            "query",
            "chromadb",
            "pinecone",
            "weaviate",
            "qdrant",
            "milvus",
        ]
    ):
        return "RETRIEVER"
    
    # Reranker patterns
    elif "rerank" in name_lower:
        return "RERANKER"
    
    # Agent patterns
    elif "agent" in name_lower:
        return "AGENT"
    
    # Tool patterns (explicit tool/function calls only, NOT HTTP)
    # HTTP spans are infrastructure layer and will be merged into parent spans
    elif any(
        keyword in name_lower
        for keyword in ["tool", "function"]
    ):
        return "TOOL"
    
    # Guardrail patterns
    elif any(
        keyword in name_lower for keyword in ["guardrail", "validate", "moderate", "safety"]
    ):
        return "GUARDRAIL"
    
    # Evaluation patterns
    elif any(keyword in name_lower for keyword in ["evaluat", "score", "metric"]):
        return "EVALUATOR"
    
    # Default to CHAIN for orchestration/workflow spans
    else:
        return "CHAIN"
