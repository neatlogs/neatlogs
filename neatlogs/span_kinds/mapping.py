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

TRACELOOP_TO_OPENINFERENCE = {
    "workflow": "CHAIN",
    "task": "CHAIN",
    "agent": "AGENT",
    "tool": "TOOL",
    "unknown": "UNKNOWN",
}


def infer_span_kind_from_name(span_name: str) -> str:
    name_lower = span_name.lower()
    if any(
        keyword in name_lower
        for keyword in ["openai", "anthropic", "cohere", "bedrock", "chat", "completion", "llm"]
    ):
        return "LLM"
    
    elif "embed" in name_lower:
        return "EMBEDDING"

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
    
    elif "rerank" in name_lower:
        return "RERANKER"
    
    elif "agent" in name_lower:
        return "AGENT"
    
    elif any(
        keyword in name_lower
        for keyword in ["tool", "function"]
    ):
        return "TOOL"
    
    elif any(
        keyword in name_lower for keyword in ["guardrail", "validate", "moderate", "safety"]
    ):
        return "GUARDRAIL"
    
    elif any(keyword in name_lower for keyword in ["evaluat", "score", "metric"]):
        return "EVALUATOR"
    
    else:
        return "CHAIN"
