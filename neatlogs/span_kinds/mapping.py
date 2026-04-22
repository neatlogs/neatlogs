def infer_span_kind_from_name(span_name: str) -> str:
    """
    Infer OpenInference span kind from span name.

    Distinguishes RETRIEVER (read ops) vs VECTOR_STORE (write ops) for vector DBs.
    """
    name_lower = span_name.lower()

    # LLM operations
    if any(
        keyword in name_lower
        for keyword in [
            "openai",
            "anthropic",
            "cohere",
            "bedrock",
            "chat",
            "completion",
            "llm",
            "gemini",
            "google_genai",
        ]
    ):
        return "LLM"

    # Embedding operations
    elif "embed" in name_lower:
        return "EMBEDDING"

    # Vector DB: Distinguish RETRIEVER (read) vs VECTOR_STORE (write)
    elif any(
        db in name_lower
        for db in [
            "chroma",
            "pinecone",
            "weaviate",
            "qdrant",
            "milvus",
            "lancedb",
            "marqo",
            "astra",
        ]
    ):
        # Check if it's a READ operation (retrieval)
        retrieval_keywords = [
            "query",
            "search",
            "get",
            "fetch",
            "find",
            "retrieve",
            "scroll",
            "peek",
            "discover",
            "recommend",
            "aggregate",
            "hybrid_search",
        ]
        if any(keyword in name_lower for keyword in retrieval_keywords):
            return "RETRIEVER"
        else:
            # WRITE operations (add, insert, upsert, update, delete, create, drop)
            return "VECTOR_STORE"

    # Generic retrieval (for custom retrieval functions)
    elif any(keyword in name_lower for keyword in ["retriev", "search", "query"]):
        return "RETRIEVER"

    # Reranker operations
    elif "rerank" in name_lower:
        return "RERANKER"

    # Agent operations
    elif "agent" in name_lower:
        return "AGENT"

    # Tool/function operations
    elif any(keyword in name_lower for keyword in ["tool", "function"]):
        return "TOOL"

    # Guardrail operations
    elif any(keyword in name_lower for keyword in ["guardrail", "validate", "moderate", "safety"]):
        return "GUARDRAIL"

    # Evaluator operations
    elif any(keyword in name_lower for keyword in ["evaluat", "score", "metric"]):
        return "EVALUATOR"

    # Default: CHAIN
    else:
        return "CHAIN"
