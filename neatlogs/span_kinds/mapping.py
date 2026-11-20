def infer_span_kind_from_name(span_name: str) -> str:
    """
    Infer OpenInference span kind from span name.

    Specific operations are checked before generic provider/model keywords so a
    span such as ``chat_history_retrieval`` is a RETRIEVER, not an LLM. Provider
    names only decide the LLM kind when no more specific operation is present.

    Distinguishes RETRIEVER (read ops) vs VECTOR_STORE (write ops) for vector DBs.
    """
    name_lower = span_name.lower()

    # Rerankers often name both the ranking operation and the searched source
    # (e.g. "rerank_search_results"), so this must precede generic retrieval.
    if "rerank" in name_lower:
        return "RERANKER"

    # Embedding operations may also mention retrieval targets ("embed_query").
    if "embed" in name_lower:
        return "EMBEDDING"

    # Vector DB: Distinguish RETRIEVER (read) vs VECTOR_STORE (write)
    if any(
        db in name_lower
        for db in [
            "chroma",
            "pinecone",
            "weaviate",
            "qdrant",
            "milvus",
            "lancedb",
            "marqo",
        ]
    ):
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
        return "VECTOR_STORE"

    # Generic retrieval, tool, guardrail, evaluation and agent operations all
    # outrank the generic LLM keywords below.
    if any(keyword in name_lower for keyword in ["retriev", "search", "query"]):
        return "RETRIEVER"

    if "agent" in name_lower:
        return "AGENT"

    if any(keyword in name_lower for keyword in ["tool", "function"]):
        return "TOOL"

    if any(keyword in name_lower for keyword in ["guardrail", "validate", "moderate", "safety"]):
        return "GUARDRAIL"

    if any(keyword in name_lower for keyword in ["evaluat", "score", "metric"]):
        return "EVALUATOR"

    # LLM operations, checked last so the specific operations above win.
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

    return "CHAIN"
