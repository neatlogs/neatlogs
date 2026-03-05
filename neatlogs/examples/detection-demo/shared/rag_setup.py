"""
Simulated RAG Setup - No External Dependencies
===============================================
Simulates retrieval and reranking without Qdrant/Cohere.
Creates proper span types (RETRIEVER, RERANKER) for trace visualization.
"""

import json
import os
from typing import List, Dict

import neatlogs


# =============================================================================
# Simulated Knowledge Bases (In-Memory)
# =============================================================================

CUSTOMER_SUPPORT_KB = [
    {"text": "Our return policy allows returns within 30 days of purchase with original packaging.", "topic": "returns"},
    {"text": "Standard shipping takes 3-5 business days. Express shipping arrives in 1-2 days.", "topic": "shipping"},
    {"text": "To track your order, use the tracking number in your confirmation email.", "topic": "tracking"},
    {"text": "Refunds are processed within 5-7 business days after receiving the returned item.", "topic": "refunds"},
    {"text": "Contact support at support@example.com or call 1-800-SUPPORT.", "topic": "contact"},
]

MODERATION_POLICIES_KB = [
    {"text": "Hate speech and discriminatory language is strictly prohibited.", "category": "prohibited"},
    {"text": "Profanity and offensive language violates community standards.", "category": "language"},
    {"text": "Reviews must be authentic and based on actual product experience.", "category": "authenticity"},
    {"text": "Personal attacks or harassment towards users is not tolerated.", "category": "conduct"},
    {"text": "NSFW or sexually explicit content is prohibited.", "category": "explicit"},
]

RESEARCH_KB = [
    {"text": "AI safety involves alignment, robustness, and interpretability.", "topic": "safety"},
    {"text": "LangGraph provides stateful multi-agent orchestration with cycles.", "topic": "frameworks"},
    {"text": "LangChain offers modular components for building LLM applications.", "topic": "frameworks"},
    {"text": "RAG improves LLM responses by incorporating external knowledge.", "topic": "techniques"},
    {"text": "Content filtering is essential for production AI systems.", "topic": "safety"},
]


# =============================================================================
# Simulated Retriever
# =============================================================================

class SimulatedRetriever:
    """Simulates vector search with keyword matching."""
    
    def __init__(self, documents: List[Dict], name: str = "knowledge_base"):
        self.documents = documents
        self.name = name
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Simulate retrieval with keyword matching.
        Creates RETRIEVER span for trace visualization.
        """
        with neatlogs.trace(name=f"{self.name}_search", kind="RETRIEVER"):
            # Simple keyword matching (simulates vector similarity)
            query_lower = query.lower()
            scored_docs = []
            
            for doc in self.documents:
                text_lower = doc["text"].lower()
                # Count matching words as "score"
                score = sum(1 for word in query_lower.split() if word in text_lower)
                if score > 0:
                    scored_docs.append({"doc": doc, "score": score})
            
            # Sort by score and return top k
            scored_docs.sort(key=lambda x: x["score"], reverse=True)
            results = [item["doc"] for item in scored_docs[:k]]
            
            # If no matches, return first k docs
            if not results:
                results = self.documents[:k]
            
            return results


# =============================================================================
# Simulated Reranker
# =============================================================================

class SimulatedReranker:
    """Simulates reranking without Cohere."""
    
    def __init__(self, top_n: int = 3):
        self.top_n = top_n
    
    def rerank(self, documents: List[Dict], query: str) -> List[Dict]:
        """
        Simulate reranking.
        Creates RERANKER span for trace visualization.
        """
        with neatlogs.trace(name="simulated_rerank", kind="RERANKER"):
            # Simple reranking: prioritize docs with more query word matches
            query_words = set(query.lower().split())
            
            scored = []
            for doc in documents:
                text_words = set(doc["text"].lower().split())
                overlap = len(query_words & text_words)
                scored.append({"doc": doc, "score": overlap})
            
            scored.sort(key=lambda x: x["score"], reverse=True)
            return [item["doc"] for item in scored[:self.top_n]]


# =============================================================================
# Factory Functions
# =============================================================================

def get_customer_support_retriever() -> SimulatedRetriever:
    """Get retriever for customer support KB."""
    return SimulatedRetriever(CUSTOMER_SUPPORT_KB, "customer_support")


def get_moderation_policies_retriever() -> SimulatedRetriever:
    """Get retriever for moderation policies KB."""
    return SimulatedRetriever(MODERATION_POLICIES_KB, "moderation_policies")


def get_research_retriever() -> SimulatedRetriever:
    """Get retriever for research KB."""
    return SimulatedRetriever(RESEARCH_KB, "research")


def get_reranker(top_n: int = 3) -> SimulatedReranker:
    """Get simulated reranker."""
    return SimulatedReranker(top_n=top_n)
