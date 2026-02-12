"""
LangChain + OpenAI + ChromaDB Test Suite
=========================================

Tests:
- Framework: LangChain (pure chains, no LangGraph)
- LLM: OpenAI
- Embeddings: OpenAI
- Vector DB: ChromaDB (upsert + retrieval)

Span types: CHAIN, LLM, EMBEDDING, VECTOR_STORE, RETRIEVER

Env:
  NEATLOGS_API_KEY, OPENAI_API_KEY

Run:
  python neatlogs/examples/61_langchain_openai_chroma.py
"""

from __future__ import annotations

import os
import sys
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("NEATLOGS_LOG_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_METRICS", "true")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_SPANS_FILE", "spans_61_langchain.jsonl")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS_FILE", "spans_raw_61_langchain.log")
os.environ.setdefault("NEATLOGS_LOG_METRICS_FILE", "metrics_61_langchain.jsonl")

from neatlogs import init, flush, shutdown, PromptTemplate, trace, workflow, chain, agent, retriever, embedding

init(
    api_key=os.getenv("NEATLOGS_API_KEY", "test-key"),
    endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:3000/api/data/v4/batch"),
    workflow_name="langchain-openai-chroma-rag",
    instrumentations=["langchain", "openai", "chromadb"],
    debug=True,
)

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Settings
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CHROMA_COLLECTION = "langchain_rag_docs"

# Components
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


@chain(name="setup_vector_store")
def setup_vector_store() -> Chroma:
    """Setup ChromaDB with documents (VECTOR_STORE + EMBEDDING spans)."""
    
    raw_docs = [
        "Neural networks are computing systems inspired by biological neural networks. Deep learning uses multiple layers to learn hierarchical representations.",
        "Transfer learning allows models to leverage knowledge from one task to improve performance on another. Pre-trained models like GPT and BERT can be fine-tuned.",
        "Reinforcement learning is where agents learn by interacting with an environment. The agent adjusts its policy to maximize cumulative reward.",
        "Attention mechanisms allow models to focus on relevant parts of input. Transformers use self-attention to process sequences in parallel.",
    ]
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    
    docs = []
    for i, text in enumerate(raw_docs):
        chunks = splitter.split_text(text.strip())
        for j, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={"source": f"doc_{i}", "chunk": j}))
    
    # Creates VECTOR_STORE and EMBEDDING spans
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        collection_name=CHROMA_COLLECTION,
    )
    
    return vector_store


@retriever(name="rag_retriever")
def retrieve_docs(vector_store: Chroma, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve documents from ChromaDB (RETRIEVER span)."""
    docs = vector_store.similarity_search(query, k=top_k)
    return [{"content": d.page_content, "metadata": d.metadata} for d in docs]


@embedding(name="embed_query")
def embed_query(text: str) -> List[float]:
    """Embed a query (EMBEDDING span)."""
    return embedding_model.embed_query(text)


# Prompt template
rag_prompt = PromptTemplate([
    {"role": "system", "content": "Answer questions using the provided context. Be concise and accurate."},
    {"role": "user", "content": "Context:\n{{context}}\n\nQuestion: {{question}}\n\nAnswer:"},
])


@agent(name="generate_answer")
def generate_answer(context: str, question: str) -> str:
    """Generate answer with LLM (AGENT span)."""
    
    with trace("rag_llm_call", kind="LLM", prompt_template=rag_prompt):
        messages = rag_prompt.compile(context=context, question=question)
        response = llm.invoke(messages)
        return response.content


@workflow(name="langchain_rag_workflow")
def run_rag_pipeline(queries: List[str]) -> List[str]:
    """Execute RAG pipeline for multiple queries."""
    
    # Setup
    vector_store = setup_vector_store()
    
    answers = []
    for query in queries:
        # Embed query (explicit embedding span)
        _ = embed_query(query)
        
        # Retrieve
        docs = retrieve_docs(vector_store, query, top_k=3)
        context = "\n\n".join([f"[{i+1}] {d['content']}" for i, d in enumerate(docs)])
        
        # Generate
        answer = generate_answer(context, query)
        answers.append(answer)
        
        print(f"Q: {query}")
        print(f"A: {answer}\n")
    
    return answers


def main():
    try:
        print("\n" + "="*60)
        print("LangChain + OpenAI + ChromaDB Test")
        print("="*60 + "\n")
        
        queries = [
            "What are neural networks?",
            "Explain transfer learning",
            "What is the attention mechanism?",
        ]
        
        run_rag_pipeline(queries)
        
        print("="*60)
        print("\nSpan checklist:")
        print("  - WORKFLOW (langchain_rag_workflow)")
        print("  - CHAIN (setup_vector_store)")
        print("  - AGENT (generate_answer - contains LLM call)")
        print("  - LLM (OpenAI with prompt template)")
        print("  - EMBEDDING (OpenAI - setup + embed_query)")
        print("  - VECTOR_STORE (ChromaDB from_documents)")
        print("  - RETRIEVER (rag_retriever)")
        
    except KeyboardInterrupt:
        raise SystemExit(130)
    finally:
        flush()
        shutdown()


if __name__ == "__main__":
    main()
