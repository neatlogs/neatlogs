"""
Example 9: RAG with ChromaDB Vector Database

Shows dual instrumentation of vector databases.
Both OpenInference and OpenLLMetry instrument ChromaDB, giving us:
- RETRIEVER span kinds (OpenInference)
- Query details, results count (both conventions)
- Embedding vectors (OpenInference)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from neatlogs.sdk.neatlogs_sdk_v4_langfuse import init, trace, flush, shutdown
from openai import OpenAI
import chromadb


def main():
    # Enable span logging
    os.environ['NEATLOGS_LOG_SPANS'] = 'true'
    
    # Initialize with explicit library instrumentation
    init(
        api_key=os.getenv("NEATLOGS_API_KEY", "test-key"),
        workflow_name="rag-chromadb",
        instrumentations=["openai", "chromadb"],
        # Only instrument the libraries we're actually using
        debug=True,
    )
    
    query = "What is Neatlogs?"
    
    try:
        with trace(
            "rag_workflow",
            prompt_template="Build knowledge base and answer: {query}",
            prompt_variables={"query": query}
        ):
            openai_client = OpenAI()
            
            # Create ChromaDB client and collection
            chroma_client = chromadb.Client()
            
            with trace("create_collection", kind="RETRIEVER") as span:
                collection = chroma_client.create_collection(
                    name="knowledge_base",
                    metadata={"description": "Company knowledge base"}
                )
                span.set_attribute("collection_name", "knowledge_base")
            
            # Add documents (this will trigger EMBEDDING spans)
            documents = [
                "Neatlogs is an AI observability platform for LLM applications.",
                "OpenInference provides semantic conventions for AI tracing.",
                "OpenLLMetry instruments popular AI frameworks and LLM providers.",
                "LangChain is a framework for building LLM applications.",
                "Vector databases enable semantic search over embeddings."
            ]
            
            with trace("add_documents", kind="CHAIN") as span:
                # Generate embeddings
                embeddings_response = openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=documents
                )
                
                # Extract embeddings
                embeddings = [item.embedding for item in embeddings_response.data]
                
                # Add to ChromaDB
                collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    ids=[f"doc_{i}" for i in range(len(documents))]
                )
                
                span.set_attribute("documents_added", len(documents))
            
            # Query the collection (RAG retrieval)
            with trace("rag_query", kind="CHAIN") as span:
                # 1. Embed the query
                with trace("embed_query", kind="EMBEDDING"):
                    query_embedding_response = openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=[query]
                    )
                    query_embedding = query_embedding_response.data[0].embedding
                
                # 2. Search in vector DB (RETRIEVER span)
                with trace("vector_search", kind="RETRIEVER") as search_span:
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=3
                    )
                    
                    retrieved_docs = results['documents'][0]
                    distances = results['distances'][0]
                    
                    search_span.set_attribute("query", query)
                    search_span.set_attribute("results_count", len(retrieved_docs))
                    search_span.set_attribute("top_distance", distances[0] if distances else None)
                    
                    print(f"\nQuery: {query}")
                    print(f"Retrieved {len(retrieved_docs)} documents:")
                    for i, (doc, dist) in enumerate(zip(retrieved_docs, distances)):
                        print(f"  {i+1}. [{dist:.4f}] {doc}")
                
                # 3. Generate answer using retrieved context
                context = "\n".join(retrieved_docs)
                
                with trace(
                    "generate_answer",
                    kind="LLM",
                    prompt_template="Context: {context}\n\nQuestion: {query}\n\nAnswer:",
                    prompt_variables={"context": context, "query": query}
                ):
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Answer based on the provided context."},
                            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
                        ]
                    )
                    
                    answer = response.choices[0].message.content
                    print(f"\nAnswer: {answer}")
    except Exception as e:
        print(f"\nError during RAG execution: {e}")
    finally:
        print("\n💾 Flushing spans...")
        flush()
        print("🛑 Shutting down SDK...")
        shutdown()
        print("✅ Done!")


if __name__ == "__main__":
    main()
