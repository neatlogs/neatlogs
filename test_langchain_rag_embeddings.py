"""
Test LangChain RAG with embeddings and vector store to verify OpenInference captures:
1. EMBEDDING spans (query + document embeddings)
2. RETRIEVER spans (vector search)
3. LLM spans (generation with context)

This demonstrates the complete RAG flow that OpenInference should capture.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import neatlogs
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

# Initialize NeatLogs
neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY", "EQZO8zBsJkDRvmIL06m-EQ7faMUHEIN5"),
    tags=["v4", "langchain", "rag", "embeddings"],
    workflow_name="langchain-rag-embeddings-test"
)

print("🚀 Starting LangChain RAG + Embeddings Test...")
print(f"📍 Backend URL: http://localhost:3000")
print()

# ============================================================================
# STEP 1: Create Sample Documents
# ============================================================================
print("=" * 80)
print("STEP 1: Creating Sample Documents for RAG")
print("=" * 80)

documents = [
    Document(
        page_content="Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum in 1991.",
        metadata={"source": "python_docs", "topic": "python"}
    ),
    Document(
        page_content="JavaScript is a programming language primarily used for web development. It runs in browsers and on servers via Node.js.",
        metadata={"source": "js_docs", "topic": "javascript"}
    ),
    Document(
        page_content="TypeScript is a strongly typed superset of JavaScript developed by Microsoft. It compiles to plain JavaScript.",
        metadata={"source": "ts_docs", "topic": "typescript"}
    ),
    Document(
        page_content="Go (Golang) is a statically typed compiled language designed at Google. It's known for its simplicity and excellent concurrency support.",
        metadata={"source": "go_docs", "topic": "go"}
    ),
    Document(
        page_content="Rust is a systems programming language focused on safety, speed, and concurrency. It has no garbage collector.",
        metadata={"source": "rust_docs", "topic": "rust"}
    ),
]

print(f"✅ Created {len(documents)} sample documents")
for i, doc in enumerate(documents, 1):
    print(f"   {i}. {doc.metadata['topic']}: {doc.page_content[:50]}...")
print()

# ============================================================================
# STEP 2: Create Embeddings Model
# ============================================================================
print("=" * 80)
print("STEP 2: Initializing OpenAI Embeddings Model")
print("=" * 80)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

print("✅ Embeddings model initialized: text-embedding-3-small")
print("   This will create EMBEDDING spans for:")
print("   - Document embeddings (batch)")
print("   - Query embeddings (single)")
print()

# ============================================================================
# STEP 3: Create Vector Store and Index Documents
# ============================================================================
print("=" * 80)
print("STEP 3: Creating Vector Store and Embedding Documents")
print("=" * 80)
print("⏳ Embedding 5 documents... (This creates EMBEDDING spans)")

# This will create an EMBEDDING span for batch embedding
vectorstore = InMemoryVectorStore.from_documents(
    documents=documents,
    embedding=embeddings
)

print("✅ Vector store created with embedded documents")
print("   Expected: 1 EMBEDDING span with 5 vectors")
print()

# ============================================================================
# STEP 4: Create Retriever
# ============================================================================
print("=" * 80)
print("STEP 4: Creating Retriever")
print("=" * 80)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}  # Return top 2 most similar documents
)

print("✅ Retriever configured: similarity search, k=2")
print()

# ============================================================================
# STEP 5: Create LLM
# ============================================================================
print("=" * 80)
print("STEP 5: Initializing LLM")
print("=" * 80)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

print("✅ LLM initialized: gpt-4o-mini")
print()

# ============================================================================
# STEP 6: Create RAG Chain
# ============================================================================
print("=" * 80)
print("STEP 6: Creating RAG Chain")
print("=" * 80)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Stuff all retrieved docs into context
    retriever=retriever,
    return_source_documents=True,
    verbose=True
)

print("✅ RAG chain created")
print("   Flow: Query → Embed → Retrieve → LLM Generate")
print()

# ============================================================================
# STEP 7: Test RAG Query
# ============================================================================
print("=" * 80)
print("STEP 7: Running RAG Queries")
print("=" * 80)

queries = [
    "What is Python and who created it?",
    "Tell me about JavaScript and where it runs",
    "What are the key features of Rust?"
]

for i, query in enumerate(queries, 1):
    print(f"\n{'─' * 80}")
    print(f"Query {i}: {query}")
    print('─' * 80)
    
    print("⏳ Processing...")
    print("   1. Embedding query (EMBEDDING span)")
    print("   2. Searching vector store (RETRIEVER span)")
    print("   3. Generating response with context (LLM span)")
    print()
    
    try:
        result = qa_chain.invoke({"query": query})
        
        print(f"✅ Answer: {result['result']}")
        print()
        print(f"📄 Source Documents ({len(result['source_documents'])}):")
        for j, doc in enumerate(result['source_documents'], 1):
            print(f"   {j}. [{doc.metadata['source']}] {doc.page_content[:80]}...")
        print()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 80)
print("✅ RAG Test Completed!")
print("=" * 80)
print()
print("📊 Check NeatLogs UI at http://localhost:3000")
print()
print("Expected Spans:")
print("  1. EMBEDDING span (document batch): 5 documents → 5 vectors")
print("  2. EMBEDDING span (query 1): 'What is Python...' → vector")
print("  3. RETRIEVER span (query 1): Returns Python + JS docs")
print("  4. LLM span (query 1): Generates answer with context")
print("  5. EMBEDDING span (query 2): 'Tell me about JavaScript...' → vector")
print("  6. RETRIEVER span (query 2): Returns JS + TS docs")
print("  7. LLM span (query 2): Generates answer")
print("  8. EMBEDDING span (query 3): 'What are key features of Rust...' → vector")
print("  9. RETRIEVER span (query 3): Returns Rust doc")
print(" 10. LLM span (query 3): Generates answer")
print()
print("🔍 Look for these attributes in spans:")
print()
print("EMBEDDING spans:")
print("  - openinference.span.kind: EMBEDDING")
print("  - embedding.model_name: text-embedding-3-small")
print("  - embedding.embeddings.N.embedding.text: [document text]")
print("  - embedding.embeddings.N.embedding.vector: [float array]")
print("  - llm.token_count.total: [token count]")
print()
print("RETRIEVER spans:")
print("  - openinference.span.kind: RETRIEVER")
print("  - retrieval.documents.N.document.content: [retrieved doc text]")
print("  - retrieval.documents.N.document.score: [similarity score]")
print("  - retrieval.documents.N.document.metadata: {...}")
print()
print("LLM spans:")
print("  - openinference.span.kind: LLM")
print("  - llm.input_messages.0.message.content: [context + query]")
print("  - llm.output_messages.0.message.content: [generated answer]")
print("  - llm.token_count.total: [total tokens]")
print()
print("🎯 This demonstrates the COMPLETE RAG flow that OpenInference captures!")
