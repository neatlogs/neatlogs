"""
CrewAI + Gemini + Qdrant Test Suite
====================================

Tests:
- Framework: CrewAI (multi-agent)
- LLM: Gemini
- Embeddings: OpenAI
- Vector DB: Qdrant (upsert + retrieval)
- Tools: Tavily web search (agentic, embedded in agent)

Span types: AGENT, TOOL (agentic), LLM, EMBEDDING, VECTOR_STORE, RETRIEVER

Env:
  NEATLOGS_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, TAVILY_API_KEY
  QDRANT_URL (default: http://localhost:6333)

Run:
  python neatlogs/examples/60_crewai_gemini_qdrant.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("NEATLOGS_LOG_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_METRICS", "true")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_SPANS_FILE", "spans_60_crewai.jsonl")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS_FILE", "spans_raw_60_crewai.log")
os.environ.setdefault("NEATLOGS_LOG_METRICS_FILE", "metrics_60_crewai.jsonl")

from neatlogs import init, flush, shutdown, PromptTemplate, trace, workflow, chain, retriever, embedding

init(
    api_key=os.getenv("NEATLOGS_API_KEY", "test-key"),
    endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:3000/api/data/v4/batch"),
    workflow_name="crewai-gemini-qdrant-research",
    instrumentations=["crewai", "google_genai", "openai", "qdrant", "langchain"],
    debug=True,
)

from crewai import Agent, Crew, Task, Process, LLM
from crewai.tools import BaseTool
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from pydantic import Field

# Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
if not GEMINI_API_KEY:
    raise SystemExit("Missing GEMINI_API_KEY or GOOGLE_API_KEY")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "crewai_research_kb"

# Initialize LLM using CrewAI's LLM wrapper (uses LiteLLM internally)
llm = LLM(
    model="gemini/gemini-2.5-flash",  # LiteLLM format: provider/model (stable model from Test 49)
    api_key=GEMINI_API_KEY,
    temperature=0,
)

# Initialize embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize Qdrant
qdrant_client = QdrantClient(url=QDRANT_URL)

try:
    qdrant_client.get_collection(collection_name=COLLECTION_NAME)
except Exception:
    dim = len(embedding_model.embed_query("test"))
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embedding_model,
)


# Seed knowledge base (VECTOR_STORE + EMBEDDING spans)
@chain(name="seed_knowledge_base")
def seed_knowledge_base():
    """Seed vector store with research documents."""
    docs = [
        Document(page_content="AI agents use memory systems for context retention. Short-term memory holds recent interactions, long-term memory persists important information.", metadata={"source": "ai-memory"}),
        Document(page_content="Vector databases enable semantic search via embeddings. Popular options: Pinecone, Qdrant, Chroma. They use similarity metrics like cosine distance.", metadata={"source": "vector-dbs"}),
        Document(page_content="RAG improves LLM accuracy by retrieving relevant context before generation. Combines vector search with language models.", metadata={"source": "rag"}),
        Document(page_content="Multi-agent systems decompose complex tasks. CrewAI, AutoGen, LangGraph enable agent orchestration with different patterns.", metadata={"source": "multi-agent"}),
    ]
    vector_store.add_documents(documents=docs, ids=[i for i in range(len(docs))])
    return len(docs)


# Retriever function (RETRIEVER span)
@retriever(name="search_knowledge_base")
def search_knowledge_base(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Search Qdrant knowledge base."""
    docs = vector_store.similarity_search(query, k=top_k)
    return [{"content": d.page_content, "source": d.metadata.get("source", "")} for d in docs]


# CrewAI Tool - Agentic tool embedded in agent (TOOL spans auto-captured)
class KnowledgeBaseSearchTool(BaseTool):
    name: str = "search_knowledge_base"
    description: str = "Search the internal knowledge base for information about AI agents, vector databases, RAG, and multi-agent systems."
    
    def _run(self, query: str) -> str:
        results = search_knowledge_base(query, top_k=3)
        return json.dumps(results, indent=2)


class TavilySearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for current information. Use for recent news, updates, or information not in knowledge base."
    
    def _run(self, query: str) -> str:
        if not TAVILY_API_KEY:
            return json.dumps({"error": "TAVILY_API_KEY not set", "results": []})
        
        try:
            from langchain_community.tools import TavilySearchResults
            tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=3)
            results = tool.invoke(query)
            return json.dumps(results, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "results": []})


# Define Agents with embedded tools
researcher = Agent(
    role="Research Specialist",
    goal="Find relevant information using knowledge base and web search",
    backstory="Expert researcher who combines internal knowledge with web sources",
    tools=[KnowledgeBaseSearchTool(), TavilySearchTool()],
    llm=llm,
    verbose=True,
)

analyst = Agent(
    role="Data Analyst", 
    goal="Analyze research findings and identify key patterns",
    backstory="Skilled at identifying patterns and extracting insights",
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="Technical Writer",
    goal="Synthesize research into clear summaries",
    backstory="Expert at distilling complex information",
    llm=llm,
    verbose=True,
)


@workflow(name="crewai_research_workflow")
def run_research_crew(query: str) -> str:
    """Execute multi-agent research workflow."""
    
    research_task = Task(
        description=f"Research: {query}. Use knowledge base first, then web search for additional context.",
        expected_output="Comprehensive research findings with sources",
        agent=researcher,
    )
    
    analysis_task = Task(
        description="Analyze research findings. Identify key patterns, themes, and relationships.",
        expected_output="Analysis with main themes and connections",
        agent=analyst,
        context=[research_task],
    )
    
    writing_task = Task(
        description="Create a concise 3-4 sentence summary synthesizing key points.",
        expected_output="Clear technical summary",
        agent=writer,
        context=[research_task, analysis_task],
    )
    
    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )
    
    result = crew.kickoff()
    return result.raw if hasattr(result, 'raw') else str(result)


def main():
    try:
        print("\n" + "="*60)
        print("CrewAI + Gemini + Qdrant Test")
        print("="*60 + "\n")
        
        # Seed KB
        n = seed_knowledge_base()
        print(f"Seeded {n} documents\n")
        
        # Run workflow
        query = "How do AI agents use memory and vector databases?"
        result = run_research_crew(query)
        
        print("\n" + "="*60)
        print(f"Result:\n{result}")
        print("="*60)
        
        print("\nSpan checklist:")
        print("  - WORKFLOW (crewai_research_workflow)")
        print("  - AGENT (CrewAI agents)")
        print("  - TOOL (agentic: search_knowledge_base, web_search)")
        print("  - LLM (Gemini)")
        print("  - EMBEDDING (OpenAI)")
        print("  - VECTOR_STORE (Qdrant upsert)")
        print("  - RETRIEVER (Qdrant query)")
        
    except KeyboardInterrupt:
        raise SystemExit(130)
    finally:
        flush()
        shutdown()


if __name__ == "__main__":
    main()
