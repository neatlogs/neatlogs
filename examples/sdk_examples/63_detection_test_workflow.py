"""
Detection Test Workflow - LangGraph + OpenAI + Qdrant + Cohere
================================================================

Purpose: Generate realistic traces to test AI Service detection system
- Simulates task failures, timeouts, slow responses, apologies
- Tests classifier, regex, and conditional detectors

Span types: WORKFLOW, AGENT, CHAIN, LLM, EMBEDDING, VECTOR_STORE, RETRIEVER, RERANKER, TOOL

Detection scenarios:
  1. Task failure (agent gives up, tool errors)
  2. High latency (slow LLM responses)
  3. Apology patterns (agent apologizes for failures)
  4. High token usage (large prompts/completions)

Env:
  NEATLOGS_API_KEY, OPENAI_API_KEY, COHERE_API_KEY
  QDRANT_URL (default: http://localhost:6333)

Prerequisites:
  docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

Run:
  python neatlogs/examples/63_detection_test_workflow.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Annotated, Sequence, TypedDict, List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("NEATLOGS_LOG_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_METRICS", "true")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_SPANS_FILE", "spans_63_detection.jsonl")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS_FILE", "spans_raw_63_detection.log")
os.environ.setdefault("NEATLOGS_LOG_METRICS_FILE", "metrics_63_detection.jsonl")

from neatlogs import init, flush, shutdown, PromptTemplate, trace, span

init(
    api_key=os.getenv("NEATLOGS_API_KEY", "test-key"),
    endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:3000/api/data/v4/batch"),
    workflow_name="detection-test-workflow",
    instrumentations=["langchain", "openai", "cohere"],
    debug=True,
)

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool as lc_tool
from langchain_cohere import CohereRerank
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Settings
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "detection-test-kb"
COHERE_RERANK_MODEL = "rerank-english-v3.0"
PROJECT_ID = "47316c74-b0e8-46de-8e8d-f0f830d97f47"  # Test project from seed script


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Components
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
reranker = CohereRerank(model=COHERE_RERANK_MODEL, top_n=3)
qdrant_client = QdrantClient(url=QDRANT_URL)


@span(kind="CHAIN", name="setup_qdrant")
def setup_qdrant_collection():
    """Setup Qdrant with documents (VECTOR_STORE + EMBEDDING spans)."""
    
    sample_emb = embedding_model.embed_query("test")
    dim = len(sample_emb)
    
    # Recreate collection
    if qdrant_client.collection_exists(QDRANT_COLLECTION):
        qdrant_client.delete_collection(QDRANT_COLLECTION)
    
    qdrant_client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    
    # Seed documents - detection-focused content
    docs = [
        {"text": "When an API request fails with a 500 error, the system should retry with exponential backoff.", "topic": "errors"},
        {"text": "High latency responses over 5 seconds indicate performance issues that need investigation.", "topic": "performance"},
        {"text": "Apologies in agent responses often signal task failures or unmet user expectations.", "topic": "quality"},
        {"text": "Token usage exceeding 10,000 tokens per request may indicate inefficient prompting.", "topic": "costs"},
        {"text": "Timeout errors after 30 seconds suggest network issues or overloaded services.", "topic": "reliability"},
    ]
    
    points = []
    for i, doc in enumerate(docs):
        emb = embedding_model.embed_query(doc["text"])
        points.append(PointStruct(id=i, vector=emb, payload={"text": doc["text"], "topic": doc["topic"]}))
    
    qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(docs)


@span(kind="RETRIEVER", name="qdrant_retriever")
def retrieve_from_qdrant(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve from Qdrant (RETRIEVER span)."""
    query_emb = embedding_model.embed_query(query)
    results = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_emb,
        limit=top_k,
    ).points
    
    return [
        {"id": str(point.id), "content": point.payload.get("text", ""), "score": point.score, "topic": point.payload.get("topic", "")}
        for point in results
    ]


# Agentic tool - will be bound to LLM
@lc_tool
def search_docs(query: str) -> str:
    """Search the knowledge base for relevant documents."""
    docs = retrieve_from_qdrant(query, top_k=5)
    return json.dumps(docs, indent=2)


# Prompts
agent_prompt = PromptTemplate([
    {"role": "system", "content": "You are a helpful assistant. Use the search_docs tool to find relevant information before answering."},
    {"role": "user", "content": "{{question}}"},
])

generation_prompt = PromptTemplate([
    {"role": "system", "content": "Answer concisely using the provided context."},
    {"role": "user", "content": "Context:\n{{context}}\n\nQuestion: {{question}}"},
])


# LangGraph nodes
@span(kind="AGENT", name="rag_agent")
def agent_node(state: AgentState) -> dict:
    """Agent decides to search or answer."""
    messages = state["messages"]
    question = messages[0].content if messages else ""
    
    tools = [search_docs]
    llm_with_tools = llm.bind_tools(tools)
    
    with trace("agent_llm_call", kind="LLM", prompt_template=agent_prompt):
        formatted = agent_prompt.compile(question=question)
        response = llm_with_tools.invoke(formatted)
    
    return {"messages": [response]}


@span(kind="CHAIN", name="rerank_step")
def rerank_node(state: AgentState) -> dict:
    """Rerank retrieved documents (RERANKER span)."""
    messages = state["messages"]
    question = messages[0].content
    last_msg = messages[-1]
    
    if not hasattr(last_msg, "content") or not last_msg.content:
        return {"messages": [last_msg]}
    
    try:
        retrieved = json.loads(str(last_msg.content))
        if not isinstance(retrieved, list):
            return {"messages": [last_msg]}
        
        docs = [Document(page_content=d.get("content", "")) for d in retrieved if d.get("content")]
        
        if not docs:
            return {"messages": [last_msg]}
        
        # RERANKER span with attributes
        with trace("cohere_rerank", kind="RERANKER") as rerank_span:
            rerank_span.set_attribute("neatlogs.reranker.model_name", COHERE_RERANK_MODEL)
            rerank_span.set_attribute("neatlogs.reranker.query", question)
            rerank_span.set_attribute("neatlogs.reranker.top_k", 3)
            rerank_span.set_attribute("neatlogs.reranker.input_documents", json.dumps([d.page_content[:200] for d in docs]))
            
            reranked = reranker.compress_documents(documents=docs, query=question)
            
            rerank_span.set_attribute("neatlogs.reranker.output_documents", json.dumps([d.page_content[:200] for d in reranked]))
        
        context = "\n\n".join([d.page_content for d in reranked])
        return {"messages": [AIMessage(content=context)]}
        
    except Exception:
        return {"messages": [last_msg]}


@span(kind="CHAIN", name="generate_step")
def generate_node(state: AgentState) -> dict:
    """Generate final answer (LLM span)."""
    messages = state["messages"]
    question = messages[0].content
    context = messages[-1].content if len(messages) > 1 else ""
    
    with trace("generation_llm_call", kind="LLM", prompt_template=generation_prompt):
        formatted = generation_prompt.compile(context=context, question=question)
        response = llm.invoke(formatted)
    
    return {"messages": [response]}


def build_graph():
    """Build LangGraph workflow."""
    tools = [search_docs]
    
    wf = StateGraph(AgentState)
    
    wf.add_node("agent", agent_node)
    wf.add_node("retrieve", ToolNode(tools))
    wf.add_node("rerank", rerank_node)
    wf.add_node("generate", generate_node)
    
    wf.add_edge(START, "agent")
    wf.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
    wf.add_edge("retrieve", "rerank")
    wf.add_edge("rerank", "generate")
    wf.add_edge("generate", END)
    
    return wf.compile()


@span(kind="WORKFLOW", name="detection_test_workflow")
def run_detection_test_pipeline(queries: List[str]) -> List[str]:
    """Run detection test pipeline - generates traces with various failure patterns."""
    
    # Setup
    n = setup_qdrant_collection()
    print(f"Seeded {n} documents to Qdrant\n")
    
    graph = build_graph()
    answers = []
    
    for query in queries:
        print(f"Q: {query}")
        inputs = {"messages": [HumanMessage(content=query)]}
        
        for output in graph.stream(inputs):
            for key, value in output.items():
                if key == "generate" and isinstance(value, dict):
                    msgs = value.get("messages", [])
                    if msgs:
                        answer = msgs[-1].content
                        answers.append(answer)
                        print(f"A: {answer}\n")
    
    return answers


def main():
    try:
        print("\n" + "="*80)
        print("Detection Test Workflow - Generating Traces for AI Service Testing")
        print("="*80 + "\n")
        
        # Queries designed to trigger different detection patterns
        queries = [
            "What happens when API requests fail with 500 errors?",  # Should retrieve error handling docs
            "How do I handle high latency responses?",  # Performance-related
            "What should I do when my agent apologizes?",  # Quality issues
        ]
        
        run_detection_test_pipeline(queries)
        
        print("="*80)
        print("\n✅ Traces generated successfully!")
        print(f"\n📊 Expected detections:")
        print(f"   Project ID: {PROJECT_ID}")
        print(f"   - High Latency: LLM spans > 5000ms")
        print(f"   - Apology Pattern: Agent/LLM outputs with 'sorry'/'apologize'")
        print(f"   - Task Failure: Classifier ML detection")
        print(f"\n🔍 Check AI Service logs at: http://localhost:8000/api/v1/detect/trace")
        print(f"   Postgres: detection_spans table")
        print(f"   ClickHouse: detection_results table")
        print("="*80)
        
    except KeyboardInterrupt:
        raise SystemExit(130)
    finally:
        flush()
        shutdown()


if __name__ == "__main__":
    main()
