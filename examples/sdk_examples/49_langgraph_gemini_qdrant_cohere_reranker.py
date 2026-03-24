"""
LangGraph + Gemini (LangChain Google GenAI) + Qdrant RAG + Cohere Reranker.

Env:
  - NEATLOGS_API_KEY
  - GEMINI_API_KEY (or GOOGLE_API_KEY)
  - QDRANT_URL (e.g. http://localhost:6333 or https://*.cloud.qdrant.io)
  - QDRANT_API_KEY (optional for local)
  - COHERE_API_KEY

Optional:
  - BLOG_URL (default: Lilian Weng agents post)
  - QUESTION (default provided)
  - QDRANT_COLLECTION (default: neatlogs_ai_blog_search)

Run:
  python neatlogs/examples/49_langgraph_gemini_qdrant_cohere_reranker.py
  python -m neatlogs.examples.49_langgraph_gemini_qdrant_cohere_reranker
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Literal, Optional, TypedDict, Annotated, Sequence
from uuid import uuid4
from functools import partial

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_classic import hub
from langgraph.graph.message import add_messages
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_cohere import CohereRerank
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from qdrant_client import QdrantClient
import google.genai as genai
from google.genai import types as genai_types

from langchain_qdrant import QdrantVectorStore

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from neatlogs import flush, init, shutdown, trace, PromptTemplate

def _env_default(k: str, v: str) -> None:
    os.environ.setdefault(k, v)


_env_default("LANGCHAIN_TRACING_V2", "false")
_env_default("LANGSMITH_TRACING", "false")
_env_default("LANGSMITH_API_KEY", "")
_env_default("NEATLOGS_LOG_SPANS", "true")
_env_default("NEATLOGS_LOG_METRICS", "true")
_env_default("NEATLOGS_LOG_RAW_SPANS", "true")
# _env_default("NEATLOGS_LOG_SPANS_FILE", "spans_49_langgraph_gemini_qdrant_cohere_reranker.jsonl")
# _env_default("NEATLOGS_LOG_RAW_SPANS_FILE", "spans_raw_49_langgraph_gemini_qdrant_cohere_reranker.jsonl")
# _env_default("NEATLOGS_LOG_METRICS_FILE", "metrics_49_langgraph_gemini_qdrant_cohere_reranker.jsonl")
_env_default("NEATLOGS_LOG_SPANS_FILE", "spans_49_langgraph_gemini_qdrant_cohere_reranker_v3.jsonl")
_env_default("NEATLOGS_LOG_RAW_SPANS_FILE", "spans_raw_49_langgraph_gemini_qdrant_cohere_reranker_v3.log")
_env_default("NEATLOGS_LOG_METRICS_FILE", "metrics_49_langgraph_gemini_qdrant_cohere_reranker_v3.jsonl")

def _require_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise SystemExit(f"Missing required env var: {key}")
    return val


def _get_gemini_api_key() -> str:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""

@dataclass(frozen=True)
class Settings:
    neatlogs_api_key: str
    neatlogs_endpoint: str
    workflow_name: str
    gemini_api_key: str
    gemini_model: str
    gemini_embedding_model: str
    qdrant_url: str
    qdrant_api_key: Optional[str]
    cohere_api_key: str
    cohere_rerank_model: str
    rerank_top_n: int
    blog_url: str
    question: str
    qdrant_collection: str
    retriever_k: int
    max_rewrite_tries: int


def load_settings() -> Settings:
    neatlogs_api_key = _require_env("NEATLOGS_API_KEY")
    gemini_api_key = _get_gemini_api_key()
    if not gemini_api_key:
        raise SystemExit("Missing required env var: GEMINI_API_KEY (or GOOGLE_API_KEY)")

    gemini_embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()
    if gemini_embedding_model and not gemini_embedding_model.startswith("models/"):
        gemini_embedding_model = f"models/{gemini_embedding_model}"

    return Settings(
        neatlogs_api_key=neatlogs_api_key,
        neatlogs_endpoint="http://localhost:3000/api/data/v4/batch",
        workflow_name="langgraph-gemini-qdrant-cohere-rerank",
        gemini_api_key=gemini_api_key,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
        gemini_embedding_model=gemini_embedding_model,
        qdrant_url="localhost:6333",
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
        cohere_api_key=_require_env("COHERE_API_KEY"),
        cohere_rerank_model="rerank-english-v3.0",
        rerank_top_n=5,
        blog_url="https://lilianweng.github.io/posts/2023-06-23-agent/",
        question="What are the main types of agent memory described, and what are the trade-offs?",
        qdrant_collection="neatlogs_ai_blog_search_1",
        retriever_k=10,
        max_rewrite_tries=2,
    )


class State(TypedDict):
    question: str
    documents: list[object]
    answer: str
    tries: int


settings = load_settings()

init(
    api_key=settings.neatlogs_api_key,
    endpoint=settings.neatlogs_endpoint,
    workflow_name=settings.workflow_name,
    instrumentations=["langchain", "qdrant", "cohere", "google_genai", "openai"],
    debug=True,
)


gemini_client = genai.Client(api_key=settings.gemini_api_key)

def initialize_components():
    """Initialize components that require API keys"""
    # Initialize embedding model with API key
    # embedding_model = GoogleGenerativeAIEmbeddings(
    #     model=settings.gemini_embedding_model,
    #     google_api_key=settings.gemini_api_key,
    # )
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    # Initialize Qdrant client
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    # Initialize vector store
    # db = QdrantVectorStore(
    #     client=client,
    #     collection_name=settings.qdrant_collection,
    #     embedding=embedding_model
    # )

    try:
        client.get_collection(collection_name=settings.qdrant_collection)
    except Exception as e:
        try:
            from qdrant_client.http.exceptions import UnexpectedResponse
            from qdrant_client.models import Distance, VectorParams

            if isinstance(e, UnexpectedResponse) and getattr(e, "status_code", None) == 404:
                dim = len(embedding_model.embed_query("neatlogs-qdrant-dimension-probe"))
                client.create_collection(
                    collection_name=settings.qdrant_collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            else:
                raise
        except Exception:
            raise

    db = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embedding_model
    )

    reranker = CohereRerank(
        cohere_api_key=settings.cohere_api_key,
        model=settings.cohere_rerank_model,
        top_n=settings.rerank_top_n,
    )

    return embedding_model, client, db, reranker

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Nodes
## agent node
def agent(state, tools):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]

    question = ""
    if messages and isinstance(messages[0], HumanMessage):
        question = str(messages[0].content)

    tool_name = "retrieve_blog_posts"
    tool_description = (
        "Search and return information about blog posts on LLMs, LLM agents, prompt engineering, and adversarial attacks on LLMs."
    )
    if tools:
        tool_name = getattr(tools[0], "name", tool_name) or tool_name
        tool_description = getattr(tools[0], "description", tool_description) or tool_description

    tool_decl = genai_types.Tool(
        function_declarations=[
            genai_types.FunctionDeclaration(
                name=tool_name,
                description=tool_description,
                parameters=genai_types.Schema(
                    type="OBJECT",
                    properties={
                        "query": genai_types.Schema(type="STRING", description="Search query"),
                        "k": genai_types.Schema(type="INTEGER", description="Number of documents to retrieve"),
                    },
                    required=["query"],
                ),
            )
        ]
    )

    cfg = genai_types.GenerateContentConfig(
        tools=[tool_decl],
        automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0,
        max_output_tokens=256,
    )

    # Define PromptTemplate inside function
    rag_agent_prompt = PromptTemplate(
        "You are a RAG agent.\n"
        "Call the tool `{{tool_name}}` exactly once to fetch context for the user question.\n"
        "Do not answer the question yet. Only call the tool.\n\n"
        "Question: {{question}}"
    )
    
    # Format prompt using PromptTemplate
    # PromptContext is automatically set by compile(), SDK will capture it
    prompt = rag_agent_prompt.compile(tool_name=tool_name, question=question)

    # Use streaming to get the response
    # Wrap in trace for orchestration context (LLM span auto-instrumented inside)
    with trace("rag_agent_call", kind="CHAIN"):
        stream = gemini_client.models.generate_content_stream(
            model=settings.gemini_model,
            contents=[genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])],
            config=cfg,
        )

    # Collect the streaming response
    resp = None
    text_parts = []
    for chunk in stream:
        resp = chunk  # Keep the last chunk which has full response metadata
        if chunk.text:
            text_parts.append(chunk.text)
            print(chunk.text, end="", flush=True)
    
    print()  # New line after streaming

    fcalls = resp.function_calls or [] if resp else []
    if not fcalls:
        return {"messages": [AIMessage(content="".join(text_parts))]}

    fcall = fcalls[0]
    tool_call_id = f"genai_{uuid4().hex}"
    ai = AIMessage(
        content="",
        tool_calls=[
            {
                "id": tool_call_id,
                "name": fcall.name,
                "args": dict(fcall.args or {}),
            }
        ],
    )
    return {"messages": [ai]}

## rewrite node
def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    # Define PromptTemplate inside function
    query_rewrite_prompt = PromptTemplate(
        "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
        "Here is the initial question:\n"
        "------- \n"
        "{{question}}\n"
        "------- \n"
        "Formulate an improved question:"
    )
    
    # Format prompt using PromptTemplate
    # PromptContext is automatically set by compile(), SDK will capture it
    prompt = query_rewrite_prompt.compile(question=question)

    # Use streaming to get the response
    # Wrap in trace for orchestration context (LLM span auto-instrumented inside)
    with trace("query_rewrite", kind="CHAIN"):
        stream = gemini_client.models.generate_content_stream(
            model=settings.gemini_model,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config={"temperature": 0},
        )
    
    # Collect the streaming response
    text_parts = []
    for chunk in stream:
        if chunk.text:
            text_parts.append(chunk.text)
            print(chunk.text, end="", flush=True)
    
    print()  # New line after streaming
    
    response_text = "".join(text_parts)
    return {"messages": [AIMessage(content=response_text)]}

## rerank node
def rerank(state, reranker: CohereRerank):
    """
    Rerank retrieved documents using Cohere Rerank.

    Args:
        state (messages): The current state
        reranker: The CohereRerank instance

    Returns:
        dict: The updated state with reranked documents
    """
    print("---RERANK DOCUMENTS---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs_payload = None
    try:
        docs_payload = json.loads(str(last_message.content or ""))
    except Exception:
        docs_payload = None

    raw_docs = []
    if isinstance(docs_payload, dict):
        raw_docs = docs_payload.get("documents") or []

    docs: list[Document] = []
    if isinstance(raw_docs, list):
        for d in raw_docs:
            if not isinstance(d, dict):
                continue
            docs.append(
                Document(
                    page_content=str(d.get("page_content") or ""),
                    metadata=dict(d.get("metadata") or {}),
                )
            )

    if not docs:
        return {"messages": [last_message]}

    reranked_docs = reranker.compress_documents(documents=docs, query=str(question))
    print(f"Reranked {len(docs)} documents to top {len(reranked_docs)}")

    context = "\n\n".join([d.page_content for d in reranked_docs])
    return {"messages": [AIMessage(content=context)]}

## check reranked results - replaces grade_documents
def check_reranked_docs(state) -> Literal["generate", "rewrite"]:
    """
    Check if reranked documents are sufficient.
    Replaces grade_documents at the same position in the graph.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether to generate or rewrite
    """
    print("---CHECK RERANKED RESULTS---")
    messages = state["messages"]
    last_message = messages[-1]
    docs = last_message.content
    
    # If we have reranked documents, proceed to generate
    if docs and len(docs) > 0:
        print("---DECISION: RERANKED DOCS AVAILABLE---")
        return "generate"
    else:
        print("---DECISION: NO RELEVANT DOCS, REWRITING---")
        return "rewrite"

## generate node
def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Define PromptTemplate inside function
    rag_generation_prompt = PromptTemplate(
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        "Question: {{question}}\n\n"
        "Context: {{docs}}\n\n"
        "Answer:"
    )
    
    # Format prompt using PromptTemplate
    prompt = rag_generation_prompt.compile(question=question, docs=docs)

    # Use streaming to get the response
    # Wrap in trace for orchestration context (LLM span auto-instrumented inside)
    with trace("rag_generation", kind="CHAIN"):
        stream = gemini_client.models.generate_content_stream(
            model=settings.gemini_model,
            contents=[genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])],
            config=genai_types.GenerateContentConfig(temperature=0, max_output_tokens=512),
        )

        # Collect the streaming response
        text_parts = []
        for chunk in stream:
            if chunk.text:
                text_parts.append(chunk.text)
                print(chunk.text, end="", flush=True)
        
        print()  # New line after streaming
        
        return {"messages": ["".join(text_parts)]}

# graph function
def get_graph(retriever_tool, reranker):
    tools = [retriever_tool]  # Create tools list here
    
    # Define a new graph
    workflow = StateGraph(AgentState)

    # Use partial to pass tools to the agent function
    workflow.add_node("agent", partial(agent, tools=tools))
    
    # Rest of the graph setup remains the same
    retrieve = ToolNode(tools)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank", partial(rerank, reranker=reranker))  # Reranking retrieved documents
    workflow.add_node("rewrite", rewrite)  # Re-writing the question
    workflow.add_node(
        "generate", generate
    )  # Generating a response after we know the documents are relevant
    # Call agent node to decide to retrieve or not
    workflow.add_edge(START, "agent")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "agent",
        # Assess agent decision
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    # After retrieve, rerank the documents
    workflow.add_edge("retrieve", "rerank")

    # Edges taken after the `rerank` node is called.
    # check_reranked_docs replaces grade_documents at the same position
    workflow.add_conditional_edges(
        "rerank",
        # Check reranked documents (replaces grade_documents)
        check_reranked_docs,
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    # Compile
    graph = workflow.compile()

    return graph

def generate_message(graph, inputs):
    generated_message = ""

    for output in graph.stream(inputs):
        for key, value in output.items():
            if key == "generate" and isinstance(value, dict):
                generated_message = value.get("messages", [""])[0]
    
    return generated_message

def add_documents_to_qdrant(url, db):
    docs = WebBaseLoader(url).load()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_chunks = text_splitter.split_documents(docs)
    uuids = [str(uuid4()) for _ in range(len(doc_chunks))]
    db.add_documents(documents=doc_chunks, ids=uuids)
    return True


def main():

    try:    
        with trace(
                "ai_blog_search_startup",
                kind="WORKFLOW",
            ):

                embedding_model, client, db, reranker = initialize_components()
                if not all([embedding_model, client, db, reranker]):
                    return

                # Add documents FIRST (creates collection and adds docs)
                add_documents_to_qdrant(settings.blog_url, db)
                
                # Initialize retriever and tools - retrieve more docs for reranking
                retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": settings.retriever_k})

                @tool("retrieve_blog_posts")
                def retrieve_blog_posts(query: str, k: int = settings.retriever_k) -> str:
                    """Search the indexed blog content and return up to `k` relevant chunks as JSON."""
                    docs = retriever.invoke(query)
                    docs = list(docs)[: int(k)]
                    payload = {
                        "documents": [
                            {"page_content": d.page_content, "metadata": dict(getattr(d, "metadata", {}) or {})}
                            for d in docs
                        ]
                    }
                    return json.dumps(payload, default=str)

                tools = [retrieve_blog_posts]
                graph = get_graph(retrieve_blog_posts, reranker)
                inputs = {"messages": [HumanMessage(content=settings.question)]}
                response = generate_message(graph, inputs)
                print(response)
    except KeyboardInterrupt:
        raise SystemExit(130)
    finally:
        flush()
        shutdown()


if __name__ == "__main__":
    main()
