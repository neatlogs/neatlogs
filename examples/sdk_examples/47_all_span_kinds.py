"""
Production-realistic support workflow emitting all span kinds:
WORKFLOW, CHAIN, AGENT, RETRIEVER, EMBEDDING, RERANKER, TOOL, MCP_TOOL, HTTP, LLM.

Prereqs:
  pip install openai chromadb haystack-ai mcp requests

Run MCP server first:
  python examples/shared_mcp_server.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from neatlogs import (
    PromptTemplate,
    agent,
    chain,
    flush,
    init,
    mcp_tool,
    retriever,
    shutdown,
    tool,
    trace,
    workflow,
)


def _env_default(k: str, v: str) -> None:
    os.environ.setdefault(k, v)


_env_default("NEATLOGS_LOG_SPANS", "true")
_env_default("NEATLOGS_LOG_METRICS", "true")


try:
    from haystack import Document, Pipeline, component

    def _cosine(a: list[float], b: list[float]) -> float:
        import math

        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a)
        nb = sum(y * y for y in b)
        denom = (math.sqrt(na) * math.sqrt(nb)) or 1e-12
        return float(dot / denom)

    @component
    class OpenAIEmbeddingSimilarityRanker:
        def __init__(self, model: str = "text-embedding-3-small") -> None:
            self.model = model
            self._client = None

        def _get_client(self):
            if self._client is None:
                from openai import OpenAI

                self._client = OpenAI()
            return self._client

        @component.output_types(documents=list[Document])
        def run(
            self, *, query: str, documents: list[Document], top_k: int = 3
        ) -> dict[str, list[Document]]:
            client = self._get_client()
            texts = [query] + [(d.content or "") for d in documents]
            resp = client.embeddings.create(model=self.model, input=texts)
            vectors = [item.embedding for item in resp.data]
            qvec = vectors[0]
            for doc, vec in zip(documents, vectors[1:]):
                doc.score = _cosine(qvec, vec)
            ranked = sorted(documents, key=lambda d: d.score or 0.0, reverse=True)
            return {"documents": ranked[:top_k]}

except Exception as e:
    raise RuntimeError("This example requires `haystack-ai` installed.") from e


@tool(name="get_product_info", tool_name="get_product_info")
def get_product_info(product_id: str) -> Dict[str, Any]:
    import requests

    r = requests.get(f"https://dummyjson.com/products/{product_id}", timeout=10)
    r.raise_for_status()
    data = r.json()
    return {
        "product_id": product_id,
        "title": data.get("title"),
        "price": data.get("price"),
        "stock": data.get("stock"),
    }


@tool(name="check_order_status", tool_name="check_order_status")
def check_order_status(order_id: str) -> Dict[str, Any]:
    import requests

    r = requests.get(f"https://dummyjson.com/carts/{order_id}", timeout=10)
    r.raise_for_status()
    data = r.json()
    return {
        "order_id": order_id,
        "total": data.get("total"),
        "items_count": len(data.get("products", []) or []),
    }


@mcp_tool(name="mcp_get_time")
async def mcp_get_time() -> str:
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    server_url = os.getenv("NEATLOGS_MCP_SERVER_URL", "http://127.0.0.1:8000/sse")
    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool("get_time", {})
            return getattr(res.content[0], "text", "") if res.content else ""


@mcp_tool(name="mcp_store_data")
async def mcp_store_data(key: str, value: str) -> str:
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    server_url = os.getenv("NEATLOGS_MCP_SERVER_URL", "http://127.0.0.1:8000/sse")
    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool("store_data", {"key": key, "value": value})
            return getattr(res.content[0], "text", "") if res.content else ""


@chain(name="setup_knowledge_base")
def setup_knowledge_base(collection):
    from openai import OpenAI

    client = OpenAI()
    docs = [
        "Return policy: returns within 30 days for a full refund.",
        "Shipping: free shipping for orders over $50 in the continental US.",
        "Support: 24/7 support via chat, email, or phone.",
    ]
    resp = client.embeddings.create(model="text-embedding-3-small", input=docs)
    embeddings = [x.embedding for x in resp.data]
    collection.add(
        ids=[f"doc_{i}" for i in range(len(docs))],
        documents=docs,
        embeddings=embeddings,
        metadatas=[{"source": "policy"} for _ in docs],
    )
    return collection


@retriever(name="retrieve_documents")
def retrieve_documents(query: str, collection) -> List[Document]:
    from openai import OpenAI

    client = OpenAI()
    q_emb = client.embeddings.create(model="text-embedding-3-small", input=[query])
    qvec = q_emb.data[0].embedding
    res = collection.query(query_embeddings=[qvec], n_results=3)
    return [
        Document(id=res["ids"][0][i], content=res["documents"][0][i], meta={})
        for i in range(len(res["documents"][0]))
    ]


@chain(name="rerank_documents")
def rerank_documents(query: str, documents: List[Document], top_k: int = 2) -> List[Document]:
    ranker = OpenAIEmbeddingSimilarityRanker(model="text-embedding-3-small")
    pipe = Pipeline()
    pipe.add_component("ranker", ranker)
    result = pipe.run(data={"ranker": {"query": query, "documents": documents, "top_k": top_k}})
    return result["ranker"]["documents"]


@agent(name="routing_agent", role="Router", goal="Route to appropriate tools")
def route_to_tools(query: str) -> Dict[str, Any]:
    tool_out: Dict[str, Any] = {}
    q_lower = query.lower()
    if "order" in q_lower or "status" in q_lower:
        tool_out["order"] = check_order_status(order_id="5")
    if "product" in q_lower:
        tool_out["product"] = get_product_info(product_id="3")
    return tool_out


@agent(name="generate_answer")
def generate_answer(query: str, context: str, tools: str, session: str) -> str:
    from openai import OpenAI

    prompt = PromptTemplate(
        [
            {
                "role": "system",
                "content": "You are customer support. Use context and tool outputs.\n\nContext:\n{{context}}\n\nTools:\n{{tools}}\n\nSession:\n{{session}}",
            },
            {"role": "user", "content": "{{query}}"},
        ]
    )
    
    with trace(name="stream_llm_response", kind="LLM", prompt_template=prompt):
        messages = prompt.compile(context=context, tools=tools, session=session, query=query)
        client = OpenAI()
        stream = client.chat.completions.create(
            model=os.getenv("NEATLOGS_OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            temperature=0.2,
        )
        chunks: List[str] = []
        for evt in stream:
            # Skip chunks without choices (final usage chunk)
            if not evt.choices:
                continue
            delta = evt.choices[0].delta
            if delta and delta.content:
                chunks.append(delta.content)
        return "".join(chunks).strip()


@workflow(name="support_workflow")
async def run_support_workflow(query: str, collection) -> str:
    ts = await mcp_get_time()
    _ = await mcp_store_data(key=f"q_{abs(hash(query))}", value=query)
    retrieved_docs = retrieve_documents(query, collection)
    ranked_docs = rerank_documents(query, retrieved_docs, top_k=2)
    tool_results = route_to_tools(query)
    context = "\n".join([d.content or "" for d in ranked_docs])
    answer = generate_answer(
        query=query,
        context=context,
        tools=json.dumps(tool_results, default=str),
        session=ts,
    )
    return answer


async def main() -> None:
    init(
        api_key=os.getenv("NEATLOGS_API_KEY", "test-key"),
        endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:3000/api/data/v4/batch"),
        workflow_name="production-support-all-span-kinds",
        instrumentations=["openai", "chromadb", "mcp", "haystack"],
        debug=True,
    )

    query = "What is your return policy and can you check my order status?"

    try:
        with trace(name="all_span_kinds"):
            import chromadb

            chroma_client = chromadb.Client()
            collection = chroma_client.get_or_create_collection(
                name="support_kb", metadata={"hnsw:space": "cosine"}
            )
            collection = setup_knowledge_base(collection)
            answer = await run_support_workflow(query, collection)
            print(f"\nQuery: {query}\nAnswer: {answer}\n")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


def run_api_server():
    """Run support workflow as HTTP API server."""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    # Initialize Neatlogs
    init(
        api_key=os.getenv("NEATLOGS_API_KEY", "test-key"),
        endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:3000/api/data/v4/batch"),
        workflow_name="support-workflow-api-server",
        instrumentations=["openai", "chromadb", "mcp", "haystack"],
        debug=True,
    )
    
    # Setup ChromaDB once at startup
    collection_global = None
    try:
        import chromadb
        client = chromadb.Client()
        collection_global = client.get_or_create_collection(
            name="support_kb", metadata={"hnsw:space": "cosine"}
        )
        collection_global = setup_knowledge_base(collection_global)
        print("✅ ChromaDB collection ready")
    except Exception as e:
        print(f"⚠️  ChromaDB setup error: {e}")
    
    @app.route('/api/support', methods=['POST'])
    def support_endpoint():
        """HTTP endpoint that runs the support workflow."""
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({"error": "query is required"}), 400
        
        try:
            # Run the support workflow
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_support_workflow(query, collection_global))
            loop.close()
            
            return jsonify({
                "query": query,
                "response": result,
                "status": "success"
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({"status": "healthy"})
    
    print("\n" + "=" * 80)
    print("🚀 Support Workflow API Server Starting...")
    print("=" * 80)
    print("  Endpoint: http://localhost:5001/api/support")
    print("  Health:   http://localhost:5001/health")
    print("=" * 80 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="All span kinds demo")
    parser.add_argument(
        '--mode',
        choices=['normal', 'server'],
        default='normal',
        help='Run normally or as API server'
    )
    args = parser.parse_args()
    
    if args.mode == 'server':
        try:
            import flask
        except ImportError:
            print("❌ Flask is required for server mode. Install with: pip install flask")
            sys.exit(1)
        run_api_server()
    else:
        try:
            asyncio.run(main())
        finally:
            flush()
            shutdown()
