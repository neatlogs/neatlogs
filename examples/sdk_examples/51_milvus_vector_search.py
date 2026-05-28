"""
Milvus Vector Search with Neatlogs Observability
=================================================
Demonstrates tracing Milvus vector database operations using neatlogs.
Creates RETRIEVER and TOOL spans for insert and search operations.

Architecture:
  init → create_collection → insert_vectors → vector_search (RETRIEVER) → cleanup

Uses MilvusClient API which supports both:
  - milvus-lite  : embedded, no server needed (set MILVUS_URI to a .db file path)
  - Milvus server: set MILVUS_URI to http://host:19530
  - Zilliz Cloud : set MILVUS_URI + MILVUS_TOKEN

Prereqs:
  pip install -r 51_requirements.txt

Run:
  python 51_milvus_vector_search.py

Optional env vars (.env):
  NEATLOGS_API_KEY
  NEATLOGS_ENDPOINT
  MILVUS_URI    (default: /tmp/neatlogs_milvus_demo.db  — uses milvus-lite)
  MILVUS_TOKEN  (only for Zilliz Cloud)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
from dotenv import load_dotenv
from opentelemetry import trace as otel_trace
from pymilvus import MilvusClient

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import neatlogs

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_DEFAULT_DB = os.path.join(tempfile.gettempdir(), "neatlogs_milvus_demo.db")
MILVUS_URI   = os.getenv("MILVUS_URI", _DEFAULT_DB)   # .db path → lite; http:// → server
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")               # Zilliz Cloud only
COLLECTION   = "neatlogs_demo"
DIM          = 128   # embedding dimension
NUM_DOCS     = 50    # documents to insert
TOP_K        = 5     # results per search

# ---------------------------------------------------------------------------
# Sample knowledge base
# ---------------------------------------------------------------------------
DOCUMENTS = [
    {
        "id": i,
        "text": f"Document {i}: sample knowledge about topic {i % 10}",
        "category": f"cat_{i % 5}",
        "vector": np.random.rand(DIM).astype("float32").tolist(),
    }
    for i in range(NUM_DOCS)
]

# ---------------------------------------------------------------------------
# Neatlogs initialisation
# ---------------------------------------------------------------------------
neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY"),
    endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://staging-cloud.neatlogs.com/api/data/v4/batch"),
    workflow_name="milvus-vector-search",
    tags=[
        "sdk-examples",
        "example-milvus-vector-search",
        "milvus",
        "vector-db",
        "rag",
    ],
    disable_export=not os.getenv("NEATLOGS_API_KEY"),
    debug=True,
)

# ---------------------------------------------------------------------------
# Milvus helpers — each wrapped in a neatlogs span
# ---------------------------------------------------------------------------

@neatlogs.span(kind="TOOL", name="connect_milvus")
def connect_milvus() -> MilvusClient:
    """Connect via MilvusClient (works with lite, server, or Zilliz Cloud)."""
    kwargs = {"uri": MILVUS_URI}
    if MILVUS_TOKEN:
        kwargs["token"] = MILVUS_TOKEN
    client = MilvusClient(**kwargs)
    mode = "milvus-lite (embedded)" if MILVUS_URI.endswith(".db") else MILVUS_URI
    print(f"  Connected — mode: {mode}")
    return client


@neatlogs.span(kind="TOOL", name="create_collection")
def create_collection(client: MilvusClient) -> None:
    """Create collection with a float-vector field."""
    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)
        print(f"  Dropped existing collection '{COLLECTION}'")
    client.create_collection(
        collection_name=COLLECTION,
        dimension=DIM,
        metric_type="L2",
        id_type="int",
    )
    print(f"  Created collection '{COLLECTION}' (dim={DIM}, metric=L2)")


@neatlogs.span(kind="TOOL", name="insert_documents")
def insert_documents(client: MilvusClient) -> None:
    """Insert sample documents with random embeddings."""
    client.insert(collection_name=COLLECTION, data=DOCUMENTS)
    print(f"  Inserted {len(DOCUMENTS)} documents")


@neatlogs.span(kind="RETRIEVER", name="milvus_vector_search")
def vector_search(client: MilvusClient, query_vector: list[float], query_text: str) -> list[dict]:
    """
    ANN search against Milvus — tagged as RETRIEVER span.
    Matches the span kind convention used for RAG pipelines.
    """
    span = otel_trace.get_current_span()
    span.set_attribute("neatlogs.retrieval.query",      query_text)
    span.set_attribute("neatlogs.retrieval.top_k",      TOP_K)
    span.set_attribute("neatlogs.retrieval.vector_db",  "milvus")
    span.set_attribute("neatlogs.retrieval.collection", COLLECTION)

    results = client.search(
        collection_name=COLLECTION,
        data=[query_vector],
        limit=TOP_K,
        output_fields=["text", "category"],
    )

    hits = [
        {
            "id":       hit["id"],
            "distance": hit["distance"],
            "text":     hit["entity"].get("text"),
            "category": hit["entity"].get("category"),
        }
        for hit in results[0]
    ]
    span.set_attribute("neatlogs.retrieval.result_count", len(hits))
    span.set_attribute("neatlogs.retrieval.documents",    json.dumps(hits))
    return hits


@neatlogs.span(kind="TOOL", name="cleanup")
def cleanup(client: MilvusClient) -> None:
    """Drop collection and close client."""
    client.drop_collection(COLLECTION)
    client.close()
    # Remove lite db file if it was ephemeral
    if MILVUS_URI.endswith(".db") and os.path.exists(MILVUS_URI):
        os.remove(MILVUS_URI)
    print(f"  Cleaned up collection '{COLLECTION}'")


# ---------------------------------------------------------------------------
# Test queries
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    "knowledge about topic 3 and related concepts",
    "information on category cat_1 items",
    "document search for topic 7",
]

# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "=" * 70)
    print("MILVUS VECTOR SEARCH — Neatlogs Observability Demo")
    print(f"  URI : {MILVUS_URI}")
    print("=" * 70)

    with neatlogs.trace("milvus_rag_pipeline", kind="WORKFLOW"):

        client = connect_milvus()
        create_collection(client)
        insert_documents(client)

        print(f"\n  Running {len(TEST_QUERIES)} search queries...\n")
        for i, query_text in enumerate(TEST_QUERIES, 1):
            query_vector = np.random.rand(DIM).astype("float32").tolist()
            hits = vector_search(client, query_vector, query_text)
            top = hits[0]["text"] if hits else "none"
            print(f"  [{i}] '{query_text[:45]}' → {len(hits)} hits | top: {top}")

        cleanup(client)

    neatlogs.flush()
    neatlogs.shutdown()

    print("\n" + "=" * 70)
    print("Done — all spans recorded")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
