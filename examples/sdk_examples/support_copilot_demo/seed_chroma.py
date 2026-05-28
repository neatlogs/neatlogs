"""Build an in-process Chroma collection and seed it from kb_data.

Returns a `chromadb.api.models.Collection.Collection`. The caller queries it.
Auto-instrumented because `instrumentations=["chromadb"]` is passed to neatlogs.init().
"""
from __future__ import annotations

import chromadb

from kb_data import KbDoc, kb_url


def build_collection(seeds: list[KbDoc], collection_name: str = "kb_docs") -> "chromadb.api.models.Collection.Collection":
    client = chromadb.EphemeralClient()
    # Drop if exists so reruns are clean.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    coll = client.create_collection(name=collection_name)
    coll.add(
        ids=[d.id for d in seeds],
        documents=[f"{d.title}\n\n{d.body}" for d in seeds],
        metadatas=[
            {
                "title": d.title,
                "url": kb_url(d),
                "last_updated": d.last_updated,
                "version": d.version or "",
            }
            for d in seeds
        ],
    )
    return coll
