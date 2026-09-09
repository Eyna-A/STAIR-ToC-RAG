"""High-level orchestration: document tree -> chunks -> embeddings -> vector store."""
from __future__ import annotations

from typing import Dict, List

from src.indexer.chunker import HierarchicalChunk, build_chunks
from src.indexer.vector_store import BaseVectorStore, EmbedFn


class HierarchicalIndexer:
    """Wires the chunker, embedder, and vector store together into one call."""

    def __init__(self, vector_store: BaseVectorStore, embed_fn: EmbedFn):
        self.vector_store = vector_store
        self.embed_fn = embed_fn

    def index_document(self, doc_tree: List[Dict]) -> List[HierarchicalChunk]:
        chunks = build_chunks(doc_tree)
        if not chunks:
            return []

        representations = [c.full_representation for c in chunks]
        vectors = self.embed_fn(representations)
        self.vector_store.upsert(chunks, vectors)
        return chunks
