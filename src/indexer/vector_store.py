"""Embedding + vector storage backends for hierarchical chunks.

Two backends ship out of the box:
  - InMemoryVectorStore: pure NumPy, zero external services. Good for tests,
    demos, and small corpora.
  - QdrantVectorStore: production backend, used behind docker-compose.

Both implement the same BaseVectorStore interface, so the indexer and
retriever never need to know which one is active.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.indexer.chunker import HierarchicalChunk

EmbedFn = Callable[[Sequence[str]], np.ndarray]


def default_embedder(model_name: str = "all-MiniLM-L6-v2") -> EmbedFn:
    """Lazily loads a sentence-transformers model and returns an embed function.

    Import is deferred to call-time so importing this module never requires
    torch/transformers to be installed unless you actually need real embeddings
    (tests, for instance, inject a fake embedder instead).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(model.encode(list(texts), normalize_embeddings=True))

    return embed


class BaseVectorStore(ABC):
    """Minimal interface a vector backend must implement."""

    @abstractmethod
    def upsert(self, chunks: List[HierarchicalChunk], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[HierarchicalChunk, float]]: ...

    @abstractmethod
    def all_chunks(self) -> List[HierarchicalChunk]: ...


class InMemoryVectorStore(BaseVectorStore):
    """NumPy-backed store. No external services required."""

    def __init__(self):
        self._chunks: List[HierarchicalChunk] = []
        self._vectors: Optional[np.ndarray] = None

    def upsert(self, chunks: List[HierarchicalChunk], vectors: np.ndarray) -> None:
        self._chunks.extend(chunks)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[HierarchicalChunk, float]]:
        if self._vectors is None or len(self._chunks) == 0:
            return []
        scores = self._vectors @ query_vector
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self._chunks[i], float(scores[i])) for i in top_indices]

    def all_chunks(self) -> List[HierarchicalChunk]:
        return list(self._chunks)


class QdrantVectorStore(BaseVectorStore):
    """Qdrant-backed store for production deployments (see docker-compose.yml)."""

    def __init__(self, url: str = "http://localhost:6333", collection: str = "stair_toc_rag", vector_size: int = 384):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels

        self._qmodels = qmodels
        self.collection = collection
        self.client = QdrantClient(url=url)
        self._chunk_cache: Dict[str, HierarchicalChunk] = {}

        existing = [c.name for c in self.client.get_collections().collections]
        if collection not in existing:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
            )

    def upsert(self, chunks: List[HierarchicalChunk], vectors: np.ndarray) -> None:
        points = []
        for chunk, vector in zip(chunks, vectors):
            self._chunk_cache[chunk.chunk_id] = chunk
            # Qdrant point IDs must be unsigned ints or UUIDs, so we derive a
            # deterministic UUID from our own human-readable chunk_id and keep
            # the original id in the payload for lookups.
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            points.append(
                self._qmodels.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "toc_address": chunk.toc_address,
                        "content": chunk.content,
                        "chapter": chunk.chapter,
                        "section": chunk.section,
                        "paragraph_index": chunk.paragraph_index,
                    },
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[HierarchicalChunk, float]]:
        hits = self.client.search(collection_name=self.collection, query_vector=query_vector.tolist(), limit=top_k)
        results = []
        for hit in hits:
            payload = hit.payload
            chunk = HierarchicalChunk(
                chunk_id=payload["chunk_id"],
                toc_address=payload["toc_address"],
                content=payload["content"],
                chapter=payload["chapter"],
                section=payload["section"],
                paragraph_index=payload["paragraph_index"],
            )
            results.append((chunk, float(hit.score)))
        return results

    def all_chunks(self) -> List[HierarchicalChunk]:
        return list(self._chunk_cache.values())
