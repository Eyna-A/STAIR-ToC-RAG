"""Unit tests for the indexing pipeline and ToC-aware retriever.

Uses a deterministic hashing-based fake embedder so the suite runs fully
offline, without downloading a real sentence-transformers model.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.indexer.chunker import build_chunks
from src.indexer.pipeline import HierarchicalIndexer
from src.indexer.vector_store import InMemoryVectorStore
from src.retriever.stair_retriever import STAIRRetriever

SAMPLE_TREE = [
    {
        "title": "Chapter 1: Introduction to RAG",
        "sections": [
            {
                "title": "Section 1.1: Chunking Methods",
                "paragraphs": [
                    "Conventional methods cut text by a fixed word count.",
                    "This severs the logical connection between related ideas.",
                ],
            }
        ],
    },
    {
        "title": "Chapter 2: The STAIR Architecture",
        "sections": [
            {
                "title": "Section 2.1: Table of Contents as Addressing",
                "paragraphs": [
                    "STAIR uses the table of contents as a precise data address.",
                    "This reduces the model's hallucination rate to under 0.05 percent.",
                    "It generalizes well even with very few training samples.",
                ],
            }
        ],
    },
]


def fake_embedder(texts):
    """Bag-of-words hashing embedder: deterministic and offline.

    Good enough to correctly rank a query above unrelated paragraphs in
    these small fixtures, without needing network access or a real model.
    """

    def vectorize(text: str) -> np.ndarray:
        vec = np.zeros(64)
        for word in text.lower().split():
            idx = hash(word) % 64
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    return np.array([vectorize(t) for t in texts])


@pytest.fixture
def indexed_store():
    store = InMemoryVectorStore()
    indexer = HierarchicalIndexer(vector_store=store, embed_fn=fake_embedder)
    indexer.index_document(SAMPLE_TREE)
    return store


def test_build_chunks_produces_correct_addresses():
    chunks = build_chunks(SAMPLE_TREE)
    assert len(chunks) == 5
    assert (
        chunks[2].toc_address
        == "Chapter 2: The STAIR Architecture > Section 2.1: Table of Contents as Addressing > Paragraph 1"
    )


def test_build_chunks_skips_empty_paragraphs():
    tree_with_blanks = [
        {"title": "Ch", "sections": [{"title": "Sec", "paragraphs": ["real text", "   ", ""]}]}
    ]
    chunks = build_chunks(tree_with_blanks)
    assert len(chunks) == 1
    assert chunks[0].content == "real text"


def test_indexer_upserts_all_chunks_into_the_store(indexed_store):
    assert len(indexed_store.all_chunks()) == 5


def test_retriever_returns_hallucination_paragraph(indexed_store):
    retriever = STAIRRetriever(vector_store=indexed_store, embed_fn=fake_embedder)
    results = retriever.retrieve("hallucination rate STAIR reduces", top_k=1)

    assert len(results) == 1
    assert "hallucination" in results[0].content.lower()
    assert results[0].toc_address.startswith("Chapter 2")


def test_retriever_parent_context_expansion(indexed_store):
    retriever = STAIRRetriever(vector_store=indexed_store, embed_fn=fake_embedder, expand_context=True)
    results = retriever.retrieve("hallucination rate STAIR reduces", top_k=1)

    result = results[0]
    assert result.context_before is not None
    assert "precise data address" in result.context_before.lower()


def test_retriever_can_disable_context_expansion(indexed_store):
    retriever = STAIRRetriever(vector_store=indexed_store, embed_fn=fake_embedder, expand_context=False)
    results = retriever.retrieve("hallucination rate STAIR reduces", top_k=1)

    assert results[0].context_before is None
    assert results[0].context_after is None


def test_retriever_returns_empty_list_on_empty_index():
    empty_store = InMemoryVectorStore()
    retriever = STAIRRetriever(vector_store=empty_store, embed_fn=fake_embedder)
    assert retriever.retrieve("anything", top_k=3) == []
