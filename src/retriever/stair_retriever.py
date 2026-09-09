"""ToC-aware retrieval engine with parent-context expansion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.indexer.chunker import HierarchicalChunk
from src.indexer.vector_store import BaseVectorStore, EmbedFn


@dataclass
class RetrievalResult:
    toc_address: str
    content: str
    score: float
    context_before: Optional[str] = None
    context_after: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "toc_address": self.toc_address,
            "content": self.content,
            "score": round(self.score, 4),
            "context_before": self.context_before,
            "context_after": self.context_after,
        }


class STAIRRetriever:
    """Retrieves paragraphs by embedding similarity and expands with sibling context.

    "Parent-Context Expansion" pulls in the immediately preceding/following
    paragraph *within the same section* so the caller (or the LLM downstream)
    never sees a matched sentence stripped of the context it depended on.
    """

    def __init__(self, vector_store: BaseVectorStore, embed_fn: EmbedFn, expand_context: bool = True):
        self.vector_store = vector_store
        self.embed_fn = embed_fn
        self.expand_context = expand_context

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        query_vector = self.embed_fn([query])[0]
        hits = self.vector_store.search(query_vector, top_k=top_k)

        results = []
        for chunk, score in hits:
            before, after = (None, None)
            if self.expand_context:
                before, after = self._sibling_context(chunk)
            results.append(
                RetrievalResult(
                    toc_address=chunk.toc_address,
                    content=chunk.content,
                    score=score,
                    context_before=before,
                    context_after=after,
                )
            )
        return results

    def _sibling_context(self, chunk: HierarchicalChunk) -> Tuple[Optional[str], Optional[str]]:
        """Finds the preceding/following paragraph within the same section."""
        siblings = [c for c in self.vector_store.all_chunks() if c.chapter == chunk.chapter and c.section == chunk.section]
        siblings.sort(key=lambda c: c.paragraph_index)

        before = next((s.content for s in reversed(siblings) if s.paragraph_index < chunk.paragraph_index), None)
        after = next((s.content for s in siblings if s.paragraph_index > chunk.paragraph_index), None)
        return before, after
