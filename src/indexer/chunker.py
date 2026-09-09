"""Turns a Chapter > Section > Paragraph tree into addressed, embeddable chunks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class HierarchicalChunk:
    """A single retrievable unit: raw paragraph text plus its ToC address."""

    chunk_id: str
    toc_address: str
    content: str
    chapter: str
    section: str
    paragraph_index: int

    @property
    def full_representation(self) -> str:
        """Address-prefixed text that actually gets embedded.

        Concatenating the address with the content is what lets the retriever
        learn positional/structural signal instead of relying on content alone.
        """
        return f"Document Address: [{self.toc_address}] | Content: {self.content}"


def build_chunks(doc_tree: List[Dict]) -> List[HierarchicalChunk]:
    """Flatten a Chapter > Section > Paragraph tree into addressed chunks.

    Args:
        doc_tree: list of {"title": str, "sections": [{"title": str, "paragraphs": [str, ...]}]}
    """
    chunks: List[HierarchicalChunk] = []
    running_id = 0

    for chapter in doc_tree:
        chap_title = chapter["title"]
        for section in chapter.get("sections", []):
            sec_title = section["title"]
            for idx, paragraph in enumerate(section.get("paragraphs", [])):
                if not paragraph or not paragraph.strip():
                    continue
                full_address = f"{chap_title} > {sec_title} > Paragraph {idx + 1}"
                chunks.append(
                    HierarchicalChunk(
                        chunk_id=f"chunk_{running_id:06d}",
                        toc_address=full_address,
                        content=paragraph.strip(),
                        chapter=chap_title,
                        section=sec_title,
                        paragraph_index=idx,
                    )
                )
                running_id += 1

    return chunks
