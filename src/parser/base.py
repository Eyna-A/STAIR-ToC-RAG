"""Base classes for document parsers.

Every parser (PDF, DOCX, Markdown, ...) must ultimately produce a
Chapter > Section > Paragraph tree so the rest of the pipeline
(indexer, retriever) never has to know what the source format was.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SectionNode:
    title: str
    paragraphs: List[str] = field(default_factory=list)


@dataclass
class ChapterNode:
    title: str
    sections: List[SectionNode] = field(default_factory=list)


DocumentTree = List[ChapterNode]


class BaseParser(ABC):
    """Interface every document parser must implement."""

    @abstractmethod
    def parse(self, file_path: str) -> DocumentTree:
        """Parse a source file into a Chapter > Section > Paragraph tree."""
        raise NotImplementedError

    @staticmethod
    def tree_to_dict(tree: DocumentTree) -> List[Dict]:
        """Convert the dataclass tree into the plain dict shape the indexer expects."""
        return [
            {
                "title": chapter.title,
                "sections": [
                    {"title": section.title, "paragraphs": list(section.paragraphs)}
                    for section in chapter.sections
                ],
            }
            for chapter in tree
        ]
