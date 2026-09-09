"""PDF structure extraction using embedded outlines or font-size heuristics.

Strategy:
  1. Prefer the PDF's embedded outline/bookmarks (doc.get_toc()) when present.
     This is ground-truth structure written by whoever authored the PDF.
  2. Fall back to a font-size heuristic when no outline exists: the most
     common font size in the document is treated as "body text"; spans
     meaningfully larger than that become chapter/section headings, and
     everything else is accumulated into paragraphs.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional, Tuple

import fitz  # PyMuPDF

from src.parser.base import BaseParser, ChapterNode, DocumentTree, SectionNode

_MIN_PARAGRAPH_CHARS = 40


class PDFHeuristicParser(BaseParser):
    def __init__(self, heading_size_ratio: float = 1.05, size_tolerance: float = 0.5):
        """
        Args:
            heading_size_ratio: a span's font size must be at least this many
                times the body-text size to even be considered a heading
                candidate (filters out bold/italic body text noise).
            size_tolerance: two spans are treated as the same heading tier if
                their sizes differ by less than this (in points).

        Rather than hard-coding "chapter = 1.35x body, section = 1.15x body",
        we collect every distinct heading-candidate size in the document and
        treat the largest one as the chapter tier and the second-largest as
        the section tier. This adapts to each document's own type scale
        instead of assuming a universal ratio, which is fragile in practice
        (e.g. a 16pt section heading in an 11pt-body doc is a bigger jump
        than 1.35x, while a 24pt chapter heading in a 18pt-body doc is a
        smaller jump than 1.35x).
        """
        self.heading_size_ratio = heading_size_ratio
        self.size_tolerance = size_tolerance

    def parse(self, file_path: str) -> DocumentTree:
        doc = fitz.open(file_path)
        try:
            toc = doc.get_toc(simple=True)
            if toc:
                return self._tree_from_outline(doc, toc)
            return self._tree_from_font_heuristics(doc)
        finally:
            doc.close()

    # ------------------------------------------------------------------ #
    # Strategy 1: embedded outline
    # ------------------------------------------------------------------ #
    def _tree_from_outline(self, doc: fitz.Document, toc: List[list]) -> DocumentTree:
        chapters: List[ChapterNode] = []
        current_chapter: Optional[ChapterNode] = None

        for i, (level, title, start_page) in enumerate(toc):
            end_page = toc[i + 1][2] - 1 if i + 1 < len(toc) else doc.page_count
            text = self._extract_page_range_text(doc, start_page - 1, end_page)
            paragraphs = self._split_paragraphs(text)

            if level == 1:
                current_chapter = ChapterNode(title=title, sections=[SectionNode(title="Overview", paragraphs=paragraphs)])
                chapters.append(current_chapter)
            else:
                if current_chapter is None:
                    current_chapter = ChapterNode(title="Untitled Chapter", sections=[])
                    chapters.append(current_chapter)
                current_chapter.sections.append(SectionNode(title=title, paragraphs=paragraphs))

        return chapters

    # ------------------------------------------------------------------ #
    # Strategy 2: font-size heuristics
    # ------------------------------------------------------------------ #
    def _tree_from_font_heuristics(self, doc: fitz.Document) -> DocumentTree:
        spans = self._collect_spans(doc)
        if not spans:
            return []

        body_size = self._most_common_size(spans)

        # Distinct candidate heading sizes, biggest first. The top one is the
        # chapter tier, the next one down is the section tier; anything
        # smaller is body text. (Deeper hierarchies flatten into the nearest
        # section — a known, documented limitation of the heuristic path.)
        heading_sizes = sorted({size for _, size in spans if size >= body_size * self.heading_size_ratio}, reverse=True)
        chapter_size = heading_sizes[0] if heading_sizes else None
        section_size = heading_sizes[1] if len(heading_sizes) > 1 else None

        chapters: List[ChapterNode] = []
        current_chapter: Optional[ChapterNode] = None
        current_section: Optional[SectionNode] = None
        buffer: List[str] = []

        def flush_buffer() -> None:
            if buffer and current_section is not None:
                current_section.paragraphs.extend(self._split_paragraphs(" ".join(buffer)))
            buffer.clear()

        def matches(size: float, target: Optional[float]) -> bool:
            return target is not None and abs(size - target) < self.size_tolerance

        for text, size in spans:
            text = text.strip()
            if not text:
                continue

            if matches(size, chapter_size):
                flush_buffer()
                current_section = SectionNode(title="Overview", paragraphs=[])
                current_chapter = ChapterNode(title=text, sections=[current_section])
                chapters.append(current_chapter)

            elif matches(size, section_size):
                flush_buffer()
                if current_chapter is None:
                    current_chapter = ChapterNode(title="Untitled Chapter", sections=[])
                    chapters.append(current_chapter)
                current_section = SectionNode(title=text, paragraphs=[])
                current_chapter.sections.append(current_section)

            else:
                buffer.append(text)

        flush_buffer()
        return chapters

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _collect_spans(doc: fitz.Document) -> List[Tuple[str, float]]:
        spans: List[Tuple[str, float]] = []
        for page in doc:
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        spans.append((span["text"], round(span["size"], 1)))
        return spans

    @staticmethod
    def _most_common_size(spans: List[Tuple[str, float]]) -> float:
        sizes = [size for _, size in spans]
        return Counter(sizes).most_common(1)[0][0]

    @staticmethod
    def _extract_page_range_text(doc: fitz.Document, start_page: int, end_page: int) -> str:
        chunks = []
        for page_no in range(max(start_page, 0), min(end_page, doc.page_count)):
            chunks.append(doc[page_no].get_text())
        return "\n".join(chunks)

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        raw_parts = re.split(r"\n\s*\n|\.\s+(?=[A-Z\u0600-\u06FF])", text)
        return [p.strip() for p in raw_parts if len(p.strip()) >= _MIN_PARAGRAPH_CHARS]
