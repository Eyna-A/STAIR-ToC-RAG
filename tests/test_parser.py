"""Unit tests for document-tree utilities in the parser layer."""
from __future__ import annotations

import fitz
import pytest

from src.parser.base import BaseParser, ChapterNode, SectionNode
from src.parser.pdf_heuristics import PDFHeuristicParser


def test_tree_to_dict_round_trip():
    tree = [
        ChapterNode(
            title="Chapter 1",
            sections=[SectionNode(title="Section 1.1", paragraphs=["Hello world.", "Second paragraph."])],
        )
    ]

    as_dict = BaseParser.tree_to_dict(tree)

    assert as_dict == [
        {
            "title": "Chapter 1",
            "sections": [
                {
                    "title": "Section 1.1",
                    "paragraphs": ["Hello world.", "Second paragraph."],
                }
            ],
        }
    ]


def test_tree_to_dict_handles_multiple_chapters_and_sections():
    tree = [
        ChapterNode(title="Chapter 1", sections=[SectionNode(title="1.1", paragraphs=["a"])]),
        ChapterNode(
            title="Chapter 2",
            sections=[
                SectionNode(title="2.1", paragraphs=["b"]),
                SectionNode(title="2.2", paragraphs=["c", "d"]),
            ],
        ),
    ]

    as_dict = BaseParser.tree_to_dict(tree)

    assert len(as_dict) == 2
    assert len(as_dict[1]["sections"]) == 2
    assert as_dict[1]["sections"][1]["paragraphs"] == ["c", "d"]


def test_tree_to_dict_handles_empty_tree():
    assert BaseParser.tree_to_dict([]) == []


@pytest.fixture
def synthetic_pdf(tmp_path) -> str:
    """Builds a minimal two-tier (chapter + section) PDF with no embedded
    outline, forcing the parser down the font-size heuristic path."""
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    page.insert_text((72, y), "Chapter 1: Test Chapter", fontsize=24)
    y += 40
    page.insert_text((72, y), "Section 1.1: Test Section", fontsize=16)
    y += 30
    for i in range(6):
        page.insert_text(
            (72, y),
            f"This is body paragraph number {i} long enough to survive the min length filter used here.",
            fontsize=11,
        )
        y += 20

    path = str(tmp_path / "synthetic.pdf")
    doc.save(path)
    doc.close()
    return path


def test_font_heuristic_parser_nests_section_under_chapter(synthetic_pdf):
    tree = PDFHeuristicParser().parse(synthetic_pdf)
    as_dict = BaseParser.tree_to_dict(tree)

    assert len(as_dict) == 1
    assert as_dict[0]["title"] == "Chapter 1: Test Chapter"
    # "Overview" (empty, since no body text preceded the section heading)
    # plus the real "Section 1.1" section with the body paragraphs.
    section_titles = [s["title"] for s in as_dict[0]["sections"]]
    assert "Section 1.1: Test Section" in section_titles

    real_section = next(s for s in as_dict[0]["sections"] if s["title"] == "Section 1.1: Test Section")
    assert len(real_section["paragraphs"]) == 6
    assert "body paragraph number 0" in real_section["paragraphs"][0]
