"""Streamlit dashboard: upload a document, inspect its ToC tree, and query it."""
from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.indexer.pipeline import HierarchicalIndexer
from src.indexer.vector_store import InMemoryVectorStore, default_embedder
from src.parser.base import BaseParser
from src.parser.pdf_heuristics import PDFHeuristicParser
from src.retriever.stair_retriever import STAIRRetriever

st.set_page_config(page_title="STAIR-ToC-RAG", page_icon="🧭", layout="wide")


@st.cache_resource
def get_pipeline():
    embed_fn = default_embedder()
    store = InMemoryVectorStore()
    indexer = HierarchicalIndexer(vector_store=store, embed_fn=embed_fn)
    retriever = STAIRRetriever(vector_store=store, embed_fn=embed_fn)
    return store, indexer, retriever


store, indexer, retriever = get_pipeline()

st.title("🧭 STAIR-ToC-RAG")
st.caption("Structure-aware retrieval: every answer comes with its exact place in the document.")

with st.sidebar:
    st.header("📄 Ingest a document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded and st.button("Index document", use_container_width=True):
        tmp_dir = Path(tempfile.gettempdir())
        tmp_path = tmp_dir / uploaded.name
        tmp_path.write_bytes(uploaded.getvalue())

        with st.spinner("Extracting structure and building the address-aware index..."):
            tree = PDFHeuristicParser().parse(str(tmp_path))
            doc_dict = BaseParser.tree_to_dict(tree)
            chunks = indexer.index_document(doc_dict)

        st.success(f"Indexed {len(chunks)} paragraphs.")

if store.all_chunks():
    with st.expander("🌳 Table of Contents tree", expanded=False):
        seen = set()
        for chunk in store.all_chunks():
            key = (chunk.chapter, chunk.section)
            if key not in seen:
                seen.add(key)
                st.markdown(f"- **{chunk.chapter}** → {chunk.section}")

st.header("🔍 Ask a question")
query_text = st.text_input("Query", placeholder="e.g. How does STAIR reduce hallucination rate?")
top_k = st.slider("Results", min_value=1, max_value=10, value=3)

if st.button("Search", type="primary") and query_text:
    if not store.all_chunks():
        st.warning("Index a document first.")
    else:
        results = retriever.retrieve(query_text, top_k=top_k)
        for r in results:
            with st.container(border=True):
                st.markdown(f"📍 `{r.toc_address}`")
                st.write(r.content)
                st.caption(f"Similarity score: {r.score:.4f}")
                if r.context_before or r.context_after:
                    with st.expander("Parent context"):
                        if r.context_before:
                            st.markdown(f"**Before:** {r.context_before}")
                        if r.context_after:
                            st.markdown(f"**After:** {r.context_after}")
