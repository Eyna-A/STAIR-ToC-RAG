"""FastAPI service exposing document ingestion and ToC-aware query endpoints."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.indexer.pipeline import HierarchicalIndexer
from src.indexer.vector_store import InMemoryVectorStore, default_embedder
from src.parser.base import BaseParser
from src.parser.pdf_heuristics import PDFHeuristicParser
from src.retriever.stair_retriever import STAIRRetriever

app = FastAPI(
    title="STAIR-ToC-RAG API",
    description="Structure-aware, table-of-contents-native retrieval.",
    version="0.1.0",
)

# Single shared embedder + store for this process.
# Swap InMemoryVectorStore for QdrantVectorStore in production
# (see src/indexer/vector_store.py) — the rest of the code is unaffected.
_embed_fn = default_embedder()
_vector_store = InMemoryVectorStore()
_indexer = HierarchicalIndexer(vector_store=_vector_store, embed_fn=_embed_fn)
_retriever = STAIRRetriever(vector_store=_vector_store, embed_fn=_embed_fn)

_PARSERS: Dict[str, BaseParser] = {".pdf": PDFHeuristicParser()}


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class QueryResult(BaseModel):
    toc_address: str
    content: str
    score: float
    context_before: Optional[str] = None
    context_after: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": len(_vector_store.all_chunks())}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    parser = _PARSERS.get(suffix)
    if parser is None:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or 'unknown'}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        tree = parser.parse(tmp_path)
        doc_dict = BaseParser.tree_to_dict(tree)
        chunks = _indexer.index_document(doc_dict)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"filename": file.filename, "chunks_indexed": len(chunks)}


@app.post("/query", response_model=List[QueryResult])
def query(request: QueryRequest) -> List[QueryResult]:
    if len(_vector_store.all_chunks()) == 0:
        raise HTTPException(status_code=400, detail="Index is empty. Call /ingest first.")
    results = _retriever.retrieve(request.query, top_k=request.top_k)
    return [QueryResult(**r.as_dict()) for r in results]
