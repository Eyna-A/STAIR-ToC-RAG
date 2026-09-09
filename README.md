# 🧭 STAIR-ToC-RAG

### Structure-Aware, Table-of-Contents-Native Retrieval Augmented Generation
**Stop chunking blind. Start retrieving by address.**

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/FastAPI-backend-009688.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-dashboard-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/VectorDB-Qdrant-purple.svg" alt="Qdrant">
  <img src="https://img.shields.io/badge/Docker-ready-2496ED.svg" alt="Docker">
</p>

---

## 📖 Overview & Motivation

Most RAG pipelines split documents into fixed-size, context-blind chunks — discarding the one structural signal every well-written document already provides: its **Table of Contents**. A paragraph in *Chapter 4 → Limitations* means something different from the same words in *Chapter 1 → Overview*, yet naive chunking erases that distinction.

**STAIR-ToC-RAG** is a document-native retrieval framework inspired by the IBM Research paper:

> Kumar, V., Pulivarthi, M., Kumar, V., Sen, J., Bhat, R. A., & Joshi, S. — **"STAIR: A novel dataset and LLM based retriever for document structure augmentation."** [arXiv:2609.03874](https://arxiv.org/abs/2609.03874)

The core idea: a generative retriever learns to output a document's **exact hierarchical address** (`Chapter 2 > Section 2.1 > Paragraph 3`) for a given query, instead of relying on flat vector similarity over anonymous text. On the paper's **SearchTome** benchmark (18 books, 6 domains), this approach reached **82.6% Recall@1** — versus 76.9% (fine-tuned DSI), 68.7% (DPR), and 59.5% (BM25) — with hallucination held **below 0.05%**.

### Why it matters
- 📉 Fewer hallucinations — answers are grounded in a verifiable document address, not just a similarity score.
- 🧩 No lost context — every chunk always knows its chapter and section.
- 🔍 Auditable answers — each result ships with a citable path (`Ch.2 > §2.1 > ¶3`).
- 📚 Built for long, structured documents — manuals, contracts, papers, internal wikis.

---

## ✨ Key Features

- 🌳 **ToC Tree Extraction** — parses PDF / DOCX / Markdown into `Chapter > Section > Paragraph` using heading/font heuristics.
- 🏷️ **Hierarchical Address Injection** — embeds `"[Path] | Content"` so the retriever is positionally aware.
- 🧠 **ToC-Aware Retriever** — vector search + **Parent-Context Expansion** to avoid losing upstream context.
- ⚡ **Pluggable Vector Store** — Qdrant by default, swappable with ChromaDB/FAISS.
- 🌐 **REST API** — FastAPI `/ingest` and `/query` endpoints.
- 📊 **Interactive Dashboard** — Streamlit UI for upload, ToC visualization, and live querying.
- 🐳 **Container-first** — one command via Docker Compose.

---

## 🏗️ Architecture

```
Documents (PDF/DOCX/MD)
        │
        ▼
1. ToC Tree Extractor  →  Chapter > Section > Paragraph (JSON)
        │
        ▼
2. Hierarchical Indexer  →  "[Address] | Content" → embed → Vector DB
        │
        ▼
3. ToC-Aware Retriever  →  query → top-k address match + parent context
        │
        ▼
4. FastAPI ⇄ Streamlit  →  /ingest  /query  Dashboard
```

**One-liner:** *Document → Tree → Address-Aware Embedding → Retrieval by Path → Cited Answer.*

---

## 📁 Project Structure

```
STAIR-ToC-RAG/
├── README.md
├── LICENSE
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/{raw,processed}/
├── src/
│   ├── parser/       # ToC & PDF extraction
│   ├── indexer/       # Addressing + vector store writes
│   ├── retriever/     # Retrieval + parent-context expansion
│   └── api/           # FastAPI endpoints
├── app/
│   └── dashboard.py   # Streamlit UI
└── tests/
```

---

## 🚀 Getting Started

**Prerequisites:** Python 3.11+, Docker (recommended for the vector DB)

```bash
git clone https://github.com/<your-username>/STAIR-ToC-RAG.git
cd STAIR-ToC-RAG
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Core dependencies: `fastapi`, `uvicorn`, `streamlit`, `sentence-transformers`, `qdrant-client`, `pymupdf`, `python-docx`.

---

## 🧪 Usage / Quickstart

```bash
# Start the API
uvicorn src.api.main:app --reload --port 8000

# Ingest a document
curl -F "file=@data/raw/sample.pdf" http://localhost:8000/ingest

# Query with ToC-aware retrieval
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does STAIR reduce hallucination rate?", "top_k": 3}'
```

```bash
# Or launch the dashboard
streamlit run app/dashboard.py
```

Each result returns a hierarchical address (e.g. `Chapter 2 > Section 2.1 > Paragraph 2`) alongside the matched content and similarity score.

---

## 🐳 Deployment

```bash
docker-compose up --build
```

| Service   | URL                       |
|-----------|---------------------------|
| API       | http://localhost:8000     |
| Dashboard | http://localhost:8501     |
| Qdrant    | http://localhost:6333     |

---

## 📚 References & Acknowledgments

> Kumar, V., Pulivarthi, M., Kumar, V., Sen, J., Bhat, R. A., & Joshi, S. (2026). **STAIR: A novel dataset and LLM based retriever for document structure augmentation.** IBM Research. [arXiv:2609.03874](https://arxiv.org/abs/2609.03874)

On the **SearchTome** benchmark (18 books, 6 domains): 82.6% Recall@1 vs. 76.9% (DSI), 68.7% (DPR), 59.5% (BM25), with hallucination below 0.05%.

> ⚠️ This is an independent, unofficial implementation inspired by the paper above — not affiliated with or endorsed by IBM or the original authors.

---

## 🤝 Contributing

Issues and PRs welcome. Please run `pytest tests/` before submitting.

## 📄 License

MIT License — see [`LICENSE`](./LICENSE).

---

<p align="center">Built with ❤️ for retrieval that remembers where it came from.</p>
