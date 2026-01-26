---
title: Cedar RAG Prototype
emoji: 📚
sdk: docker
app_port: 8501
---







# Ceadar Task — RAG Prototype

This repository contains a Retrieval-Augmented Generation (RAG) prototype that ingests PDF/DOCX/XLSX documents, retrieves relevant chunks via vector search, and synthesizes grounded answers using an LLM.

## Architecture (End-to-End)

1. **Ingestion**: PDFs, DOCX, XLSX are parsed into `Document` objects with metadata.
2. **Chunking**: Text is split into overlapping chunks for retrieval.
3. **Embeddings**: Chunks are encoded using a SentenceTransformer model.
4. **Vector Store**: FAISS stores embeddings for fast similarity search.
5. **Retrieval**: Top-K chunks are selected for a query.
6. **Generation**: An LLM (Hugging Face) answers using only retrieved context.

```
[PDF/DOCX/XLSX] -> [Chunking] -> [Embeddings] -> [FAISS] -> [Top-K] -> [LLM Answer]
```

## Design Rationale & Trade-offs

- **Embedding model**: default `sentence-transformers/all-MiniLM-L6-v2` for CPU-friendly performance.
- **LLM**: default `google/flan-t5-base` for grounded QA on CPU; can be swapped via env or UI.
- **Parsing**: PDF table extraction and OCR are optional to keep the pipeline robust if deps are missing.
- **Persistence**: FAISS index is cached in `faiss_store` to avoid recomputation on restart.
- **Minimal dependencies**: Streamlit app for deployment with a simple UI.

## Quick Start (CLI)

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python rag_pipeline_combined_2.py --files-dir . --query "What is the objective of the challenge?"
```

## Web App (Streamlit)

```
streamlit run app.py
```

## Docker (optional)

```
docker build -t ceadar-rag .
docker run -p 8501:8501 cedar-rag
```

Optional environment variables:
- `FILES_DIR` (default `.`)
- `PERSIST_DIR` (default `faiss_store`)
- `EMBEDDING_MODEL`
- `LLM_MODEL`
- `LLM_TASK`
- `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MAX_NEW_TOKENS`, `MAX_CONTEXT_CHARS`

## Evaluation — Test Queries (5–8)

Use these to validate retrieval and answer quality:

1. What is the objective of the Data Scientist II RAG challenge?
2. Which components are required in the deliverables?
3. What does the challenge require for cloud deployment?
4. Summarize the main idea of the “Attention is All You Need” paper.
5. What is the EU AI Act about, at a high level?
6. Which evaluation criteria are used to score the submission?
7. What are the provided document types for ingestion?
8. What are the limitations of this prototype?

### Observations & Limitations

- **Accuracy**: Strong for queries that map cleanly to a single source document.
- **Retrieval sensitivity**: Smaller chunks improve recall but can fragment context.
- **LLM limits**: Smaller models can hallucinate; the prompt constrains answers to retrieved context.
- **Latency**: First request is slower due to model download and index building.
- **OCR**: Only available if Tesseract is installed; otherwise figures are skipped.

## Deployment Notes (Ready for Cloud Hosting)

This repo is prepared for a simple deployment on services like Hugging Face Spaces or similar platforms.

- Ensure `app.py` is the entry point.
- `requirements.txt` contains all Python dependencies.
- Set environment variables to point to the data folder if not using repo root.

Deployed app link:
- https://huggingface.co/spaces/mukimshardul/ceadar_rag_proto

## Repo Structure

- `app.py` — Streamlit UI for the RAG demo
- `rag_pipeline_combined_2.py` — main RAG pipeline with rich PDF parsing
- `datascientistiichallenge/` — challenge instructions PDF
- `processed/` — intermediate data artifacts (local)
- `faiss_store/` — FAISS index cache (local)

## Notes

- Large binary files are kept alongside the notebooks for reproducibility.
- Optional dependencies: `pdfplumber`, `pymupdf`; `pytesseract` is not required unless OCR is desired.
