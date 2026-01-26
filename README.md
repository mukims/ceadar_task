---
title: Cedar RAG Prototype
emoji: 📚
sdk: docker
app_port: 8501
---







# Ceadar Task — RAG Prototype

This repository contains a Retrieval-Augmented Generation (RAG) prototype that ingests PDF/DOCX/XLSX documents, retrieves relevant chunks via vector search, and synthesizes grounded answers using an LLM.

## Architecture Overview and Design Rationale

1. **Ingestion**: PDFs, DOCX, XLSX are parsed into `Document` objects with metadata.
2. **Chunking**: Text is split into overlapping chunks for retrieval.
3. **Embeddings**: Chunks are encoded using a SentenceTransformer model.
4. **Vector Store**: FAISS stores embeddings for fast similarity search.
5. **Retrieval**: Top-K chunks are selected for a query.
6. **Generation**: An LLM (Hugging Face) answers using only retrieved context.

```
[PDF/DOCX/XLSX] -> [Chunking] -> [Embeddings] -> [FAISS] -> [Top-K] -> [LLM Answer]
```

## Implementation Decisions and Trade-offs

- **Embedding model**: default `sentence-transformers/all-MiniLM-L6-v2` for CPU-friendly performance. Advanced models are used in huggingface deployment. 
- **LLM**: default `google/flan-t5-base` for grounded QA on CPU; can be swapped via env or UI. Advanced models are used in huggingface deployment. 
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


###  Test Queries (Cross-Doc + Sanity Checks)

1. What is the difference between DeepSeek-R1-Zero and DeepSeek-R1?
2. How does GRPO differ from PPO, and why was it chosen?
3. What is the "aha moment" described in the paper, and why is it significant?
4. Compare reinforcement learning in DeepSeek-R1 with supervised training in the original Transformer paper.
5. How does the AI Act's definition of "general purpose AI" differ from how researchers define foundation models?
6. Which annex defines high-risk AI use cases in the AI Act?
7. Does the Inflation Calculator show that Transformers cause inflation?
8. Which article of the AI Act explains scaled dot-product attention?

### Observations & Limitations on local system

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
