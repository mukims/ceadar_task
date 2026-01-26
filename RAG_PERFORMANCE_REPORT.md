RAG Pipeline Performance Report
Date: 2026-01-26

Scope
- Pipeline: `rag_pipeline_combined_2.py` (PDF/DOCX/XLSX ingestion, chunking, embeddings, FAISS, retrieval, generation).
- Dataset: local documents in repo root.
- Goal: measure cold start, warm start, retrieval latency, and generation latency on CPU.

Test Environment
- Host: Linux x86_64
- CPU: 13th Gen Intel(R) Core(TM) i7-13700 (24 logical CPUs)
- Memory: 31 GiB RAM
- Python: 3.13.7 (Anaconda)
- Models: `sentence-transformers/all-MiniLM-L6-v2` (embeddings), `google/flan-t5-base` (LLM)
- Config: `chunk_size=1000`, `chunk_overlap=150`, `top_k=5`, `max_new_tokens=256`, `max_context_chars=4000`

Dataset Summary
- Files ingested: 4 (2 PDF, 1 DOCX, 1 XLSX)
- Total size: 3.41 MiB
- Raw documents produced by loader: 1653
- Chunks after splitting: 1728
- FAISS index size on disk: ~3.5 MiB

Methodology
- Cold start: run with a fresh persistence directory to force embedding + index build.
- Warm start: reuse the existing index (same directory) to measure load time.
- Retrieval timing: 8 default evaluation queries from `main.py`.
- Generation timing: first 2 evaluation queries to limit runtime.
- Timing source: `time.perf_counter()`; memory: `ru_maxrss` (peak RSS, Linux KB).

Results (CPU)
Cold Start (fresh index)
- Load docs: 4.86 s
- Chunking: 2.38 s
- Embedding + FAISS build: 9.02 s
- Total cold start: 16.26 s
- Peak RSS after index: 1.34 GiB

Warm Start (index already built)
- FAISS load + compatibility check: 1.67 s
- Peak RSS after warm start: 1.34 GiB

Retrieval (embedding query + FAISS search, top_k=5)
- Queries: 8
- Avg latency: 1.72 s
- p50: 1.69 s | p95: 2.08 s
- Min/Max: 1.64 s / 1.97 s
- Peak RSS after retrieval: 1.42 GiB

Generation (FLAN-T5, 2 queries)
- Avg latency: 1.70 s
- Min/Max: 1.55 s / 1.85 s
- Peak RSS after generation: 2.15 GiB

Throughput Estimates
- Ingestion: ~340 docs/s (1653 docs / 4.86 s)
- Chunking: ~725 chunks/s (1728 chunks / 2.38 s)
- Embedding + indexing: ~192 chunks/s (1728 / 9.02 s)
- Retrieval QPS: ~0.58 queries/s (1 / 1.72 s)

Key Observations
- Retrieval latency is dominated by query embedding; the embedding model is instantiated inside `retrieve_documents` each call.
- Cold-start cost is primarily embedding + index build (~55% of cold start time).
- Memory grows from ~0.35 GiB after load to ~2.15 GiB after generation, consistent with model loading.

Performance Risks / Bottlenecks
- Per-query model initialization for embeddings adds ~1–2 s latency and wastes memory.
- Generation uses a full seq2seq model on CPU; latency scales with context length and token budget.
- Document loaders can be heavy for large PDFs (tables/OCR paths), though dataset here is small.

Recommendations
1) Cache embedding model instances (e.g., `@lru_cache` around `_import_embeddings()` + model creation) to cut retrieval latency.
2) Move retrieval to a long-lived service process so model weights stay warm.
3) Consider smaller or distilled models for faster CPU generation, or GPU inference where available.
4) If throughput matters, batch query embedding or add async queueing for retrieval.

Notes
- Metrics reflect a single run on 2026-01-26 and will vary with hardware, OS, and cache state.
- The first run also includes model weight loading; subsequent runs may be faster if weights are already cached.
