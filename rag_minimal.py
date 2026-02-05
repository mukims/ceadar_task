"""
Minimal RAG pipeline for testing token budget effects (TOKEN-BASED).

- Ingest PDF/DOCX/XLSX via your load_documents()
- Naive chunking by character length
- Ollama embeddings + FAISS retrieval
- Prompt assembly constrained by max input tokens (using HF tokenizer)
- Optional generation via Ollama local endpoint
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import requests

from rag_pipeline_combined_2 import Document, load_documents  # your existing loader

logger = logging.getLogger(__name__)
OLLAMA = "http://localhost:11434"


@dataclass(frozen=True)
class Config:
    files_dir: Path
    chunk_size: int = 1000
    chunk_overlap: int = 150
    embedding_model: str = "nomic-embed-text"     # OLLAMA embedding model name
    llm_model: str = "qwen2.5:7b"                 # OLLAMA generation model name
    tokenizer_model: str = "Qwen/Qwen2.5-7B-Instruct"  # HF tokenizer id for token counting
    top_k: int = 5
    max_input_tokens: Optional[int] = None
    max_new_tokens: int = 256


def chunk_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Naive character-based chunking to keep dependencies minimal."""
    chunks: List[Document] = []
    step = max(1, chunk_size - chunk_overlap)

    for doc in docs:
        text = (doc.page_content or "").strip()
        if not text:
            continue

        start = 0
        chunk_idx = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_text = text[start:end].strip()
            if chunk_text:
                meta = dict(doc.metadata or {})
                meta.update({"chunk_index": chunk_idx, "chunk_start": start, "chunk_end": end})
                chunks.append(Document(page_content=chunk_text, metadata=meta))
                chunk_idx += 1
            start += step

    return chunks


def _faiss_normalize(v: np.ndarray) -> np.ndarray:
    # Normalize for cosine similarity via inner product
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def ollama_embed_texts(model_name: str, texts: List[str]) -> np.ndarray:
    """Embeddings via Ollama /api/embeddings."""
    vecs = []
    for t in texts:
        r = requests.post(
            f"{OLLAMA}/api/embeddings",
            json={"model": model_name, "prompt": t},
            timeout=120,
        )
        r.raise_for_status()
        vecs.append(r.json()["embedding"])

    arr = np.array(vecs, dtype="float32")
    arr = _faiss_normalize(arr)
    return arr


def build_faiss_index(embeddings: np.ndarray):
    import faiss
    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine via normalized dot product
    index.add(embeddings)
    return index


def retrieve(index, docs: List[Document], embed_model: str, query: str, top_k: int) -> List[Document]:
    q = ollama_embed_texts(embed_model, [query])[0:1]
    distances, indices = index.search(q, top_k)

    results: List[Document] = []
    for idx in indices[0]:
        if 0 <= idx < len(docs):
            results.append(docs[idx])
    return results


def _truncate_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_prompt(
    query: str,
    docs: List[Document],
    tokenizer,
    max_input_tokens: int,
) -> Tuple[str, int, int]:
    """
    Build a prompt that fits within max_input_tokens (INPUT ONLY).
    max_new_tokens is not subtracted here because it's output budget.
    """
    prefix = (
        "You are a helpful assistant.\n"
        "Answer the question using ONLY the context below.\n"
        "If the answer is not in the context, say: \"I don't know based on the provided documents.\"\n\n"
        "Context:\n"
    )
    suffix = f"\n\nQuestion: {query}\nAnswer:"

    prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
    suffix_tokens = len(tokenizer.encode(suffix, add_special_tokens=False))

    budget = max_input_tokens - prefix_tokens - suffix_tokens
    if budget < 64:
        # if the user chooses tiny budgets, keep it minimally usable
        budget = max(64, max_input_tokens // 2)

    parts: List[str] = []
    used = 0

    for i, doc in enumerate(docs, start=1):
        # include lightweight metadata for later debugging/citations
        src = doc.metadata.get("source", "") if doc.metadata else ""
        entry = f"[{i}] ({src}) {doc.page_content.strip()}" if src else f"[{i}] {doc.page_content.strip()}"

        entry_tokens = len(tokenizer.encode(entry, add_special_tokens=False))
        if used + entry_tokens > budget:
            remaining = budget - used
            if remaining <= 0:
                break
            entry = _truncate_to_tokens(entry, tokenizer, remaining)
            entry_tokens = len(tokenizer.encode(entry, add_special_tokens=False))

        parts.append(entry)
        used += entry_tokens
        if used >= budget:
            break

    context = "\n\n".join(parts)
    prompt = f"{prefix}{context}{suffix}"
    total_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    return prompt, used, total_tokens


def ollama_generate(llm_model: str, prompt: str, max_new_tokens: int) -> str:
    """
    Generation via Ollama /api/generate.
    Ollama uses options.num_predict as a max tokens to generate.
    """
    r = requests.post(
        f"{OLLAMA}/api/generate",
        json={
            "model": llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_new_tokens},
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["response"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal RAG with token budget control (Ollama + FAISS)")
    p.add_argument("--files-dir", type=Path, default=Path("."))
    p.add_argument("--query", type=str, required=True)

    # Ollama model names
    p.add_argument("--embedding-model", type=str, default="nomic-embed-text")
    p.add_argument("--llm-model", type=str, default="qwen2.5:7b")

    # HF tokenizer id just for counting
    p.add_argument("--tokenizer-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")

    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--chunk-overlap", type=int, default=150)
    p.add_argument("--top-k", type=int, default=5)

    p.add_argument("--max-input-tokens", type=int, default=2048, help="INPUT token budget for prompt")
    p.add_argument("--max-new-tokens", type=int, default=256)

    p.add_argument("--no-generate", action="store_true", help="Skip LLM generation (print prompt stats)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    cfg = Config(
        files_dir=args.files_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        tokenizer_model=args.tokenizer_model,
        top_k=args.top_k,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
    )

    logger.info("Loading documents from %s", cfg.files_dir)
    raw_docs = load_documents(cfg.files_dir)

    logger.info("Chunking %d documents", len(raw_docs))
    chunks = chunk_documents(raw_docs, cfg.chunk_size, cfg.chunk_overlap)
    if not chunks:
        raise SystemExit("No chunks produced. Check your loader or file types.")

    logger.info("Embedding %d chunks using Ollama model '%s'", len(chunks), cfg.embedding_model)
    embeddings = ollama_embed_texts(cfg.embedding_model, [c.page_content for c in chunks])

    logger.info("Building FAISS index")
    index = build_faiss_index(embeddings)

    logger.info("Retrieving top %d chunks", cfg.top_k)
    retrieved = retrieve(index, chunks, cfg.embedding_model, args.query, cfg.top_k)

    from transformers import AutoTokenizer
    logger.info("Loading tokenizer '%s' (HF) for token counting", cfg.tokenizer_model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_model, use_fast=True)

    max_input_tokens = int(cfg.max_input_tokens or 2048)

    prompt, ctx_tokens, total_tokens = build_prompt(
        args.query,
        retrieved,
        tokenizer,
        max_input_tokens,
    )

    print("\n=== Retrieval Summary ===")
    print(f"Retrieved chunks: {len(retrieved)}")
    print(f"Context tokens used: {ctx_tokens}")
    print(f"Prompt tokens total: {total_tokens}")
    print(f"Token budget (input): {max_input_tokens} | max_new_tokens: {cfg.max_new_tokens}")
    print(f"Embed model (Ollama): {cfg.embedding_model} | LLM model (Ollama): {cfg.llm_model}")
    print(f"Tokenizer (HF): {cfg.tokenizer_model}")

    if args.no_generate:
        print("\n=== Prompt (truncated) ===")
        print(prompt[:2000])
        return

    answer = ollama_generate(cfg.llm_model, prompt, cfg.max_new_tokens)
    print("\n=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
