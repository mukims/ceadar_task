"""
Minimal RAG pipeline for testing token budget effects.

- Ingest PDF/DOCX/XLSX
- Naive chunking by character length
- Embeddings + FAISS retrieval
- Prompt assembly constrained by max input tokens
- Optional generation with an HF model
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from rag_pipeline_combined_2 import Document, load_documents

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    files_dir: Path
    chunk_size: int = 1000
    chunk_overlap: int = 150
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "google/flan-t5-base"
    top_k: int = 5
    max_input_tokens: int | None = None
    max_new_tokens: int = 256
    max_context_chars: int = 4000


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


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_texts(model_name: str, texts: List[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return _l2_normalize(emb.astype(np.float32))


def build_faiss_index(embeddings: np.ndarray):
    import faiss

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve(index, docs: List[Document], model_name: str, query: str, top_k: int) -> List[Document]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    query_emb = model.encode([query], convert_to_numpy=True)
    query_emb = _l2_normalize(query_emb.astype(np.float32))
    distances, indices = index.search(query_emb, top_k)
    results: List[Document] = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(docs):
            continue
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
    max_new_tokens: int,
) -> Tuple[str, int, int]:
    prefix = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Context:\n"
    )
    suffix = f"\n\nQuestion: {query}\nAnswer:"

    prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
    suffix_tokens = len(tokenizer.encode(suffix, add_special_tokens=False))
    budget = max_input_tokens - max_new_tokens - prefix_tokens - suffix_tokens
    if budget < 16:
        budget = max(16, max_input_tokens // 4)

    parts: List[str] = []
    used = 0
    for i, doc in enumerate(docs, start=1):
        entry = f"[{i}] {doc.page_content.strip()}"
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


def generate_answer(llm_model: str, prompt: str, max_new_tokens: int) -> str:
    from transformers import pipeline

    task = "text2text-generation" if "t5" in llm_model.lower() or "flan" in llm_model.lower() else "text-generation"
    try:
        generator = pipeline(task, model=llm_model)
    except KeyError:
        task = "text-generation"
        generator = pipeline(task, model=llm_model)
    outputs = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_return_sequences=1,
        eos_token_id=getattr(generator.tokenizer, "eos_token_id", None),
    )
    generated = outputs[0].get("generated_text", "").strip()
    if task == "text-generation" and generated.startswith(prompt):
        return generated[len(prompt):].strip()
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal RAG with token budget control")
    parser.add_argument("--files-dir", type=Path, default=Path("."))
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--llm-model", type=str, default="google/flan-t5-base")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-input-tokens", type=int, default=None, help="Total input token budget")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-generate", action="store_true", help="Skip LLM generation")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    cfg = Config(
        files_dir=args.files_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        top_k=args.top_k,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
    )

    logger.info("Loading documents from %s", cfg.files_dir)
    raw_docs = load_documents(cfg.files_dir)
    chunks = chunk_documents(raw_docs, cfg.chunk_size, cfg.chunk_overlap)

    logger.info("Embedding %d chunks", len(chunks))
    embeddings = embed_texts(cfg.embedding_model, [c.page_content for c in chunks])

    logger.info("Building FAISS index")
    index = build_faiss_index(embeddings)

    logger.info("Retrieving top %d chunks", cfg.top_k)
    retrieved = retrieve(index, chunks, cfg.embedding_model, args.query, cfg.top_k)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model)
    model_max = getattr(tokenizer, "model_max_length", None)
    max_input_tokens = cfg.max_input_tokens or (model_max if isinstance(model_max, int) else 512)

    prompt, ctx_tokens, total_tokens = build_prompt(
        args.query,
        retrieved,
        tokenizer,
        max_input_tokens,
        cfg.max_new_tokens,
    )

    print("\n=== Retrieval Summary ===")
    print(f"Retrieved chunks: {len(retrieved)}")
    print(f"Context tokens used: {ctx_tokens}")
    print(f"Prompt tokens total: {total_tokens}")
    print(f"Token budget (input): {max_input_tokens} | max_new_tokens: {cfg.max_new_tokens}")

    if args.no_generate:
        print("\n=== Prompt (truncated) ===")
        print(prompt[:2000])
        return

    answer = generate_answer(cfg.llm_model, prompt, cfg.max_new_tokens)
    print("\n=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
