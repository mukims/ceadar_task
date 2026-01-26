"""
Interpretable RAG Skeleton
==========================

This script provides a clean, readable reference implementation of the
end-to-end RAG flow required by the challenge:

1) Ingest multi-format docs (PDF/DOCX/XLSX)
2) Chunk and embed
3) Build / load FAISS index
4) Retrieve top-k chunks
5) Generate a grounded answer
6) Run a small evaluation set

The heavy lifting is delegated to `rag_pipeline_combined_2.py` so the
logic here stays easy to follow.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from rag_pipeline_combined_2 import (
    RAGConfig,
    build_faiss_index,
    chunk_documents,
    generate_answer,
    load_documents,
    retrieve_documents,
)

logger = logging.getLogger(__name__)


DEFAULT_TEST_QUERIES = [
    "What is the objective of the Data Scientist II RAG challenge?",
    "Which components are required in the deliverables?",
    "What does the challenge require for cloud deployment?",
    "Summarize the main idea of the 'Attention is All You Need' paper.",
    "What is the EU AI Act about, at a high level?",
    "Which evaluation criteria are used to score the submission?",
    "What are the provided document types for ingestion?",
    "What are the limitations of this prototype?",
]


@dataclass(frozen=True)
class AppConfig:
    files_dir: Path
    persist_dir: Path
    embedding_model: str
    llm_model: str
    llm_task: str | None
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_new_tokens: int
    max_context_chars: int

    def to_rag_config(self) -> RAGConfig:
        return RAGConfig(
            files_dir=self.files_dir,
            persist_dir=self.persist_dir,
            embedding_model=self.embedding_model,
            llm_model=self.llm_model,
            llm_task=self.llm_task,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            top_k=self.top_k,
            max_new_tokens=self.max_new_tokens,
            max_context_chars=self.max_context_chars,
        )


def ingest_and_index(cfg: AppConfig):
    """Load docs, chunk them, then build/load a FAISS index."""
    rag_cfg = cfg.to_rag_config()
    logger.info("Loading documents from %s", rag_cfg.files_dir)
    raw_docs = load_documents(rag_cfg.files_dir)

    logger.info("Chunking %d documents", len(raw_docs))
    chunks = chunk_documents(rag_cfg, raw_docs)

    logger.info("Building/Loading FAISS index at %s", rag_cfg.persist_dir)
    index, docs = build_faiss_index(rag_cfg, chunks)
    return rag_cfg, index, docs


def answer_query(rag_cfg: RAGConfig, index, docs, query: str) -> str:
    """Retrieve top-k chunks and generate an answer."""
    retrieved = retrieve_documents(index, docs, rag_cfg, query, rag_cfg.top_k)
    return generate_answer(rag_cfg, query, retrieved)


def run_evaluation(rag_cfg: RAGConfig, index, docs, queries: List[str]) -> None:
    """Run a small evaluation set and print results."""
    for i, query in enumerate(queries, start=1):
        logger.info("[Eval %d/%d] %s", i, len(queries), query)
        answer = answer_query(rag_cfg, index, docs, query)
        print("\n=== Query ===")
        print(query)
        print("\n=== Answer ===")
        print(answer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpretable RAG skeleton entrypoint")
    parser.add_argument("--files-dir", type=Path, default=Path("."))
    parser.add_argument("--persist-dir", type=Path, default=Path("faiss_store"))
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--llm-model", type=str, default="google/flan-t5-base")
    parser.add_argument("--llm-task", type=str, default=None)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument("--query", type=str, default=None, help="Ask a single question")
    parser.add_argument("--eval", action="store_true", help="Run default evaluation queries")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    cfg = AppConfig(
        files_dir=args.files_dir,
        persist_dir=args.persist_dir,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        llm_task=args.llm_task,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        max_context_chars=args.max_context_chars,
    )

    rag_cfg, index, docs = ingest_and_index(cfg)

    if args.query:
        answer = answer_query(rag_cfg, index, docs, args.query)
        print("\n=== Answer ===")
        print(answer)
        return

    if args.eval:
        run_evaluation(rag_cfg, index, docs, DEFAULT_TEST_QUERIES)
        return

    print("No query provided. Use --query to ask a question or --eval to run the test set.")


if __name__ == "__main__":
    main()
