import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st

from rag_pipeline_combined_2 import (
    RAGConfig,
    _build_prompt,
    _import_embeddings,
    _import_pipeline,
    _infer_llm_task,
    build_faiss_index,
    chunk_documents,
    load_documents,
)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


@st.cache_resource(show_spinner=False)
def _load_index(
    files_dir: str,
    persist_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
) -> Tuple[RAGConfig, object, List[object]]:
    cfg = RAGConfig(
        files_dir=Path(files_dir),
        persist_dir=Path(persist_dir),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
    )
    raw_docs = load_documents(cfg.files_dir)
    chunks = chunk_documents(cfg, raw_docs)
    index, docs = build_faiss_index(cfg, chunks)
    return cfg, index, docs


@st.cache_resource(show_spinner=False)
def _load_embedding_model(model_name: str):
    SentenceTransformer = _import_embeddings()
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False)
def _load_generator(model_name: str, task_override: str | None):
    task = task_override or _infer_llm_task(model_name)
    pipeline = _import_pipeline()
    generator = pipeline(task, model=model_name)
    return generator, task


def _retrieve_documents(index, docs, model, query: str, top_k: int):
    query_emb = model.encode([query], convert_to_numpy=True)
    query_emb = _l2_normalize(query_emb.astype(np.float32))
    distances, indices = index.search(query_emb, top_k)
    results = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(docs):
            continue
        results.append(docs[idx])
    return results


def _generate_answer(generator, task: str, query: str, docs, max_new_tokens: int, max_context_chars: int) -> str:
    max_len = getattr(generator.tokenizer, "model_max_length", 512)
    if not isinstance(max_len, int) or max_len > 100000:
        max_len = 512
    input_budget = max_len - max_new_tokens
    if input_budget < 16:
        input_budget = max_len
    prompt = _build_prompt(
        query,
        docs,
        max_context_chars,
        tokenizer=generator.tokenizer,
        max_input_tokens=input_budget,
    )
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


st.set_page_config(page_title="Cedar RAG Demo", layout="wide")

st.title("Cedar RAG Prototype")
st.write("Retrieval-augmented QA over the provided PDFs/DOCX/XLSX files.")

with st.sidebar:
    st.header("Configuration")
    files_dir = st.text_input("Files directory", os.getenv("FILES_DIR", "."))
    persist_dir = st.text_input("FAISS directory", os.getenv("PERSIST_DIR", "faiss_store"))
    embedding_model = st.text_input(
        "Embedding model",
        os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )
    llm_model = st.text_input("LLM model", os.getenv("LLM_MODEL", "google/flan-t5-base"))
    llm_task = st.text_input("LLM task override (optional)", os.getenv("LLM_TASK", "")).strip() or None
    top_k = st.slider("Top K", 1, 8, int(os.getenv("TOP_K", "5")))
    chunk_size = st.slider("Chunk size", 300, 2000, int(os.getenv("CHUNK_SIZE", "1000")), 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, int(os.getenv("CHUNK_OVERLAP", "150")), 10)
    max_new_tokens = st.slider("Max new tokens", 64, 512, int(os.getenv("MAX_NEW_TOKENS", "256")), 16)
    max_context_chars = st.slider("Max context chars", 1000, 8000, int(os.getenv("MAX_CONTEXT_CHARS", "4000")), 250)

query = st.text_input("Ask a question")
run = st.button("Run")

if run and query.strip():
    files_path = Path(files_dir)
    if not files_path.exists():
        st.error(f"Files directory not found: {files_path}")
        st.stop()
    with st.spinner("Loading index and models..."):
        try:
            cfg, index, docs = _load_index(
                str(files_path),
                str(Path(persist_dir)),
                int(chunk_size),
                int(chunk_overlap),
                embedding_model,
            )
        except Exception as exc:
            st.error(f"Failed to load documents or index: {exc}")
            st.stop()
        embed_model = _load_embedding_model(embedding_model)
        generator, task = _load_generator(llm_model, llm_task)
    with st.spinner("Retrieving context..."):
        retrieved = _retrieve_documents(index, docs, embed_model, query, top_k)
    with st.spinner("Generating answer..."):
        answer = _generate_answer(generator, task, query, retrieved, int(max_new_tokens), int(max_context_chars))

    st.subheader("Answer")
    st.write(answer if answer else "No answer generated.")

    st.caption(f"Indexed chunks: {len(docs)} | Retrieved: {len(retrieved)}")

    st.subheader("Retrieved context")
    for i, doc in enumerate(retrieved, start=1):
        meta = doc.metadata or {}
        source = meta.get("filename") or meta.get("source") or "unknown"
        label = f"[{i}] {source}"
        with st.expander(label):
            st.code(doc.page_content[:2000])
            st.json(meta)
elif run:
    st.warning("Enter a question to run the query.")
