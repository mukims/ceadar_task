Python File Contents Report
Date: 2026-01-26

Total .py files: 7

## app.py
Module docstring: (none)
Objective: Streamlit web UI that runs the RAG pipeline with cached indexing, retrieval, and answer generation.
Imports:
os, pathlib:Path, typing:List,Tuple, numpy, streamlit, rag_pipeline_combined_2:RAGConfig,_build_prompt,_import_embeddings,_import_pipeline,_infer_llm_task,build_faiss_index,chunk_documents,load_documents
Classes: _Seq2SeqGenerator
Functions: _l2_normalize, _load_index, _load_embedding_model, _is_seq2seq_model, _load_generator, _retrieve_documents, _generate_answer

## main.py
Module docstring:
Interpretable RAG Skeleton
==========================

Objective: CLI entrypoint that orchestrates ingestion, indexing, retrieval, generation, and optional evaluation using `rag_pipeline_combined_2.py`.
Imports:
__future__:annotations, argparse, logging, dataclasses:dataclass, pathlib:Path, typing:List, rag_pipeline_combined_2:RAGConfig,build_faiss_index,chunk_documents,generate_answer,load_documents,retrieve_documents
Classes: AppConfig
Functions: ingest_and_index, answer_query, run_evaluation, parse_args, main

## rag_minimal.py
Module docstring:
Minimal RAG pipeline for testing token budget effects.

- Ingest PDF/DOCX/XLSX
Objective: Minimal RAG pipeline to test token budget constraints with naive chunking and optional generation.
Imports:
__future__:annotations, argparse, logging, dataclasses:dataclass, pathlib:Path, typing:List,Tuple, numpy, rag_pipeline_combined_2:Document,load_documents
Classes: Config
Functions: chunk_documents, _l2_normalize, embed_texts, build_faiss_index, retrieve, _truncate_to_tokens, build_prompt, generate_answer, parse_args, main

## rag_pipeline.py
Module docstring: (none)
Objective: Baseline end-to-end RAG pipeline (ingestion, chunking, embeddings, FAISS retrieval, and generation) for PDF/DOCX/XLSX.
Imports:
argparse, logging, re, shutil, dataclasses:dataclass, pathlib:Path, typing:Any,List,Optional,Tuple, warnings, docx:Document, langchain_core.documents:Document, numpy, openpyxl:load_workbook, pypdf:PdfReader
Classes: RAGConfig
Functions: _patch_importlib_metadata, _import_text_splitter, _import_embeddings, _import_faiss, _import_pipeline, _import_pdfplumber, _import_fitz, _import_pytesseract, _import_pil_image, _equations, _split_text_in_blocks, _multi_line_math, _extract_tables_from_page, _clean_table, _table_to_documents, _figure_captions_from_blocks, _ocr_image, _load_pdf_basic, _load_pdf, _load_docx, _load_xlsx, load_documents, chunk_documents, _faiss_index_path, _faiss_docs_path, _l2_normalize, _encode_chunks, _save_docs, _load_docs, build_faiss_index, retrieve_documents, _build_prompt, _infer_llm_task, generate_answer, run_pipeline, parse_args, main

## rag_pipeline_2.py
Module docstring:
RAG Pipeline Example
====================

Objective: Offline RAG example using TF-IDF retrieval and extractive summarization without heavy neural models.
Imports:
os, re, dataclasses:dataclass,field, typing:List,Tuple,Dict, fitz, pandas, sklearn.feature_extraction.text:TfidfVectorizer, sklearn.metrics.pairwise:cosine_similarity
Classes: RAGEngine
Functions: split_into_sentences, extract_text_from_pdf, extract_text_from_docx, extract_text_from_excel, ingest_documents, build_index

## rag_pipeline_combined.py
Module docstring:
Extended RAG Pipeline
=====================

Objective: Extended RAG pipeline with richer PDF parsing (equations, tables, figures, OCR) plus FAISS retrieval and LLM generation.
Imports:
argparse, logging, dataclasses:dataclass, pathlib:Path, typing:List,Optional,Dict,Any,Tuple, warnings, numpy, openpyxl:load_workbook, re
Classes: RAGConfig
Functions: _patch_importlib_metadata, _import_text_splitter, _import_embeddings, _import_faiss, _import_pipeline, equations, split_text_in_blocks, multi_line_math, extract_tables_from_page, clean_table, table_to_docs, figure_captions_from_blocks, ocr_image, ingest_pdf_to_documents, _load_pdf, _load_docx, _load_xlsx, load_documents, chunk_documents, _faiss_index_path, _faiss_docs_path, _l2_normalize, _encode_chunks, _save_docs, _load_docs, build_faiss_index, retrieve_documents, _build_prompt, _infer_llm_task, generate_answer, run_pipeline, parse_args, main

## rag_pipeline_combined_2.py
Module docstring:
Extended RAG Pipeline
=====================

Objective: Primary extended RAG pipeline with rich PDF parsing, FAISS persistence/compat checks, and LLM generation.
Imports:
argparse, logging, dataclasses:dataclass, functools:lru_cache, pathlib:Path, typing:List,Optional,Dict,Any,Tuple, warnings, numpy, openpyxl:load_workbook, re
Classes: RAGConfig
Functions: _patch_importlib_metadata, _import_text_splitter, _import_embeddings, _import_faiss, _import_pipeline, equations, split_text_in_blocks, multi_line_math, extract_tables_from_page, clean_table, table_to_docs, figure_captions_from_blocks, ocr_image, ingest_pdf_to_documents, _load_pdf, _load_docx, _load_xlsx, load_documents, chunk_documents, _faiss_index_path, _faiss_docs_path, _faiss_meta_path, _l2_normalize, _encode_chunks, _save_docs, _load_docs, _load_index_meta, _is_index_compatible, _save_index_meta, build_faiss_index, retrieve_documents, _build_prompt, _infer_llm_task, _is_seq2seq_model, _load_seq2seq_model, generate_answer, run_pipeline, parse_args, main
