# AI Agent Instructions for Cedar RAG Task

## Project Overview
This is a **Retrieval Augmented Generation (RAG)** implementation challenge. The system builds a semantic search pipeline to extract information from PDF documents, chunk them intelligently, compute embeddings, and enable similarity-based retrieval.

## Architecture & Data Flow
The pipeline follows this sequence (all code is in `task1.ipynb`):

1. **PDF Ingestion** → `extract_text_from_pdfs()` - Reads all pages from PDFs using pypdf library
2. **Corpus Creation** → Store raw text in `corpus.jsonl` (path: `corpus_path()`)
3. **Tokenization & Chunking** → Split text respecting token limits with overlap using `AutoTokenizer` from transformers
4. **Embedding Generation** → Convert chunks to vectors using `SentenceTransformer` (path: `embeddings_path()`)
5. **Indexing** → Build FAISS index for similarity search (path: `faiss_index_path()`)
6. **Storage** → Save chunks metadata in `chunks.jsonl` and `metadata.jsonl`

## Key Configuration Object
The `config` dataclass (frozen, immutable) centralizes all parameters:
```python
@dataclass(frozen=True)
class config:
    files_dir: Path              # Source PDF directory
    output_dir: Path             # Where to save corpus, chunks, embeddings, index
    max_tokens: int              # Token limit per chunk
    token_overlap: int           # Sliding window overlap for chunks
    tokenizer_name: str          # HuggingFace tokenizer ID (e.g., "gpt2")
    embedding_model: str         # SentenceTransformer model name
    llm_model: str               # LLM for final retrieval/generation
```
This is passed throughout the pipeline; always preserve immutability when modifying.

## Output File Artifacts
All outputs are JSONL or binary with path helpers:
- `corpus.jsonl` → Full extracted text
- `chunks.jsonl` → Tokenized document chunks
- `embeddings.npy` → NumPy array of embeddings
- `metadata.jsonl` → Chunk metadata (source, position, etc.)
- `faiss_index.index` → FAISS binary index for vector similarity

Path helpers auto-create output directories: `corpus_path()`, `chunks_path()`, `embeddings_path()`, `metadata_path()`, `faiss_index_path()`.

## Dependencies & Models
- **PDF Handling**: `pypdf` (PdfReader, PdfWriter)
- **Tokenization**: `transformers.AutoTokenizer` (HuggingFace)
- **Embeddings**: `sentence_transformers.SentenceTransformer`
- **Indexing**: `FAISS` (for vector similarity search)
- **Data**: `pandas`, `numpy`

Models are lazy-loaded via HuggingFace model IDs in config—assume first-run downloads to cache.

### Recommended Model Configuration
- **Tokenizer**: `"gpt2"` or `"bert-base-uncased"` - Fast, lightweight, suitable for chunking logic
- **Embedding Model**: `"all-MiniLM-L6-v2"` - 384-dim vectors, excellent quality/speed for semantic search
- **LLM Model**: For retrieval-augmented generation (future stages)—consider `"gpt2"` for inference

These balance latency and quality for RAG; adjust if embedding dimensionality or inference speed becomes a constraint.

## Common Workflows

### Tokenization & Chunking Strategy
The `token_overlap` parameter enables context preservation between chunks. Implementation flow:
1. Load tokenizer via `AutoTokenizer.from_pretrained(config.tokenizer_name)`
2. Tokenize full corpus: `tokenizer.encode(corpus_text)`
3. Split token list into chunks respecting `max_tokens`
4. Create overlapping windows: for each chunk, prepend the last `token_overlap` tokens from previous chunk
5. Decode overlapped token sequences back to text strings
6. Save to `chunks.jsonl` (one JSON object per line with chunk text and metadata)

Example chunk metadata structure:
```json
{"chunk_id": 0, "text": "...", "source_file": "document.pdf", "page": 1, "token_count": 512}
```

### FAISS Indexing & Similarity Search
Once embeddings are computed:
1. Load embeddings from `embeddings.npy` as 2D NumPy array (shape: `num_chunks × embedding_dim`)
2. Create FAISS index: `faiss.IndexFlatL2(embedding_dim)` for L2 distance or `IndexFlatIP` for cosine
3. Add vectors: `index.add(embeddings)`
4. Save index: `faiss.write_index(index, str(faiss_index_path()))`
5. **Query workflow**: encode query text → compute embedding → `index.search(query_embedding, k)` → returns k nearest neighbors with distances
6. Retrieve full chunk text using returned indices into `chunks.jsonl`

### Output File Formats

**corpus.jsonl** - Raw extracted text:
```json
{"source": "Attention_is_all_you_need.pdf", "page": 1, "text": "...", "length": 1250}
```

**chunks.jsonl** - Tokenized document chunks:
```json
{"chunk_id": 0, "text": "...", "source_file": "doc.pdf", "page": 2, "token_count": 512, "start_token": 0, "end_token": 512}
{"chunk_id": 1, "text": "...", "source_file": "doc.pdf", "page": 2, "token_count": 512, "start_token": 256, "end_token": 768}
```
(Notice overlapping token ranges due to `token_overlap`)

**metadata.jsonl** - Embedding metadata for FAISS linkage:
```json
{"chunk_id": 0, "embedding_index": 0, "embedding_dim": 384, "model": "all-MiniLM-L6-v2"}
```

**embeddings.npy** - Binary NumPy array:
- Shape: `(num_chunks, embedding_dim)` - e.g., `(1000, 384)` for 1000 chunks with 384-dim vectors
- Dtype: `float32`
- Index alignment: row `i` in array = chunk with `chunk_id = i`

### Extending the Pipeline
When adding new stages (e.g., reranking, filtering):
1. Add corresponding `*_path()` helper for output artifacts
2. Accept config as parameter to access model names and hyperparameters
3. Follow JSONL format for human-readable intermediate outputs
4. Ensure chunking respects `max_tokens` and `token_overlap` from config

### Handling Text Extraction Failures
`extract_text_from_pdfs()` uses `page.extract_text() or ""` to gracefully handle corrupted pages. If enhancing, maintain this pattern to avoid pipeline breakage.

## Project-Specific Patterns

- **Immutable Config**: Never modify `config` after instantiation; create new instances if needed
- **Path-First Output**: Always use path helpers to maintain consistent artifact locations
- **JSONL Format**: Store structured data (chunks, metadata) line-by-line for streaming/scanning
- **Type Hints**: Code uses Path types and return type annotations—maintain consistency
- **Error Handling**: Graceful defaults (empty strings for failed extractions) over exceptions

## Testing & Iteration
The notebook is incomplete (cells 3-6 are unexecuted). When implementing:
1. Execute cells sequentially to test each pipeline stage
2. Verify output files exist at expected `*_path()` locations
3. Inspect JSONL files to ensure chunk quality and metadata integrity
4. Validate embedding dimensions match the chosen model

## Challenge Context
See `datascientistiichallenge/Data Scientist II Challenge Instructions.pdf` for full requirements and evaluation criteria.

---
*Last updated: January 2026 | Framework: Jupyter Notebook with HuggingFace/SentenceTransformers/FAISS*
