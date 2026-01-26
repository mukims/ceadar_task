"""
RAG Pipeline Example
====================

This script implements a simple Retrieval‑Augmented Generation (RAG) pipeline
tailored for offline environments where heavy neural embeddings or external
language models may not be available.  The goal is to demonstrate the
end‑to‑end flow from document ingestion and indexing through retrieval and
answer synthesis.

Key Features
------------
* **Multi‑format ingestion** – support for PDF, DOCX and XLSX files.  PDF
  pages are extracted via PyMuPDF (`fitz`), DOCX is parsed by reading its
  internal XML, and Excel is converted into human‑readable sentences using
  `pandas`.  Embedded hyperlinks in DOCX are preserved as part of the
  extracted text.
* **Chunking and preprocessing** – each document is broken into smaller
  chunks (e.g. pages or paragraphs) to allow fine‑grained retrieval.  The
  text is normalised by collapsing whitespace and lower‑casing while still
  retaining mathematical symbols, tables and figure captions.
* **Vectorisation with TF–IDF** – instead of semantic embeddings requiring
  external models, the script uses scikit‑learn’s `TfidfVectorizer` to
  represent chunks and queries in a high‑dimensional term space.  This
  lexical approach is robust and does not require network access.
* **Similarity search** – similarity between a user query and document
  chunks is computed via cosine similarity.  The top‑K relevant chunks are
  returned as context for the generation step.
* **Simple generation** – to synthesise an answer, the script implements a
  naive extractive summariser.  Sentences from the retrieved context are
  ranked by term overlap with the query, and the top sentences are
  concatenated.  This mechanism avoids dependency on large language models
  while still producing a coherent response.

How to Use
----------
1. Place the documents you wish to index in a directory.  Supported
   extensions are `.pdf`, `.docx`, and `.xlsx`.
2. Run `build_index()` specifying the directory; it will return a
   `RAGEngine` instance containing the indexed data.
3. Call `engine.answer_query(query, top_k=3, max_sentences=3)` to retrieve
   relevant information and generate a short answer.

Example:

```python
from rag_pipeline import build_index

# Build an index from sample files in the `data` directory
engine = build_index("/path/to/data")

# Ask a question
response = engine.answer_query(
    "What does the challenge ask you to do?",
    top_k=3,
    max_sentences=4,
)
print(response)
```

The returned string contains the synthesised answer along with the
corresponding source identifiers.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import fitz  # PyMuPDF for PDF processing
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# A simple sentence splitter that avoids NLTK's external dependencies.
def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation.

    This naive implementation splits on a period, exclamation mark or question
    mark followed by whitespace.  It does not handle abbreviations but
    avoids the need for NLTK data downloads.

    Parameters
    ----------
    text : str
        The input text.

    Returns
    -------
    List[str]
        A list of sentences with surrounding whitespace stripped.
    """
    # Ensure text is a string and normalise newlines
    if not text:
        return []
    text = text.replace('\n', ' ')
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def extract_text_from_pdf(path: str) -> List[Tuple[str, Dict[str, str]]]:
    """Extract text from a PDF file page by page.

    Each page constitutes a separate chunk with metadata capturing the page
    number.  Figures, tables and mathematical formulae are preserved as
    they appear in the text extracted by PyMuPDF.

    Parameters
    ----------
    path : str
        Path to the PDF file.

    Returns
    -------
    List[Tuple[str, Dict[str, str]]]
        A list of tuples `(text, metadata)` where `metadata` contains the
        source file name and page number.
    """
    doc = fitz.open(path)
    chunks = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            metadata = {"source": os.path.basename(path), "page": str(page_num)}
            chunks.append((text, metadata))
    return chunks


def extract_text_from_docx(path: str) -> List[Tuple[str, Dict[str, str]]]:
    """Extract text from a DOCX file.

    This function reads the internal XML of the Word document rather than
    relying on external libraries.  It preserves hyperlinks by including
    their targets inline in the text.  Each paragraph becomes a separate
    chunk.

    Parameters
    ----------
    path : str
        Path to the DOCX file.

    Returns
    -------
    List[Tuple[str, Dict[str, str]]]
        A list of paragraph texts with metadata including the paragraph
        index.
    """
    import zipfile
    import xml.etree.ElementTree as ET

    chunks: List[Tuple[str, Dict[str, str]]] = []
    with zipfile.ZipFile(path) as z:
        with z.open('word/document.xml') as doc_xml:
            tree = ET.parse(doc_xml)
            root = tree.getroot()
            ns = {
                'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
                'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
            }
            rels = {}
            try:
                with z.open('word/_rels/document.xml.rels') as rels_file:
                    rels_tree = ET.parse(rels_file)
                    for rel in rels_tree.getroot():
                        rel_id = rel.attrib.get('Id')
                        rel_target = rel.attrib.get('Target')
                        if rel_id and rel_target:
                            rels[rel_id] = rel_target
            except KeyError:
                pass
            paragraphs = root.findall('.//w:p', ns)
            for i, p in enumerate(paragraphs):
                texts: List[str] = []
                for run in p.findall('.//w:r', ns):
                    text_elems = run.findall('.//w:t', ns)
                    text_content = ''.join(t.text for t in text_elems if t.text)
                    texts.append(text_content)
                for hl in p.findall('.//w:hyperlink', ns):
                    anchor = ''.join(t.text for t in hl.findall('.//w:t', ns) if t.text)
                    rel_id = hl.attrib.get(f'{{{ns["r"]}}}id')
                    if rel_id and rel_id in rels:
                        texts.append(f" {anchor} ({rels[rel_id]}) ")
                paragraph_text = ' '.join(texts).strip()
                paragraph_text = re.sub(r"\s+", " ", paragraph_text)
                if paragraph_text:
                    metadata = {
                        "source": os.path.basename(path),
                        "paragraph": str(i + 1)
                    }
                    chunks.append((paragraph_text, metadata))
    return chunks


def extract_text_from_excel(path: str) -> List[Tuple[str, Dict[str, str]]]:
    """Convert an Excel file into textual chunks.

    Each row is converted into a descriptive sentence where column names
    and cell values are concatenated.  Only numeric or string values are
    included.  This provides a human‑readable representation suitable for
    indexing.

    Parameters
    ----------
    path : str
        Path to the XLSX file.

    Returns
    -------
    List[Tuple[str, Dict[str, str]]]
        A list of row descriptions with metadata including the sheet name
        and row number.
    """
    chunks = []
    excel = pd.ExcelFile(path)
    for sheet_name in excel.sheet_names:
        df = excel.parse(sheet_name)
        for idx, row in df.iterrows():
            parts = []
            for col, value in row.items():
                if pd.isna(value):
                    continue
                if isinstance(value, (int, float)):
                    value_str = str(value)
                else:
                    value_str = str(value)
                parts.append(f"{col}: {value_str}")
            if parts:
                text = '; '.join(parts)
                metadata = {
                    "source": os.path.basename(path),
                    "sheet": sheet_name,
                    "row": str(idx + 1)
                }
                chunks.append((text, metadata))
    return chunks


def ingest_documents(directory: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Walk through a directory and ingest all supported documents.

    Supported extensions: `.pdf`, `.docx`, `.xlsx`.

    Parameters
    ----------
    directory : str
        Directory containing documents.

    Returns
    -------
    Tuple[List[str], List[Dict[str, str]]]
        A tuple of (texts, metadata) lists.
    """
    texts: List[str] = []
    metadata: List[Dict[str, str]] = []
    for root, _, files in os.walk(directory):
        for fname in files:
            fpath = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            try:
                if ext == '.pdf':
                    chunks = extract_text_from_pdf(fpath)
                elif ext == '.docx':
                    chunks = extract_text_from_docx(fpath)
                elif ext == '.xlsx':
                    chunks = extract_text_from_excel(fpath)
                else:
                    continue
                for text, meta in chunks:
                    texts.append(text)
                    metadata.append(meta)
            except Exception as e:
                print(f"Warning: failed to ingest {fname}: {e}")
    return texts, metadata


@dataclass
class RAGEngine:
    """A simple retrieval and generation engine."""
    texts: List[str]
    metadata: List[Dict[str, str]]
    vectorizer: TfidfVectorizer = field(init=False)
    embeddings: any = field(init=False)

    def __post_init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        self.embeddings = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, Dict[str, str], float]]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.embeddings).flatten()
        if top_k <= 0:
            top_k = 1
        idxs = sims.argsort()[::-1][:top_k]
        results = []
        for idx in idxs:
            results.append((self.texts[idx], self.metadata[idx], float(sims[idx])))
        return results

    @staticmethod
    def _score_sentence(sentence: str, query_tokens: set) -> float:
        words = set(re.findall(r"\w+", sentence.lower()))
        if not words:
            return 0.0
        overlap = query_tokens.intersection(words)
        return len(overlap) / len(query_tokens) if query_tokens else 0.0

    def answer_query(self, query: str, top_k: int = 3, max_sentences: int = 4) -> str:
        retrieved = self.retrieve(query, top_k)
        sentences: List[Tuple[str, Dict[str, str]]] = []
        query_tokens = set(re.findall(r"\w+", query.lower()))
        for text, meta, _ in retrieved:
            for sent in split_into_sentences(text):
                sentences.append((sent, meta))
        scored = [
            (self._score_sentence(sent, query_tokens), sent, meta)
            for (sent, meta) in sentences
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen: List[Tuple[str, Dict[str, str]]] = []
        for score, sent, meta in scored:
            if score == 0.0:
                continue
            chosen.append((sent.strip(), meta))
            if len(chosen) >= max_sentences:
                break
        if not chosen and retrieved:
            fallback_text, fallback_meta, _ = retrieved[0]
            first_sentences = split_into_sentences(fallback_text)
            first_sentence = first_sentences[0] if first_sentences else fallback_text
            chosen = [(first_sentence, fallback_meta)]
        answer_sentences = [c[0] for c in chosen]
        sources = [
            f"{c[1]['source']} (section {c[1].get('page', c[1].get('paragraph', c[1].get('row', '?')))})"
            for c in chosen
        ]
        answer = ' '.join(answer_sentences)
        answer += "\n\nSources: " + '; '.join(sources)
        return answer


def build_index(directory: str) -> RAGEngine:
    texts, metadata = ingest_documents(directory)
    if not texts:
        raise ValueError(f"No supported documents found in {directory}")
    return RAGEngine(texts=texts, metadata=metadata)