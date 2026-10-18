"""
Repository indexer module for chunking and lexical indexing.

Ingests the codebase and documentation, processes files with AST and Markdown
chunkers, and persists BM25 indices to disk according to Chapter VI.1.
"""

import os
import json
import pickle
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from .code_chunker import CodeChunker
from .doc_chunker import DocChunker
from .tokenizer import tokenize_for_bm25


class RepositoryIndexer:
    """
    Main Indexer for the vLLM repository.
    Discovers relevant source and documentation files, chunks them with
    exact character tracking, tokenizes for lexical search, and builds
    searchable indices.
    """

    ROOT_FILES = ["CMakeLists.txt", "RELEASE.md", "SECURITY.md", "README.md"]

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        max_chunk_size: int = 1800,
        prefix_path: str = "data/raw/vllm-0.10.1",
    ):
        """
        Initialize the repository indexer.

        Args:
            raw_dir: Root directory containing raw sources.
            processed_dir: Directory where processed indices are stored.
            max_chunk_size: Maximum chunk size in characters (strictly <= 2000).
            prefix_path: Prefix path written to retrieved sources.
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        # Ensure max_chunk_size does not exceed 2000
        safe_max_size = min(max_chunk_size, 1950)
        self.code_chunker = CodeChunker(
            max_chunk_len=safe_max_size, prefix_path=prefix_path
        )
        self.doc_chunker = DocChunker(
            max_chunk_len=safe_max_size, prefix_path=prefix_path
        )
        self.prefix_path = prefix_path

        self.code_chunks: List[Dict[str, Any]] = []
        self.doc_chunks: List[Dict[str, Any]] = []
        self.unified_chunks: List[Dict[str, Any]] = []

        self.bm25_code: Optional[BM25Okapi] = None
        self.bm25_docs: Optional[BM25Okapi] = None
        self.bm25_unified: Optional[BM25Okapi] = None

    def _resolve_vllm_dir(self) -> str:
        """Resolve the path to the vllm repository root."""
        candidate = os.path.join(self.raw_dir, "vllm-0.10.1")
        if os.path.exists(candidate):
            return candidate
        if os.path.exists(self.raw_dir):
            return self.raw_dir
        return "vllm-0.10.1"

    def build_index(self, verbose: bool = True) -> Dict[str, int]:
        """Index all files in scope and build BM25 indices."""
        self.code_chunks.clear()
        self.doc_chunks.clear()
        repo_dir = self._resolve_vllm_dir()

        # 1. Index Code: vllm/
        vllm_dir = os.path.join(repo_dir, "vllm")
        py_files: List[str] = []
        if os.path.exists(vllm_dir):
            for root, _, files in os.walk(vllm_dir):
                for f in files:
                    if f.endswith(".py"):
                        py_files.append(os.path.join(root, f))

        iterator = (
            tqdm(py_files, desc="Chunking Code", unit="file")
            if verbose
            else py_files
        )
        for full_p in iterator:
            rel_p = os.path.relpath(full_p, repo_dir)
            try:
                with open(
                    full_p, "r", encoding="utf-8", errors="ignore"
                ) as fp:
                    content = fp.read()
                chunks = self.code_chunker.chunk_file(content, rel_p)
                self.code_chunks.extend(chunks)
            except Exception:
                pass

        # 2. Index Docs: docs/
        docs_dir = os.path.join(repo_dir, "docs")
        doc_files: List[str] = []
        if os.path.exists(docs_dir):
            for root, _, files in os.walk(docs_dir):
                for f in files:
                    if f.endswith(".md"):
                        doc_files.append(os.path.join(root, f))

        doc_iter = (
            tqdm(doc_files, desc="Chunking Docs", unit="file")
            if verbose
            else doc_files
        )
        for full_p in doc_iter:
            rel_p = os.path.relpath(full_p, repo_dir)
            try:
                with open(
                    full_p, "r", encoding="utf-8", errors="ignore"
                ) as fp:
                    content = fp.read()
                chunks = self.doc_chunker.chunk_file(content, rel_p)
                self.doc_chunks.extend(chunks)
            except Exception:
                pass

        # 3. Index Root files (CMakeLists.txt, RELEASE.md, SECURITY.md,
        # README.md)
        for rf in self.ROOT_FILES:
            rf_path = os.path.join(repo_dir, rf)
            if os.path.exists(rf_path):
                try:
                    with open(
                        rf_path, "r", encoding="utf-8", errors="ignore"
                    ) as fp:
                        content = fp.read()
                    chunks = self.doc_chunker.chunk_file(content, rf)
                    self.doc_chunks.extend(chunks)
                except Exception:
                    pass

        self.unified_chunks = self.code_chunks + self.doc_chunks

        # Build BM25 models
        if verbose:
            print(f"Tokenizing {len(self.unified_chunks)} chunks for BM25...")

        code_corpus = [
            tokenize_for_bm25(c["enriched_text"]) for c in self.code_chunks
        ]
        self.bm25_code = BM25Okapi(code_corpus)

        docs_corpus = [
            tokenize_for_bm25(c["enriched_text"]) for c in self.doc_chunks
        ]
        self.bm25_docs = BM25Okapi(docs_corpus)

        unified_corpus = [
            tokenize_for_bm25(c["enriched_text"]) for c in self.unified_chunks
        ]
        self.bm25_unified = BM25Okapi(unified_corpus)

        return {
            "code_chunks": len(self.code_chunks),
            "doc_chunks": len(self.doc_chunks),
            "unified_chunks": len(self.unified_chunks),
        }

    def save(self, output_dir: Optional[str] = None) -> None:
        """Persist chunks and tokenized data to disk."""
        target_dir = output_dir or self.processed_dir
        os.makedirs(target_dir, exist_ok=True)

        with open(
            os.path.join(target_dir, "code_chunks.jsonl"),
            "w",
            encoding="utf-8",
        ) as f:
            for c in self.code_chunks:
                f.write(json.dumps(c) + "\n")

        with open(
            os.path.join(target_dir, "doc_chunks.jsonl"), "w", encoding="utf-8"
        ) as f:
            for c in self.doc_chunks:
                f.write(json.dumps(c) + "\n")

        with open(os.path.join(target_dir, "indexer_state.pkl"), "wb") as f:
            pickle.dump(
                {
                    "code_chunks": self.code_chunks,
                    "doc_chunks": self.doc_chunks,
                    "unified_chunks": self.unified_chunks,
                    "bm25_code": self.bm25_code,
                    "bm25_docs": self.bm25_docs,
                    "bm25_unified": self.bm25_unified,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load(self, input_dir: Optional[str] = None) -> None:
        """Load pre-built indices from disk."""
        target_dir = input_dir or self.processed_dir
        state_file = os.path.join(target_dir, "indexer_state.pkl")
        if not os.path.exists(state_file):
            raise FileNotFoundError(
                f"Index not found at {state_file}. Please run index command first."
            )

        with open(state_file, "rb") as f:
            state = pickle.load(f)

        self.code_chunks = state["code_chunks"]
        self.doc_chunks = state["doc_chunks"]
        self.unified_chunks = state["unified_chunks"]
        self.bm25_code = state["bm25_code"]
        self.bm25_docs = state["bm25_docs"]
        self.bm25_unified = state["bm25_unified"]
