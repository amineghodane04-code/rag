"""Index builder for BM25 and optional semantic vector embeddings."""
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from src.chunking import Chunk, chunk_file, read_file

SUPPORTED_EXTENSIONS = {".py", ".md", ".txt"}
INDEX_DIR = "data/processed"
INDEX_FILE = "index.pkl"


def tokenize(text: str) -> List[str]:
    """Split text into words, plus snake_case and camelCase subwords.

    Args:
        text: Input string to tokenize.

    Returns:
        List of extracted tokens.
    """
    raw_words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    tokens: List[str] = []

    for word in raw_words:
        tokens.append(word)
        # Split snake_case (e.g. lora_adapter -> lora, adapter)
        if "_" in word:
            parts = [p for p in word.split("_") if p]
            if len(parts) > 1:
                tokens.extend(parts)

        # Split camelCase (e.g. apiServer -> api, server)
        camel_parts = re.findall(r"[a-z]+|[A-Z][a-z]*|[0-9]+", word)
        if len(camel_parts) > 1:
            tokens.extend([p.lower() for p in camel_parts])

    return tokens


def collect_corpus_files(corpus_dir: str) -> List[str]:
    """Find all indexable files in the corpus directory.

    Args:
        corpus_dir: Root directory of raw corpus.

    Returns:
        Sorted list of valid file paths.
    """
    return sorted(
        str(p)
        for p in Path(corpus_dir).rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_index(
    corpus_dir: str = "data/raw",
    max_chunk_size: int = 2000,
    compute_embeddings: bool = False,
) -> Tuple[List[Chunk], List[List[str]], Optional[Any], Dict[str, float]]:
    """Ingest corpus files and build the search indices.

    Args:
        corpus_dir: Path to corpus files.
        max_chunk_size: Maximum character length per chunk.
        compute_embeddings: If True, generate dense vector embeddings.

    Returns:
        Tuple of (chunks, tokenized_texts, optional_embeddings, file_mtimes).
    """
    files = collect_corpus_files(corpus_dir)
    print(f"Found {len(files)} files to index.")

    # Bonus 3: Check existing index for incremental update
    previous_data: Dict[str, Any] = {}
    old_file_mtimes: Dict[str, float] = {}
    index_path = os.path.join(INDEX_DIR, INDEX_FILE)

    if os.path.exists(index_path):
        try:
            with open(index_path, "rb") as f:
                previous_data = pickle.load(f)
                old_file_mtimes = previous_data.get("file_mtimes", {})
        except Exception:
            previous_data = {}

    current_mtimes = {f: os.path.getmtime(f) for f in files}

    # If nothing changed, return previous index immediately (Bonus 3)
    if previous_data and old_file_mtimes == current_mtimes:
        print("Incremental check: No files changed. Using cached index.")
        return (
            previous_data["chunks"],
            previous_data["tokenized"],
            previous_data.get("embeddings"),
            current_mtimes,
        )

    # Chunk all files
    all_chunks: List[Chunk] = []
    for file_path in tqdm(files, desc="Chunking files"):
        content = read_file(file_path)
        if content.strip():
            all_chunks.extend(chunk_file(file_path, content, max_chunk_size))

    print(f"Created {len(all_chunks)} chunks.")

    # Tokenize each chunk + file path for BM25
    tokenized: List[List[str]] = []
    for file_path, _start, _end, chunk_text in tqdm(all_chunks, desc="Tokenizing chunks"):
        path_tokens = tokenize(" ".join(Path(file_path).parts))
        tokens = tokenize(chunk_text) + path_tokens
        tokenized.append(tokens)

    # Bonus 1: Optional Semantic Embeddings
    embeddings = None
    if compute_embeddings:
        try:
            from sentence_transformers import SentenceTransformer

            print("Bonus 1: Computing semantic embeddings with all-MiniLM-L6-v2...")
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            chunk_texts = [c[3] for c in all_chunks]
            embeddings = embedder.encode(chunk_texts, show_progress_bar=True)
        except Exception as e:
            print(f"Notice: Semantic embedding generation skipped: {e}")

    return all_chunks, tokenized, embeddings, current_mtimes


def save_index(
    chunks: List[Chunk],
    tokenized: List[List[str]],
    embeddings: Optional[Any],
    file_mtimes: Dict[str, float],
    index_dir: str = INDEX_DIR,
) -> None:
    """Save the index data to disk.

    Args:
        chunks: List of Chunk tuples.
        tokenized: Pre-tokenized corpus.
        embeddings: Optional dense embeddings array.
        file_mtimes: Dictionary of file modification times.
        index_dir: Target directory for saving.
    """
    os.makedirs(index_dir, exist_ok=True)
    out_file = os.path.join(index_dir, INDEX_FILE)
    payload = {
        "chunks": chunks,
        "tokenized": tokenized,
        "embeddings": embeddings,
        "file_mtimes": file_mtimes,
    }
    with open(out_file, "wb") as f:
        pickle.dump(payload, f)
    print(f"Index saved to {out_file}")


def load_index(index_dir: str = INDEX_DIR) -> Dict[str, Any]:
    """Load index dictionary from disk.

    Args:
        index_dir: Directory containing index.pkl.

    Returns:
        Index payload dictionary.

    Raises:
        FileNotFoundError: If index.pkl is not found.
    """
    path = os.path.join(index_dir, INDEX_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index not found at {path}. Run 'index' first.")
    with open(path, "rb") as f:
        data: Dict[str, Any] = pickle.load(f)
        return data


def run_index(
    corpus_dir: str = "data/raw",
    max_chunk_size: int = 2000,
    with_embeddings: bool = False,
) -> None:
    """Index command implementation.

    Args:
        corpus_dir: Root directory of raw files.
        max_chunk_size: Maximum allowed chunk size.
        with_embeddings: If True, also build dense embeddings.
    """
    if not os.path.isdir(corpus_dir):
        print(f"Error: corpus directory '{corpus_dir}' not found.")
        return
    try:
        chunks, tokenized, embeddings, mtimes = build_index(
            corpus_dir=corpus_dir,
            max_chunk_size=max_chunk_size,
            compute_embeddings=with_embeddings,
        )
        save_index(chunks, tokenized, embeddings, mtimes)
        print("Ingestion complete! Indices saved under data/processed/")
    except Exception as e:
        print(f"Error during indexing: {e}")
