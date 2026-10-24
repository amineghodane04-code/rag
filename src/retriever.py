"""Retrieval engine supporting BM25, Hybrid search, and Query Caching."""
import json
import os
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.chunking import Chunk
from src.indexer import load_index, tokenize
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

# Bonus 4: Simple in-memory query cache
_QUERY_CACHE: Dict[str, List[MinimalSource]] = {}


def build_bm25(tokenized_corpus: List[List[str]]) -> BM25Okapi:
    """Instantiate BM25Okapi index.

    Args:
        tokenized_corpus: List of token lists for every chunk.

    Returns:
        Prepared BM25Okapi ranker.
    """
    return BM25Okapi(tokenized_corpus)


def _reciprocal_rank_fusion(
    bm25_indices: List[int],
    dense_indices: List[int],
    k_fusion: int = 60,
    top_k: int = 10,
) -> List[int]:
    """Bonus 2: Combine two rankings using Reciprocal Rank Fusion (RRF).

    Args:
        bm25_indices: Chunk indices sorted by BM25.
        dense_indices: Chunk indices sorted by dense similarity.
        k_fusion: Constant smoothing factor.
        top_k: Number of combined indices to return.

    Returns:
        Merged top-k indices.
    """
    scores: Dict[int, float] = {}

    for rank, idx in enumerate(bm25_indices):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k_fusion + rank + 1)

    for rank, idx in enumerate(dense_indices):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k_fusion + rank + 1)

    sorted_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
    return sorted_indices[:top_k]


def search(
    query: str,
    chunks: List[Chunk],
    bm25: BM25Okapi,
    k: int = 10,
    embeddings: Optional[Any] = None,
    hybrid: bool = False,
) -> List[MinimalSource]:
    """Search the index for the top-k most relevant chunks.

    Args:
        query: Query string.
        chunks: List of indexed chunk tuples.
        bm25: BM25 index.
        k: Number of results to return.
        embeddings: Optional dense embeddings matrix.
        hybrid: If True, combine BM25 with dense embeddings.

    Returns:
        List of top-k MinimalSource instances.
    """
    # Degenerate input handling
    if k <= 0 or not query.strip():
        return []

    # Bonus 4: Return from cache if query was already computed
    cache_key = f"{query.strip()}__k={k}__hybrid={hybrid}"
    if cache_key in _QUERY_CACHE:
        return _QUERY_CACHE[cache_key]

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    bm25_scores = bm25.get_scores(query_tokens)
    if not any(s > 0 for s in bm25_scores):
        return []

    ranked_bm25 = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )

    # Bonus 2: Hybrid search if embeddings are available and requested
    final_indices: List[int] = ranked_bm25[:k]
    if hybrid and embeddings is not None:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            q_emb = embedder.encode([query])[0]
            dense_scores = np.dot(embeddings, q_emb) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-9
            )
            ranked_dense = sorted(
                range(len(dense_scores)), key=lambda i: dense_scores[i], reverse=True
            )
            final_indices = _reciprocal_rank_fusion(ranked_bm25, ranked_dense, top_k=k)
        except Exception:
            final_indices = ranked_bm25[:k]

    results = [
        MinimalSource(
            file_path=chunks[i][0],
            first_character_index=chunks[i][1],
            last_character_index=chunks[i][2],
        )
        for i in final_indices
    ]

    # Save to query cache (Bonus 4)
    _QUERY_CACHE[cache_key] = results
    return results


def run_search_dataset(
    dataset_path: str,
    k: int = 10,
    save_directory: str = "data/output/search_results",
    hybrid: bool = False,
) -> None:
    """Execute search over an entire question dataset.

    Args:
        dataset_path: Path to dataset JSON.
        k: Top-k sources per query.
        save_directory: Output directory.
        hybrid: Whether to use hybrid retrieval.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: dataset not found at '{dataset_path}'")
        return

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = RagDataset.model_validate(json.load(f))
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return

    try:
        index_data = load_index()
        chunks: List[Chunk] = index_data["chunks"]
        tokenized: List[List[str]] = index_data["tokenized"]
        embeddings: Optional[Any] = index_data.get("embeddings")
        bm25 = build_bm25(tokenized)
    except Exception as e:
        print(f"Error loading index: {e}")
        return

    results = [
        MinimalSearchResults(
            question_id=q.question_id,
            question=q.question,
            retrieved_sources=search(
                q.question, chunks, bm25, k, embeddings=embeddings, hybrid=hybrid
            ),
        )
        for q in tqdm(dataset.rag_questions, desc="Searching dataset")
    ]

    output = StudentSearchResults(search_results=results, k=k)
    os.makedirs(save_directory, exist_ok=True)
    out_file = os.path.join(save_directory, os.path.basename(dataset_path))
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))
    print(f"Saved student_search_results to {out_file}")
