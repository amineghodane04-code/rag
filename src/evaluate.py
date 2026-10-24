"""Evaluation tool calculating Recall@k using character range IoU >= 0.05."""
import json
import os
from typing import Dict, List

from src.models import AnsweredQuestion, MinimalSource, RagDataset, StudentSearchResults


def compute_iou(s1: int, e1: int, s2: int, e2: int) -> float:
    """Compute Intersection over Union of two character spans.

    Args:
        s1: Start index of span 1.
        e1: End index of span 1.
        s2: Start index of span 2.
        e2: End index of span 2.

    Returns:
        IoU value between 0.0 and 1.0.
    """
    intersection = max(0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - intersection
    return intersection / union if union > 0 else 0.0


def is_source_found(truth: MinimalSource, retrieved: List[MinimalSource]) -> bool:
    """Check if truth source has same file and IoU >= 0.05 with any retrieved source.

    Args:
        truth: Ground-truth MinimalSource.
        retrieved: List of candidate retrieved sources.

    Returns:
        True if matched, False otherwise.
    """
    return any(
        r.file_path == truth.file_path
        and compute_iou(
            truth.first_character_index,
            truth.last_character_index,
            r.first_character_index,
            r.last_character_index,
        )
        >= 0.05
        for r in retrieved
    )


def recall_at_k(
    truth_sources: List[MinimalSource], retrieved: List[MinimalSource], k: int
) -> float:
    """Compute fraction of ground-truth sources found in top-k results.

    Args:
        truth_sources: Ground-truth sources for this query.
        retrieved: All retrieved sources for this query.
        k: Cut-off rank.

    Returns:
        Recall score between 0.0 and 1.0.
    """
    if not truth_sources:
        return 1.0
    top_k = retrieved[:k]
    found = sum(1 for s in truth_sources if is_source_found(s, top_k))
    return found / len(truth_sources)


def run_evaluate(student_search_results_path: str, dataset_path: str) -> None:
    """Evaluate student search results against ground-truth dataset.

    Args:
        student_search_results_path: Path to StudentSearchResults JSON.
        dataset_path: Path to AnsweredQuestions dataset JSON.
    """
    if not os.path.exists(student_search_results_path):
        print(f"Error: file not found '{student_search_results_path}'")
        return
    if not os.path.exists(dataset_path):
        print(f"Error: dataset not found at '{dataset_path}'")
        return

    try:
        with open(student_search_results_path, "r", encoding="utf-8") as f:
            student = StudentSearchResults.model_validate(json.load(f))
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = RagDataset.model_validate(json.load(f))
    except Exception as e:
        print(f"Error reading evaluation files: {e}")
        return

    truth_map: Dict[str, List[MinimalSource]] = {
        q.question_id: q.sources
        for q in dataset.rag_questions
        if isinstance(q, AnsweredQuestion)
    }

    for k in [1, 3, 5, 10]:
        scores = [
            recall_at_k(truth_map[item.question_id], item.retrieved_sources, k)
            for item in student.search_results
            if item.question_id in truth_map
        ]
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"Recall@{k}: {avg:.3f}", end="  ")
    print()
