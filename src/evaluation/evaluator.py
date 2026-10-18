"""
Local evaluation module for calculating Recall@k.

Evaluates student search results against ground-truth datasets using the
Intersection-over-Union (IoU) metric without relying on external binaries,
strictly adhering to the evaluation rules specified in Chapter VII.
"""

import json
from typing import Dict, List, Any, Optional


def calculate_iou(start1: int, end1: int, start2: int, end2: int) -> float:
    """
    Calculate 1D Intersection-over-Union (IoU) between two character spans.

    Args:
        start1: First span start offset.
        end1: First span end offset.
        start2: Second span start offset.
        end2: Second span end offset.

    Returns:
        float: IoU value in range [0.0, 1.0].
    """
    intersection = max(0, min(end1, end2) - max(start1, start2))
    len1 = max(0, end1 - start1)
    len2 = max(0, end2 - start2)
    union = len1 + len2 - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def compare_sources(pred: Dict[str, Any], truth: Dict[str, Any]) -> float:
    """
    Compare a predicted source to a ground-truth source.

    Args:
        pred: Predicted source with file_path and character offsets.
        truth: Ground-truth source with file_path and character offsets.

    Returns:
        float: IoU overlap if file paths match exactly, else 0.0.
    """
    if pred.get("file_path") != truth.get("file_path"):
        return 0.0

    p_start = pred.get("first_character_index", -1)
    p_end = pred.get("last_character_index", -1)
    t_start = truth.get("first_character_index", -1)
    t_end = truth.get("last_character_index", -1)

    return calculate_iou(p_start, p_end, t_start, t_end)


def calculate_recall_at_k(
    student_results_path: str,
    dataset_path: str,
    minimal_iou_threshold: float = 0.05,
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Calculate Recall@k across all questions in a dataset.

    Args:
        student_results_path: Path to student_search_results.json.
        dataset_path: Path to ground truth dataset.json.
        minimal_iou_threshold: Minimum IoU overlap (default: 0.05).
        k_values: List of k values to compute (default: [1, 3, 5, 10]).

    Returns:
        Dict[str, float]: Average Recall@k scores for each k.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    with open(student_results_path, "r", encoding="utf-8") as f:
        student_data = json.load(f)

    with open(dataset_path, "r", encoding="utf-8") as f:
        ground_truth_data = json.load(f)

    pred_map = {
        res["question_id"]: res.get("retrieved_sources", [])
        for res in student_data.get("search_results", [])
    }

    recall_accumulator: Dict[int, List[float]] = {k: [] for k in k_values}
    questions = ground_truth_data.get("rag_questions", [])

    for q in questions:
        qid = q.get("question_id")
        true_sources = q.get("sources", [])
        pred_sources = pred_map.get(qid, [])

        if not true_sources:
            for k in k_values:
                recall_accumulator[k].append(1.0)
            continue

        for k in k_values:
            top_k_preds = pred_sources[:k]
            matched_sources = 0

            for t_src in true_sources:
                hit = False
                for p_src in top_k_preds:
                    if compare_sources(p_src, t_src) > minimal_iou_threshold:
                        hit = True
                        break
                if hit:
                    matched_sources += 1

            recall = matched_sources / len(true_sources)
            recall_accumulator[k].append(recall)

    avg_results: Dict[str, float] = {}
    for k in k_values:
        scores = recall_accumulator[k]
        avg = sum(scores) / len(scores) if scores else 0.0
        avg_results[f"recall@{k}"] = avg

    return avg_results
