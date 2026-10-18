#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path

# Ensure moulinette and src are in path
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("moulinette_py"))

from indexing.indexer import RepositoryIndexer
from indexing.searcher import RAGSearcher

from moulinette.evaluate_retrieval import calculate_recall_at_k_on_dataset
from moulinette.validate_student_data import validate_student_data
from moulinette.models import RagDataset, StudentSearchResults

def main():
    print("=" * 60)
    print("🚀 vLLM RAG Indexing & Retrieval Evaluation Pipeline")
    print("=" * 60)

    # 1. Initialize and build index
    indexer = RepositoryIndexer(repo_dir="vllm-0.10.1")
    indexer.build_index(verbose=True)
    
    # Save index to disk
    os.makedirs("output/index", exist_ok=True)
    os.makedirs("output/search_results", exist_ok=True)
    print("\n💾 Saving index state to output/index/...")
    indexer.save("output/index")

    searcher = RAGSearcher(indexer)

    # 2. Evaluate Code Dataset
    code_gt_path = "datasets_public/public/AnsweredQuestions/dataset_code_public.json"
    print(f"\n🔍 Processing Code Questions ({code_gt_path})...")
    code_results = searcher.search_dataset(code_gt_path, k=10, mode="code")
    code_out_path = "output/search_results/dataset_code_public.json"
    with open(code_out_path, "w", encoding="utf-8") as f:
        json.dump(code_results, f, indent=2)
    print(f"  Saved code search results to {code_out_path}")

    # Validate
    student_code_obj = StudentSearchResults(**code_results)
    is_valid_code = validate_student_data(student_code_obj, max_context_length=2000, k=10)
    print(f"  Student Code Data Validated: {is_valid_code}")

    with open(code_gt_path, "r", encoding="utf-8") as f:
        code_gt_obj = RagDataset(**json.load(f))
    
    code_metrics = calculate_recall_at_k_on_dataset(
        student_code_obj, code_gt_obj, minimal_iou_threshold=0.05, k_values=[1, 3, 5, 10]
    )
    code_pass = code_metrics["recall@5"] >= 0.50
    print(f"  Code Pass Criteria (Recall@5 >= 50%): {'✅ PASSED' if code_pass else '❌ FAILED'} ({code_metrics['recall@5']*100:.1f}%)")

    # 3. Evaluate Docs Dataset
    docs_gt_path = "datasets_public/public/AnsweredQuestions/dataset_docs_public.json"
    print(f"\n🔍 Processing Docs Questions ({docs_gt_path})...")
    docs_results = searcher.search_dataset(docs_gt_path, k=10, mode="docs")
    docs_out_path = "output/search_results/dataset_docs_public.json"
    with open(docs_out_path, "w", encoding="utf-8") as f:
        json.dump(docs_results, f, indent=2)
    print(f"  Saved docs search results to {docs_out_path}")

    # Validate
    student_docs_obj = StudentSearchResults(**docs_results)
    is_valid_docs = validate_student_data(student_docs_obj, max_context_length=2000, k=10)
    print(f"  Student Docs Data Validated: {is_valid_docs}")

    with open(docs_gt_path, "r", encoding="utf-8") as f:
        docs_gt_obj = RagDataset(**json.load(f))
    
    docs_metrics = calculate_recall_at_k_on_dataset(
        student_docs_obj, docs_gt_obj, minimal_iou_threshold=0.05, k_values=[1, 3, 5, 10]
    )
    docs_pass = docs_metrics["recall@5"] >= 0.80
    print(f"  Docs Pass Criteria (Recall@5 >= 80%): {'✅ PASSED' if docs_pass else '❌ FAILED'} ({docs_metrics['recall@5']*100:.1f}%)")

    print("\n" + "=" * 60)
    print("🏁 FINAL SUMMARY")
    print("=" * 60)
    print(f"Code Recall@5: {code_metrics['recall@5']*100:.1f}% (Required >= 50%) -> {'PASSED ✅' if code_pass else 'FAILED ❌'}")
    print(f"Docs Recall@5: {docs_metrics['recall@5']*100:.1f}% (Required >= 80%) -> {'PASSED ✅' if docs_pass else 'FAILED ❌'}")
    print("=" * 60)

if __name__ == "__main__":
    main()
