"""
Command-line interface (CLI) for the RAG against the machine project.

Exposes the mandatory CLI commands built with Python Fire as required by
Chapter VI.6:
  - index --max_chunk_size <int>
  - search <query> --k <int>
  - search_dataset --dataset_path <path> --k <int> --save_directory <dir>
  - answer <query> --k <int>
  - answer_dataset --student_search_results_path <path> --save_directory <dir>
  - evaluate --student_search_results_path <path> --dataset_path <path>
"""

import os
import sys
import json
from typing import Optional, List, Any, Literal
from tqdm import tqdm
import fire

from .models import (
    MinimalSource,
    MinimalSearchResults,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from .indexing.indexer import RepositoryIndexer
from .indexing.searcher import RAGSearcher
from .evaluation.evaluator import calculate_recall_at_k


class RAGCLI:
    """Main CLI class exposed via Python Fire."""

    def __init__(self) -> None:
        """Initialize CLI with shared state."""
        self._indexer: Optional[RepositoryIndexer] = None
        self._generator: Optional[Any] = None

    def _get_indexer(
        self, processed_dir: str = "data/processed"
    ) -> RepositoryIndexer:
        """Load or create repository indexer."""
        if self._indexer is None:
            self._indexer = RepositoryIndexer(processed_dir=processed_dir)
            self._indexer.load(processed_dir)
        return self._indexer

    def _get_generator(self, model_name: str = "Qwen/Qwen3-0.6B") -> Any:
        """Load or create answer generator."""
        from .generation.generator import AnswerGenerator

        if (
            self._generator is None
            or getattr(self._generator, "model_name", None) != model_name
        ):
            self._generator = AnswerGenerator(model_name=model_name)
        return self._generator

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ) -> None:
        """
        Ingest data/raw/ and build the index under data/processed/.

        Args:
            max_chunk_size: Maximum chunk size (default: 2000 characters).
            raw_dir: Source corpus directory (default: data/raw).
            processed_dir: Directory where processed indices are stored.
        """
        try:
            print(
                f"Ingesting corpus from '{raw_dir}' (max_chunk_size={max_chunk_size})..."
            )
            indexer = RepositoryIndexer(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                max_chunk_size=max_chunk_size,
            )
            stats = indexer.build_index(verbose=True)
            indexer.save(processed_dir)
            print(
                f"Ingestion complete! Indexed {stats['unified_chunks']} chunks "
                f"under {processed_dir}/"
            )
        except Exception as e:
            print(f"Error during indexing: {e}", file=sys.stderr)

    def search(
        self,
        query: str = "",
        k: int = 10,
        processed_dir: str = "data/processed",
    ) -> None:
        """
        Return the top-k sources for a single query.

        Args:
            query: The search query string.
            k: Number of source locations to return.
            processed_dir: Path to directory containing indexed data.
        """
        if not query or not query.strip():
            print("Empty query provided. Please provide a valid search query.")
            return

        if k <= 0:
            print("k must be greater than 0.")
            return

        try:
            indexer = self._get_indexer(processed_dir=processed_dir)
            searcher = RAGSearcher(indexer)
            res = searcher.search_question(
                question_id="cli_query", question=query, k=k
            )

            sources = res.get("retrieved_sources", [])
            if not sources:
                print("No relevant sources found.")
                return

            for src in sources:
                fp = src["file_path"]
                first = src["first_character_index"]
                last = src["last_character_index"]
                print(f"{fp} [{first}:{last}]")
        except Exception as e:
            print(f"Error during search: {e}", file=sys.stderr)

    def search_dataset(
        self,
        dataset_path: str = "",
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """
        Run search over a whole dataset and write a StudentSearchResults JSON file.

        Args:
            dataset_path: Path to dataset JSON file.
            k: Maximum number of sources per question (default: 10).
            save_directory: Output directory where results JSON is saved.
        """
        if not dataset_path:
            print("Error: --dataset_path is required.", file=sys.stderr)
            return

        if not os.path.exists(dataset_path):
            print(
                f"Error: Dataset file not found at '{dataset_path}'",
                file=sys.stderr,
            )
            return

        if k <= 0:
            print("Error: k must be greater than 0.", file=sys.stderr)
            return

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error: Failed to parse dataset JSON: {e}", file=sys.stderr)
            return

        questions = data.get("rag_questions", [])
        if not questions:
            print("Warning: Dataset contains no questions.")

        indexer = self._get_indexer()
        searcher = RAGSearcher(indexer)

        # Detect dataset domain to select optimal index
        is_code = "code" in os.path.basename(dataset_path).lower()
        search_mode: Literal["code", "docs", "unified"] = (
            "code" if is_code else "docs"
        )

        results_list: List[MinimalSearchResults] = []
        for q in tqdm(questions, desc="Searching questions", unit="q"):
            qid = q.get("question_id", "")
            qtext = q.get("question", "")
            raw_res = searcher.search_question(
                question_id=qid, question=qtext, k=k, mode=search_mode
            )
            sources = [
                MinimalSource(
                    file_path=s["file_path"],
                    first_character_index=s["first_character_index"],
                    last_character_index=s["last_character_index"],
                )
                for s in raw_res.get("retrieved_sources", [])
            ]
            results_list.append(
                MinimalSearchResults(
                    question_id=qid, question=qtext, retrieved_sources=sources
                )
            )

        student_output = StudentSearchResults(search_results=results_list, k=k)

        os.makedirs(save_directory, exist_ok=True)
        filename = os.path.basename(dataset_path)
        out_path = os.path.join(save_directory, filename)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(student_output.model_dump_json(indent=2))

        print(f"Saved student_search_results to {out_path}")

    def answer(
        self,
        query: str = "",
        k: int = 10,
        processed_dir: str = "data/processed",
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """
        Answer a single query using the retrieved context.

        Args:
            query: Question string.
            k: Number of retrieved sources to use (default: 10).
            processed_dir: Path to directory containing indexed data.
            model_name: Model identifier for generation.
        """
        if not query or not query.strip():
            print("Empty query provided. Please provide a valid question.")
            return

        try:
            indexer = self._get_indexer(processed_dir=processed_dir)
            searcher = RAGSearcher(indexer)
            res = searcher.search_question(
                question_id="cli_query", question=query, k=k
            )
            sources = res.get("retrieved_sources", [])

            generator = self._get_generator(model_name=model_name)
            ans = generator.generate_answer(question=query, sources=sources)
            print(f"\nAnswer:\n{ans}")
        except Exception as e:
            print(f"Error during answering: {e}", file=sys.stderr)

    def answer_dataset(
        self,
        student_search_results_path: str = "",
        save_directory: str = "data/output/search_results_and_answer",
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """
        Generate answers for a dataset, producing a StudentSearchResultsAndAnswer JSON file.

        Args:
            student_search_results_path: Path to StudentSearchResults JSON file.
            save_directory: Output directory for saving answers.
            model_name: Generation model to use.
        """
        if not student_search_results_path:
            print(
                "Error: --student_search_results_path is required.",
                file=sys.stderr,
            )
            return

        if not os.path.exists(student_search_results_path):
            print(
                f"Error: Search results file not found at '{student_search_results_path}'",
                file=sys.stderr,
            )
            return

        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                search_data = json.load(f)
        except Exception as e:
            print(
                f"Error: Failed to parse search results JSON: {e}",
                file=sys.stderr,
            )
            return

        results = search_data.get("search_results", [])
        k = search_data.get("k", 10)
        generator = self._get_generator(model_name=model_name)

        answered_list: List[MinimalAnswer] = []
        print(f"Loaded {len(results)} questions ...")

        for item in tqdm(results, desc="Generating answers", unit="q"):
            qid = item.get("question_id", "")
            qtext = item.get("question", "")
            raw_sources = item.get("retrieved_sources", [])

            sources = [
                MinimalSource(
                    file_path=s["file_path"],
                    first_character_index=s["first_character_index"],
                    last_character_index=s["last_character_index"],
                )
                for s in raw_sources
            ]

            ans = generator.generate_answer(
                question=qtext, sources=raw_sources
            )
            answered_list.append(
                MinimalAnswer(
                    question_id=qid,
                    question=qtext,
                    retrieved_sources=sources,
                    answer=ans,
                )
            )

        output_obj = StudentSearchResultsAndAnswer(
            search_results=answered_list, k=k
        )

        os.makedirs(save_directory, exist_ok=True)
        filename = os.path.basename(student_search_results_path)
        out_path = os.path.join(save_directory, filename)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output_obj.model_dump_json(indent=2))

        print(f"Saved student_search_results_and_answer to {out_path}")

    def evaluate(
        self, student_search_results_path: str = "", dataset_path: str = ""
    ) -> None:
        """
        Report recall@k against a ground-truth dataset for local testing.

        Args:
            student_search_results_path: Path to student_search_results JSON.
            dataset_path: Path to ground-truth AnsweredQuestions dataset JSON.
        """
        if not student_search_results_path or not dataset_path:
            print(
                "Error: Both --student_search_results_path and --dataset_path are required.",
                file=sys.stderr,
            )
            return

        if not os.path.exists(student_search_results_path):
            print(
                f"File not found: {student_search_results_path}",
                file=sys.stderr,
            )
            return

        if not os.path.exists(dataset_path):
            print(f"File not found: {dataset_path}", file=sys.stderr)
            return

        try:
            metrics = calculate_recall_at_k(
                student_results_path=student_search_results_path,
                dataset_path=dataset_path,
                minimal_iou_threshold=0.05,
                k_values=[1, 3, 5, 10],
            )

            print("\n🎯 Local Evaluation Results")
            print("========================================")
            for k in [1, 3, 5, 10]:
                val = metrics.get(f"recall@{k}", 0.0)
                print(f"Recall@{k}: {val:.3f} ({val * 100:.1f}%)")
        except Exception as e:
            print(f"Error during evaluation: {e}", file=sys.stderr)


def main() -> None:
    """Entrypoint function for Fire CLI."""
    fire.Fire(RAGCLI)


if __name__ == "__main__":
    main()
