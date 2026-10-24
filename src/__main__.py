"""CLI entry point built with Python Fire."""
import sys

import fire

from src.answerer import generate_answer, run_answer_dataset
from src.api import run_server
from src.evaluate import run_evaluate
from src.indexer import load_index, run_index
from src.retriever import build_bm25, run_search_dataset, search


class CLI:
    """Command Line Interface for RAG against the machine."""

    def index(
        self,
        max_chunk_size: int = 2000,
        corpus_dir: str = "data/raw",
        with_embeddings: bool = False,
    ) -> None:
        """Ingest corpus files and build the search index.

        Args:
            max_chunk_size: Maximum chunk size in characters.
            corpus_dir: Path to raw files directory.
            with_embeddings: Compute dense embeddings for Bonus 1 & 2.
        """
        run_index(
            corpus_dir=corpus_dir,
            max_chunk_size=max_chunk_size,
            with_embeddings=with_embeddings,
        )

    def search(self, query: str = "", k: int = 10, hybrid: bool = False) -> None:
        """Return the top-k sources for a query.

        Args:
            query: The search query string.
            k: Number of sources to retrieve.
            hybrid: Whether to use hybrid BM25 + dense search (Bonus 2).
        """
        try:
            data = load_index()
            bm25 = build_bm25(data["tokenized"])
            results = search(
                query,
                data["chunks"],
                bm25,
                k=k,
                embeddings=data.get("embeddings"),
                hybrid=hybrid,
            )
            if not results:
                print("No results found.")
                return
            for s in results:
                print(
                    f"{s.file_path} "
                    f"[{s.first_character_index}:{s.last_character_index}]"
                )
        except Exception as e:
            print(f"Error during search: {e}")

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        hybrid: bool = False,
    ) -> None:
        """Run search over a whole dataset and save JSON results.

        Args:
            dataset_path: Path to question dataset JSON.
            k: Number of results per question.
            save_directory: Output directory.
            hybrid: Whether to use hybrid retrieval.
        """
        run_search_dataset(
            dataset_path=dataset_path,
            k=k,
            save_directory=save_directory,
            hybrid=hybrid,
        )

    def answer(self, query: str = "", k: int = 10) -> None:
        """Answer a single query using retrieved context.

        Args:
            query: Question to answer.
            k: Number of sources to retrieve.
        """
        try:
            data = load_index()
            bm25 = build_bm25(data["tokenized"])
            sources = search(query, data["chunks"], bm25, k=k)
            print(generate_answer(query, sources))
        except Exception as e:
            print(f"Error during answer: {e}")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answers for a dataset from search results.

        Args:
            student_search_results_path: Path to StudentSearchResults JSON.
            save_directory: Output directory for answers.
        """
        run_answer_dataset(
            student_search_results_path=student_search_results_path,
            save_directory=save_directory,
        )

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Report recall@k against ground-truth dataset.

        Args:
            student_search_results_path: Path to StudentSearchResults JSON.
            dataset_path: Path to AnsweredQuestions dataset JSON.
        """
        run_evaluate(student_search_results_path, dataset_path)

    def serve(self, port: int = 8000) -> None:
        """Start local HTTP API server (Bonus 5).

        Args:
            port: Local TCP port.
        """
        run_server(port=port)


def main() -> None:
    """CLI entrypoint."""
    try:
        fire.Fire(CLI)
    except SystemExit:
        pass
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
