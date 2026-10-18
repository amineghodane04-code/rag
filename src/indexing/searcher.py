import json
from typing import Dict, Any, Literal
from .indexer import RepositoryIndexer
from .tokenizer import tokenize_for_bm25


class RAGSearcher:
    """
    Searcher for retrieving sources for RAG questions.
    Outputs results strictly adhering to Moulinette's StudentSearchResults schema.
    """

    def __init__(self, indexer: RepositoryIndexer):
        self.indexer = indexer

    def search_question(
        self,
        question_id: str,
        question: str,
        k: int = 10,
        mode: Literal["code", "docs", "unified"] = "unified",
    ) -> Dict[str, Any]:
        """Search top-k sources for a single question."""
        tokens = tokenize_for_bm25(question)

        if mode == "code":
            assert self.indexer.bm25_code is not None, "Code index not built"
            scores = self.indexer.bm25_code.get_scores(tokens)
            chunks = self.indexer.code_chunks
        elif mode == "docs":
            assert self.indexer.bm25_docs is not None, "Docs index not built"
            scores = self.indexer.bm25_docs.get_scores(tokens)
            chunks = self.indexer.doc_chunks
        else:
            assert (
                self.indexer.bm25_unified is not None
            ), "Unified index not built"
            scores = self.indexer.bm25_unified.get_scores(tokens)
            chunks = self.indexer.unified_chunks

        # Get top-k indices
        top_indices = scores.argsort()[::-1][:k]

        retrieved_sources = []
        for idx in top_indices:
            c = chunks[idx]
            retrieved_sources.append(
                {
                    "file_path": c["file_path"],
                    "first_character_index": c["first_character_index"],
                    "last_character_index": c["last_character_index"],
                }
            )

        return {
            "question_id": question_id,
            "question": question,
            "retrieved_sources": retrieved_sources,
        }

    def search_dataset(
        self,
        questions_file: str,
        k: int = 10,
        mode: Literal["code", "docs", "unified"] = "unified",
    ) -> Dict[str, Any]:
        """Search top-k sources for all questions in a dataset JSON file."""
        with open(questions_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = data.get("rag_questions", [])
        search_results = []

        for q in questions:
            res = self.search_question(
                question_id=q["question_id"],
                question=q["question"],
                k=k,
                mode=mode,
            )
            search_results.append(res)

        return {"search_results": search_results, "k": k}
