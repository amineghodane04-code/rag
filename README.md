*This project has been created as part of the 42 curriculum by okhouya, hchiar.*

# RAG Against the Machine

A clean, robust, and performant Retrieval-Augmented Generation (RAG) system built from scratch in Python to answer complex technical questions about a codebase (`vLLM`).

---

## Table of Contents
- [Description](#description)
- [System Architecture](#system-architecture)
- [Chunking Strategy](#chunking-strategy)
- [Retrieval Method](#retrieval-method)
- [Performance Analysis](#performance-analysis)
- [Design Decisions & Trade-offs](#design-decisions--trade-offs)
- [Challenges Faced](#challenges-faced)
- [Instructions & Setup](#instructions--setup)
- [Example Usage](#example-usage)
- [Bonus Part (5/5)](#bonus-part-55)
- [Resources & AI Usage](#resources--ai-usage)

---

## Description

Large Language Models (LLMs) have static knowledge cutoffs and frequently hallucinate when queried about proprietary codebases, recent library updates, or specialized internal APIs. Instead of attempting costly fine-tuning or retraining, **Retrieval-Augmented Generation (RAG)** supplies the model with authoritative, real-time context fetched dynamically at inference time.

In this project, the system indexes the entire `vLLM` repository (~2,000 files, tens of thousands of lines of code and documentation). Given an arbitrary technical question, the system retrieves the top-$k$ most relevant text segments and prompts a local, lightweight model (`Qwen/Qwen3-0.6B`) to synthesize an answer strictly grounded in the retrieved sources.

---

## System Architecture

```text
+-----------------------------------------------------------------------------+
|                                1. INGESTION                                 |
|                                                                             |
|   Corpus (data/raw/vllm-0.10.1)                                             |
|          │                                                                  |
|          ▼                                                                  |
|   Chunking Engine (chunking.py)                                             |
|     ├── Python: AST-boundary regex (def/class/@)                            |
|     └── Markdown/Text: Heading & paragraph splitting                        |
|          │                                                                  |
|          ▼                                                                  |
|   Tokenizer & Index Builder (indexer.py)                                    |
|     ├── Sub-word decomposition (camelCase & snake_case)                     |
|     └── Path-token enrichment (filename & directory tokens)                 |
|          │                                                                  |
|          ▼                                                                  |
|   Serialized BM25 Index (data/processed/index.pkl)                          |
+-----------------------------------------------------------------------------+
                                      │
                                      ▼
+-----------------------------------------------------------------------------+
|                                2. RETRIEVAL                                 |
|                                                                             |
|   Query ("How to configure OpenAI server?")                                 |
|          │                                                                  |
|          ▼                                                                  |
|   BM25Okapi Ranker + Optional Hybrid / Cache (retriever.py)                 |
|     ├── Scored over inverted index & cached queries                         |
|     └── Top-k MinimalSource locations (file_path, start_idx, end_idx)       |
+-----------------------------------------------------------------------------+
                                      │
                                      ▼
+-----------------------------------------------------------------------------+
|                          3. AUGMENTATION & ANSWER                           |
|                                                                             |
|   Context Formatter (answerer.py)                                           |
|     └── Extract source spans within token budget (<= 4000 chars)            |
|          │                                                                  |
|          ▼                                                                  |
|   Local LLM (Qwen/Qwen3-0.6B)                                               |
|     ├── System prompt enforcing strict grounding                            |
|     ├── Thinking tags disabled (<think>...</think> stripped)                |
|     └── Structured JSON output conforming to Pydantic models                |
+-----------------------------------------------------------------------------+
```

### Modular Components
1. **`src/models.py`**: Pydantic v2 data models ensuring strict schema validation and JSON compliance.
2. **`src/chunking.py`**: Domain-specific document segmenters respecting code semantics and document hierarchy while strictly guaranteeing chunk spans $\le 2000$ characters.
3. **`src/indexer.py`**: Traverses the corpus, creates text chunks with exact file offsets, extracts enriched tokens, supports incremental updates, and persists the index.
4. **`src/retriever.py`**: Evaluates queries against the corpus using BM25Okapi ranking, query caching, and optional dense vector hybrid fusion.
5. **`src/answerer.py`**: Formats retrieved evidence into a grounded prompt and manages CPU inference with `Qwen/Qwen3-0.6B`.
6. **`src/evaluate.py`**: Computes Recall@k using character-range Intersection over Union ($\text{IoU} \ge 0.05$) matching against ground-truth source spans.
7. **`src/api.py`**: Lightweight local HTTP server for interactive querying.
8. **`src/__main__.py`**: Exposes the complete CLI using Python Fire with graceful exception handling and `tqdm` progress monitoring.

---

## Chunking Strategy

Standard fixed-character chunking breaks code and markdown sentences mid-word or mid-function, destroying syntactic and semantic coherence. This project implements two specialized chunking strategies:

### 1. Python Code Chunking
- **Logical boundaries**: Splits source files at top-level construct boundaries (`def `, `class `, `async def `, `@decorator`).
- **Packing**: Consecutive small blocks are packed into a single chunk up to `--max_chunk_size` (default: 2000 characters).
- **Oversized construct splitting**: If an individual function or class exceeds 2000 characters, it is cleanly subdivided along newline boundaries.
- **Span preservation**: The start and end character offsets in the original file are tracked exactly.

### 2. Markdown & Documentation Chunking
- **Logical boundaries**: Splits documents on Markdown header boundaries (`# `, `## `, `### `) and blank lines.
- **Packing**: Related paragraphs and subheadings are merged up to the 2000-character ceiling.
- **Guarantee**: No chunk ever exceeds 2000 characters, ensuring 100% compliance with the moulinette context window threshold.

---

## Retrieval Method

### Algorithm: BM25Okapi
The retrieval engine uses BM25Okapi, a probabilistic term-matching algorithm superior to basic TF-IDF. 
BM25 calculates the relevance of a document $D$ to a query $Q$ using:

$$\text{score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

- **Term frequency saturation ($k_1 = 1.5$)**: Repeated occurrences of a term provide diminishing returns, avoiding domination by files that repeat a keyword.
- **Document length normalization ($b = 0.75$)**: Compensates for long files by normalizing against average chunk length.

### Identifier Tokenization & Path Enrichment
1. **Compound splitting**:
   - `snake_case` words are decomposed (e.g., `openai_api_server` yields `openai`, `api`, `server`, and the full string).
   - `camelCase` words are decomposed (e.g., `apiServer` yields `api`, `server`).
2. **Filepath token enrichment**:
   - Directory and file names (e.g., `vllm/entrypoints/openai/api_server.py`) are tokenized and appended to the chunk's bag of words. This guarantees that queries asking about specific files or modules score highly even if the chunk text itself does not repeat the filename.

---

## Performance Analysis

All benchmarks were evaluated on the reference public datasets on standard campus-equivalent CPU hardware.

### 1. Retrieval Quality (Recall@k)
The evaluation metric measures the fraction of ground-truth sources retrieved in the top-$k$ results with $\text{IoU} \ge 0.05$ within the exact source file.

| Dataset | Metric | Subject Threshold | Achieved Score | Margin / Status |
|---|---|:---:|:---:|:---:|
| **Documentation (`dataset_docs_public.json`)** | Recall@1 | - | **62.0%** | - |
| **Documentation (`dataset_docs_public.json`)** | Recall@3 | - | **77.0%** | - |
| **Documentation (`dataset_docs_public.json`)** | **Recall@5** | **$\ge$ 80%** | **82.0%** | **+2.0% (PASSED)** |
| **Documentation (`dataset_docs_public.json`)** | Recall@10 | - | **87.0%** | - |
| **Source Code (`dataset_code_public.json`)** | Recall@1 | - | **40.4%** | - |
| **Source Code (`dataset_code_public.json`)** | Recall@3 | - | **56.6%** | - |
| **Source Code (`dataset_code_public.json`)** | **Recall@5** | **$\ge$ 50%** | **69.7%** | **+19.7% (PASSED)** |
| **Source Code (`dataset_code_public.json`)** | Recall@10 | - | **78.8%** | - |

### 2. Runtime Benchmarks

| Metric | Allowed Maximum | Measured Performance | Status |
|---|:---:|:---:|:---:|
| **Full Corpus Indexing Time** | $\le 300\text{ s}$ (5 min) | **~2.6 seconds** | **PASSED (115x faster)** |
| **Retrieval Throughput (200 questions)** | $\le 90\text{ s}$ | **~12.7 seconds** | **PASSED (7x faster)** |

---

## Design Decisions & Trade-offs

1. **BM25 vs. Dense Vector Embeddings (for baseline mandatory retrieval)**:
   - *Choice*: BM25Okapi over neural embeddings for baseline.
   - *Rationale*: BM25 indexes the entire 2,000-file repository in under 3 seconds on CPU without multi-gigabyte models, while achieving 82% docs recall and 69.7% code recall.
2. **Sub-word Tokenization without Heavy Dependencies**:
   - *Choice*: Regex-based identifier splitting (`snake_case` + `camelCase`) directly in pure Python.
   - *Rationale*: Eliminates external heavy tokenizer overhead while preserving exact variable names alongside component keywords.
3. **Chunk Size Capped at 2000 Characters**:
   - *Choice*: Default of 2000 characters packed to maximum density.
   - *Trade-off*: Smaller chunks increase precision but fragment function contexts; packing to 2000 characters maximized Recall@5.
4. **Context Budgeting for `Qwen3-0.6B`**:
   - *Choice*: Truncate context at 4,000 characters and disable generation `<think>` tokens.
   - *Rationale*: Keeps prompt token length small enough to ensure fast CPU inference while maintaining sufficient grounding text.

---

## Challenges Faced

1. **Exact Character Index Preservation**:
   - When splitting blocks and packing them, whitespace between consecutive blocks could cause character offsets to drift.
   - *Solution*: Maintained exact file offsets by reading the raw file content once and calculating start and end offsets directly against original file slices.
2. **Handling Degenerate and Malformed Inputs**:
   - An empty query `""`, random gibberish `"asdfghjkl"`, or missing files previously risked unhandled exceptions or returning random files with score 0.0.
   - *Solution*: Wrapped all CLI endpoints in defensive try/except blocks, checked for positive BM25 scores before returning results, and validated datasets with Pydantic.
3. **Mypy Static Typing & Stubs**:
   - Third-party packages (`rank_bm25`, `fire`) lack inline type stubs.
   - *Solution*: Configured `types-tqdm` and explicit type overrides in `pyproject.toml`, achieving full passing status under `mypy . --strict`.

---

## Instructions & Setup

### Requirements
- Python 3.10+
- `uv` package manager

### Installation

Clone the repository and install all dependencies:
```bash
uv sync
```

### Makefile Rules

The included `Makefile` provides the standard interface:
- `make install`: Install project dependencies using `uv sync`.
- `make run ARGS="<command>"`: Execute the main script via Python interpreter.
- `make debug ARGS="<command>"`: Run the main script under Python's built-in `pdb` debugger.
- `make clean`: Clean temporary cache files (`__pycache__`, `.mypy_cache`).
- `make lint`: Run code quality checks (`flake8` and `mypy` with project flags).
- `make lint-strict`: Run strict type checking (`flake8` and `mypy --strict`).

---

## Example Usage

### 1. Ingest & Index the Corpus
```bash
uv run python -m src index --max_chunk_size 2000
```

### 2. Single Query Search
```bash
uv run python -m src search "How to configure OpenAI server?" --k 10
```

### 3. Batch Search over a Dataset
```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

### 4. Single Query Answer Generation
```bash
uv run python -m src answer "How to configure OpenAI server?" --k 10
```

### 5. Batch Answer Generation over Search Results
```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

### 6. Evaluate Retrieval Recall Locally
```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

---

## Bonus Part (5/5)

1. **Semantic Embeddings**:
   Build vector index with CPU model `all-MiniLM-L6-v2`:
   ```bash
   uv run python -m src index --with_embeddings True
   ```
2. **Hybrid Retrieval**:
   Combines BM25 and semantic rankings using Reciprocal Rank Fusion (RRF):
   ```bash
   uv run python -m src search "How to configure OpenAI server?" --k 10 --hybrid True
   ```
3. **Incremental Indexing**:
   Re-indexing tracks file modification times (`mtime`); if files are unchanged, indexing completes in $<0.1$ seconds.
4. **Caching**:
   Query results and index objects are cached in-memory, making repeated queries instantaneous.
5. **Local HTTP API**:
   Launch a local HTTP API server exposing `/search` and `/answer`:
   ```bash
   uv run python -m src serve --port 8000
   ```
   Query it via `curl`:
   ```bash
   curl "http://127.0.0.1:8000/search?q=How+to+configure+OpenAI+server&k=2"
   ```

---

## Resources & AI Usage

### References
- Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Information Retrieval, 11(4).
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
- Qwen Team (2024). *Qwen2.5 / Qwen3 Technical Report*. Alibaba Group.
- Pydantic Documentation: https://docs.pydantic.dev/
- Python Fire Guide: https://github.com/google/python-fire

### AI Usage Disclosure
- **Brainstorming & Architecture**: AI was used to compare BM25 parameter tuning ($k_1, b$) and explore regex patterns for sub-word token splitting.
- **Code Linting & Types**: AI assisted in verifying untyped third-party library boundaries for `mypy --strict`.
- **Validation**: All logic, character span indexing, IoU calculations, and CLI commands were thoroughly tested and benchmarked against the reference public datasets.
